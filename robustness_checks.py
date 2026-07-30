"""Robustness checks for gate-surviving factors.

robustness_checks v1.0 (2026-07-30)
(a) Value-weighted (mktcap at formation month t) Q5-Q1 long-short.
(b) EW Q5-Q1 excluding imputed observations. A NaN in the raw batch2
    checkpoint means that stock-month was industry-median imputed.
    Only ASSET_GROWTH and REV_1M among survivors are batch2 factors;
    BM and CFP (batch1) have no raw checkpoint -- noted as a limitation.
Baseline = saved EW series from ls_returns.parquet, same stats applied.
Output: data/processed/robustness_results.csv
"""
import resource
import sys

import numpy as np
import pandas as pd

from multiple_testing import ff5_alpha

VERSION = "robustness_checks v1.0 (2026-07-30)"
PROC = "data/processed"
N_BINS = 5
MIN_STOCKS_PER_MONTH = 100  # must match run_factor_tests.py
SURVIVORS = {"BM": 1, "ASSET_GROWTH": -1, "REV_1M": -1, "CFP": 1}
BATCH2 = ["ASSET_GROWTH", "REV_1M"]


def add_forward_returns(df):
    """Replicates run_factor_tests.py exactly."""
    df = df.sort_values(["permno", "date"], kind="mergesort").reset_index(drop=True)
    df["_midx"] = (df["date"].dt.year * 12 + df["date"].dt.month).astype("int32")
    g = df.groupby("permno", sort=False)
    df["ret_fwd"] = g["ret"].shift(-1).where(g["_midx"].shift(-1) - df["_midx"] == 1)
    return df.drop(columns=["_midx"])


def ls_series(df, zcol, weighted=False, mask=None):
    cols = ["date", "ret_fwd", zcol] + (["mktcap"] if weighted else [])
    sub = (df if mask is None else df.loc[mask])[cols].dropna()
    out = {}
    for dt, grp in sub.groupby("date", sort=True):
        if len(grp) < MIN_STOCKS_PER_MONTH:
            continue
        try:
            bins = pd.qcut(grp[zcol], N_BINS, labels=False, duplicates="drop")
        except ValueError:
            continue
        if bins.nunique() < N_BINS:
            continue
        if weighted:
            w = grp["mktcap"]
            m = grp["ret_fwd"].mul(w).groupby(bins).sum() / w.groupby(bins).sum()
        else:
            m = grp["ret_fwd"].groupby(bins).mean()
        out[dt] = float(m.loc[N_BINS - 1] - m.loc[0])
    return pd.Series(out).sort_index()


def stats(s, sign, fac, variant, ff):
    r = (s * sign).rename("ret")
    r.index = pd.PeriodIndex(pd.to_datetime(r.index), freq="M") + 1  # t -> t+1
    res = ff5_alpha(r, ff, use_momentum=True)
    return {"factor": fac, "variant": variant, "months": len(r),
            "ann_mean": r.mean() * 12,
            "ann_SR": r.mean() / r.std() * np.sqrt(12),
            "ff5mom_alpha_ann": res["alpha_annual"],
            "alpha_t": res["alpha_tstat"]}


print(VERSION)
ff = pd.read_parquet("data/raw/ff_factors_monthly.parquet")
ff.index = pd.PeriodIndex(pd.to_datetime(ff["date"]), freq="M")
ren = {"mkt_rf": "Mkt-RF", "smb": "SMB", "hml": "HML", "rmw": "RMW",
       "cma": "CMA", "mom": "MOM"}
ff = ff.rename(columns=ren)[list(ren.values())]  # RF excluded: zero-investment

usecols = ["permno", "date", "ret", "mktcap"] + [f + "_z" for f in SURVIVORS]
df = pd.read_parquet(f"{PROC}/factors_all.parquet", columns=usecols)
df = add_forward_returns(df)
print(f"panel: {len(df):,} rows, ret_fwd available "
      f"{df['ret_fwd'].notna().mean():.1%}")

raw = pd.read_parquet(f"{PROC}/_ckpt_batch2_raw.parquet",
                      columns=["permno", "date"] + BATCH2)
raw = raw.rename(columns={f: f + "_raw" for f in BATCH2})
df = df.merge(raw, on=["permno", "date"], how="left", validate="1:1")

rows = []
base = pd.read_parquet(f"{PROC}/ls_returns.parquet")
for fac, sign in SURVIVORS.items():
    rows.append(stats(base[fac].dropna(), sign, fac, "EW_baseline", ff))
    rows.append(stats(ls_series(df, fac + "_z", weighted=True), sign,
                      fac, "VW", ff))
    if fac in BATCH2:
        zc, rc = fac + "_z", fac + "_raw"
        share = (df[zc].notna() & df[rc].isna()).sum() / df[zc].notna().sum()
        print(f"{fac}: excluding {share:.1%} imputed obs")
        rows.append(stats(ls_series(df, zc, mask=df[rc].notna()), sign,
                          fac, "EW_no_imputed", ff))

out = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print("\n=== Robustness summary (flipped to documented direction) ===")
print(out.round(3).to_string(index=False))
out.to_csv(f"{PROC}/robustness_results.csv", index=False)
print(f"\nsaved -> {PROC}/robustness_results.csv")
rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
print(f"peak RSS: {rss_gb:.2f} GB")
