"""Microcap-exclusion robustness: EW quintiles among stocks at or above
the NYSE 20th-percentile mktcap breakpoint (Fama-French microcap cut).

robustness_microcap v1.0 (2026-07-30)
Also validates mktcap > 0 (a precondition for valid VW weights).
Output: data/processed/robustness_micro.csv
"""
import resource

import numpy as np
import pandas as pd

from multiple_testing import ff5_alpha

VERSION = "robustness_microcap v1.0 (2026-07-30)"
PROC = "data/processed"
N_BINS = 5
MIN_STOCKS_PER_MONTH = 100
SURVIVORS = {"BM": 1, "ASSET_GROWTH": -1, "REV_1M": -1, "CFP": 1}


def add_forward_returns(df):
    df = df.sort_values(["permno", "date"], kind="mergesort").reset_index(drop=True)
    df["_midx"] = (df["date"].dt.year * 12 + df["date"].dt.month).astype("int32")
    g = df.groupby("permno", sort=False)
    df["ret_fwd"] = g["ret"].shift(-1).where(g["_midx"].shift(-1) - df["_midx"] == 1)
    return df.drop(columns=["_midx"])


def ls_series(df, zcol, mask):
    sub = df.loc[mask, ["date", "ret_fwd", zcol]].dropna()
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
        m = grp["ret_fwd"].groupby(bins).mean()
        out[dt] = float(m.loc[N_BINS - 1] - m.loc[0])
    return pd.Series(out).sort_index()


def stats(s, sign, fac, variant, ff):
    r = (s * sign).rename("ret")
    r.index = pd.PeriodIndex(pd.to_datetime(r.index), freq="M") + 1
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
ff = ff.rename(columns=ren)[list(ren.values())]

usecols = ["permno", "date", "ret", "mktcap"] + [f + "_z" for f in SURVIVORS]
df = pd.read_parquet(f"{PROC}/factors_all.parquet", columns=usecols)
df = add_forward_returns(df)

mc = df["mktcap"].dropna()
print(f"mktcap check: min={mc.min():.4f}, n<=0: {(mc <= 0).sum()}")
assert (mc > 0).all(), "non-positive mktcap found -- VW run is suspect!"

exch = pd.read_parquet(f"{PROC}/panel.parquet",
                       columns=["permno", "date", "primaryexch"])
df = df.merge(exch, on=["permno", "date"], how="left", validate="1:1")
print("primaryexch counts:", df["primaryexch"].value_counts().to_dict())
assert (df["primaryexch"] == "N").sum() > 0, "no NYSE stocks found"

nyse = df[df["primaryexch"] == "N"]
p20 = nyse.groupby("date")["mktcap"].quantile(0.20).rename("p20")
df = df.merge(p20, on="date", how="left")
keep = df["mktcap"] >= df["p20"]
valid = df["mktcap"].notna() & df["p20"].notna()
n_share = keep[valid].mean()
cap_share = df.loc[keep & valid, "mktcap"].sum() / df.loc[valid, "mktcap"].sum()
print(f"kept above NYSE p20: {n_share:.1%} of stock-months, "
      f"{cap_share:.1%} of total mktcap")

rows = [stats(ls_series(df, f + "_z", keep), s, f, "EW_ex_micro", ff)
        for f, s in SURVIVORS.items()]
out = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print("\n=== EW excluding microcaps (below NYSE p20) ===")
print(out.round(3).to_string(index=False))
out.to_csv(f"{PROC}/robustness_micro.csv", index=False)
print(f"\nsaved -> {PROC}/robustness_micro.csv")
print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
