"""
D3: second factor batch (momentum/reversal/risk/liquidity/accruals/asset
growth, 8 factors) -> merged into the full 13-factor panel.

Factor          Definition                                    Dir  Source
------          ----------                                    ---  ------
MOM_12_2        cumulative return t-12..t-2 (skip last month)  +   CRSP
REV_1M          previous-month return                          -   CRSP
VOL_36M         std of returns over past 36 months             -   CRSP
BETA_60M        rolling 60m slope on market return (min 36)    -   CRSP+index
TURNOVER        monthly volume/shares out, 12m mean            -   CRSP
SIZE_CTRL       ln(mktcap) (control only, not in main table)   -   CRSP
ACCRUALS        (ni_a - oancf_a)/at_a                          -   Compustat (in panel)
ASSET_GROWTH    yoy growth of at_a                             -   Compustat (in panel)

Notes:
- BETA/VOL use monthly windows (standard FF practice), no daily data needed.
- TURNOVER: NASDAQ volume convention differs from NYSE (dealer double
  count); halved for exchange Q before computing -- a coarse patch, noted
  in the report. Neutralization (industry+size) already absorbs most of
  the exchange effect; this is an extra correction on top.
- ASSET_GROWTH needs last year's at_a: per permno, the value 12 months back.
- SIZE_CTRL is NOT neutralized (it IS the regressor in the neutralization;
  the residual would be float noise re-inflated by the z-score).

--------------------------------------------------------------------------
v2 (2026-07-30) revision notes -- why v1 got OOM-killed (zsh: killed)
--------------------------------------------------------------------------
1. v1 `pd.read_parquet(panel.parquet)` read ALL columns. The panel holds a
   pile of string columns (ticker / sharetype / securitytype / exchange
   codes ...); at 1.04M rows every object column costs hundreds of MB in
   pandas. v2 reads only the 10 columns actually used.
2. v1 process_panel did `panel.copy()`, doubling the whole wide table,
   then ran 216 groupby(date).apply passes, each slicing a full-column
   subframe. v2 works in place with numpy positional indexing.
3. v1 line 66 held `g = panel.groupby("permno")` while line 83 later
   rebound `panel = panel.merge(...)` -- g kept a reference to the OLD
   panel, so the whole old table could never be collected. v2 uses map
   instead of merge, the panel object never changes identity, and every
   step ends with del + gc.
4. v2 runs in 3 steps, each checkpointed to disk. If any step dies, the
   rerun does not recompute what already finished.
   On an 8 GB machine, strongly prefer running the steps one by one
   (see usage at the bottom of this file).
5. [important] This file now ships its OWN cross-sectional preprocessing
   kernel and no longer imports preprocess.py.
   -> Only this one file needed replacing; preprocess.py stays untouched
   (so batch1's methodology remains exactly reproducible).
   The kernel matches preprocess.py v1 value for value:
     - residuals via np.linalg.lstsq vs statsmodels sm.OLS(...).fit().resid
       (pinv solution): max diff 7e-15;
     - rolling stats via groupby().rolling() (Cython), element-identical
       to v1's transform(lambda).

Outputs:
  data/processed/_ckpt_batch2_raw.parquet   step1: 8 RAW factors (kept for robustness checks)
  data/processed/coverage_batch2.csv        step1: raw coverage table (for the README)
  data/processed/factors_batch2.parquet     step2: 8 *_z columns
  data/processed/factors_all.parquet        step3: batch1 + batch2 = 13 *_z
"""
from __future__ import annotations

import gc
import os
import resource
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

CKPT_RAW = PROC / "_ckpt_batch2_raw.parquet"
OUT_B2 = PROC / "factors_batch2.parquet"
OUT_ALL = PROC / "factors_all.parquet"
COVERAGE_CSV = PROC / "coverage_batch2.csv"

FACTORS = ["MOM_12_2", "REV_1M", "VOL_36M", "BETA_60M",
           "TURNOVER", "ACCRUALS", "ASSET_GROWTH", "SIZE_CTRL"]
NO_NEUTRALIZE = ("SIZE_CTRL",)          # rationale: last bullet of Notes above

# columns of panel.parquet actually used -- read only these
PANEL_COLS = ["permno", "date", "ret", "mktcap", "vol", "shrout",
              "primaryexch", "ni_a", "oancf_a", "at_a"]

FORCE = os.environ.get("FORCE", "") == "1"


# ==========================================================================
# Cross-sectional preprocessing kernel (equivalent to preprocess.py v1,
# rewritten in numpy to avoid blowing up memory)
#   winsorize -> industry-median fill -> zscore -> industry+size neutralize
#   -> zscore again
#   The order is fixed: outliers must be handled before standardization or
#   they contaminate the cross-section's mean/std; neutralization runs
#   after z-scoring (comparable coefficients) and changes the residual
#   distribution, hence the outer re-z-score.
# ==========================================================================
MAD_N = 3.0                 # MAD winsorization multiple
MIN_OBS_NEUTRALIZE = 50     # skip neutralization below this cross-section size


def _winsorize_mad(x: np.ndarray, n: float = MAD_N) -> np.ndarray:
    """MAD winsorize: median +- n*1.4826*MAD. More robust than 3-sigma
    (sigma itself is inflated by outliers). No clipping when MAD==0
    (more than half the sample identical)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(x)
        if not np.isfinite(med):
            return x
        mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return x
    return np.clip(x, med - n * 1.4826 * mad, med + n * 1.4826 * mad)  # NaN stays NaN


def _fill_industry_median(x: np.ndarray, ind: np.ndarray) -> np.ndarray:
    """Industry-median fill; an all-missing industry falls back to the
    overall market median (computed BEFORE filling)."""
    if not np.isnan(x).any():
        return x
    s = pd.Series(x)
    s = s.fillna(s.groupby(pd.Series(ind)).transform("median"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        overall = np.nanmedian(x)
    if np.isfinite(overall):
        s = s.fillna(overall)
    return s.to_numpy(dtype="float64")


def _zscore(x: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu, sd = np.nanmean(x), np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return x * 0.0
    return (x - mu) / sd


def _neutralize(y: np.ndarray, ind: np.ndarray, size: np.ndarray) -> np.ndarray:
    """factor ~ const + industry_dummies + ln(mktcap), keep residual.
    Without this, "BM works" may just be "small caps / financials work"
    in disguise. Returned unchanged when n < 50 or fewer than 2
    industries."""
    ok = np.isfinite(y) & np.isfinite(size) & np.isfinite(ind)
    n = int(ok.sum())
    if n < MIN_OBS_NEUTRALIZE:
        return y
    u = np.unique(ind[ok])
    if u.size < 2:
        return y
    yo, io, so = y[ok], ind[ok], size[ok]
    D = (io[:, None] == u[None, 1:]).astype(np.float64)   # same as drop_first=True
    X = np.empty((n, D.shape[1] + 2), dtype=np.float64)
    X[:, 0], X[:, 1:-1], X[:, -1] = 1.0, D, so
    coef, *_ = np.linalg.lstsq(X, yo, rcond=None)
    out = np.full(y.shape, np.nan, dtype="float64")
    out[ok] = yo - X @ coef
    return out


def _pipeline(raw, ind, size, do_neutralize=True):
    x = _winsorize_mad(raw)
    x = _fill_industry_median(x, ind)
    x = _zscore(x)
    if do_neutralize:
        x = _neutralize(x, ind, size)
    return _zscore(x)


def process_panel(panel: pd.DataFrame, factor_cols: list[str],
                  date_col: str = "date", industry_col: str = "gsector",
                  mktcap_col: str = "mktcap",
                  skip_neutralize: tuple[str, ...] = ()) -> pd.DataFrame:
    """Run the pipeline for every factor and date, adding {name}_z columns
    in place. Only three 1-D arrays (y, gsector, ln_mktcap) plus per-date
    positional indices are touched -- no subframe slicing -> flat peak
    memory."""
    def f64(s):
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")

    ln_mc = np.log(np.clip(f64(panel[mktcap_col]), 1e-6, None))
    ind = f64(panel[industry_col])
    ind = np.where(np.isfinite(ind), ind, -1.0)
    groups = list(panel.groupby(date_col, sort=True).indices.values())
    print(f"  cross-sections: {len(groups)} dates x {len(factor_cols)} factors")

    for f in factor_cols:
        do_neut = f not in skip_neutralize
        print(f"  processing {f} ...{'' if do_neut else '  [skip neutralization]'}", flush=True)
        y = f64(panel[f])
        z = np.full(y.shape, np.nan, dtype="float64")
        for pos in groups:
            z[pos] = _pipeline(y[pos], ind[pos], ln_mc[pos], do_neut)
        panel[f + "_z"] = z.astype("float32")
        del y, z
        gc.collect()
    return panel


# ======================================================================= utils
def mem(tag: str = "") -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gb = rss / 1024**3 if sys.platform == "darwin" else rss / 1024**2
    print(f"      [peak RSS {gb:.2f} GB] {tag}", flush=True)


def _grolling(df: pd.DataFrame, key: str, col: str,
              window: int, minp: int, how: str) -> pd.Series:
    """Grouped rolling statistic via pandas' Cython path -- an order of
    magnitude faster than apply(lambda)."""
    gb = df.groupby(key, sort=False)[col].rolling(window, min_periods=minp)
    r = getattr(gb, how)().droplevel(0)
    if not r.index.equals(df.index):    # safety: realign explicitly if order differs
        r = r.reindex(df.index)
    return r


def _check_cols(path: Path, want: list[str]) -> list[str]:
    have = set(pq.read_schema(path).names)
    missing = [c for c in want if c not in have]
    if missing:
        print(f"\n!! {path.name} missing columns: {missing}")
        print(f"   columns actually present: {sorted(have)}")
        raise SystemExit("Column mismatch -- update the column-name mapping to match the list above.")
    return want


def _assert_unique_keys(df: pd.DataFrame, name: str) -> None:
    n_dup = df.duplicated(["permno", "date"]).sum()
    if n_dup:
        raise SystemExit(f"!! {name} has {n_dup:,} duplicated (permno,date) rows -- a merge would multiply rows. Investigate first.")


def load_market_index() -> pd.Series:
    """CRSP NYSE Value-Weighted Market Index, monthly total return."""
    idx = pd.read_csv(RAW / "crsp_indexes.csv.gz", parse_dates=["MthCalDt"],
                      low_memory=False)
    idx.columns = idx.columns.str.lower()
    m = idx[idx["indnm"].str.contains("Value-Weighted Market", case=False, na=False)
            & idx["_name_"].eq("MthTotRet")]
    if m.empty:                          # fallback: any market index flagged vw
        m = idx[idx["_name_"].eq("MthTotRet")]
    m = m.drop_duplicates("mthcaldt").set_index("mthcaldt")["col1"].astype("float64")
    m.name = "mkt_ret"
    print(f"  market index: {len(m)} months ({m.index.min():%Y-%m} ~ {m.index.max():%Y-%m})")
    del idx
    gc.collect()
    return m


def rolling_beta(g: pd.DataFrame, window: int = 60, min_p: int = 36) -> pd.Series:
    """Single-stock rolling beta = cov(r, mkt)/var(mkt)."""
    r, m = g["ret"], g["mkt_ret"]
    cov = r.rolling(window, min_periods=min_p).cov(m)
    var = m.rolling(window, min_periods=min_p).var()
    return cov / var


# ================================================================== STEP 1
def step1_raw_factors() -> None:
    if CKPT_RAW.exists() and not FORCE:
        print(f"[step1] skipped -- {CKPT_RAW.name} exists (FORCE=1 to recompute)")
        return

    print("[step1] computing 8 raw factors")
    panel_path = PROC / "panel.parquet"
    b1_path = PROC / "factors_batch1.parquet"

    want = list(PANEL_COLS)
    have = set(pq.read_schema(panel_path).names)
    has_gsector_in_panel = "gsector" in have
    if has_gsector_in_panel:
        want.append("gsector")
    _check_cols(panel_path, PANEL_COLS)

    print(f"  reading panel.parquet ({len(want)} columns only, not the full table)...")
    panel = pd.read_parquet(panel_path, columns=want)
    panel["primaryexch"] = panel["primaryexch"].astype("category")
    panel = panel.sort_values(["permno", "date"], kind="mergesort").reset_index(drop=True)
    print(f"  {len(panel):,} rows, {panel['permno'].nunique():,} stocks, "
          f"{panel['date'].nunique()} months")
    mem("panel loaded")

    # ---- momentum / reversal ----
    # log1p(-1) = -inf would poison the whole rolling window; delisting
    # months (ret = -1) must be clipped first
    print("  momentum/reversal ...")
    r = panel["ret"].astype("float64")
    n_clip = int((r <= -0.9999).sum())
    panel["_logret"] = np.log1p(r.clip(lower=-0.9999))
    if n_clip:
        print(f"    ({n_clip:,} observations with ret <= -99.99% clipped before log, to avoid -inf)")
    del r

    cum11 = np.expm1(_grolling(panel, "permno", "_logret", 11, 8, "sum"))
    panel["MOM_12_2"] = cum11.groupby(panel["permno"]).shift(1).astype("float32")
    panel["REV_1M"] = panel["ret"].astype("float32")
    panel.drop(columns=["_logret"], inplace=True)
    del cum11
    gc.collect()

    # ---- volatility ----
    print("  volatility ...")
    panel["VOL_36M"] = _grolling(panel, "permno", "ret", 36, 24, "std").astype("float32")
    gc.collect()
    mem("momentum+vol done")

    # ---- beta ----
    print("  beta (rolling 60m, ~1-2 minutes) ...")
    mkt = load_market_index()
    panel["mkt_ret"] = panel["date"].map(mkt).astype("float64")   # map keeps panel identity
    cov_mkt = panel["mkt_ret"].notna().mean()
    print(f"    market return match rate: {cov_mkt:.1%}")
    if cov_mkt < 0.90:
        print("    !! match rate too low -- panel date and index mthcaldt are probably")
        print("       different month-end conventions. Stop and check this number first.")
    sub = panel[["permno", "ret", "mkt_ret"]]
    gb = sub.groupby("permno", group_keys=False)
    try:
        beta = gb.apply(rolling_beta, include_groups=False)
    except TypeError:                    # older pandas: no include_groups
        beta = gb.apply(rolling_beta)
    panel["BETA_60M"] = beta.astype("float32")
    panel.drop(columns=["mkt_ret"], inplace=True)
    del sub, mkt, gb, beta
    gc.collect()
    mem("beta done")

    # ---- turnover ----
    print("  turnover ...")
    vol_adj = panel["vol"].astype("float64").where(
        ~panel["primaryexch"].eq("Q"), panel["vol"].astype("float64") / 2.0)
    panel["_to"] = vol_adj / (panel["shrout"].astype("float64") * 1000).where(
        panel["shrout"] > 0)
    panel["TURNOVER"] = _grolling(panel, "permno", "_to", 12, 6, "mean").astype("float32")
    panel.drop(columns=["_to"], inplace=True)
    del vol_adj
    gc.collect()

    # ---- size (control) ----
    panel["SIZE_CTRL"] = np.log(panel["mktcap"].astype("float64").clip(lower=1e-6)).astype("float32")

    # ---- accruals / asset growth ----
    print("  accruals/asset growth ...")
    at_a = panel["at_a"].astype("float64")
    panel["ACCRUALS"] = ((panel["ni_a"].astype("float64")
                          - panel["oancf_a"].astype("float64"))
                         / at_a.where(at_a > 0)).astype("float32")
    at_lag = panel.groupby("permno", sort=False)["at_a"].shift(12).astype("float64")
    panel["ASSET_GROWTH"] = (at_a / at_lag.where(at_lag > 0) - 1).astype("float32")
    del at_a, at_lag
    gc.collect()
    mem("8 factors computed")

    # ---- gsector ----
    if has_gsector_in_panel and panel["gsector"].notna().mean() > 0.5:
        print(f"  gsector taken from the panel (coverage {panel['gsector'].notna().mean():.1%})")
    else:
        print("  gsector carried over from factors_batch1 ...")
        b1g = pd.read_parquet(b1_path, columns=["permno", "date", "gsector"])
        _assert_unique_keys(b1g, "factors_batch1.parquet")
        panel = panel.merge(b1g, on=["permno", "date"], how="left",
                            suffixes=("_drop", ""))
        panel.drop(columns=[c for c in panel.columns if c.endswith("_drop")],
                   inplace=True, errors="ignore")
        del b1g
        gc.collect()
    panel["gsector"] = pd.to_numeric(panel["gsector"], errors="coerce").fillna(-1).astype("int16")

    # ---- coverage + save ----
    print("\nraw factor coverage:")
    rows = []
    for f in FACTORS:
        c = panel[f].notna().mean()
        rows.append({"factor": f, "raw_coverage": round(float(c), 4)})
        print(f"  {f:13} {c:.1%}")
    pd.DataFrame(rows).to_csv(COVERAGE_CSV, index=False)

    keep = ["permno", "date", "ret", "mktcap", "gsector"] + FACTORS
    panel[keep].to_parquet(CKPT_RAW, index=False)
    print(f"\n[step1] saved {CKPT_RAW.name} ({len(panel):,} rows)")
    print(f"        raw coverage table -> {COVERAGE_CSV.name}")
    mem("step1 done")
    del panel
    gc.collect()


# ================================================================== STEP 2
def step2_cross_section() -> None:
    if OUT_B2.exists() and not FORCE:
        print(f"[step2] skipped -- {OUT_B2.name} exists (FORCE=1 to recompute)")
        return

    print("\n[step2] cross-sectional preprocessing (winsorize -> industry-median fill -> zscore -> neutralize -> zscore)")
    df = pd.read_parquet(CKPT_RAW)
    mem("checkpoint loaded")

    raw_na = {f: df[f].isna().to_numpy() for f in FACTORS}   # for the imputed-share stat

    df = process_panel(df, FACTORS, skip_neutralize=NO_NEUTRALIZE)
    gc.collect()
    mem("cross-sections done")

    print("\nfactor QC (cross-sectional mean should be ~0, std ~1):")
    print(f"  {'factor':<14}{'raw cov':>9}{'z cov':>11}{'imputed':>9}"
          f"{'cs mean':>10}{'cs std':>9}")
    flags = []
    for f in FACTORS:
        zc = f + "_z"
        z = df[zc]
        ok = z.notna()
        by_date = df.loc[ok].groupby("date")[zc]
        m, s = by_date.mean().mean(), by_date.std(ddof=1).mean()
        raw_cov = 1 - raw_na[f].mean()
        z_cov = ok.mean()
        imputed = float((raw_na[f] & ok.to_numpy()).mean())
        print(f"  {f:<14}{raw_cov:>8.1%}{z_cov:>11.1%}{imputed:>9.1%}"
              f"{m:>10.3f}{s:>9.3f}")
        if abs(m) > 0.01 or abs(s - 1) > 0.05:
            flags.append(f"{f}: mean={m:.4f}, std={s:.4f}")
        if f in NO_NEUTRALIZE:
            continue
        if imputed > 0.20:
            flags.append(f"{f}: {imputed:.1%} of observations are industry-median fills, not real values")

    if flags:
        print("\n  !! needs attention:")
        for x in flags:
            print(f"     - {x}")
    else:
        print("\n  OK: all checks passed")
    print("\n  note: SIZE_CTRL is NOT neutralized -- it is the ln(mktcap) regressor"
          "\n      itself; regressing it on itself leaves ~1e-14 float noise, which a"
          "\n      z-score would re-inflate into a fake std=1 factor."
          "\n      It is not comparable to the other 7 and never enters the main table.")

    keep = ["permno", "date", "ret", "mktcap", "gsector"] + [f + "_z" for f in FACTORS]
    df[keep].to_parquet(OUT_B2, index=False)
    print(f"\n[step2] saved {OUT_B2.name} ({len(df):,} rows, {len(FACTORS)} *_z columns)")
    mem("step2 done")
    del df, raw_na
    gc.collect()


# ================================================================== STEP 3
def step3_merge() -> None:
    if OUT_ALL.exists() and not FORCE:
        print(f"[step3] skipped -- {OUT_ALL.name} exists (FORCE=1 to recompute)")
        return

    print("\n[step3] merging batch1 + batch2 -> factors_all.parquet")
    b2 = pd.read_parquet(OUT_B2)
    b1_cols = pq.read_schema(PROC / "factors_batch1.parquet").names
    z1 = [c for c in b1_cols if c.endswith("_z")]
    b1 = pd.read_parquet(PROC / "factors_batch1.parquet",
                         columns=["permno", "date"] + z1)
    print(f"  batch1: {len(z1)} factors {z1}")

    _assert_unique_keys(b1, "factors_batch1.parquet")
    _assert_unique_keys(b2, "factors_batch2.parquet")

    n_before = len(b2)
    out = b2.merge(b1, on=["permno", "date"], how="left")
    del b1, b2
    gc.collect()
    assert len(out) == n_before, f"row count changed in merge: {n_before:,} -> {len(out):,}"

    out.to_parquet(OUT_ALL, index=False)
    zc = [c for c in out.columns if c.endswith("_z")]
    print(f"\n[step3] saved {OUT_ALL.name} ({len(out):,} rows, {len(zc)} factors)")
    print("        factor list:", ", ".join(c[:-2] for c in zc))
    print(f"        date range: {out['date'].min():%Y-%m} ~ {out['date'].max():%Y-%m}")
    mem("step3 done")


# ======================================================================= main
def main() -> None:
    print("=" * 62)
    print("  factors_batch2.py  v2  (2026-07-30, self-contained kernel, no preprocess import)")
    print("=" * 62)
    PROC.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"1": [step1_raw_factors], "2": [step2_cross_section],
             "3": [step3_merge],
             "all": [step1_raw_factors, step2_cross_section, step3_merge]}
    if which not in steps:
        raise SystemExit("usage: python3 factors_batch2.py [1|2|3|all]")
    for fn in steps[which]:
        fn()
        gc.collect()
    print("\ndone.")


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------
# Usage (on an 8 GB machine run the steps one at a time; each step is an
# independent process, so the OS reclaims all memory in between):
#   python3 factors_batch2.py 1     # 8 raw factors  -> _ckpt_batch2_raw.parquet
#   python3 factors_batch2.py 2     # cross-sections -> factors_batch2.parquet
#   python3 factors_batch2.py 3     # merge batch1   -> factors_all.parquet
# Finished steps are skipped automatically; to redo one:
#   FORCE=1 python3 factors_batch2.py 2
# --------------------------------------------------------------------------
