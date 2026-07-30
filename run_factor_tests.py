"""
Single-factor long-short tests with multiple-testing corrections.

Portfolio construction (per factor):
  - Formation month t: rank stocks by the factor z-score, split into
    quintiles (Q5 = highest z).
  - Holding month t+1: equal-weighted average return of Q5 minus Q1.
    The signal is lagged one month relative to the return -- no look-ahead.
  - The forward return is taken only when a stock's next observation is
    exactly one calendar month ahead; delisting months keep their CRSP
    delisting-adjusted return, and gaps produce NaN instead of silently
    stitching non-adjacent months together.
  - Each L-S return is indexed by the FORMATION month t (the cash flow
    happens in t+1; the labeling convention does not affect any statistic).

The 12 monthly L-S series then go through
multiple_testing.factor_significance_report:
Lo (2002) SE -> naive |t|>2 vs Bonferroni vs BHY (FDR) vs DSR.

SIZE_CTRL_z is EXCLUDED from the candidate set: it is the control variable
inside the neutralization (different treatment than the other factors), and
including it would also distort the trial count M used by Bonferroni/DSR.

Outputs:
  data/processed/ls_returns.parquet   monthly L-S return per factor
  data/processed/factor_tests.csv     the full significance table
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiple_testing import factor_significance_report  # noqa: E402

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"

N_BINS = 5                    # quintiles
MIN_STOCKS_PER_MONTH = 100    # skip a formation month with fewer ranked stocks

# Expected direction from the literature (+1: high z -> high future return)
EXPECTED_SIGN = {
    "BM": +1, "EP_TTM": +1, "CFP": +1, "GP_AT": +1, "ROE": +1, "MOM_12_2": +1,
    "REV_1M": -1, "VOL_36M": -1, "BETA_60M": -1, "TURNOVER": -1,
    "ACCRUALS": -1, "ASSET_GROWTH": -1,
}
EXCLUDED = ("SIZE_CTRL",)     # neutralization control, not a candidate factor


def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ret_fwd = next-calendar-month return per stock.

    NaN when the next observation is not exactly one month ahead, so a
    portfolio formed at t only earns returns actually observable in t+1."""
    df = df.sort_values(["permno", "date"], kind="mergesort").reset_index(drop=True)
    midx = (df["date"].dt.year * 12 + df["date"].dt.month).astype("int32")
    df["_midx"] = midx
    g = df.groupby("permno", sort=False)
    ret_next = g["ret"].shift(-1)
    midx_next = g["_midx"].shift(-1)
    df["ret_fwd"] = ret_next.where(midx_next - df["_midx"] == 1)
    df.drop(columns=["_midx"], inplace=True)
    return df


def long_short_series(df: pd.DataFrame, zcol: str) -> pd.Series:
    """Equal-weighted Q5-minus-Q1 forward return, one value per formation month."""
    sub = df[["date", "ret_fwd", zcol]].dropna()
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
    return pd.Series(out, name=zcol[:-2]).sort_index()


def main() -> None:
    path = PROC / "factors_all.parquet"
    df = pd.read_parquet(path)
    zcols = [c for c in df.columns if c.endswith("_z")]
    factors = [c[:-2] for c in zcols if c[:-2] not in EXCLUDED]
    print(f"Loaded {path.name}: {len(df):,} rows, {len(zcols)} z-columns")
    print(f"Candidate factors: {len(factors)}   "
          f"excluded as controls: {[c[:-2] for c in zcols if c[:-2] in EXCLUDED]}")

    df = add_forward_returns(df)
    print(f"Forward 1-month returns available for {df['ret_fwd'].notna().mean():.1%} "
          f"of stock-months")

    print("\nBuilding equal-weighted quintile long-short portfolios "
          "(signal at t, return at t+1):")
    ls = {}
    for f in factors:
        s = long_short_series(df, f + "_z")
        ls[f] = s
        print(f"  {f:13} months={len(s):3d}   ann.mean={s.mean()*12:+7.1%}   "
              f"ann.vol={s.std(ddof=1)*np.sqrt(12):6.1%}")
    ls = pd.DataFrame(ls)
    ls.to_parquet(PROC / "ls_returns.parquet")   # raw Q5-minus-Q1, unflipped

    # Test each factor in its DOCUMENTED direction: flip short-side factors so
    # that "works as documented" shows up as a positive Sharpe. Two-sided
    # stats (t, p, Bonferroni, BHY) are unchanged by the flip; DSR is
    # one-sided by construction and would otherwise auto-fail every
    # short-side factor no matter how strong.
    signs = pd.Series({f: EXPECTED_SIGN.get(f, 1) for f in ls.columns})
    rep = factor_significance_report(ls * signs)
    rep["expected_sign"] = [EXPECTED_SIGN.get(f, 0) for f in rep.index]
    rep["works_as_documented"] = rep["sharpe_annual"] > 0

    pd.set_option("display.width", 200)
    cols = ["n", "sharpe_annual", "se_annual", "t_stat", "p_value",
            "pass_naive_t2", "pass_bonferroni", "pass_bhy", "pass_dsr",
            "expected_sign", "works_as_documented"]
    print("\n=== Single-factor long-short tests "
          "(EW quintiles, documented direction, 2007-2024) ===")
    print(f"Bonferroni t threshold for {rep.attrs['n_tests']} tests: "
          f"{rep.attrs['bonferroni_t_threshold']:.2f}\n")
    print(rep[cols].round(3).to_string())
    print(f"\n  naive |t|>2 : {int(rep['pass_naive_t2'].sum())}/{len(rep)}")
    print(f"  Bonferroni  : {int(rep['pass_bonferroni'].sum())}/{len(rep)}")
    print(f"  BHY (FDR)   : {int(rep['pass_bhy'].sum())}/{len(rep)}")
    print(f"  DSR         : {int(rep['pass_dsr'].sum())}/{len(rep)}")

    rep.to_csv(PROC / "factor_tests.csv")
    print("\nSaved: ls_returns.parquet, factor_tests.csv")


if __name__ == "__main__":
    main()
