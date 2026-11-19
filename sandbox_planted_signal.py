"""
Pipeline validation on synthetic data: planted signal, look-ahead canary,
gap handling.

sandbox_planted_signal v1.0 (2026-11-19)

Purpose: prove that the portfolio-formation code in run_factor_tests.py
(add_forward_returns + long_short_series) and the statistics in
multiple_testing.py (factor_significance_report)
  (1) FIND a predictive signal that is really there,
  (2) do NOT find one that is not there,
  (3) do NOT reward information from the holding month itself (look-ahead),
  (4) do NOT stitch returns across a missing month.

No WRDS data is needed; everything is simulated with a fixed seed, so the
numbers below are exactly reproducible.

Synthetic panel
---------------
N_STOCKS x N_MONTHS, monthly returns

    ret[i, t+1] = PREMIUM * PLANTED[i, t] + noise[i, t+1]

  PLANTED  cross-sectional N(0,1), known at t, truly predicts t+1.
  NOISE    cross-sectional N(0,1), unrelated to any return (placebo).
  CANARY   = same-month return ret[i, t] plus a little noise. It contains
             NO information about t+1. A correctly aligned pipeline
             (signal at t, return at t+1) must report t ~ 0. If the
             pipeline ever paired a signal with its own month's return,
             CANARY would light up with a huge t.

Checks
------
A. Correct pipeline: PLANTED large t, NOISE ~ 0, CANARY ~ 0.
B. Deliberately broken pipeline (signal at t paired with return at t):
   CANARY must explode. This shows the canary is able to detect the bug
   it is meant to detect.
C. Gap test: delete one month for some stocks; the forward return of the
   month before the hole must be NaN, never the return from two months on.

One simulated panel proves little (a placebo lands above |t| = 2 one time
in twenty by construction), so A and B are repeated over N_SIMS independent
panels and summarized: the planted signal must be recovered every time,
and the placebo and the canary must be flagged at roughly the nominal 5%
rate -- no more.

Usage:
    python3 sandbox_planted_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiple_testing import factor_significance_report  # noqa: E402
from run_factor_tests import add_forward_returns, long_short_series  # noqa: E402

VERSION = "sandbox_planted_signal v1.0 (2026-11-19)"

SEED = 20261119
N_STOCKS = 2000
N_MONTHS = 120
NOISE_SD = 0.10        # idiosyncratic monthly return volatility
PREMIUM = 0.005        # planted: +0.5% next-month return per 1 sd of signal
CANARY_NOISE_SD = 0.02
N_GAP_STOCKS = 50      # stocks that get one month deleted in the gap test
N_SIMS = 100           # independent simulated panels (seed, seed+1, ...)


def build_panel(rng: np.random.Generator) -> pd.DataFrame:
    """Simulate the panel described in the module docstring."""
    dates = pd.date_range("2010-01-31", periods=N_MONTHS, freq="ME")
    planted = rng.standard_normal((N_MONTHS, N_STOCKS))
    noise_sig = rng.standard_normal((N_MONTHS, N_STOCKS))
    eps = rng.normal(0.0, NOISE_SD, (N_MONTHS, N_STOCKS))

    ret = eps.copy()
    ret[1:] += PREMIUM * planted[:-1]          # signal at t pays off in t+1
    canary = ret + rng.normal(0.0, CANARY_NOISE_SD, (N_MONTHS, N_STOCKS))

    return pd.DataFrame({
        "permno": np.tile(np.arange(1, N_STOCKS + 1), N_MONTHS),
        "date": np.repeat(dates.values, N_STOCKS),
        "ret": ret.ravel(),
        "PLANTED_z": planted.ravel(),
        "NOISE_z": noise_sig.ravel(),
        "CANARY_z": canary.ravel(),
    })


def t_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Long-short series for the three signals -> the project's own t-stats."""
    ls = pd.DataFrame({c[:-2]: long_short_series(df, c)
                       for c in ("PLANTED_z", "NOISE_z", "CANARY_z")})
    rep = factor_significance_report(ls)
    out = rep[["n", "sharpe_annual", "t_stat"]].copy()
    out["ann_mean"] = ls.mean() * 12
    out.insert(0, "pipeline", label)
    return out.loc[["PLANTED", "NOISE", "CANARY"]]


def one_sim(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate one panel; return the t-tables of the correct and the
    deliberately misaligned pipeline."""
    panel = build_panel(np.random.default_rng(seed))
    df = add_forward_returns(panel)
    a = t_table(df, "correct (t -> t+1)")
    broken = df.copy()
    broken["ret_fwd"] = broken["ret"]       # look-ahead on purpose
    b = t_table(broken, "BROKEN (t -> t)")
    return a, b


def gap_test(seed: int) -> tuple[int, int, int, str, str]:
    rng = np.random.default_rng(seed)
    panel = build_panel(rng)
    gap_stocks = rng.choice(np.arange(1, N_STOCKS + 1), N_GAP_STOCKS, replace=False)
    all_dates = np.sort(panel["date"].unique())
    hole = all_dates[N_MONTHS // 2]             # the month to delete
    before = all_dates[N_MONTHS // 2 - 1]       # the month just before the hole
    holed = panel[~(panel["permno"].isin(gap_stocks) & panel["date"].eq(hole))]
    g = add_forward_returns(holed.copy())
    at_before = g[g["permno"].isin(gap_stocks) & g["date"].eq(before)]
    others = g[~g["permno"].isin(gap_stocks) & g["date"].eq(before)]
    return (int(at_before["ret_fwd"].isna().sum()),
            int(others["ret_fwd"].isna().sum()), len(others),
            f"{pd.Timestamp(hole):%Y-%m}", f"{pd.Timestamp(before):%Y-%m}")


def main() -> None:
    print(VERSION)
    print(f"each panel: {N_STOCKS} stocks x {N_MONTHS} months = "
          f"{N_STOCKS * N_MONTHS:,} rows; {N_SIMS} panels, seeds "
          f"{SEED}..{SEED + N_SIMS - 1}")
    print(f"planted premium {PREMIUM:.1%}/month per 1 sd, "
          f"idiosyncratic vol {NOISE_SD:.0%}/month\n")

    rows = []
    for k in range(N_SIMS):
        a, b = one_sim(SEED + k)
        if k == 0:
            pd.set_option("display.width", 160)
            print("=== example: first panel, correct pipeline ===")
            print(a.round(3).to_string())
            print("\n=== example: first panel, deliberately misaligned pipeline ===")
            print(b.round(3).to_string())
            print()
        rows.append({"planted": a.loc["PLANTED", "t_stat"],
                     "placebo": a.loc["NOISE", "t_stat"],
                     "canary": a.loc["CANARY", "t_stat"],
                     "canary_broken": b.loc["CANARY", "t_stat"]})
        if (k + 1) % 20 == 0:
            print(f"  ... {k + 1}/{N_SIMS} panels done", flush=True)
    t = pd.DataFrame(rows).astype(float)

    summ = pd.DataFrame({
        "min_t": t.min(), "median_t": t.median(), "max_t": t.max(),
        "mean_abs_t": t.abs().mean(),
        "share_abs_t_gt_2": (t.abs() > 2).mean(),
    })
    print(f"\n=== t-statistics across {N_SIMS} simulated panels ===")
    print(summ.round(3).to_string())

    n_nan, n_other_nan, n_others, hole, before = gap_test(SEED)
    print("\n=== gap test ===")
    print(f"deleted {hole} for {N_GAP_STOCKS} stocks")
    print(f"forward return at {before} is NaN for {n_nan}/{N_GAP_STOCKS} "
          f"of those stocks (must be all)")
    print(f"and NaN for {n_other_nan}/{n_others} untouched stocks (must be 0)")

    fp_placebo = float((t["placebo"].abs() > 2).mean())
    fp_canary = float((t["canary"].abs() > 2).mean())
    checks = {
        "planted signal recovered in every panel (min t > 5)": t["planted"].min() > 5,
        "placebo flagged at about the nominal rate (<= 12%)": fp_placebo <= 0.12,
        "look-ahead canary flagged at about the nominal rate (<= 12%)": fp_canary <= 0.12,
        "canary fires in every misaligned panel (min t > 5)": t["canary_broken"].min() > 5,
        "no stitching across a gap": n_nan == N_GAP_STOCKS and n_other_nan == 0,
    }
    print("\n=== verdict ===")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nsummary: planted median t = {t['planted'].median():.1f} "
          f"(min {t['planted'].min():.1f}); "
          f"|t| > 2 in {fp_placebo:.0%} of placebo and {fp_canary:.0%} of canary "
          f"panels (nominal 5%); misaligned canary median t = "
          f"{t['canary_broken'].median():.1f}")
    if not all(checks.values()):
        raise SystemExit("validation FAILED")


if __name__ == "__main__":
    main()
