"""
Multiple-testing corrections and statistical significance -- what separates
this project from a typical course assignment.

The core problem
----------------
Test 15 factors at |t| > 2 and pure noise alone delivers ~0.75 "significant"
ones. Test 100 and you are guaranteed a few pretty ones. This is where the
factor zoo comes from.

Lines of defense:
  sharpe_standard_error  -- every Sharpe gap must pass through here first
  sharpe_difference_test -- is the Sharpe gap between two strategies real?
  bonferroni_threshold   -- controls FWER, most conservative
  bhy_procedure          -- controls FDR, Harvey-Liu-Zhu (2016) recommended
  deflated_sharpe_ratio  -- Bailey & Lopez de Prado (2014)
  ff5_alpha              -- how much alpha survives known style factors

References
----------
Lo (2002), "The Statistics of Sharpe Ratios", FAJ
Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected Returns", RFS
Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", JPM
Benjamini & Yekutieli (2001), Annals of Statistics
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# Sharpe standard error -- the gatekeeper for every conclusion
# ============================================================
def sharpe_standard_error(sharpe: float, n_periods: int,
                          skew: float = 0.0, kurt: float = 3.0) -> float:
    """
    Lo (2002), with higher-moment correction:

        SE(SR) = sqrt( [1 + SR^2/2 - skew*SR + (kurt-3)/4 * SR^2] / T )

    Reduces to sqrt((1 + SR^2/2)/T) under normality.

    ! Frequencies must match: `sharpe` and `n_periods` at the same
      frequency (monthly Sharpe with number of months). For an annualized
      SE, compute the monthly SE first and multiply by sqrt(12) -- see
      annualized_sharpe_se().

    `kurt` is plain (non-excess) kurtosis (normal = 3), NOT scipy's
    default excess kurtosis.
    """
    if n_periods < 2:
        raise ValueError("n_periods must be at least 2")
    var = 1 + sharpe**2 / 2 - skew * sharpe + (kurt - 3) / 4 * sharpe**2
    if var <= 0:
        var = 1 + sharpe**2 / 2        # extreme moments: fall back to normal approx
    return float(np.sqrt(var / n_periods))


def annualized_sharpe_se(returns, periods_per_year: int = 12) -> dict:
    """Annualized Sharpe, standard error and 95% CI from a return series."""
    r = pd.Series(returns).dropna()
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    se = sharpe_standard_error(sr, T, stats.skew(r), stats.kurtosis(r, fisher=False))
    k = np.sqrt(periods_per_year)
    return {"sharpe_period": float(sr), "sharpe_annual": float(sr*k),
            "se_annual": float(se*k),
            "ci95": (float((sr-1.96*se)*k), float((sr+1.96*se)*k)),
            "t_stat": float(sr/se), "n": T}


def sharpe_difference_test(r1, r2, periods_per_year: int = 12) -> dict:
    """
    Is the Sharpe difference between two strategies significant?
    Jobson-Korkie with the Memmel (2003) correction.

    * Mandatory whenever claiming "X beats Y". With 180 months of data,
      a Sharpe gap of 0.3 is usually NOT significant.
    """
    df = pd.concat([pd.Series(r1).rename("a"), pd.Series(r2).rename("b")], axis=1).dropna()
    if len(df) < 10:
        raise ValueError("too few overlapping observations")
    x, y = df["a"].values, df["b"].values
    T = len(x)
    s1, s2 = x.mean()/x.std(ddof=1), y.mean()/y.std(ddof=1)
    rho = float(np.corrcoef(x, y)[0, 1])
    v = (2*(1-rho) + 0.5*(s1**2 + s2**2 - 2*s1*s2*rho**2)) / T
    z = (s1 - s2) / np.sqrt(max(v, 1e-16))
    k = np.sqrt(periods_per_year)
    return {"sharpe1_annual": float(s1*k), "sharpe2_annual": float(s2*k),
            "diff_annual": float((s1-s2)*k), "z_stat": float(z),
            "p_value": float(2*(1-stats.norm.cdf(abs(z)))),
            "correlation": rho, "n": T, "significant_5pct": bool(abs(z) > 1.96)}


# ============================================================
# Multiple-testing corrections
# ============================================================
def bonferroni_threshold(n_tests: int, alpha: float = 0.05,
                         df: int | None = None) -> dict:
    """Controls the family-wise error rate. 15 factors -> t threshold ~2.94."""
    a = alpha / n_tests
    t = stats.norm.ppf(1 - a/2) if df is None else stats.t.ppf(1 - a/2, df)
    return {"alpha_adjusted": float(a), "t_threshold": float(t)}


def bhy_procedure(pvalues, alpha: float = 0.05, yekutieli: bool = True) -> pd.DataFrame:
    """
    Benjamini-Hochberg(-Yekutieli) -- controls FDR, looser than Bonferroni.
    This is what Harvey-Liu-Zhu (2016) use for factor tests.

    yekutieli=True divides by the harmonic sum c(M) = sum(1/i), which is
    valid under arbitrary dependence between test statistics -- factors
    are of course correlated, so it is on by default.
    """
    p = pd.Series(pvalues).dropna().sort_values()
    M = len(p)
    c = float(np.sum(1.0/np.arange(1, M+1))) if yekutieli else 1.0
    ranks = np.arange(1, M+1)
    crit = ranks / (M*c) * alpha
    below = p.values <= crit
    kmax = int(np.max(np.where(below)[0]) + 1) if below.any() else 0
    return pd.DataFrame({"pvalue": p.values, "rank": ranks, "critical": crit,
                         "reject": ranks <= kmax}, index=p.index)


def deflated_sharpe_ratio(sharpe: float, n_periods: int, n_trials: int,
                          skew: float = 0.0, kurt: float = 3.0,
                          sharpe_variance_across_trials: float | None = None) -> dict:
    """
    Bailey & Lopez de Prado (2014).

    After N trials, even pure noise produces a max Sharpe well above 0.
    First compute that "max Sharpe reachable by luck" SR0, then measure
    how far the candidate exceeds it.

        SR0 = sqrt(V) * [ (1-g)*Phi^-1(1-1/N) + g*Phi^-1(1-1/(N*e)) ]
        DSR = Phi( (SR-SR0)*sqrt(T-1) / sqrt(1 - skew*SR + (kurt-1)/4*SR^2) )

    g = Euler-Mascheroni constant. V = variance of Sharpe across trials;
    approximated by 1/T when not supplied.
    """
    g = 0.5772156649
    V = sharpe_variance_across_trials if sharpe_variance_across_trials else 1.0/n_periods
    N = max(int(n_trials), 2)
    sr0 = np.sqrt(V) * ((1-g)*stats.norm.ppf(1 - 1.0/N)
                        + g*stats.norm.ppf(1 - 1.0/(N*np.e)))
    denom = np.sqrt(max(1 - skew*sharpe + (kurt-1)/4*sharpe**2, 1e-12))
    dsr = stats.norm.cdf((sharpe - sr0)*np.sqrt(n_periods-1)/denom)
    return {"sr0": float(sr0), "dsr": float(dsr), "significant_5pct": bool(dsr > 0.95)}


# ============================================================
# Alpha after controlling for known styles
# ============================================================
def ff5_alpha(strategy_returns, ff_factors: pd.DataFrame,
              use_momentum: bool = True, nw_lags: int = 6,
              periods_per_year: int = 12) -> dict:
    """
    FF5(+MOM) regression with Newey-West robust standard errors.

    ff_factors must contain columns Mkt-RF, SMB, HML, RMW, CMA, RF, (MOM),
    in decimal units -- Ken French's raw CSVs are in percent, divide by 100.
    Note: if an RF column is present, it is subtracted from the strategy
    return; drop RF from ff_factors for zero-investment long-short returns.

    * An alpha t < 2 means the strategy is essentially a linear combination
      of known factors. Have the nerve to write that down -- it is far more
      honest than an unadjusted pretty Sharpe.
    """
    import statsmodels.api as sm
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    if use_momentum and "MOM" in ff_factors.columns:
        cols.append("MOM")
    df = pd.concat([pd.Series(strategy_returns).rename("ret"), ff_factors], axis=1).dropna()
    if len(df) < len(cols) + 10:
        raise ValueError(f"too few observations: {len(df)} periods")
    y = df["ret"] - df["RF"] if "RF" in df.columns else df["ret"]
    fit = sm.OLS(y, sm.add_constant(df[cols])).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    a = float(fit.params["const"])
    return {"alpha_period": a,
            "alpha_annual": float((1+a)**periods_per_year - 1),
            "alpha_tstat": float(fit.tvalues["const"]),
            "alpha_pvalue": float(fit.pvalues["const"]),
            "betas": {c: float(fit.params[c]) for c in cols},
            "beta_tstats": {c: float(fit.tvalues[c]) for c in cols},
            "r_squared": float(fit.rsquared), "n": int(len(df))}


# ============================================================
# Summary report
# ============================================================
def factor_significance_report(factor_returns: pd.DataFrame, alpha: float = 0.05,
                               periods_per_year: int = 12) -> pd.DataFrame:
    """
    Run the full test battery on all factors' long-short returns at once.

    factor_returns : index = dates, columns = factor names,
                     values = per-period long-short return of each factor
    """
    rows, pvals = [], {}
    M = factor_returns.shape[1]
    bonf = bonferroni_threshold(M, alpha)
    for f in factor_returns.columns:
        r = factor_returns[f].dropna()
        T = len(r)
        if T < 12:
            continue
        sk = float(stats.skew(r)); ku = float(stats.kurtosis(r, fisher=False))
        sr = r.mean()/r.std(ddof=1)
        se = sharpe_standard_error(sr, T, sk, ku)
        t = sr/se
        p = 2*(1 - stats.norm.cdf(abs(t)))
        pvals[f] = p
        d = deflated_sharpe_ratio(sr, T, M, sk, ku)
        rows.append({"factor": f, "n": T,
                     "sharpe_annual": sr*np.sqrt(periods_per_year),
                     "se_annual": se*np.sqrt(periods_per_year),
                     "t_stat": t, "p_value": p,
                     "pass_naive_t2": abs(t) > 2,
                     "pass_bonferroni": abs(t) > bonf["t_threshold"],
                     "dsr": d["dsr"], "pass_dsr": d["significant_5pct"]})
    out = pd.DataFrame(rows).set_index("factor")
    bhy = bhy_procedure(pd.Series(pvals), alpha)
    out["pass_bhy"] = bhy["reject"].reindex(out.index).fillna(False)
    out.attrs["bonferroni_t_threshold"] = bonf["t_threshold"]
    out.attrs["n_tests"] = M
    return out.sort_values("t_stat", key=abs, ascending=False)


if __name__ == "__main__":
    # Self-test: 15 pure-noise factors -- do the gates block false positives?
    rng = np.random.default_rng(11)
    T, M = 180, 15
    noise = pd.DataFrame(rng.normal(0, 0.03, (T, M)),
                         columns=[f"noise_{i:02d}" for i in range(M)])
    rep = factor_significance_report(noise)
    print(f"{M} pure-noise factors, {T} months")
    print(f"Bonferroni t threshold = {rep.attrs['bonferroni_t_threshold']:.2f}\n")
    print(rep[["sharpe_annual", "se_annual", "t_stat", "pass_naive_t2",
               "pass_bonferroni", "pass_bhy", "pass_dsr"]].head(4).round(3).to_string())
    print(f"\n  naive |t|>2 : {rep['pass_naive_t2'].sum()}/{M}  <- false positives")
    print(f"  Bonferroni : {rep['pass_bonferroni'].sum()}/{M}")
    print(f"  BHY        : {rep['pass_bhy'].sum()}/{M}")
    print(f"  DSR        : {rep['pass_dsr'].sum()}/{M}")

    print("\n--- Sharpe standard error examples ---")
    for sr in [0.5, 1.0, 1.5]:
        for T2 in [96, 180]:
            se = sharpe_standard_error(sr/np.sqrt(12), T2)*np.sqrt(12)
            print(f"  ann. SR={sr:.1f}, {T2} months -> SE={se:.2f}, "
                  f"95%CI=[{sr-1.96*se:+.2f}, {sr+1.96*se:+.2f}]")
