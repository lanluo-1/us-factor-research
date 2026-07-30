"""
多重检验校正与统计显著性 —— 这个项目区别于普通课程作业的地方。

核心问题
--------
测 15 个因子,按 |t| > 2 的门槛,纯随机也会有约 0.75 个"显著"。
测 100 个,就必然能挑出几个漂亮的。这就是 factor zoo 的来源。

五道防线:
  sharpe_standard_error  —— 任何两个 Sharpe 的差距,先过这里
  sharpe_difference_test —— 两个策略的 Sharpe 差异是否显著
  bonferroni_threshold   —— 控制 FWER,最保守
  bhy_procedure          —— 控制 FDR,Harvey-Liu-Zhu (2016) 推荐
  deflated_sharpe_ratio  —— Bailey & Lopez de Prado (2014)
  ff5_alpha              —— 控制已知风格后 alpha 还剩多少

参考文献
--------
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
# Sharpe 的标准误 —— 所有结论的守门员
# ============================================================
def sharpe_standard_error(sharpe: float, n_periods: int,
                          skew: float = 0.0, kurt: float = 3.0) -> float:
    """
    Lo (2002),含高阶矩修正:

        SE(SR) = sqrt( [1 + SR^2/2 - skew*SR + (kurt-3)/4 * SR^2] / T )

    正态时退化为 sqrt((1 + SR^2/2)/T)。

    ! 口径必须一致: sharpe 与 n_periods 同频率。月频 Sharpe 配月数。
      要年化标准误,先算月频 SE 再乘 sqrt(12) —— 见 annualized_sharpe_se()。

    kurt 传非超额峰度(正态=3),不是 scipy 默认的 excess kurtosis。
    """
    if n_periods < 2:
        raise ValueError("n_periods 至少为 2")
    var = 1 + sharpe**2 / 2 - skew * sharpe + (kurt - 3) / 4 * sharpe**2
    if var <= 0:
        var = 1 + sharpe**2 / 2        # 极端矩下退回正态近似
    return float(np.sqrt(var / n_periods))


def annualized_sharpe_se(returns, periods_per_year: int = 12) -> dict:
    """从收益序列直接算年化 Sharpe、标准误、95% 置信区间。"""
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
    两个策略的 Sharpe 差异是否显著?Jobson-Korkie,Memmel (2003) 修正。

    ★ 回答"动态筛选比静态等权好多少"时必须跑这个。
      180 个月样本下,Sharpe 差 0.3 通常不显著。
    """
    df = pd.concat([pd.Series(r1).rename("a"), pd.Series(r2).rename("b")], axis=1).dropna()
    if len(df) < 10:
        raise ValueError("重叠样本太少")
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
# 多重检验校正
# ============================================================
def bonferroni_threshold(n_tests: int, alpha: float = 0.05,
                         df: int | None = None) -> dict:
    """控制 family-wise error rate。15 个因子 -> t 门槛约 2.94。"""
    a = alpha / n_tests
    t = stats.norm.ppf(1 - a/2) if df is None else stats.t.ppf(1 - a/2, df)
    return {"alpha_adjusted": float(a), "t_threshold": float(t)}


def bhy_procedure(pvalues, alpha: float = 0.05, yekutieli: bool = True) -> pd.DataFrame:
    """
    Benjamini-Hochberg(-Yekutieli) —— 控制 FDR,比 Bonferroni 宽松。
    Harvey-Liu-Zhu (2016) 因子检验用的就是这个。

    yekutieli=True 除以调和级数 c(M)=sum(1/i),允许检验统计量任意相关
    —— 因子之间当然相关,所以默认开启。
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
    Bailey & Lopez de Prado (2014)。

    试了 N 次,即使全是噪声,最大的 Sharpe 也会明显大于 0。
    先算这个"随机可达的最大 Sharpe" SR0,再看你超出多少。

        SR0 = sqrt(V) * [ (1-g)*Phi^-1(1-1/N) + g*Phi^-1(1-1/(N*e)) ]
        DSR = Phi( (SR-SR0)*sqrt(T-1) / sqrt(1 - skew*SR + (kurt-1)/4*SR^2) )

    g = 欧拉常数。V = 各次试验 Sharpe 的方差,未给时用 1/T 近似。
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
# 控制已知风格后的 alpha
# ============================================================
def ff5_alpha(strategy_returns, ff_factors: pd.DataFrame,
              use_momentum: bool = True, nw_lags: int = 6,
              periods_per_year: int = 12) -> dict:
    """
    对 FF5(+MOM)回归,Newey-West 稳健标准误。

    ff_factors 需含列 Mkt-RF, SMB, HML, RMW, CMA, RF, (MOM),
    单位为小数 —— Ken French 原始 CSV 是百分数,记得除以 100。

    ★ alpha 的 t < 2,说明你的策略基本就是已知因子的线性组合。
      这个结论要敢写,它比一个未调整的漂亮 Sharpe 诚实得多。
    """
    import statsmodels.api as sm
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    if use_momentum and "MOM" in ff_factors.columns:
        cols.append("MOM")
    df = pd.concat([pd.Series(strategy_returns).rename("ret"), ff_factors], axis=1).dropna()
    if len(df) < len(cols) + 10:
        raise ValueError(f"样本太少: {len(df)} 期")
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
# 汇总报告
# ============================================================
def factor_significance_report(factor_returns: pd.DataFrame, alpha: float = 0.05,
                               periods_per_year: int = 12) -> pd.DataFrame:
    """
    对所有因子的多空组合收益一次性做完整检验。

    factor_returns : index=日期, columns=因子名, 值=该因子多空组合的期收益
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
    # 自检: 15 个纯噪声因子,看各道防线能不能挡住假阳性
    rng = np.random.default_rng(11)
    T, M = 180, 15
    noise = pd.DataFrame(rng.normal(0, 0.03, (T, M)),
                         columns=[f"noise_{i:02d}" for i in range(M)])
    rep = factor_significance_report(noise)
    print(f"纯噪声 {M} 个因子, {T} 个月")
    print(f"Bonferroni t 门槛 = {rep.attrs['bonferroni_t_threshold']:.2f}\n")
    print(rep[["sharpe_annual", "se_annual", "t_stat", "pass_naive_t2",
               "pass_bonferroni", "pass_bhy", "pass_dsr"]].head(4).round(3).to_string())
    print(f"\n  朴素 |t|>2 : {rep['pass_naive_t2'].sum()}/{M}  <- 假阳性")
    print(f"  Bonferroni : {rep['pass_bonferroni'].sum()}/{M}")
    print(f"  BHY        : {rep['pass_bhy'].sum()}/{M}")
    print(f"  DSR        : {rep['pass_dsr'].sum()}/{M}")

    print("\n--- Sharpe 标准误示例 ---")
    for sr in [0.5, 1.0, 1.5]:
        for T2 in [96, 180]:
            se = sharpe_standard_error(sr/np.sqrt(12), T2)*np.sqrt(12)
            print(f"  年化 SR={sr:.1f}, {T2} 个月 -> SE={se:.2f}, "
                  f"95%CI=[{sr-1.96*se:+.2f}, {sr+1.96*se:+.2f}]")
