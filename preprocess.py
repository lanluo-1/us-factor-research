"""
Cross-sectional preprocessing pipeline -- winsorize -> fill -> z-score ->
industry+size neutralize -> z-score again.

The order matters; see individual docstrings for why.
Every function operates on a single cross-section (one date); process_panel
schedules them via groupby.

Industry classification: gsector (GICS sectors, 11 of them), from
industry_sic_naics.csv.gz.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm


def winsorize_mad(s: pd.Series, n: float = 3.0) -> pd.Series:
    """
    MAD winsorization: median +- n * 1.4826 * MAD.
    More robust than 3-sigma -- sigma itself gets inflated by outliers,
    MAD does not. Must run BEFORE standardization, otherwise outliers
    contaminate the whole cross-section's mean and std.
    """
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return s
    lo, hi = med - n * 1.4826 * mad, med + n * 1.4826 * mad
    return s.clip(lo, hi)


def fill_industry_median(s: pd.Series, industry: pd.Series) -> pd.Series:
    """Industry-median fill; an all-missing industry falls back to the
    market median; anything still missing stays NaN (stock dropped that
    month)."""
    out = s.copy()
    ind_med = s.groupby(industry).transform("median")
    out = out.fillna(ind_med)
    return out.fillna(s.median())


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - s.mean()) / sd


def neutralize(s: pd.Series, industry: pd.Series, log_mktcap: pd.Series) -> pd.Series:
    """
    Industry + size neutralization: factor ~ industry_dummies + ln(mktcap),
    keep the residual.

    Why it is mandatory: without it, "BM works" may just be "small caps /
    financials work" in disguise. What survives neutralization is the
    factor's own cross-sectional information.
    Runs after z-scoring (comparable coefficients); the residual
    distribution changes, hence the outer re-z-score.
    """
    df = pd.DataFrame({"y": s, "ind": industry, "size": log_mktcap}).dropna()
    if len(df) < 50 or df["ind"].nunique() < 2:
        return s
    X = pd.get_dummies(df["ind"], drop_first=True, dtype=float)
    X["size"] = df["size"]
    X = sm.add_constant(X)
    resid = sm.OLS(df["y"], X).fit().resid
    out = pd.Series(np.nan, index=s.index)
    out.loc[resid.index] = resid
    return out


def process_factor_cross_section(raw: pd.Series, industry: pd.Series,
                                 log_mktcap: pd.Series) -> pd.Series:
    """Full pipeline for a single cross-section."""
    x = winsorize_mad(raw)
    x = fill_industry_median(x, industry)
    x = zscore(x)
    x = neutralize(x, industry, log_mktcap)
    return zscore(x)


def process_panel(panel: pd.DataFrame, factor_cols: list[str],
                  date_col: str = "date", industry_col: str = "gsector",
                  mktcap_col: str = "mktcap") -> pd.DataFrame:
    """
    Run the pipeline for every factor and every cross-section date.
    Returns: the original panel + a {name}_z column per factor.
    """
    out = panel.copy()
    out["_ln_mktcap"] = np.log(out[mktcap_col].clip(lower=1e-6))
    for f in factor_cols:
        print(f"  processing {f} ...")
        out[f + "_z"] = (
            out.groupby(date_col, group_keys=False)
               .apply(lambda g: process_factor_cross_section(
                   g[f], g[industry_col], g["_ln_mktcap"]),
                   include_groups=False)
        )
    return out.drop(columns=["_ln_mktcap"])
