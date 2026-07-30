"""
截面预处理流水线 —— 去极值 -> 填充 -> 标准化 -> 行业+市值中性化 -> 再标准化

顺序不能乱,理由见各函数 docstring。
所有函数都作用在「单个截面日期」的 Series 上,由 process_panel 统一 groupby 调度。

行业分类用 gsector (GICS Sector, 11 个),来自 industry_sic_naics.csv.gz。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm


def winsorize_mad(s: pd.Series, n: float = 3.0) -> pd.Series:
    """
    MAD 去极值: median ± n * 1.4826 * MAD。
    比 3σ 稳健 —— σ 本身会被极端值撑大,MAD 不会。
    必须在标准化之前做,否则极端值污染整个截面的均值方差。
    """
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return s
    lo, hi = med - n * 1.4826 * mad, med + n * 1.4826 * mad
    return s.clip(lo, hi)


def fill_industry_median(s: pd.Series, industry: pd.Series) -> pd.Series:
    """行业中位数填充;整行业全缺退化到全市场中位数;仍缺保留 NaN(该期剔除该股)。"""
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
    行业 + 市值中性化: factor ~ industry_dummies + ln(mktcap), 取残差。

    为什么必须做: 不做的话,"BM 有效" 很可能只是 "小盘股/金融股有效" 的伪装。
    中性化后剩下的才是因子自身的截面信息。
    在 zscore 之后做(回归系数可比),做完残差分布变了,所以外层还要再 zscore 一次。
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
    """单个截面的完整流水线。"""
    x = winsorize_mad(raw)
    x = fill_industry_median(x, industry)
    x = zscore(x)
    x = neutralize(x, industry, log_mktcap)
    return zscore(x)


def process_panel(panel: pd.DataFrame, factor_cols: list[str],
                  date_col: str = "date", industry_col: str = "gsector",
                  mktcap_col: str = "mktcap") -> pd.DataFrame:
    """
    对面板中每个因子、每个截面日期跑流水线。
    Returns: 原面板 + 每个因子的 {name}_z 列(处理后的标准化得分)。
    """
    out = panel.copy()
    out["_ln_mktcap"] = np.log(out[mktcap_col].clip(lower=1e-6))
    for f in factor_cols:
        print(f"  处理 {f} ...")
        out[f + "_z"] = (
            out.groupby(date_col, group_keys=False)
               .apply(lambda g: process_factor_cross_section(
                   g[f], g[industry_col], g["_ln_mktcap"]),
                   include_groups=False)
        )
    return out.drop(columns=["_ln_mktcap"])
