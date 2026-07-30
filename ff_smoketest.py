"""
Ken French 数据冒烟测试 —— WRDS 批下来之前就能跑,验证多重检验模块。

做什么
------
拉 Ken French 的 49 个行业组合 + FF5 因子(全免费),把每个行业当成一个"策略",
跑一遍完整的显著性检验。这是个真问题: 49 个行业里,有几个的超额收益
能在多重检验校正后活下来?

答案通常是: 很少。这正是你在正式项目里要对 15 个因子做的同一件事。
"""
from __future__ import annotations
import io, zipfile, urllib.request
import numpy as np, pandas as pd
import sys; sys.path.insert(0, "src")
from evaluation.multiple_testing import (
    factor_significance_report, annualized_sharpe_se,
    sharpe_difference_test, ff5_alpha)

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def fetch(name: str) -> str:
    with urllib.request.urlopen(BASE + name, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
        return z.read(z.namelist()[0]).decode("latin-1")


def parse_block(txt: str, start_marker: str = "") -> pd.DataFrame:
    """Ken French 的 CSV 有多个区块,取第一个月频区块。"""
    lines = txt.splitlines()
    i0 = next(i for i, l in enumerate(lines)
              if l.strip() and l.split(",")[0].strip().isdigit()
              and len(l.split(",")[0].strip()) == 6)
    rows = []
    for l in lines[i0:]:
        p = l.split(",")
        k = p[0].strip()
        if not (k.isdigit() and len(k) == 6):
            break
        rows.append([k] + [float(x) for x in p[1:]])
    hdr = [h.strip() for h in lines[i0-1].split(",")]
    df = pd.DataFrame(rows, columns=["date"] + hdr[1:] if len(hdr) > 1 else None)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    return df.set_index("date").replace([-99.99, -999], np.nan) / 100.0


if __name__ == "__main__":
    print("下载 Ken French 数据...")
    ind = parse_block(fetch("49_Industry_Portfolios_CSV.zip"))
    ff = parse_block(fetch("F-F_Research_Data_5_Factors_2x3_CSV.zip"))
    ind, ff = ind["1970":], ff["1970":]
    ind = ind.loc[:, ind.notna().mean() > 0.95]
    print(f"  行业组合 {ind.shape[1]} 个, {len(ind)} 个月 ({ind.index[0]:%Y-%m} ~ {ind.index[-1]:%Y-%m})")

    # 每个行业相对市场的超额收益,当成一个"策略"
    excess = ind.sub(ff["Mkt-RF"] + ff["RF"], axis=0).dropna(how="all")

    print("\n" + "="*76)
    print("49 个行业超额收益的显著性 —— 多重检验前后")
    print("="*76)
    rep = factor_significance_report(excess)
    cols = ["sharpe_annual","se_annual","t_stat","pass_naive_t2","pass_bonferroni","pass_bhy","pass_dsr"]
    print(rep[cols].head(8).round(3).to_string())
    M = rep.shape[0]
    print(f"\n  Bonferroni t 门槛 = {rep.attrs['bonferroni_t_threshold']:.2f}")
    print(f"  朴素 |t|>2 : {rep['pass_naive_t2'].sum():2d}/{M}")
    print(f"  Bonferroni : {rep['pass_bonferroni'].sum():2d}/{M}")
    print(f"  BHY        : {rep['pass_bhy'].sum():2d}/{M}")
    print(f"  DSR        : {rep['pass_dsr'].sum():2d}/{M}")
    print(f"\n  纯随机下 |t|>2 的期望个数 = {M} x 0.05 = {M*0.05:.1f}")

    # 最好的两个行业,Sharpe 差异是否显著
    top2 = rep.index[:2]
    d = sharpe_difference_test(excess[top2[0]], excess[top2[1]])
    print(f"\n  {top2[0]} vs {top2[1]}: Sharpe {d['sharpe1_annual']:.2f} vs "
          f"{d['sharpe2_annual']:.2f}, p={d['p_value']:.3f}"
          f"  -> {'显著' if d['significant_5pct'] else '不显著'}")

    # FF5 回归: 最好的那个行业,控制风格后还剩多少
    try:
        a = ff5_alpha(excess[top2[0]], ff)
        print(f"\n  {top2[0]} 的 FF5 alpha: 年化 {a['alpha_annual']*100:+.2f}%, "
              f"t={a['alpha_tstat']:+.2f}, R2={a['r_squared']:.2f}")
        print(f"    -> {'控制风格后仍有 alpha' if abs(a['alpha_tstat'])>2 else '控制风格后 alpha 不显著'}")
    except ImportError:
        print("\n  (需要 statsmodels 才能跑 FF5 回归: pip install statsmodels)")
