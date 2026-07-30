"""
rdq 缺失诊断 —— 回答三个问题:
  1. 缺失集中在哪些年份?(早年集中 = 已知现象,放心)
  2. 缺失的是大公司还是小公司?(小公司 = 影响有限)
  3. 有 rdq 的行里,rdq - datadate 实际间隔多少天?(检验 90 天保守规则合不合理)
"""
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parent / "data" / "raw"

q = pd.read_csv(RAW / "compustat_quarterly_2005_2024.csv.gz",
                parse_dates=["datadate", "rdq"], low_memory=False)
q = q[q["curcdq"].eq("USD")]

print(f"总行数 {len(q):,}, rdq 缺失 {q['rdq'].isna().sum():,} ({q['rdq'].isna().mean()*100:.1f}%)\n")

# ---- 1. 按年份 ----
q["year"] = q["datadate"].dt.year
by_year = q.groupby("year")["rdq"].apply(lambda s: s.isna().mean() * 100).round(1)
print("按年份的缺失率 (%):")
print(by_year.to_string())

# ---- 2. 按公司规模 (用总资产 atq 分位) ----
q_valid = q.dropna(subset=["atq"])
q_valid = q_valid.assign(
    size_bucket=pd.qcut(q_valid["atq"], 4, labels=["小(Q1)", "中小(Q2)", "中大(Q3)", "大(Q4)"]))
by_size = q_valid.groupby("size_bucket", observed=True)["rdq"] \
                 .apply(lambda s: s.isna().mean() * 100).round(1)
print("\n按总资产四分位的缺失率 (%):")
print(by_size.to_string())

# ---- 3. rdq - datadate 的实际分布 ----
gap = (q["rdq"] - q["datadate"]).dt.days.dropna()
print(f"\n有 rdq 的行, 公告滞后天数分布:")
print(f"  中位数 {gap.median():.0f} 天, 均值 {gap.mean():.0f} 天")
print(f"  90% 分位 {gap.quantile(0.9):.0f} 天, 99% 分位 {gap.quantile(0.99):.0f} 天")
print(f"  超过 90 天的占比: {(gap > 90).mean()*100:.1f}%")
print(f"\n结论提示: 若真实滞后的 90 分位 <= 90 天, 则 datadate+90 的保守规则")
print(f"对缺失行是安全的(宁可晚用数据,不会提前偷看)。")
