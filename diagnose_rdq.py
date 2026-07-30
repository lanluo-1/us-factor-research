"""
rdq missingness diagnosis -- answers three questions:
  1. Which years is the missingness concentrated in? (early years =
     known phenomenon, fine)
  2. Is it large or small firms that are missing? (small = limited impact)
  3. For rows that HAVE rdq, what is the actual rdq - datadate gap in
     days? (tests whether the 90-day conservative rule is reasonable)
"""
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parent / "data" / "raw"

q = pd.read_csv(RAW / "compustat_quarterly_2005_2024.csv.gz",
                parse_dates=["datadate", "rdq"], low_memory=False)
q = q[q["curcdq"].eq("USD")]

print(f"total rows {len(q):,}, rdq missing {q['rdq'].isna().sum():,} ({q['rdq'].isna().mean()*100:.1f}%)\n")

# ---- 1. by year ----
q["year"] = q["datadate"].dt.year
by_year = q.groupby("year")["rdq"].apply(lambda s: s.isna().mean() * 100).round(1)
print("missing rate by year (%):")
print(by_year.to_string())

# ---- 2. by firm size (total assets atq quartiles) ----
q_valid = q.dropna(subset=["atq"])
q_valid = q_valid.assign(
    size_bucket=pd.qcut(q_valid["atq"], 4, labels=["small(Q1)", "mid-small(Q2)", "mid-large(Q3)", "large(Q4)"]))
by_size = q_valid.groupby("size_bucket", observed=True)["rdq"] \
                 .apply(lambda s: s.isna().mean() * 100).round(1)
print("\nmissing rate by total-assets quartile (%):")
print(by_size.to_string())

# ---- 3. actual rdq - datadate distribution ----
gap = (q["rdq"] - q["datadate"]).dt.days.dropna()
print(f"\nrows with rdq, announcement-lag distribution (days):")
print(f"  median {gap.median():.0f}, mean {gap.mean():.0f}")
print(f"  90th pct {gap.quantile(0.9):.0f}, 99th pct {gap.quantile(0.99):.0f}")
print(f"  share above 90 days: {(gap > 90).mean()*100:.1f}%")
print(f"\nconclusion hint: if the true lag's 90th percentile is <= 90 days,")
print(f"the datadate+90 fallback is safe for missing rows (data may be used")
print(f"late, but is never peeked at early).")
