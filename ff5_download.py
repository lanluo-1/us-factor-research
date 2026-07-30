"""Download Ken French FF5 + Momentum monthly factors, save as parquet.

ff5_download v1.0 (2026-07-30)
Output: data/raw/ff_factors_monthly.parquet
Raw files are in PERCENT; divided by 100 here (decimal returns).
"""
import io
import re
import sys
import urllib.request
import zipfile

import pandas as pd

VERSION = "ff5_download v1.0 (2026-07-30)"
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF5_ZIP = BASE + "F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_ZIP = BASE + "F-F_Momentum_Factor_CSV.zip"


def fetch_text(url):
    print(f"Downloading {url.rsplit('/', 1)[-1]} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        return z.read(z.namelist()[0]).decode("latin1")


def parse_monthly_block(text):
    """Parse the FIRST contiguous block of rows starting with 6-digit yyyymm.

    Ken French CSVs list monthly data first, annual data later, so the
    first block is the monthly one.
    """
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^\d{6},", ln.strip()):
            start = i
            break
    if start is None:
        sys.exit("ERROR: no monthly rows found -- file format changed?")
    header = [h.strip() for h in lines[start - 1].split(",")]
    rows = []
    for ln in lines[start:]:
        ln = ln.strip()
        if not re.match(r"^\d{6},", ln):
            break
        rows.append([p.strip() for p in ln.split(",")])
    ncol = len(rows[0])
    if len(header) != ncol:
        header = ["yyyymm"] + [f"col{j}" for j in range(1, ncol)]
    header[0] = "yyyymm"
    df = pd.DataFrame(rows, columns=header)
    df["yyyymm"] = df["yyyymm"].astype(int)
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


ff5 = parse_monthly_block(fetch_text(FF5_ZIP))
mom = parse_monthly_block(fetch_text(MOM_ZIP))
mom = mom.rename(columns={mom.columns[-1]: "MOM"})[["yyyymm", "MOM"]]

df = ff5.merge(mom, on="yyyymm", how="inner")
df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]
factor_cols = [c for c in df.columns if c != "yyyymm"]
df[factor_cols] = df[factor_cols].replace([-99.99, -999.0], pd.NA)
df[factor_cols] = df[factor_cols].astype(float) / 100.0
df["date"] = pd.to_datetime(df["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)

print(f"\n{VERSION}")
print(f"columns: {list(df.columns)}")
print(f"rows: {len(df)}, range: {df.yyyymm.min()} .. {df.yyyymm.max()}")

sub = df[(df.yyyymm >= 200701) & (df.yyyymm <= 202412)]
print(f"2007-01..2024-12 rows: {len(sub)} (expect 216)")
assert len(sub) == 216, "missing months inside sample range"
assert not sub[factor_cols].isna().any().any(), "NaN inside sample range"

print("\nSample-period monthly mean / std (decimal):")
print(sub[factor_cols].agg(["mean", "std"]).T.round(4))

oct08 = df.loc[df.yyyymm == 200810, "mkt_rf"].iloc[0]
print(f"\nSanity check Oct-2008 mkt_rf: {oct08:.4f} (expect approx -0.17)")

out = "data/raw/ff_factors_monthly.parquet"
df.to_parquet(out, index=False)
print(f"saved -> {out}")
