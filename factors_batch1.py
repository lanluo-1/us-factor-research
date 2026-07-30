"""
First factor batch (value + quality, 5 factors) -- uses only columns
already present in panel.parquet.

Factor         Definition                           Direction
------         ----------                           ---------
BM             be_a / mktcap                        +   (Fama-French 1992)
EP_TTM         TTM net income / mktcap              +
CFP            TTM operating cash flow / mktcap     +
GP_AT          (revt_a - cogs_a) / at_a             +   (Novy-Marx 2013)
ROE            TTM net income / ceqq_q              +

TTM: rolling 4-quarter sum of quarterly niq.
* oancfy is fiscal-year-to-date cumulative; single quarter = current -
  previous (within the fiscal year), first fiscal quarter = current.
  For simplicity CFP uses ANNUAL oancf here (quarterly TTM left as future work).

Usage:
    python3 factors_batch1.py
Output:
    data/processed/factors_batch1.parquet  (permno, date, five *_z columns)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import process_panel

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"


def build_ttm(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build TTM from the quarterly columns in the panel.
    Quarterly columns carry the _q suffix, and the panel is monthly --
    the same fiscal quarter repeats across months, so first reduce to
    unique (permno, datadate_q) rows, then roll.
    """
    q = (panel[["permno", "datadate_q", "niq_q"]]
         .dropna(subset=["datadate_q"])
         .drop_duplicates(["permno", "datadate_q"])
         .sort_values(["permno", "datadate_q"]))
    q["ni_ttm"] = (q.groupby("permno")["niq_q"]
                     .transform(lambda s: s.rolling(4, min_periods=4).sum()))
    return q[["permno", "datadate_q", "ni_ttm"]]


def main():
    print("loading panel.parquet ...")
    panel = pd.read_parquet(PROC / "panel.parquet")

    # ---- attach industry (gsector) ----
    print("attaching industry classification ...")
    ind = pd.read_csv(RAW / "industry_sic_naics.csv.gz", low_memory=False)
    ind = ind[["gvkey", "datadate", "gsector", "sic"]].copy()
    ind["datadate"] = pd.to_datetime(ind["datadate"])
    # latest industry record per gvkey (industries rarely change; good enough)
    ind = ind.sort_values("datadate").drop_duplicates("gvkey", keep="last")
    # the panel has no gvkey (comp columns got _q/_a suffixes in the merge);
    # go through ccm once more
    ccm = pd.read_csv(RAW / "ccm_link.csv.gz", low_memory=False)
    ccm.columns = ccm.columns.str.lower()
    ccm = ccm[ccm["linkprim"].isin(["P", "C"])][["gvkey", "lpermno"]] \
             .drop_duplicates("lpermno").rename(columns={"lpermno": "permno"})
    ind_map = ccm.merge(ind[["gvkey", "gsector", "sic"]], on="gvkey", how="left")
    panel = panel.merge(ind_map[["permno", "gsector", "sic"]], on="permno", how="left")
    print(f"  gsector coverage: {panel['gsector'].notna().mean():.1%}")
    panel["gsector"] = panel["gsector"].fillna(-1)   # missing industry = own group

    # ---- TTM net income ----
    print("building TTM ...")
    ttm = build_ttm(panel)
    panel = panel.merge(ttm, on=["permno", "datadate_q"], how="left")

    # ---- raw factors ----
    print("computing raw factors ...")
    mcap = panel["mktcap"].where(panel["mktcap"] > 0)
    panel["BM"] = panel["be_a"].where(panel["be_a"] > 0) / mcap
    panel["EP_TTM"] = panel["ni_ttm"] / mcap
    panel["CFP"] = panel["oancf_a"] / mcap
    panel["GP_AT"] = (panel["revt_a"] - panel["cogs_a"]) / panel["at_a"].where(panel["at_a"] > 0)
    panel["ROE"] = panel["ni_ttm"] / panel["ceqq_q"].where(panel["ceqq_q"] > 0)

    facs = ["BM", "EP_TTM", "CFP", "GP_AT", "ROE"]
    print("\nraw factor coverage:")
    for f in facs:
        print(f"  {f:8} {panel[f].notna().mean():.1%}")

    # ---- preprocessing pipeline ----
    print("\nrunning cross-sectional preprocessing (216 months x 5 factors, a few minutes) ...")
    panel = process_panel(panel, facs)

    out_cols = ["permno", "date", "mktcap", "gsector"] + [f + "_z" for f in facs]
    out = panel[out_cols]
    out.to_parquet(PROC / "factors_batch1.parquet")
    print(f"\nsaved {PROC/'factors_batch1.parquet'}  ({len(out):,} rows)")

    # ---- quick check: cross-sectional mean ~0, std ~1 per factor ----
    chk = out.groupby("date")[[f + "_z" for f in facs]].agg(["mean", "std"])
    print("\nhealth check (latest cross-section):")
    print(chk.tail(1).T.round(3).to_string())


if __name__ == "__main__":
    main()
