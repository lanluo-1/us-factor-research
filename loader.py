"""
Data loading layer -- written against the real column names of the four
files downloaded from a WRDS Day Pass on 2026-07-29.
(English translation 2026-07-30; logic unchanged.)

Files (in data/raw/):
    crsp_monthly_2007_2024.csv.gz        1,667,100 rows  CRSP monthly (CIZ)
    compustat_quarterly_2005_2024.csv.gz   926,569 rows  includes RDQ
    compustat_annual_2005_2024.csv.gz      230,823 rows
    ccm_link.csv.gz                         32,814 rows  LC/LU links

Usage:
    from loader import load_crsp, load_comp_q, load_comp_a, load_ccm, build_merged_panel
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).resolve().parent / "data" / "raw"


# ============================================================
# CRSP monthly (CIZ format)
# ============================================================
def load_crsp(path=None) -> pd.DataFrame:
    """
    Actual header:
    PERMNO,HdrCUSIP,PrimaryExch,SecurityType,ShareType,DelActionType,
    DelStatusType,DelReasonType,DelPaymentType,Ticker,PERMCO,MthCalDt,
    MthPrc,MthDelFlg,MthCap,MthPrevCap,MthRet,MthRetx,MthVol,ShrOut

    CIZ vs legacy filter mapping (important!):
        legacy SHRCD in (10,11)  -> new SecurityType='EQTY' and ShareType='NS'
                                    (approximate; the strict version also
                                     needs SecuritySubType='COM', not in this
                                     download -- the sanity check below tests
                                     whether the approximation is good enough)
        legacy EXCHCD in (1,2,3) -> new PrimaryExch in ('N','A','Q')
    """
    path = path or RAW / "crsp_monthly_2007_2024.csv.gz"
    df = pd.read_csv(
        path,
        parse_dates=["MthCalDt"],
        dtype={"PERMNO": "int64", "PERMCO": "Int64"},
        low_memory=False,
    )
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"mthcaldt": "date", "mthret": "ret", "mthretx": "retx",
                            "mthprc": "prc", "mthcap": "mktcap", "mthvol": "vol",
                            "shrout": "shrout"})

    # numeric coercion (defensive: CIZ is usually numeric already; harmless)
    for c in ["ret", "retx", "prc", "mktcap", "vol", "shrout"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # common stock + major exchanges
    n0 = len(df)
    df = df[df["securitytype"].eq("EQTY") & df["sharetype"].eq("NS")]
    df = df[df["primaryexch"].isin(["N", "A", "Q"])]
    print(f"[crsp] {n0:,} -> {len(df):,} rows (EQTY/NS + NYSE/AMEX/NASDAQ)")

    # sanity: stocks per monthly cross-section; US major-exchange common
    # stocks should run ~3500-5500 over 2007-2024.
    cs = df.groupby(df["date"].dt.to_period("M"))["permno"].nunique()
    print(f"[crsp] stocks per month: min={cs.min()}, median={int(cs.median())}, max={cs.max()}")
    if cs.median() > 6500:
        print("  !! cross-section too LARGE -> EQTY/NS approximation may include ADRs/funds; tighten")
    if cs.median() < 3000:
        print("  !! cross-section too SMALL -> filter too strict; inspect sharetype values")

    # delisting-month flag (CIZ MthRet already folds in delisting returns;
    # the flag is kept for diagnostics only)
    df["is_delist_month"] = df["mthdelflg"].astype(str).str.upper().eq("Y")
    # [fix 2026-07-30] The raw CIZ csv contains ~2k security-months with fully
    # identical duplicated rows; the two merge passes below (quarterly + annual)
    # amplify k copies into k**4 rows (2 -> 16, 3 -> 81), which once added
    # 34,784 excess rows to the panel. Deduplicate at the source.
    n_dup = df.duplicated(["permno", "date"]).sum()
    if n_dup:
        print(f"[crsp] !! {n_dup:,} fully duplicated (permno,date) rows -> deduplicated at source")
        df = df.drop_duplicates(["permno", "date"], keep="first")

    return df[["permno", "date", "ret", "retx", "prc", "mktcap", "vol", "shrout",
               "primaryexch", "is_delist_month", "delactiontype"]].reset_index(drop=True)


# ============================================================
# Compustat quarterly (the PIT core)
# ============================================================
def load_comp_q(path=None) -> pd.DataFrame:
    """
    Actual header:
    costat,curcdq,datafmt,indfmt,consol,tic,datadate,gvkey,rdq,
    atq,ceqq,cogsq,niq,revtq,seqq,oancfy

    PIT rule: a row's financials become usable at available_date = rdq;
              when rdq is missing, conservatively datadate + 90 days.
    oancfy is fiscal-year-to-date cumulative; single-quarter conversion
    requires differencing -- handled at the factor layer, only loaded here.
    """
    path = path or RAW / "compustat_quarterly_2005_2024.csv.gz"
    df = pd.read_csv(path, parse_dates=["datadate", "rdq"],
                     dtype={"gvkey": "int64"}, low_memory=False)
    df = df[df["curcdq"].eq("USD")]          # drop non-USD reports
    n_missing_rdq = df["rdq"].isna().sum()
    print(f"[comp_q] {len(df):,} rows (USD), rdq missing {n_missing_rdq:,} "
          f"({n_missing_rdq/len(df)*100:.1f}%)")
    df["available_date"] = df["rdq"].fillna(df["datadate"] + pd.Timedelta(days=90))
    keep = ["gvkey", "datadate", "available_date", "rdq",
            "atq", "ceqq", "seqq", "niq", "revtq", "cogsq", "oancfy"]
    return df[keep].sort_values(["gvkey", "datadate"]).reset_index(drop=True)


# ============================================================
# Compustat annual
# ============================================================
def load_comp_a(path=None) -> pd.DataFrame:
    """
    Actual header:
    costat,curcd,datafmt,indfmt,consol,tic,datadate,gvkey,at,ceq,lt,
    pstk,pstkl,pstkrv,seq,txditc,cogs,ni,revt,oancf

    Annual filings have no RDQ -> conservative rule
    available_date = datadate + 90 days.
    Also builds Fama-French book equity:
        BE = (seq or ceq+pstk or at-lt) + txditc - preferred (pstkrv>pstkl>pstk)
    """
    path = path or RAW / "compustat_annual_2005_2024.csv.gz"
    df = pd.read_csv(path, parse_dates=["datadate"],
                     dtype={"gvkey": "int64"}, low_memory=False)
    df = df[df["curcd"].eq("USD")]
    df["available_date"] = df["datadate"] + pd.Timedelta(days=90)

    se = df["seq"].fillna(df["ceq"] + df["pstk"].fillna(0)).fillna(df["at"] - df["lt"])
    ps = df["pstkrv"].fillna(df["pstkl"]).fillna(df["pstk"]).fillna(0)
    df["be"] = se + df["txditc"].fillna(0) - ps
    neg = (df["be"] <= 0).sum()
    print(f"[comp_a] {len(df):,} rows (USD), BE<=0 in {neg:,} rows (dropped for BM)")
    keep = ["gvkey", "datadate", "available_date", "at", "lt", "ceq", "seq",
            "ni", "revt", "cogs", "oancf", "be"]
    return df[keep].sort_values(["gvkey", "datadate"]).reset_index(drop=True)


# ============================================================
# CCM link table
# ============================================================
def load_ccm(path=None) -> pd.DataFrame:
    """
    Actual header:
    gvkey,tic,LINKPRIM,LIID,LINKTYPE,LPERMNO,LPERMCO,LINKDT,LINKENDDT

    The WRDS query already filtered LINKTYPE in (LC,LU); LINKPRIM in (P,C)
    is applied here. Missing LINKENDDT = link still active -> 2099-12-31.
    """
    path = path or RAW / "ccm_link.csv.gz"
    df = pd.read_csv(path, parse_dates=["LINKDT"],
                     dtype={"gvkey": "int64"}, low_memory=False)
    df.columns = df.columns.str.lower()
    df = df[df["linkprim"].isin(["P", "C"])]
    df["linkenddt"] = pd.to_datetime(df["linkenddt"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    df = df.rename(columns={"lpermno": "permno"})
    df["permno"] = df["permno"].astype("int64")
    print(f"[ccm] {len(df):,} active links (LC/LU + P/C)")
    return df[["gvkey", "permno", "linkdt", "linkenddt"]]


# ============================================================
# Merge: CRSP monthly x latest available financials (merge_asof = PIT)
# ============================================================
def attach_fundamentals(crsp: pd.DataFrame, comp: pd.DataFrame,
                        ccm: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    For every (permno, date), take the LATEST financials whose
    available_date <= date. This single step is the entire point-in-time
    alignment -- merge_asof(direction='backward').
    """
    # 1) attach permno to comp (filtered by link validity window)
    m = comp.merge(ccm, on="gvkey", how="inner")
    m = m[(m["datadate"] >= m["linkdt"]) & (m["datadate"] <= m["linkenddt"])]
    dup = m.duplicated(["permno", "datadate"]).sum()
    if dup:
        print(f"  !! {suffix}: {dup} duplicated permno-datadate links, keeping latest available_date")
        m = (m.sort_values("available_date")
               .drop_duplicates(["permno", "datadate"], keep="last"))

    # 2) merge_asof aligns on available_date to the trading month
    left = crsp[["permno", "date"]].sort_values(["date", "permno"])
    right = m.drop(columns=["gvkey", "linkdt", "linkenddt"]) \
             .sort_values(["available_date", "permno"])
    out = pd.merge_asof(left, right,
                        left_on="date", right_on="available_date",
                        by="permno", direction="backward",
                        tolerance=pd.Timedelta(days=550))   # >1.5y-old financials = missing
    out = out.drop(columns=["available_date"])
    out = out.rename(columns={c: f"{c}_{suffix}" for c in out.columns
                              if c not in ("permno", "date")})
    return crsp.merge(out, on=["permno", "date"], how="left")


def build_merged_panel(save_to: str | Path | None = None) -> pd.DataFrame:
    """One call: load all four tables -> PIT merge -> optionally save parquet."""
    crsp = load_crsp()
    ccm = load_ccm()
    panel = attach_fundamentals(crsp, load_comp_q(), ccm, "q")
    panel = attach_fundamentals(panel, load_comp_a(), ccm, "a")
    cov = panel["atq_q"].notna().mean()
    print(f"[panel] {len(panel):,} rows, quarterly coverage {cov:.1%} "
          f"(normal >85%; if clearly lower, check the ccm date filter)")
    n_dup = panel.duplicated(["permno", "date"]).sum()
    assert n_dup == 0, f"panel has {n_dup:,} duplicated (permno,date) rows -- an upstream merge multiplied rows; stop and investigate"
    if save_to:
        panel.to_parquet(save_to)
        print(f"[panel] saved {save_to}")
    return panel


if __name__ == "__main__":
    (RAW.parent / "processed").mkdir(parents=True, exist_ok=True)
    build_merged_panel(RAW.parent / "processed" / "panel.parquet")
