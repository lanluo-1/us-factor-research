"""
数据加载层 —— 按 2026-07-29 从 WRDS Day Pass 下载的四个文件的真实列名编写。

文件(放在 data/raw/):
    crsp_monthly_2007_2024.csv.gz       1,667,100 行  CRSP CIZ 月频
    compustat_quarterly_2005_2024.csv.gz  926,569 行  含 RDQ
    compustat_annual_2005_2024.csv.gz     230,823 行
    ccm_link.csv.gz                        32,814 行  LC/LU 链接

用法:
    from loader import load_crsp, load_comp_q, load_comp_a, load_ccm, build_merged_panel
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).resolve().parent / "data" / "raw"


# ============================================================
# CRSP 月频 (CIZ 版)
# ============================================================
def load_crsp(path=None) -> pd.DataFrame:
    """
    真实表头:
    PERMNO,HdrCUSIP,PrimaryExch,SecurityType,ShareType,DelActionType,
    DelStatusType,DelReasonType,DelPaymentType,Ticker,PERMCO,MthCalDt,
    MthPrc,MthDelFlg,MthCap,MthPrevCap,MthRet,MthRetx,MthVol,ShrOut

    CIZ 与旧版过滤口径的对照(重要!):
        旧 SHRCD in (10,11)  -> 新 SecurityType='EQTY' 且 ShareType='NS'
                                (近似;严格版还需 SecuritySubType='COM',本次未下载,
                                 见下方 sanity check 验证近似是否够用)
        旧 EXCHCD in (1,2,3) -> 新 PrimaryExch in ('N','A','Q')
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

    # 数值化(防御:CIZ 一般已是数值,但留着无害)
    for c in ["ret", "retx", "prc", "mktcap", "vol", "shrout"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 普通股 + 主板
    n0 = len(df)
    df = df[df["securitytype"].eq("EQTY") & df["sharetype"].eq("NS")]
    df = df[df["primaryexch"].isin(["N", "A", "Q"])]
    print(f"[crsp] {n0:,} -> {len(df):,} 行 (EQTY/NS + NYSE/AMEX/NASDAQ)")

    # sanity: 每月截面股票数。2007-2024 美国主板普通股应在 3500~5500 之间。
    cs = df.groupby(df["date"].dt.to_period("M"))["permno"].nunique()
    print(f"[crsp] 每月股票数: min={cs.min()}, median={int(cs.median())}, max={cs.max()}")
    if cs.median() > 6500:
        print("  !! 截面偏大 -> EQTY/NS 近似可能放进了 ADR/基金,考虑收紧")
    if cs.median() < 3000:
        print("  !! 截面偏小 -> 过滤过严,检查 sharetype 取值分布")

    # 退市月标记(CIZ 的 MthRet 已并入退市收益,这里只留 flag 供检验)
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
# Compustat 季频 (PIT 核心)
# ============================================================
def load_comp_q(path=None) -> pd.DataFrame:
    """
    真实表头:
    costat,curcdq,datafmt,indfmt,consol,tic,datadate,gvkey,rdq,
    atq,ceqq,cogsq,niq,revtq,seqq,oancfy

    PIT 规则: 该行财务数据在 available_date = rdq 之后才可用;
              rdq 缺失时保守取 datadate + 90 天。
    oancfy 是**年初至今累计**口径,转单季度要差分 —— 在因子层处理,这里只加载。
    """
    path = path or RAW / "compustat_quarterly_2005_2024.csv.gz"
    df = pd.read_csv(path, parse_dates=["datadate", "rdq"],
                     dtype={"gvkey": "int64"}, low_memory=False)
    df = df[df["curcdq"].eq("USD")]          # 剔加元报告
    n_missing_rdq = df["rdq"].isna().sum()
    print(f"[comp_q] {len(df):,} 行 (USD), rdq 缺失 {n_missing_rdq:,} "
          f"({n_missing_rdq/len(df)*100:.1f}%)")
    df["available_date"] = df["rdq"].fillna(df["datadate"] + pd.Timedelta(days=90))
    keep = ["gvkey", "datadate", "available_date", "rdq",
            "atq", "ceqq", "seqq", "niq", "revtq", "cogsq", "oancfy"]
    return df[keep].sort_values(["gvkey", "datadate"]).reset_index(drop=True)


# ============================================================
# Compustat 年频
# ============================================================
def load_comp_a(path=None) -> pd.DataFrame:
    """
    真实表头:
    costat,curcd,datafmt,indfmt,consol,tic,datadate,gvkey,at,ceq,lt,
    pstk,pstkl,pstkrv,seq,txditc,cogs,ni,revt,oancf

    年报无 RDQ -> 保守规则 available_date = datadate + 90 天。
    顺便算 Fama-French 口径的 book equity:
        BE = (seq 或 ceq+pstk 或 at-lt) + txditc - 优先股(pstkrv>pstkl>pstk)
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
    print(f"[comp_a] {len(df):,} 行 (USD), BE<=0 有 {neg:,} 行 (BM 因子时剔除)")
    keep = ["gvkey", "datadate", "available_date", "at", "lt", "ceq", "seq",
            "ni", "revt", "cogs", "oancf", "be"]
    return df[keep].sort_values(["gvkey", "datadate"]).reset_index(drop=True)


# ============================================================
# CCM 链接表
# ============================================================
def load_ccm(path=None) -> pd.DataFrame:
    """
    真实表头:
    gvkey,tic,LINKPRIM,LIID,LINKTYPE,LPERMNO,LPERMCO,LINKDT,LINKENDDT

    查询时已筛 LINKTYPE in (LC,LU);这里再筛 LINKPRIM in (P,C)。
    LINKENDDT 缺失 = 链接仍有效 -> 填 2099-12-31。
    """
    path = path or RAW / "ccm_link.csv.gz"
    df = pd.read_csv(path, parse_dates=["LINKDT"],
                     dtype={"gvkey": "int64"}, low_memory=False)
    df.columns = df.columns.str.lower()
    df = df[df["linkprim"].isin(["P", "C"])]
    df["linkenddt"] = pd.to_datetime(df["linkenddt"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    df = df.rename(columns={"lpermno": "permno"})
    df["permno"] = df["permno"].astype("int64")
    print(f"[ccm] {len(df):,} 条有效链接 (LC/LU + P/C)")
    return df[["gvkey", "permno", "linkdt", "linkenddt"]]


# ============================================================
# 合并: CRSP 月频 x 最新可用财务 (merge_asof 实现 PIT)
# ============================================================
def attach_fundamentals(crsp: pd.DataFrame, comp: pd.DataFrame,
                        ccm: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    对每个 (permno, date),取 available_date <= date 的**最新一期**财务。
    这一步就是 point-in-time 对齐的全部实现 —— merge_asof(direction='backward')。
    """
    # 1) comp 挂上 permno(按链接有效期过滤)
    m = comp.merge(ccm, on="gvkey", how="inner")
    m = m[(m["datadate"] >= m["linkdt"]) & (m["datadate"] <= m["linkenddt"])]
    dup = m.duplicated(["permno", "datadate"]).sum()
    if dup:
        print(f"  !! {suffix}: {dup} 条 permno-datadate 重复,保留 available_date 最新")
        m = (m.sort_values("available_date")
               .drop_duplicates(["permno", "datadate"], keep="last"))

    # 2) merge_asof 按 available_date 对齐到交易月
    left = crsp[["permno", "date"]].sort_values(["date", "permno"])
    right = m.drop(columns=["gvkey", "linkdt", "linkenddt"]) \
             .sort_values(["available_date", "permno"])
    out = pd.merge_asof(left, right,
                        left_on="date", right_on="available_date",
                        by="permno", direction="backward",
                        tolerance=pd.Timedelta(days=550))   # 财务过旧(>1.5年)视为缺失
    out = out.drop(columns=["available_date"])
    out = out.rename(columns={c: f"{c}_{suffix}" for c in out.columns
                              if c not in ("permno", "date")})
    return crsp.merge(out, on=["permno", "date"], how="left")


def build_merged_panel(save_to: str | Path | None = None) -> pd.DataFrame:
    """一键: 加载四表 -> PIT 合并 -> (可选) 存 parquet。"""
    crsp = load_crsp()
    ccm = load_ccm()
    panel = attach_fundamentals(crsp, load_comp_q(), ccm, "q")
    panel = attach_fundamentals(panel, load_comp_a(), ccm, "a")
    cov = panel["atq_q"].notna().mean()
    print(f"[panel] {len(panel):,} 行, 季频财务覆盖率 {cov:.1%} "
          f"(正常应 >85%; 明显偏低先查 ccm 日期过滤)")
    n_dup = panel.duplicated(["permno", "date"]).sum()
    assert n_dup == 0, f"panel has {n_dup:,} duplicated (permno,date) rows -- an upstream merge multiplied rows; stop and investigate"
    if save_to:
        panel.to_parquet(save_to)
        print(f"[panel] 已存 {save_to}")
    return panel


if __name__ == "__main__":
    (RAW.parent / "processed").mkdir(parents=True, exist_ok=True)
    build_merged_panel(RAW.parent / "processed" / "panel.parquet")
