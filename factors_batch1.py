"""
D2: 第一批因子(价值 + 质量,5 个)—— 全部只用 panel.parquet 里已有的列。

因子           定义                                方向
----           ----                                ----
BM             be_a / mktcap                        +   (Fama-French 1992)
EP_TTM         TTM净利润 / mktcap                   +
CFP            TTM经营现金流 / mktcap               +
GP_AT          (revt_a - cogs_a) / at_a             +   (Novy-Marx 2013)
ROE            TTM净利润 / ceqq_q                   +

TTM 的实现: 季频 niq 滚动 4 期求和。
★ oancfy 是年初至今累计口径,单季度 = 本期 - 上期(财年内),财年首季 = 本期。
  这里为省事直接用年频 oancf 做 CFP(D3 再升级成季频 TTM)。

用法:
    python3 factors_batch1.py
产出:
    data/processed/factors_batch1.parquet   (permno, date, 5个 *_z 列)
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
    从面板里的季频列构造 TTM。
    panel 中季频列带 _q 后缀,且已按 (permno,date) 是月频观测 ——
    同一财季会在多个月重复出现,先取每个 (permno, datadate_q) 唯一行再滚动。
    """
    q = (panel[["permno", "datadate_q", "niq_q"]]
         .dropna(subset=["datadate_q"])
         .drop_duplicates(["permno", "datadate_q"])
         .sort_values(["permno", "datadate_q"]))
    q["ni_ttm"] = (q.groupby("permno")["niq_q"]
                     .transform(lambda s: s.rolling(4, min_periods=4).sum()))
    return q[["permno", "datadate_q", "ni_ttm"]]


def main():
    print("加载 panel.parquet ...")
    panel = pd.read_parquet(PROC / "panel.parquet")

    # ---- 挂行业 (gsector) ----
    print("挂行业分类 ...")
    ind = pd.read_csv(RAW / "industry_sic_naics.csv.gz", low_memory=False)
    ind = ind[["gvkey", "datadate", "gsector", "sic"]].copy()
    ind["datadate"] = pd.to_datetime(ind["datadate"])
    # 每个 gvkey 取最新一条行业记录(行业极少变,足够)
    ind = ind.sort_values("datadate").drop_duplicates("gvkey", keep="last")
    # panel 里没有 gvkey,但 comp 合并时的列带 _q/_a 后缀;走 ccm 再挂一次
    ccm = pd.read_csv(RAW / "ccm_link.csv.gz", low_memory=False)
    ccm.columns = ccm.columns.str.lower()
    ccm = ccm[ccm["linkprim"].isin(["P", "C"])][["gvkey", "lpermno"]] \
             .drop_duplicates("lpermno").rename(columns={"lpermno": "permno"})
    ind_map = ccm.merge(ind[["gvkey", "gsector", "sic"]], on="gvkey", how="left")
    panel = panel.merge(ind_map[["permno", "gsector", "sic"]], on="permno", how="left")
    print(f"  gsector 覆盖率: {panel['gsector'].notna().mean():.1%}")
    panel["gsector"] = panel["gsector"].fillna(-1)   # 缺行业单独成组

    # ---- TTM 净利润 ----
    print("构造 TTM ...")
    ttm = build_ttm(panel)
    panel = panel.merge(ttm, on=["permno", "datadate_q"], how="left")

    # ---- 原始因子 ----
    print("计算原始因子 ...")
    mcap = panel["mktcap"].where(panel["mktcap"] > 0)
    panel["BM"] = panel["be_a"].where(panel["be_a"] > 0) / mcap
    panel["EP_TTM"] = panel["ni_ttm"] / mcap
    panel["CFP"] = panel["oancf_a"] / mcap
    panel["GP_AT"] = (panel["revt_a"] - panel["cogs_a"]) / panel["at_a"].where(panel["at_a"] > 0)
    panel["ROE"] = panel["ni_ttm"] / panel["ceqq_q"].where(panel["ceqq_q"] > 0)

    facs = ["BM", "EP_TTM", "CFP", "GP_AT", "ROE"]
    print("\n原始因子覆盖率:")
    for f in facs:
        print(f"  {f:8} {panel[f].notna().mean():.1%}")

    # ---- 预处理流水线 ----
    print("\n跑截面预处理 (216 个月 x 5 因子,几分钟) ...")
    panel = process_panel(panel, facs)

    out_cols = ["permno", "date", "mktcap", "gsector"] + [f + "_z" for f in facs]
    out = panel[out_cols]
    out.to_parquet(PROC / "factors_batch1.parquet")
    print(f"\n已存 {PROC/'factors_batch1.parquet'}  ({len(out):,} 行)")

    # ---- 快速体检: 每因子的截面均值应≈0, std≈1 ----
    chk = out.groupby("date")[[f + "_z" for f in facs]].agg(["mean", "std"])
    print("\n体检(最近一期截面):")
    print(chk.tail(1).T.round(3).to_string())


if __name__ == "__main__":
    main()
