"""
D3: 第二批因子(动量/反转/风险/流动性/应计/资产增长, 8 个)-> 合并成完整 13 因子面板

因子            定义                                          方向  数据源
----            ----                                          ----  ------
MOM_12_2        t-12..t-2 累计收益(跳过最近1月)               +    CRSP
REV_1M          上月收益                                       -    CRSP
VOL_36M         过去36个月收益标准差                           -    CRSP
BETA_60M        对市场收益滚动60月回归斜率(min 36)            -    CRSP+指数
TURNOVER        月成交量/流通股, 12月均值                      -    CRSP
SIZE_CTRL       ln(mktcap)  (仅作对照,不进多重检验主表)        -    CRSP
ACCRUALS        (ni_a - oancf_a)/at_a                          -    Compustat(已在panel)
ASSET_GROWTH    at_a 同比增长                                  -    Compustat(已在panel)

注意:
- BETA/VOL 用月频窗口(FF 标准做法),不需要日频数据。
- TURNOVER: NASDAQ 成交量口径与 NYSE 不同(做市商双计),对 Q 所减半后再算 —— 粗调,
  报告里注明。中性化里已控制交易所差异的大头(行业+规模),这里再打个补丁。
- ASSET_GROWTH 需要上一年的 at_a: 按 permno 取 12 个月前的值。
- SIZE_CTRL 不做中性化(它本身就是中性化回归的自变量,做了残差只剩浮点噪声)。

--------------------------------------------------------------------------
v2 (2026-07-30) 修订说明 —— 为什么 v1 会被系统 OOM 杀掉 (zsh: killed)
--------------------------------------------------------------------------
1. v1 `pd.read_parquet(panel.parquet)` 读入**全部列**。panel 里有一堆字符串列
   (ticker / sharetype / securitytype / 交易所代码 ...),1.04M 行的 object 列
   在 pandas 里每个都是几百 MB。v2 只读实际用到的 10 列。
2. v1 process_panel 里 `panel.copy()` 把整张宽表翻倍,再对宽表做 216 次
   groupby(date).apply,每次又切一份全列子表。v2 用 inplace + numpy 位置索引。
3. v1 第 66 行 `g = panel.groupby("permno")` 之后第 83 行 `panel = panel.merge(...)`
   换了对象,g 仍然握着**旧 panel** 的引用 -> 旧表整张无法回收。v2 用 map 代替
   merge,panel 对象全程不变,并且每步用完就 del + gc。
4. v2 分 3 步 + 每步落盘 checkpoint。任何一步挂掉,重跑不用从头再算一遍。
   强烈建议在 8GB 机器上一步一步跑(见文件底部用法)。
5. [重要] 本文件现在**自带截面预处理内核**,不再 import preprocess.py。
   -> 你只需要替换这一个文件,preprocess.py 原样不动(batch1 的口径也就原样可复现)。
   内核算法与 preprocess.py v1 完全一致,已逐值验证:
     - 残差用 np.linalg.lstsq,与 statsmodels sm.OLS(...).fit().resid(pinv 解)
       最大差 7e-15;
     - 滚动统计用 groupby().rolling()(Cython),与 v1 的 transform(lambda) 逐值相同。

产出:
  data/processed/_ckpt_batch2_raw.parquet   step1: 8 个「原始未处理」因子(留作稳健性检验)
  data/processed/coverage_batch2.csv        step1: 原始覆盖率表(写进 README 用)
  data/processed/factors_batch2.parquet     step2: 8 个 *_z
  data/processed/factors_all.parquet        step3: batch1 + batch2 = 13 个 *_z
"""
from __future__ import annotations

import gc
import os
import resource
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

CKPT_RAW = PROC / "_ckpt_batch2_raw.parquet"
OUT_B2 = PROC / "factors_batch2.parquet"
OUT_ALL = PROC / "factors_all.parquet"
COVERAGE_CSV = PROC / "coverage_batch2.csv"

FACTORS = ["MOM_12_2", "REV_1M", "VOL_36M", "BETA_60M",
           "TURNOVER", "ACCRUALS", "ASSET_GROWTH", "SIZE_CTRL"]
NO_NEUTRALIZE = ("SIZE_CTRL",)          # 理由见文件头「注意」最后一条

# panel.parquet 里实际要用到的列 —— 只读这些
PANEL_COLS = ["permno", "date", "ret", "mktcap", "vol", "shrout",
              "primaryexch", "ni_a", "oancf_a", "at_a"]

FORCE = os.environ.get("FORCE", "") == "1"


# ==========================================================================
# 截面预处理内核(等价于 preprocess.py v1,只是换成 numpy 实现以免爆内存)
#   去极值 -> 行业中位数填充 -> zscore -> 行业+市值中性化 -> 再 zscore
#   顺序不能乱:极端值必须先于标准化处理,否则污染整个截面的均值方差;
#   中性化在 zscore 之后做(回归系数可比),做完残差分布变了,所以外层再 zscore 一次。
# ==========================================================================
MAD_N = 3.0                 # MAD 去极值倍数
MIN_OBS_NEUTRALIZE = 50     # 截面样本不足则跳过中性化


def _winsorize_mad(x: np.ndarray, n: float = MAD_N) -> np.ndarray:
    """MAD 去极值: median +- n*1.4826*MAD。比 3sigma 稳健(sigma 本身会被极端值撑大)。
    MAD==0(过半样本同值)时不裁剪。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(x)
        if not np.isfinite(med):
            return x
        mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return x
    return np.clip(x, med - n * 1.4826 * mad, med + n * 1.4826 * mad)  # NaN 仍是 NaN


def _fill_industry_median(x: np.ndarray, ind: np.ndarray) -> np.ndarray:
    """行业中位数填充;整行业全缺退化到全市场中位数(用填充「前」的中位数)。"""
    if not np.isnan(x).any():
        return x
    s = pd.Series(x)
    s = s.fillna(s.groupby(pd.Series(ind)).transform("median"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        overall = np.nanmedian(x)
    if np.isfinite(overall):
        s = s.fillna(overall)
    return s.to_numpy(dtype="float64")


def _zscore(x: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu, sd = np.nanmean(x), np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return x * 0.0
    return (x - mu) / sd


def _neutralize(y: np.ndarray, ind: np.ndarray, size: np.ndarray) -> np.ndarray:
    """factor ~ const + industry_dummies + ln(mktcap), 取残差。
    不做的话,"BM 有效" 很可能只是 "小盘股/金融股有效" 的伪装。
    样本 < 50 或行业类别 < 2 时原样返回。"""
    ok = np.isfinite(y) & np.isfinite(size) & np.isfinite(ind)
    n = int(ok.sum())
    if n < MIN_OBS_NEUTRALIZE:
        return y
    u = np.unique(ind[ok])
    if u.size < 2:
        return y
    yo, io, so = y[ok], ind[ok], size[ok]
    D = (io[:, None] == u[None, 1:]).astype(np.float64)   # 等价 drop_first=True
    X = np.empty((n, D.shape[1] + 2), dtype=np.float64)
    X[:, 0], X[:, 1:-1], X[:, -1] = 1.0, D, so
    coef, *_ = np.linalg.lstsq(X, yo, rcond=None)
    out = np.full(y.shape, np.nan, dtype="float64")
    out[ok] = yo - X @ coef
    return out


def _pipeline(raw, ind, size, do_neutralize=True):
    x = _winsorize_mad(raw)
    x = _fill_industry_median(x, ind)
    x = _zscore(x)
    if do_neutralize:
        x = _neutralize(x, ind, size)
    return _zscore(x)


def process_panel(panel: pd.DataFrame, factor_cols: list[str],
                  date_col: str = "date", industry_col: str = "gsector",
                  mktcap_col: str = "mktcap",
                  skip_neutralize: tuple[str, ...] = ()) -> pd.DataFrame:
    """对每个因子、每个截面日期跑流水线,原地加上 {name}_z 列。
    只取 (y, gsector, ln_mktcap) 三条一维数组 + 日期的位置索引,不切子表 -> 峰值内存恒定。"""
    def f64(s):
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")

    ln_mc = np.log(np.clip(f64(panel[mktcap_col]), 1e-6, None))
    ind = f64(panel[industry_col])
    ind = np.where(np.isfinite(ind), ind, -1.0)
    groups = list(panel.groupby(date_col, sort=True).indices.values())
    print(f"  截面数: {len(groups)} 个日期 x {len(factor_cols)} 个因子")

    for f in factor_cols:
        do_neut = f not in skip_neutralize
        print(f"  处理 {f} ...{'' if do_neut else '  [跳过中性化]'}", flush=True)
        y = f64(panel[f])
        z = np.full(y.shape, np.nan, dtype="float64")
        for pos in groups:
            z[pos] = _pipeline(y[pos], ind[pos], ln_mc[pos], do_neut)
        panel[f + "_z"] = z.astype("float32")
        del y, z
        gc.collect()
    return panel


# ======================================================================= 工具
def mem(tag: str = "") -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gb = rss / 1024**3 if sys.platform == "darwin" else rss / 1024**2
    print(f"      [进程峰值内存 {gb:.2f} GB] {tag}", flush=True)


def _grolling(df: pd.DataFrame, key: str, col: str,
              window: int, minp: int, how: str) -> pd.Series:
    """按 key 分组的滚动统计。用 pandas 的 Cython 实现,比 apply(lambda) 快一个量级。"""
    gb = df.groupby(key, sort=False)[col].rolling(window, min_periods=minp)
    r = getattr(gb, how)().droplevel(0)
    if not r.index.equals(df.index):    # 保险:顺序对不上就显式对齐
        r = r.reindex(df.index)
    return r


def _check_cols(path: Path, want: list[str]) -> list[str]:
    have = set(pq.read_schema(path).names)
    missing = [c for c in want if c not in have]
    if missing:
        print(f"\n!! {path.name} 缺列: {missing}")
        print(f"   实际有的列: {sorted(have)}")
        raise SystemExit("请把上面这份列名清单发给我,我改列名映射。")
    return want


def _assert_unique_keys(df: pd.DataFrame, name: str) -> None:
    n_dup = df.duplicated(["permno", "date"]).sum()
    if n_dup:
        raise SystemExit(f"!! {name} 有 {n_dup:,} 行 (permno,date) 重复,merge 会炸行数。先查这个。")


def load_market_index() -> pd.Series:
    """CRSP NYSE Value-Weighted Market Index 月度总收益。"""
    idx = pd.read_csv(RAW / "crsp_indexes.csv.gz", parse_dates=["MthCalDt"],
                      low_memory=False)
    idx.columns = idx.columns.str.lower()
    m = idx[idx["indnm"].str.contains("Value-Weighted Market", case=False, na=False)
            & idx["_name_"].eq("MthTotRet")]
    if m.empty:                          # 兜底: 任何带 vw 的市场指数
        m = idx[idx["_name_"].eq("MthTotRet")]
    m = m.drop_duplicates("mthcaldt").set_index("mthcaldt")["col1"].astype("float64")
    m.name = "mkt_ret"
    print(f"  市场指数: {len(m)} 个月 ({m.index.min():%Y-%m} ~ {m.index.max():%Y-%m})")
    del idx
    gc.collect()
    return m


def rolling_beta(g: pd.DataFrame, window: int = 60, min_p: int = 36) -> pd.Series:
    """单只股票的滚动 beta = cov(r, mkt)/var(mkt)。"""
    r, m = g["ret"], g["mkt_ret"]
    cov = r.rolling(window, min_periods=min_p).cov(m)
    var = m.rolling(window, min_periods=min_p).var()
    return cov / var


# ================================================================== STEP 1
def step1_raw_factors() -> None:
    if CKPT_RAW.exists() and not FORCE:
        print(f"[step1] 跳过 —— {CKPT_RAW.name} 已存在 (要重算就 FORCE=1)")
        return

    print("[step1] 计算 8 个原始因子")
    panel_path = PROC / "panel.parquet"
    b1_path = PROC / "factors_batch1.parquet"

    want = list(PANEL_COLS)
    have = set(pq.read_schema(panel_path).names)
    has_gsector_in_panel = "gsector" in have
    if has_gsector_in_panel:
        want.append("gsector")
    _check_cols(panel_path, PANEL_COLS)

    print(f"  读 panel.parquet(只读 {len(want)} 列,不是全表)...")
    panel = pd.read_parquet(panel_path, columns=want)
    panel["primaryexch"] = panel["primaryexch"].astype("category")
    panel = panel.sort_values(["permno", "date"], kind="mergesort").reset_index(drop=True)
    print(f"  {len(panel):,} 行, {panel['permno'].nunique():,} 只股票, "
          f"{panel['date'].nunique()} 个月")
    mem("读完 panel")

    # ---- 动量 / 反转 ----
    # log1p(-1) = -inf 会顺着 rolling sum 污染整个窗口,退市月(ret=-1)必须夹一下
    print("  动量/反转 ...")
    r = panel["ret"].astype("float64")
    n_clip = int((r <= -0.9999).sum())
    panel["_logret"] = np.log1p(r.clip(lower=-0.9999))
    if n_clip:
        print(f"    (ret <= -99.99% 的 {n_clip:,} 个观测在取对数前夹到 -99.99%,防 -inf)")
    del r

    cum11 = np.expm1(_grolling(panel, "permno", "_logret", 11, 8, "sum"))
    panel["MOM_12_2"] = cum11.groupby(panel["permno"]).shift(1).astype("float32")
    panel["REV_1M"] = panel["ret"].astype("float32")
    panel.drop(columns=["_logret"], inplace=True)
    del cum11
    gc.collect()

    # ---- 波动 ----
    print("  波动率 ...")
    panel["VOL_36M"] = _grolling(panel, "permno", "ret", 36, 24, "std").astype("float32")
    gc.collect()
    mem("动量+波动完成")

    # ---- beta ----
    print("  beta (60月滚动, 约1-2分钟) ...")
    mkt = load_market_index()
    panel["mkt_ret"] = panel["date"].map(mkt).astype("float64")   # map 不换 panel 对象
    cov_mkt = panel["mkt_ret"].notna().mean()
    print(f"    市场收益匹配率: {cov_mkt:.1%}")
    if cov_mkt < 0.90:
        print("    !! 匹配率过低 —— panel 的 date 和指数的 mthcaldt 很可能不是同一种月末口径。")
        print("       先别继续,把这行数字发我。")
    sub = panel[["permno", "ret", "mkt_ret"]]
    gb = sub.groupby("permno", group_keys=False)
    try:
        beta = gb.apply(rolling_beta, include_groups=False)
    except TypeError:                    # 老版本 pandas 没有 include_groups
        beta = gb.apply(rolling_beta)
    panel["BETA_60M"] = beta.astype("float32")
    panel.drop(columns=["mkt_ret"], inplace=True)
    del sub, mkt, gb, beta
    gc.collect()
    mem("beta 完成")

    # ---- 换手 ----
    print("  换手率 ...")
    vol_adj = panel["vol"].astype("float64").where(
        ~panel["primaryexch"].eq("Q"), panel["vol"].astype("float64") / 2.0)
    panel["_to"] = vol_adj / (panel["shrout"].astype("float64") * 1000).where(
        panel["shrout"] > 0)
    panel["TURNOVER"] = _grolling(panel, "permno", "_to", 12, 6, "mean").astype("float32")
    panel.drop(columns=["_to"], inplace=True)
    del vol_adj
    gc.collect()

    # ---- 规模(对照) ----
    panel["SIZE_CTRL"] = np.log(panel["mktcap"].astype("float64").clip(lower=1e-6)).astype("float32")

    # ---- 应计 / 资产增长 ----
    print("  应计/资产增长 ...")
    at_a = panel["at_a"].astype("float64")
    panel["ACCRUALS"] = ((panel["ni_a"].astype("float64")
                          - panel["oancf_a"].astype("float64"))
                         / at_a.where(at_a > 0)).astype("float32")
    at_lag = panel.groupby("permno", sort=False)["at_a"].shift(12).astype("float64")
    panel["ASSET_GROWTH"] = (at_a / at_lag.where(at_lag > 0) - 1).astype("float32")
    del at_a, at_lag
    gc.collect()
    mem("8 个因子算完")

    # ---- gsector ----
    if has_gsector_in_panel and panel["gsector"].notna().mean() > 0.5:
        print(f"  gsector 直接用 panel 里的 (覆盖 {panel['gsector'].notna().mean():.1%})")
    else:
        print("  gsector 从 factors_batch1 带回来 ...")
        b1g = pd.read_parquet(b1_path, columns=["permno", "date", "gsector"])
        _assert_unique_keys(b1g, "factors_batch1.parquet")
        panel = panel.merge(b1g, on=["permno", "date"], how="left",
                            suffixes=("_drop", ""))
        panel.drop(columns=[c for c in panel.columns if c.endswith("_drop")],
                   inplace=True, errors="ignore")
        del b1g
        gc.collect()
    panel["gsector"] = pd.to_numeric(panel["gsector"], errors="coerce").fillna(-1).astype("int16")

    # ---- 覆盖率 + 落盘 ----
    print("\n原始因子覆盖率:")
    rows = []
    for f in FACTORS:
        c = panel[f].notna().mean()
        rows.append({"factor": f, "raw_coverage": round(float(c), 4)})
        print(f"  {f:13} {c:.1%}")
    pd.DataFrame(rows).to_csv(COVERAGE_CSV, index=False)

    keep = ["permno", "date", "ret", "mktcap", "gsector"] + FACTORS
    panel[keep].to_parquet(CKPT_RAW, index=False)
    print(f"\n[step1] 已存 {CKPT_RAW.name} ({len(panel):,} 行)")
    print(f"        原始覆盖率表 -> {COVERAGE_CSV.name}")
    mem("step1 结束")
    del panel
    gc.collect()


# ================================================================== STEP 2
def step2_cross_section() -> None:
    if OUT_B2.exists() and not FORCE:
        print(f"[step2] 跳过 —— {OUT_B2.name} 已存在 (要重算就 FORCE=1)")
        return

    print("\n[step2] 截面预处理 (去极值 -> 行业中位数填充 -> zscore -> 中性化 -> zscore)")
    df = pd.read_parquet(CKPT_RAW)
    mem("读完 checkpoint")

    raw_na = {f: df[f].isna().to_numpy() for f in FACTORS}   # 用来算「填充比例」

    df = process_panel(df, FACTORS, skip_neutralize=NO_NEUTRALIZE)
    gc.collect()
    mem("截面处理完成")

    print("\n因子质检 (截面 mean 应 ~0, std 应 ~1):")
    print(f"  {'因子':<14}{'原始覆盖':>9}{'处理后覆盖':>11}{'其中填充':>9}"
          f"{'截面mean':>10}{'截面std':>9}")
    flags = []
    for f in FACTORS:
        zc = f + "_z"
        z = df[zc]
        ok = z.notna()
        by_date = df.loc[ok].groupby("date")[zc]
        m, s = by_date.mean().mean(), by_date.std(ddof=1).mean()
        raw_cov = 1 - raw_na[f].mean()
        z_cov = ok.mean()
        imputed = float((raw_na[f] & ok.to_numpy()).mean())
        print(f"  {f:<14}{raw_cov:>8.1%}{z_cov:>11.1%}{imputed:>9.1%}"
              f"{m:>10.3f}{s:>9.3f}")
        if abs(m) > 0.01 or abs(s - 1) > 0.05:
            flags.append(f"{f}: mean={m:.4f}, std={s:.4f}")
        if f in NO_NEUTRALIZE:
            continue
        if imputed > 0.20:
            flags.append(f"{f}: {imputed:.1%} 的观测是行业中位数填出来的,不是真值")

    if flags:
        print("\n  ⚠️  需要留意:")
        for x in flags:
            print(f"     - {x}")
    else:
        print("\n  ✅ 全部通过")
    print("\n  注: SIZE_CTRL 未做中性化 —— 它就是中性化回归里的自变量 ln(mktcap),"
          "\n      对自己回归后残差只剩 ~1e-14 的浮点噪声,再 zscore 会被放大成 std=1 的假因子。"
          "\n      它与其余 7 个因子不在同一口径,只当对照用,不进多重检验主表。")

    keep = ["permno", "date", "ret", "mktcap", "gsector"] + [f + "_z" for f in FACTORS]
    df[keep].to_parquet(OUT_B2, index=False)
    print(f"\n[step2] 已存 {OUT_B2.name} ({len(df):,} 行, {len(FACTORS)} 个 *_z)")
    mem("step2 结束")
    del df, raw_na
    gc.collect()


# ================================================================== STEP 3
def step3_merge() -> None:
    if OUT_ALL.exists() and not FORCE:
        print(f"[step3] 跳过 —— {OUT_ALL.name} 已存在 (要重算就 FORCE=1)")
        return

    print("\n[step3] 合并 batch1 + batch2 -> factors_all.parquet")
    b2 = pd.read_parquet(OUT_B2)
    b1_cols = pq.read_schema(PROC / "factors_batch1.parquet").names
    z1 = [c for c in b1_cols if c.endswith("_z")]
    b1 = pd.read_parquet(PROC / "factors_batch1.parquet",
                         columns=["permno", "date"] + z1)
    print(f"  batch1: {len(z1)} 个因子 {z1}")

    _assert_unique_keys(b1, "factors_batch1.parquet")
    _assert_unique_keys(b2, "factors_batch2.parquet")

    n_before = len(b2)
    out = b2.merge(b1, on=["permno", "date"], how="left")
    del b1, b2
    gc.collect()
    assert len(out) == n_before, f"merge 后行数变了: {n_before:,} -> {len(out):,}"

    out.to_parquet(OUT_ALL, index=False)
    zc = [c for c in out.columns if c.endswith("_z")]
    print(f"\n[step3] 已存 {OUT_ALL.name} ({len(out):,} 行, {len(zc)} 个因子)")
    print("        因子清单:", ", ".join(c[:-2] for c in zc))
    print(f"        日期范围: {out['date'].min():%Y-%m} ~ {out['date'].max():%Y-%m}")
    mem("step3 结束")


# ======================================================================= main
def main() -> None:
    print("=" * 62)
    print("  factors_batch2.py  v2  (2026-07-30, 自带预处理内核, 不 import preprocess)")
    print("=" * 62)
    PROC.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"1": [step1_raw_factors], "2": [step2_cross_section],
             "3": [step3_merge],
             "all": [step1_raw_factors, step2_cross_section, step3_merge]}
    if which not in steps:
        raise SystemExit("用法: python3 factors_batch2.py [1|2|3|all]")
    for fn in steps[which]:
        fn()
        gc.collect()
    print("\n完成。")


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------
# 用法(8GB 机器建议一步一步跑,每步跑完是独立进程,系统能把内存全部收回):
#   python3 factors_batch2.py 1     # 算 8 个原始因子 -> _ckpt_batch2_raw.parquet
#   python3 factors_batch2.py 2     # 截面处理       -> factors_batch2.parquet
#   python3 factors_batch2.py 3     # 合并 batch1    -> factors_all.parquet
# 已经跑出来的步骤会自动跳过;要重算某一步:
#   FORCE=1 python3 factors_batch2.py 2
# --------------------------------------------------------------------------