"""
Ken French smoke test -- runnable before WRDS access arrives; validates
the multiple-testing module end to end.

What it does
------------
Pull Ken French's 49 industry portfolios + FF5 factors (all free), treat
every industry as a "strategy", and run the full significance battery.
A real question: out of 49 industries, how many excess returns survive
multiple-testing correction?

The usual answer: very few. This is exactly what the real project later
does to its own factor panel.
"""
from __future__ import annotations
import io, zipfile, urllib.request
import numpy as np, pandas as pd
from multiple_testing import (
    factor_significance_report, annualized_sharpe_se,
    sharpe_difference_test, ff5_alpha)

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def fetch(name: str) -> str:
    with urllib.request.urlopen(BASE + name, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
        return z.read(z.namelist()[0]).decode("latin-1")


def parse_block(txt: str, start_marker: str = "") -> pd.DataFrame:
    """Ken French CSVs contain several blocks; take the first monthly one."""
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
    print("downloading Ken French data...")
    ind = parse_block(fetch("49_Industry_Portfolios_CSV.zip"))
    ff = parse_block(fetch("F-F_Research_Data_5_Factors_2x3_CSV.zip"))
    ind, ff = ind["1970":], ff["1970":]
    ind = ind.loc[:, ind.notna().mean() > 0.95]
    print(f"  {ind.shape[1]} industry portfolios, {len(ind)} months ({ind.index[0]:%Y-%m} ~ {ind.index[-1]:%Y-%m})")

    # each industry's excess return over the market, treated as a "strategy"
    excess = ind.sub(ff["Mkt-RF"] + ff["RF"], axis=0).dropna(how="all")

    print("\n" + "="*76)
    print("significance of 49 industry excess returns -- before and after multiple testing")
    print("="*76)
    rep = factor_significance_report(excess)
    cols = ["sharpe_annual","se_annual","t_stat","pass_naive_t2","pass_bonferroni","pass_bhy","pass_dsr"]
    print(rep[cols].head(8).round(3).to_string())
    M = rep.shape[0]
    print(f"\n  Bonferroni t threshold = {rep.attrs['bonferroni_t_threshold']:.2f}")
    print(f"  naive |t|>2 : {rep['pass_naive_t2'].sum():2d}/{M}")
    print(f"  Bonferroni : {rep['pass_bonferroni'].sum():2d}/{M}")
    print(f"  BHY        : {rep['pass_bhy'].sum():2d}/{M}")
    print(f"  DSR        : {rep['pass_dsr'].sum():2d}/{M}")
    print(f"\n  expected |t|>2 count under pure noise = {M} x 0.05 = {M*0.05:.1f}")

    # top two industries: is the Sharpe gap significant?
    top2 = rep.index[:2]
    d = sharpe_difference_test(excess[top2[0]], excess[top2[1]])
    print(f"\n  {top2[0]} vs {top2[1]}: Sharpe {d['sharpe1_annual']:.2f} vs "
          f"{d['sharpe2_annual']:.2f}, p={d['p_value']:.3f}"
          f"  -> {'significant' if d['significant_5pct'] else 'not significant'}")

    # FF5 regression: does the best industry keep alpha after styles?
    try:
        a = ff5_alpha(excess[top2[0]], ff)
        print(f"\n  {top2[0]} FF5 alpha: annualized {a['alpha_annual']*100:+.2f}%, "
              f"t={a['alpha_tstat']:+.2f}, R2={a['r_squared']:.2f}")
        print(f"    -> {'alpha survives style adjustment' if abs(a['alpha_tstat'])>2 else 'alpha not significant after style adjustment'}")
    except ImportError:
        print("\n  (statsmodels required for the FF5 regression: pip install statsmodels)")
