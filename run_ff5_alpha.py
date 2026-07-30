"""FF5(+MOM) alpha regressions for gate-surviving factors.

run_ff5_alpha v1.0 (2026-07-30)
Inputs : data/processed/ls_returns.parquet  (raw Q5-minus-Q1, unflipped;
         indexed by SIGNAL month t, value = return realized in t+1)
         data/raw/ff_factors_monthly.parquet (decimal units)
Output : data/processed/ff5_alpha_results.csv

Key decisions (README material):
- Alignment: LS return stored at signal month t is realized in t+1,
  so the index is shifted +1 month to match FF factors of the return month.
- Zero-investment: a long-short portfolio is self-financing, so RF must
  NOT be subtracted; the RF column is dropped before calling ff5_alpha
  (which would otherwise subtract it automatically).
- Direction: raw Q5-Q1 flipped to the documented literature direction
  so the alpha sign is interpretable.
"""
import pandas as pd

from multiple_testing import ff5_alpha

VERSION = "run_ff5_alpha v1.0 (2026-07-30)"

# +1: high characteristic -> high return (long Q5); -1: literature says
# low-minus-high (low asset growth wins; short-term reversal).
SURVIVORS = {"BM": 1, "ASSET_GROWTH": -1, "REV_1M": -1, "CFP": 1}
EXPECTED_SR = {"BM": 0.90, "ASSET_GROWTH": 0.94, "REV_1M": 0.72, "CFP": None}


def to_period(idx):
    if pd.api.types.is_integer_dtype(idx):
        return pd.PeriodIndex(pd.to_datetime(idx.astype(str), format="%Y%m"),
                              freq="M")
    return pd.PeriodIndex(pd.to_datetime(idx), freq="M")


print(VERSION)
ls = pd.read_parquet("data/processed/ls_returns.parquet")
print(f"ls_returns: {ls.shape}, index {ls.index[0]} .. {ls.index[-1]}")
ls.index = to_period(ls.index) + 1  # signal month t -> return month t+1

ff = pd.read_parquet("data/raw/ff_factors_monthly.parquet")
ff.index = pd.PeriodIndex(pd.to_datetime(ff["date"]), freq="M")
ren = {"mkt_rf": "Mkt-RF", "smb": "SMB", "hml": "HML", "rmw": "RMW",
       "cma": "CMA", "mom": "MOM"}
ff = ff.rename(columns=ren)[list(ren.values())]  # RF deliberately excluded

canary = pd.concat([(ls["BM"] * SURVIVORS["BM"]).rename("BM_ls"),
                    ff["HML"]], axis=1).dropna()
c = canary["BM_ls"].corr(canary["HML"])
print(f"Alignment canary: corr(BM L-S, HML same month) = {c:.2f} "
      f"(expect clearly > 0; near 0 would mean misalignment)")

rows = []
for fac, sign in SURVIVORS.items():
    r = (ls[fac].dropna() * sign).rename("ret")
    n_overlap = pd.concat([r, ff], axis=1).dropna().shape[0]
    assert n_overlap >= 200, f"{fac}: only {n_overlap} overlapping months"
    sr = r.mean() / r.std() * 12 ** 0.5
    exp = EXPECTED_SR[fac]
    tag = f"(expect approx {exp})" if exp else "(no prior)"
    print(f"\n{fac}: n={len(r)}, overlap={n_overlap}, ann SR={sr:.2f} {tag}")
    for label, use_mom in [("FF5", False), ("FF5+MOM", True)]:
        res = ff5_alpha(r, ff, use_momentum=use_mom)
        row = {"factor": fac, "model": label,
               "alpha_ann": res["alpha_annual"],
               "alpha_t": res["alpha_tstat"],
               "alpha_p": res["alpha_pvalue"],
               "r2": res["r_squared"], "n": res["n"]}
        for k, v in res["betas"].items():
            row[f"b_{k}"] = v
        rows.append(row)
        print(f"  {label:8s} alpha_ann={res['alpha_annual']:+.2%} "
              f"t={res['alpha_tstat']:+.2f} R2={res['r_squared']:.2f}")

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n=== FF5(+MOM) alpha summary (annualized alpha, NW t, lags=6) ===")
print(out.round(3).to_string(index=False))
out.to_csv("data/processed/ff5_alpha_results.csv", index=False)
print("\nsaved -> data/processed/ff5_alpha_results.csv")
