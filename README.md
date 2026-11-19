# US Equity Multi-Factor Research (2007-2024)

<!-- README v1.0 (2026-07-30) -->

Cross-sectional factor research on a point-in-time CRSP/Compustat panel:
12 standard characteristics, multiple-testing corrections, FF5+MOM alpha
tests, and size-based robustness checks.

**Headline result:** of 12 candidate factors, 3 survive every
multiple-testing gate (BM, ASSET_GROWTH, REV_1M) and carry significant
FF5+MOM alphas in equal-weighted quintile portfolios -- but nearly all of
the premium lives in microcaps. Value-weighting kills every factor;
excluding stocks below the NYSE 20th-percentile size breakpoint leaves
only ASSET_GROWTH robust (alpha t = 3.7). The pipeline detects real
structure, and honest weighting shows it is not scalable capacity.

## Data

- CRSP monthly (CIZ format), 2007-01..2024-12; delisting returns already
  folded into monthly returns. Compustat quarterly (has rdq) + annual;
  CCM link table (LC/LU, P/C). Final panel: 1,009,540 stock-months.
- Point-in-time rule: fundamentals become available at rdq, or
  datadate + 90 days when rdq is missing. rdq is missing for 29.7% of
  quarterly records overall but only 0.7% for large caps; observed
  announcement lag p90 = 77 days < 90, so the fallback is conservative.
- Dedup: the raw CIZ csv contained ~2k security-months with fully
  identical duplicate rows; two merge passes amplified k copies to k^4.
  Fixed by dedup at source + a uniqueness assert at pipeline exit.
- NASDAQ share volume halved before TURNOVER (dealer double-count).
- Book equity fallback chain: seq -> ceq + pstk -> at - lt.
- Delisting months clipped at -99.99% before log-momentum.

## Factor construction

Preprocessing per monthly cross-section: MAD winsorize ->
industry-median fill -> z-score -> neutralize on industry (gsector) +
ln(mktcap) -> re-z-score.

Batch 1 (value/profitability): BM, EP_TTM, CFP, GP_AT, ROE.
Batch 2 (price/volume/accounting dynamics): MOM_12_2, REV_1M, VOL_36M,
BETA_60M, TURNOVER, ACCRUALS, ASSET_GROWTH. SIZE_CTRL is the
neutralization regressor and is excluded from candidates.
Max imputed share: BETA_60M 15.9%, ASSET_GROWTH 15.2%.

## Testing methodology

- Equal-weighted Q5-minus-Q1 forward return per factor; signal at t,
  return at t+1 only if the next observation is exactly the next
  calendar month (gaps -> NaN, no stitching). Min 100 ranked stocks
  per formation month.
- Multiple testing on M = 12 candidates: Bonferroni (|t| > 2.87),
  BHY (Benjamini-Hochberg-Yekutieli FDR), and Deflated Sharpe Ratio
  (Bailey & Lopez de Prado). Sharpe SEs use Lo (2002) with higher
  moments. Factors flipped to their documented direction before the
  one-sided DSR; two-sided stats are unaffected.
- Alpha tests: FF5 and FF5+MOM (Ken French library), Newey-West lags 6.
  Long-short is zero-investment, so RF is NOT subtracted. LS returns
  realized at t+1 are matched to factor returns of month t+1.

## Results

Funnel (12 candidates): naive |t|>2: 5 -> Bonferroni: 3 -> BHY: 4 ->
DSR: 3. Survive all gates: BM, ASSET_GROWTH, REV_1M. CFP passes BHY
only (t = 2.79 vs Bonferroni threshold 2.87). GP_AT passes only the
naive threshold (t = 2.69) and fails all three gates. Fail everything: ROE
(1.36), ACCRUALS (1.15), TURNOVER (0.98), MOM_12_2 (0.96), EP_TTM
(0.85), VOL_36M (0.26), BETA_60M (0.06, sign flipped vs literature).

FF5+MOM alphas, EW baseline (annualized, NW t):

| factor       | alpha  | t    | notes                                  |
|--------------|--------|------|----------------------------------------|
| BM           | +10.7% | 5.15 | b_MOM = -0.45: alpha RISES vs FF5-only |
| ASSET_GROWTH | +7.3%  | 3.60 | b_CMA = 0.55, still independent alpha  |
| REV_1M       | +8.8%  | 2.94 | cost-fragile by construction           |
| CFP          | +6.3%  | 3.13 | b_RMW = 0.81, b_HML = 0.56             |

Robustness (ann. Sharpe / FF5+MOM alpha t):

| factor       | EW          | EW no-imputed | EW ex-microcap | VW            |
|--------------|-------------|---------------|----------------|---------------|
| BM           | 0.90 / 5.15 | --            | 0.18 / 0.93    | -0.00 / 0.26  |
| ASSET_GROWTH | 0.94 / 3.60 | 0.95 / 3.70   | 0.48 / 3.72    | -0.02 / 0.10  |
| REV_1M       | 0.72 / 2.94 | 0.72 / 2.93   | 0.26 / 0.21    | -0.24 / -1.49 |
| CFP          | 0.71 / 3.13 | --            | 0.39 / 2.00    | -0.02 / 0.41  |

Ex-microcap = stocks at or above the NYSE 20th-percentile mktcap
breakpoint (keeps 49.4% of stock-months, 98.6% of total mktcap).
Consistent with the large-scale replication evidence (Hou-Xue-Zhang
2020): most EW anomalies do not survive value-weighting.

## Validation

- Planted-signal sandbox (sandbox_planted_signal.py; 100 simulated
  panels of 2,000 stocks x 120 months, fixed seeds): the planted
  signal is recovered in every panel (median t = 12.8, min 8.1);
  a placebo and a look-ahead canary exceed |t| = 2 in 5% and 2% of
  panels (nominal 5%); the canary fires in every deliberately
  misaligned run (median t = 15.7); forward returns are NaN across
  a deleted month (no stitching).
- Alignment canary: corr(BM long-short, HML, same month) = 0.51.
- Ken French data check: Oct-2008 Mkt-RF = -17.2%; 216 months, no NaN.
- Batch-2 numpy kernel verified against the statsmodels reference
  pipeline (max coefficient diff 7e-15). Peak RSS 0.62 GB (8 GB Mac).
- mktcap strictly positive (validates VW weights).

## Limitations (read before quoting any number)

1. Post-selection inference: alpha t-stats are computed on factors that
   already passed the multiple-testing funnel; they are conditional,
   not fresh evidence.
2. No transaction costs; REV_1M would be hit hardest.
3. EW baseline overweights microcaps by design; that is the point of
   the robustness section, not a hidden flaw.
4. BM here is industry-neutralized EW -- its t = 4.5 is NOT "value is
   back" for plain HML (HML's own 2007-24 mean is negative).
5. BETA_60M weakness partly reflects its 15.9% imputation share.
6. No raw checkpoint exists for batch-1 factors, so the no-imputed
   robustness column covers batch-2 survivors only.

## Reproduce

    python3 loader.py               # PIT panel (dedup + uniqueness assert)
    python3 factors_batch1.py       # value/profitability z-scores
    python3 factors_batch2.py       # dynamics z-scores (3 checkpointed steps)
    python3 run_factor_tests.py     # EW quintile LS + multiple testing
    python3 ff5_download.py         # Ken French FF5 + MOM (free)
    python3 run_ff5_alpha.py        # FF5(+MOM) alphas, NW t
    python3 robustness_checks.py    # VW + no-imputed variants
    python3 robustness_microcap.py  # NYSE-p20 ex-microcap variant
    python3 sandbox_planted_signal.py  # pipeline validation, simulated data (no WRDS needed)

Dependencies: pandas, numpy, statsmodels, pyarrow.
