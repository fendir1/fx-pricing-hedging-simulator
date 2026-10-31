# Project state

- Current stage: Notebook 01 and Notebook 01a review complete.
- ETL run: `etl_01_v3`.
- Notebook 00: full original-history audit executed.
- Notebook 01: full-history machine validation passed.
- Notebook 01a: review reports generated.
- Human ETL review: `accepted`.
- Size denomination: millions of EUR, confirmed by Raymond.

## Locations

- Original CSVs: `data/`.
- Downloaded exports: `data/2017 2018 investigation/`.
- Observed quotes: `data/processed/etl_01_v3/observed/`.
- Analysis quotes: `data/processed/etl_01_v3/analysis/`.
- Review report: `reports/tables/etl/etl_01_v3/01a_output_validation.html`.
- Review answers: `reports/tables/etl/etl_01_v3/01a_review_answers.json`.

## Outstanding matters

- CSV size denomination is confirmed by Raymond as millions of EUR.
- Exact historical provider closure completeness remains a documented limitation.
- Locked quotes and zero-size observations are excluded from P&L modeling inputs.
- Execution and valuation handling will be designed during accounting.
- Locked-quote decision: `retain_flagged`.
- Boundary-anchor decision: `allow_current`.

## Outputs affected

No additional outputs invalidated by this review.

## Next step

Discuss Notebook 02 sampling, candles and indicators before building.
