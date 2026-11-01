# Project state

- Notebooks 00, 01 and 01a: executed and reviewed.
- Notebook 02: full-history machine checks and Codex technical review completed.
- Notebook 03: full-history machine checks and Codex technical review completed.
- Current stage: Notebook 04 output review.
- Notebook 04 run: `entropy_04_v1`.
- Execution scope: prefix through 2016-01-31.
- Machine checks: passed for the expanded prefix.
- Notebook 04 human review: pending.

## Locations

- Input: `data/processed/sampling_02_v1/sampled_10s/`.
- Features: `outputs/features/entropy/entropy_04_v1/`.
- Reports: `reports/tables/entropy/entropy_04_v1/`.
- Figures: `reports/figures/entropy/entropy_04_v1/`.

## Outstanding matters

- Review entropy values, support, missing estimates, timing and runtime.
- Review the initial prefix before expanding processing.
- Exact historical provider closure completeness remains a limitation.
- Interpolation can introduce measured regularity.
- Descriptive entropy bands are not validated trading thresholds.
- Compare entropy candidates after regression is available.

## Next step

Review Notebook 04 saved outputs.
Then expand processing gradually using complete continuation cells.

## Outputs invalidated

None from the accepted upstream definitions.
Notebook 04 creates a separate versioned feature output.
Changed inputs or feature definitions require compatible regenerated outputs.
