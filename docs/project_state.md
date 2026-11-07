# Project state

- Notebooks 00, 01 and 01a: executed and reviewed.
- Notebook 02: full-history machine checks and Codex technical review completed.
- Notebook 03: full-history machine checks and Codex technical review completed.
- Current stage: Notebook 05 design discussion.
- Notebook 04 run: `entropy_04_v1`.
- Execution scope: full available history through 2026-10-16.
- Full-history machine checks: passed.
- January technical review: completed by Codex.
- Full-history technical review: passed by Codex on 2026-11-01.
- Notebook 04 human review: pending.

## Locations

- Input: `data/processed/sampling_02_v1/sampled_10s/`.
- Features: `outputs/features/entropy/entropy_04_v1/`.
- Reports: `reports/tables/entropy/entropy_04_v1/`.
- January figures: `reports/figures/entropy/entropy_04_v1/`.

## Outstanding matters

- Short-window sample entropy can be undefined or saturated.
- Exact historical provider closure completeness remains a limitation.
- Interpolation can introduce measured regularity.
- Descriptive bands are not validated trading thresholds.
- Compare entropy candidates after regression is available.

## Next step

Discuss Notebook 05 forecasting targets and chronological validation.

## Outputs invalidated

None from the agreed upstream definitions.
Changing inputs or feature definitions requires compatible regenerated outputs.

<!-- risk-interpretation-and-targets-2026-11-05:start -->
## Earlier target discussion
Superseded by the completed Notebook 05 and current Notebook 06 records.
The side-dependent interpretation and 1/30/60-second exit agenda remains for 09-10.
<!-- risk-interpretation-and-targets-2026-11-05:end -->

<!-- notebook_05_state_v2:start -->
## Notebook 05 — current state
- Full-history labels: targets_05_v2, 130 months through October 16, 2026.
- Codex reviewed saved checks, coverage and checkpoint/file metadata.
- No reconciliation failures found; upstream label definitions unchanged.
- Notebook 06 uses the reviewed outputs and an additional April 1 information cutoff.
<!-- notebook_05_state_v2:end -->

<!-- notebook_06_state_v1:start -->
## Notebook 06 — current state
- Run: regression_06_v1; fingerprint: cef05f2e7a7f593b25c126cc8b06cb46e58872e273c34cb1b87062e937814c36.
- Features: outputs/features/regression_06_v1.
- Models: outputs/models/regression_06_v1.
- Forecasts: outputs/forecasts/regression_06_v1.
- Reports: reports/tables/regression/regression_06_v1.
- Models fit on 2018; early stopping January-March 2019.
- Comparison forecasts cover April-June 2019; final 2026 was not modeled.
- Machine checks passed; output review remains pending.
- Next: review forecasting performance and coverage before further experiments.
- No pricing, inventory or exit policy was implemented.
- No upstream numerical outputs were changed.
<!-- notebook_06_state_v1:end -->

<!-- short_ma_variation_v1:start -->
## Current model-input experiment
- Notebook: 06_gradient_boosted_regression.ipynb, appended short-MA section.
- Run: regression_06_short_ma_v1; fingerprint: 650d88ee4e8ae4272136e4e9936f039aee522d00a6482c8e0bc93292cf4c4faf.
- Models: outputs/models/regression_06_short_ma_v1.
- Fresh input tables: outputs/features/regression_06_short_ma_v1.
- Fresh forecasts: outputs/forecasts/regression_06_short_ma_v1.
- Reports: reports/tables/regression/regression_06_short_ma_v1.
- Machine checks passed; performance interpretation remains pending.
- Original regression_06_v1 models and data were preserved.
- No upstream numerical outputs were invalidated.
- Next: review matched MAE/RMSE, monthly stability and baseline-relative training/later errors.
- Deferred diagnostics: real/interpolated outcome splits, eligible-row missing rates and median calibration.
- No risk-action rules were changed; 2026 remains reserved.
<!-- short_ma_variation_v1:end -->

<!-- notebook06_group_tests_regression_06_group_tests_v1:start -->
- Experiment: regression_06_group_tests_v1; fingerprint f761bd52ecbf0f47277b5b617626c45a75644d7519394a01c3887bb8fc9138b1.
- Raymond Fendi requested input-removal tests to investigate MAE improvement.
- Agreed: 21-input reference, six grouped omissions at each horizon; 2018 training and January-March 2019 stopping remain fixed.
- Q4 2019 selects removals; Q1 2020 checks the combined candidate. 2026 remains reserved.
- MAE is primary, RMSE secondary. Undefined inputs remain missing.
- Five-active-date paired bootstrap is primary; one-date sensitivity; Holm correction across 12 tests and a separate two-test confirmation family.
- Intervals are pointwise and approximate, conditional on fixed models and the evaluation quarter. Inconclusive does not mean irrelevant.
- Selected groups: {"30s": [], "60s": []}.
- Execution checks passed. Human interpretation and any model promotion remain pending; no upstream definitions or models were replaced.
<!-- notebook06_group_tests_regression_06_group_tests_v1:end -->
