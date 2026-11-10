# Contributions

## Agreed design

Raymond Fendi defined the research objectives and supplied FX domain direction.

The agreed initial implementation is notebook-first, uses Europe/Athens,
preserves original observations, and audits the complete available history.

OpenAI Codex prepared the implementation, inline validation code, and
documentation for Notebook 00.

## Execution and review status

Notebook 01 and Notebook 01a machine checks passed. Raymond accepted the ETL outputs for the agreed market analysis. Trading results remain pending simulation and evaluation.

## Notebook 00 execution — 20261029T002746_883835Z

The full-history audit was executed and its structural accounting checks passed.
Calendar interpretation, quality findings, and research validation await
Raymond's review. No gap threshold was selected.

Reports: reports/tables/data_audit/20261029T002746_883835Z/

## Notebook 01 execution — etl_01_v3

- Raymond Fendi defined the FX research objectives and processing direction.
- OpenAI Codex supplied notebook implementation, validation code and documentation.
- The executed notebook reports passing full-history machine checks.
- Raymond's subsequent acceptance is recorded in the Notebook 01 output review below.

<!-- notebook-01-review-etl_01_v3:start -->
## Notebook 01 output review — etl_01_v3

Raymond Fendi:
- Reviewed the remaining 88 dead-zone
  intervals and accepted their exclusion from the experiment.
- Interpreted the dates as consistent with holidays, weekends and thin
  pre-holiday conditions. This is domain interpretation rather than
  independently verified evidence of each gap's cause.
- Accepted the holiday-related research treatment while retaining the
  limitation around exact historical provider closure coverage.
- Reviewed interpolation examples showing right anchor B.
- Agreed that synthetic observations must retain availability at B.
- Approved zero-size observations for price-based regime, trend,
  volatility and entropy analysis, and excluded them from P&L modeling inputs.
- Supplied the locked-quote decision: `retain_flagged`.
- Supplied the excluded-hour anchor decision: `allow_current`.

Locked-quote notes: Retain locked quotes for regime, trend, entropy and volatility analysis. Exclude them from P&L simulation inputs. The recorded flag means bid equals ask; it does not independently establish that prices were stuck.

Boundary-anchor notes: excluded hours are still actual market data, we just choose not to make it part of our analyses

CSV size denomination: `millions`.

Size-unit evidence: it's definitely millions of EUR

OpenAI Codex:
- Supplied the ETL implementation, compatibility corrections and validation code.
- Supplied Notebook 01a and the detailed review reports.
- Distinguished measured checks, user interpretation and unresolved evidence.

Machine validation passed: `True`.
Human review status: `accepted`.
Requested data changes are not implemented by Notebook 01a.
<!-- notebook-01-review-etl_01_v3:end -->

<!-- notebook-02-sampling_02_v1:start -->
## Notebook 02 — sampling_02_v1

Raymond Fendi:
- Selected three-minute mid-price candles and a separate ten-second grid.
- Selected SMA and EMA periods 15, 30, 60, 100 and 200.
- Required MA history to persist across all market-data breaks.
- Required all-hours prices to contribute to MA history.
- Confirmed the <=300-second interpolation rule takes precedence for
  wholly interpolated candles.
- Selected OHLC construction from every eligible observation.
- Accepted delayed reconstructed sampling with explicit availability.
- Requested separate close-to-average distance columns.

OpenAI Codex:
- Prepared Notebook 02, its shared helper, inline validation and documentation.
- Preserved the distinction between event time and information availability.
- Supplied synthetic anchor, month-boundary and indicator-continuity checks.
- Identified interpolation as an assumed path whose statistical effects
  require review during volatility and entropy design.

Execution scope: full available history.
Processed-prefix machine checks passed when this documentation cell ran.
Raymond's review of Notebook 02 execution outputs remains pending.
Trading results remain pending simulation and evaluation.
<!-- notebook-02-sampling_02_v1:end -->

<!-- notebook-03-volatility_03_v1:start -->
## Notebook 03 — volatility_03_v1

Raymond Fendi:
- Defined volatility's intended use in execution urgency and regime research.
- Selected 30-second, 1-minute, 5-minute and 15-minute evaluations.
- Agreed to ten-second inputs and updates.
- Agreed to RMS and population SD as separate saved measurements.
- Required fresh full lookbacks after missing-market periods.
- Retained available excluded-hour data and eligible interpolation.
- Agreed to fast/slow and largest-move diagnostics.
- Described forecast-priced limit exits and multiple clips as future applications.

OpenAI Codex:
- Prepared the implementation, inline synthetic checks and documentation.
- Distinguished movement magnitude, dispersion, direction and execution assumptions.
- Preserved input availability and context across processing boundaries.

Execution scope: full available history.
Machine checks passed when this record was written.
Raymond's acceptance of Notebook 03 execution outputs remains pending.
<!-- notebook-03-volatility_03_v1:end -->

<!-- notebook-04-entropy_04_v1:start -->
## Notebook 04 — entropy_04_v1

Raymond Fendi:
- Defined entropy's intended use in regime and risk-management research.
- Selected conditional, sample and transfer entropy as candidates.
- Selected 3, 5, 15, 30 and 60-minute lookbacks.
- Required ten-second updates and fresh history after missing periods.
- Selected five-minute spread evolution as the transfer-entropy source.
- Agreed 20% sample-entropy matching tolerance.
- Required decimal scores and low/medium/high descriptive labels.
- Agreed fixed-third label bands.
- Requested a later controlled comparison after regression is available.

OpenAI Codex:
- Prepared the notebook, calculation helper and validation code.
- Distinguished uncertainty, pattern irregularity and information transfer.
- Preserved interpolation eligibility and information availability.
- Added support diagnostics and bounded-memory processing.

Execution scope: prefix through 2016-01-04.
Machine checks passed when this record was written.
Raymond's review of Notebook 04 execution outputs remains pending.
<!-- notebook-04-entropy_04_v1:end -->

<!-- notebook-04-january-progress-entropy_04_v1:start -->
## Notebook 04 — January expansion

- Raymond Fendi executed the initial entropy notebook and supplied its outputs.
- Codex reviewed the saved January 4 reports, code, provenance and figure.
- The first-day review found no material implementation inconsistency.
- Short-window sample-entropy support limitations were identified.
- The agreed estimator definitions were retained.
- A continuous, fully warmed boundary check was added.
- Processing scope is now prefix through 2016-01-31.
- Expanded-prefix machine checks passed when this record was written.
- Expanded output review and human acceptance remain pending.
<!-- notebook-04-january-progress-entropy_04_v1:end -->

<!-- notebook-04-full-history-entropy_04_v1:start -->
## Notebook 04 — full-history execution

- Raymond Fendi executed the January extension and supplied its outputs.
- Codex reviewed the saved January reports, figure and provenance.
- All 4,144 January recorded checks were zero.
- Weekend resets and fresh lookbacks reconciled.
- Short-window sample-entropy support limitations were documented.
- Estimator definitions were retained for the planned later comparison.
- Execution has expanded to full available history through 2026-10-16.
- Full-history machine checks passed when this record was written.
- Previously completed January checkpoints were preserved.
- Full-history output review and human acceptance remain pending.
<!-- notebook-04-full-history-entropy_04_v1:end -->

<!-- notebook-04-technical-review-entropy_04_v1:start -->
## Notebook 04 — full-history technical review

Review date: 2026-11-01.

Raymond Fendi:
- Executed the full-history entropy calculations and validation cells.
- Supplied execution outputs and reported interrupted runs.
- Executed checkpoint-based recovery and full-history reporting.

OpenAI Codex:
- Reviewed saved validation reports, helper logic and file inventory.
- Checked source/dependency hashes and aggregate reconciliation.
- Confirmed all 582,972 recorded daily check values were zero.
- Documented persistent short-window sample-entropy support limitations.
- Supplied recovery code addressing temporary-file readback and sharing locks.

Technical review passed.
No feature recalculation or parameter change is required.
Estimator usefulness remains for the planned later comparison.
<!-- notebook-04-technical-review-entropy_04_v1:end -->

<!-- risk-interpretation-and-targets-2026-11-05:start -->
## Research direction — 2026-11-05

Raymond Fendi:
- Reaffirmed that the simulator should choose risk-exit timing after simulated trades.
- Defined intended exit horizons of 1, 30 and 60 seconds.
- Required joint interpretation of moving-average distances, multiscale
  volatility, entropy and gradient-boosted EURUSD forecasts.
- Required good/medium/bad interpretations relative to the dealer's exposure.
- Supplied an illustrative combination of low volatility, low entropy,
  an upward 60-second forecast and price above two SMA and two EMA lines.
- Required exits to permit one or more transactions and record execution and P&L.
- Requested a dedicated interpretation-policy discussion before implementation.
- Agreed initial targets of future mid-price changes over 30 and 60 seconds.
- Required continuity-eligible real and interpolated prices to be eligible outcomes.
- Selected the entire 2018 calendar year as the initial training period.

OpenAI Codex:
- Clarified the measurement, forecast, interpretation and action layers.
- Identified tick-level execution timing as necessary for the one-second exit option.
- Required interpolated labels to retain their complete information-availability times.
- Prepared this documentation cell.

Operational thresholds and exit mappings have not been finalized.
No forecasting model or exit policy has been implemented by this record.
<!-- risk-interpretation-and-targets-2026-11-05:end -->

<!-- notebook_05_contributions_v2:start -->
## Notebook 05 — Raymond Fendi's agreed design
- Defined the simulator's purpose: use trend, MA distances, volatility and entropy
  to decide how quickly to dispose of a simulated position.
- Requested side-dependent good/medium/bad interpretation and intended exits at
  1, 30 or 60 seconds, potentially in several clips.
- Requested a dedicated interpretation discussion before trading rules are built.
- Chose 10-second examples, 30/60-second forecasts, eligible synthetic outcomes,
  2018 training, first-half 2019 validation and the 2026 final evaluation period.
- Requested both predicted change in pips and the corresponding expected rate.
- OpenAI Codex supplied the replacement notebook, helper code and validation code.
- Execution evidence for this run: full history, 3939 days through 2026-10-16; machine checks passed.
- Human acceptance of these replacement outputs remains pending.
- Good/medium/bad thresholds and their exit mapping remain provisional.
- Results pending execution and validation.
<!-- notebook_05_contributions_v2:end -->

<!-- notebook_06_contributions_v1:start -->
## Notebook 06 — agreed contributions
- Raymond Fendi chose LightGBM and two joint models using all six feature groups.
- Requested every agreed SMA and EMA lookback, plus sample and conditional entropy.
- Chose MAE primary, RMSE secondary and MAE training loss.
- Agreed the six price-change lookbacks and the two 2019 validation roles.
- Chose to retain valid examples with missing inputs.
- Requested concise performance tables and a 30-second forecast-price example.
- Clarified that P&L must come from simulated trades and accounting.
- Requested later evaluation of each group's additional forecasting value.
- OpenAI Codex supplied the notebook, implementation and inline validation code.
- This executed run produced two LightGBM models and benchmark comparisons.
- Output interpretation and human review remain pending.
<!-- notebook_06_contributions_v1:end -->

<!-- short_ma_variation_v1:start -->
## Short-MA variation
- Raymond Fendi proposed input experiments to improve forecasting and selected the shortest MA pair.
- Agreed July-September 2019 as the fresh validation period.
- Requested keeping the experiment in Notebook 06 with an explanatory Markdown cell.
- OpenAI Codex verified the shortest pair is SMA 15 / EMA 15 and supplied the appended cells.
- This executed variation retrained two models and produced matched comparison reports.
- Interpretation, improvement and any overfitting conclusion remain pending review.
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

<!-- notebook06_regression_06_sizes_spread_v1:start -->
- Run: regression_06_sizes_spread_v1; fingerprint ba65a1f9828279dc1e34fa16cc94b8eb92f6c167fd414ee17ca8a97ff095f62b.
- Raymond Fendi requested a two-model variation using only bid size, ask size and spread following the reported group-removal results.
- This is a requested joint-removal experiment; the omitted groups have not been established as irrelevant.
- Training remains all of 2018; stopping remains January-March 2019. Parameters, MAE objective and valid training rows remain fixed.
- April-June 2020 is the fresh development comparison; Q4 2019 and Q1 2020 informed this recipe and are already used development evidence. 2026 remains reserved.
- Compare both horizons with frozen 21-input models and zero change; paired five-active-date bootstrap, one-date sensitivity, Holm family of four.
- MAE is primary and RMSE secondary. Missing inputs remain missing. Forecasts retain availability and information cutoffs.
- Execution checks passed. Results await human interpretation; no model was promoted and no upstream output was replaced.
<!-- notebook06_regression_06_sizes_spread_v1:end -->
