# Design decisions

| Decision | Rationale | Component | Status |
|---|---|---|---|
| Audit every original CSV | Establish complete input coverage | 00 | Agreed |
| Source and display timezone Europe/Athens | Preserve agreed source-clock convention | All timing | Agreed baseline; validate against observations |
| Use actual elapsed time | Avoid DST arithmetic errors | All timing | Agreed |
| Preserve repeated observations | Repetition is not automatically corruption | 00–01 | Agreed |
| Use 56 specified gap buckets | Inspect the distribution before choosing a cutoff | 00 | Agreed |
| Keep public, settlement, and quote-closure calendars distinct | Avoid false closure exclusions | 00–01 | Agreed |
| 24/31 December are open holiday-affected EURUSD sessions | Retain thin-session observations | 00–01 | Agreed baseline |
| Disruption threshold unset | Requires distribution review and a chronological reference period | 00–02 | Pending |
| No quote filtering or gap filling in audit | ETL choices require discussion | 00–01 | Agreed |
| Holiday context scope is EUR/US plus agreed year-end dates | Other regional and exceptional events may remain | 00 | Provisional |

## Notebook 01 — etl_01_v3

- Decision: combine original and downloaded EURUSD exports.
  Rationale: use recovered observations before evaluating remaining gaps.
  Component: ETL.
  Status: agreed.

- Decision: collapse equal timestamps only when all four parsed quote fields agree.
  Rationale: remove overlapping copies while preserving later repeated quotes.
  Component: observed quotes.
  Status: agreed; conflicting values require review.

- Decision: gaps of 300 seconds or less qualify for linear interpolation.
  Rationale: agreed research treatment of short quote gaps.
  Component: analysis output.
  Status: agreed.

- Decision: synthetic timestamps start 10 seconds after A; B remains fixed.
  Rationale: preserve the agreed spacing and anchors.
  Component: interpolation.
  Status: agreed.

- Decision: synthetic availability is B, with no execution use.
  Rationale: B is future information until it arrives.
  Component: features and historical replay.
  Status: agreed correctness requirement.

- Decision: raw gaps above 300 seconds are never filled.
  Rationale: keep dead zones unavailable to market-based analysis.
  Component: continuity.
  Status: agreed.

- Decision: exclude first two and last one New York session hours,
  expected weekends, January 1 in Athens and verified quote closures.
  Rationale: agreed analysis scope.
  Component: analysis output.
  Status: agreed.

- Decision: retain only allowed synthetic timestamps, while permitting anchors
  just inside excluded periods.
  Rationale: complete eligible boundary gaps using real observations.
  Component: interpolation boundaries.
  Status: provisional assumption for human review.

- Decision: retain locked quotes and zero displayed sizes with flags;
  invalid inputs stop processing.
  Rationale: preserve evidence without silently inventing replacement quotes.
  Component: quote quality.
  Status: conservative implementation; human review pending.

- Decision: retain source size units.
  Rationale: absolute size denomination remains unconfirmed.
  Component: sizes.
  Status: provisional.

- Validation: full-history machine checks passed for this run.
  Human review: pending.

<!-- notebook-01-review-etl_01_v3:start -->
## Accepted ETL output review — etl_01_v3

This review supersedes earlier pending or provisional review labels
for the same ETL run.

- Reviewed dead-zone exclusions: accepted by Raymond.
- Holiday treatment: accepted with exact historical provider closure
  completeness retained as a documented limitation.
- Locked quotes: retained with flags for regime, trend, entropy and
  volatility analysis; excluded from P&L simulation inputs.
- Zero-size observations: retained for agreed market analysis;
  excluded from P&L simulation inputs.
- Displayed-size denomination: millions of EUR, confirmed by Raymond.
- Real anchors inside excluded hours: permitted.
- Generated timestamps inside exclusions: remain excluded.
- Synthetic availability: must equal right anchor B.
- Machine validation: passed.
- Human review: accepted.
- Analysis regeneration: not required.

Locked means bid equals ask; that flag alone does not establish stuck prices.
P&L execution and valuation handling will be designed during accounting.
<!-- notebook-01-review-etl_01_v3:end -->

<!-- notebook-02-sampling_02_v1:start -->
## Notebook 02 — sampling_02_v1

This section records the agreed Notebook 02 definitions.
It extends feature-input scope without changing accepted Notebook 01 files.

- Decision: form fixed, clock-aligned three-minute mid OHLC candles from
  every real and eligible interpolated observation.
  Rationale: preserve available intraperiod highs and lows.
  Component: candles.
  Status: agreed.

- Decision: retain a separate clock-aligned ten-second sampled table,
  using backward lookup from the reconstructed history.
  Rationale: provide a regular research grid without replacing tick OHLC.
  Component: sampling.
  Status: agreed.

- Decision: use all-hours prices for candle and moving-average history.
  Rationale: project-excluded hours still contain useful observed prices.
  Component: candle and MA input scope.
  Status: agreed.

- Decision: interpolate actual real-anchor intervals of at most 300 seconds,
  including exactly 300 seconds; synthetic timestamps remain A+10 seconds.
  Rationale: preserve the agreed short-gap reconstruction.
  Component: interpolation.
  Status: agreed.

- Decision: allow wholly interpolated candles with anchors outside their window.
  Rationale: the 300-second eligibility rule takes precedence.
  Component: candles and later analytical inputs.
  Status: agreed; retain interpolation flags.

- Decision: never fill intervals longer than 300 seconds.
  Rationale: preserve missing periods rather than inventing prices.
  Component: missing data.
  Status: agreed.

- Decision: SMA and EMA use 15, 30, 60, 100 and 200 populated completed closes.
  Rationale: agreed trend measurements.
  Component: indicators.
  Status: agreed.

- Decision: retain MA history across every break; empty windows do not count.
  Rationale: continue the observed candle sequence after reopening.
  Component: indicator continuity.
  Status: agreed; supersedes the earlier proposed dead-zone reset.

- Decision: seed each EMA from its first N-candle SMA and use alpha=2/(N+1).
  Rationale: transparent initialization with a full lookback.
  Component: EMA.
  Status: agreed implementation.

- Decision: record close-minus-average distances in EURUSD pips.
  Rationale: measure position relative to historical price averages.
  Component: indicators.
  Status: agreed; trading interpretation remains for later design.

- Decision: retain exclusion and long-gap-overlap flags separately.
  Rationale: MA history uses all hours while experiment scope remains visible.
  Component: output interface.
  Status: agreed.

- Implementation: publication conservatively resolves the trailing real interval,
  using B or the first nanosecond beyond the 300-second limit.
  Indicator rows inherit prior dependencies using exact integer nanoseconds.
  Component: information availability.
  Status: correctness rule; machine checks supplied.

- Execution scope: full available history.
- Machine checks for the processed prefix: passed.
- Human review: pending.
- Fingerprint: `ad5235dc14f854d9f26ef04dd2c1fdf846faf05cff43eb9f84afccc229628e54`.
- No volatility estimator, entropy estimator or trading policy is selected here.
<!-- notebook-02-sampling_02_v1:end -->

<!-- notebook-03-volatility_03_v1:start -->
## Notebook 03 — volatility_03_v1

- Input: all-hours ten-second sampled mids from `sampling_02_v1`.
- Lookbacks: 30 seconds, 1 minute, 5 minutes and 15 minutes.
- Measurements: RMS movement and centred population standard deviation.
- Units: pips of ten-second change; no annualisation.
- Diagnostics: largest-change share and interpolation-supported-change fraction.
- Comparison: RMS 1-minute / RMS 15-minute, undefined at a zero denominator.
- History: require complete consecutive windows after missing periods.
- Excluded hours with available prices continue the measurement history.
- Interpolation remains eligible, including entirely reconstructed support.
- Availability: conservative exact integer maximum over supporting prices.
- No learned thresholds, regime classifications or execution rules are fitted.
- Status of these definitions: agreed.
- Forecast-priced limit exits and execution clips remain provisional applications.
- Execution scope: full available history.
- Saved machine checks: passed.
- Human review of Notebook 03 outputs: pending.
- Fingerprint: `23cd20f31245e3dd1ff03f508c7f3cd978d581ac41838901cd18a062207499ad`.
<!-- notebook-03-volatility_03_v1:end -->

<!-- notebook-04-entropy_04_v1:start -->
## Notebook 04 — entropy_04_v1

Status: agreed definitions; execution-output human review pending.

- Inputs: Notebook 02 all-hours ten-second reconstructed market series.
- Lookbacks: 3, 5, 15, 30 and 60 minutes; update every ten seconds.
- Price symbols: down, unchanged, up.
- Conditional entropy uses the previous price symbol.
- Save normalized conditional uncertainty and residual-uncertainty diagnostic.
- Sample entropy uses length-two patterns extended to length three.
- Its tolerance is 20% of trailing population price-change SD.
- Save raw sample entropy and bounded irregularity score 1-A/B.
- Preserve undefined and infinite-raw cases with support counts.
- Transfer source: five-minute spread-change sign at the preceding step.
- Transfer target: following ten-second price-change sign.
- Condition transfer entropy on the preceding price-change sign.
- High TE denotes additional spread information.
- Labels use fixed thirds and are descriptive, not trading rules.
- Restart after unfilled missing periods; require fresh full support.
- Available excluded-hour data and eligible interpolation remain included.
- Preserve exact supporting availability timestamps.
- Numerical comparison tolerance: 1e-08 pips.
- No full-history fitted bins, thresholds or normalizations.

Validation scope: prefix through 2016-01-04.
Machine checks passed when this record was written.
Short-window statistical usefulness remains for empirical evaluation.

Provisional application:
Combine measurements with volatility, trend forecasts and MA distances.
No inventory, pricing or exit action is implemented here.

Later comparison:
After Notebook 06, compare a no-entropy baseline with each entropy candidate
on a one-month development experiment. Use matching observations and
chronological evaluation. Define success independently of each entropy score.
Confirm promising findings on later untouched data.
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
## Inventory-aware interpretation and forecasting design

Status:
- Business objective and initial forecasting choices: agreed.
- Indicator thresholds, combined grades and operational exit mappings: provisional.

Decision:
After a simulated trade, interpret the available indicators jointly and
relative to the dealer's position. Produce good/medium/bad assessments
and select an intended exit horizon from 1, 30 or 60 seconds.

Rationale:
The research investigates whether indicator-guided risk retention and
disposal improve outcomes under explicit simulation assumptions.

Execution:
Allow one or more clips. Record desired exits separately from actual fills,
remaining inventory and P&L.

Required design discussion:
- Notebook 09: interpretation's interaction with inventory retention.
- Notebook 10: dedicated indicator criteria, aggregation, conflicting
  signals, risk overrides, exit timing and clip execution.
- Notebook 11: controlled policy comparison and evaluation.

The illustrative low-volatility/low-entropy/upward-forecast case is
a candidate policy pattern, not a finalized rule.

Notebook 05 agreements:
- Signed future mid-price changes in pips.
- Forecast horizons: 30 and 60 seconds.
- Continuity-eligible real and interpolated outcome prices.
- Initial training period: calendar year 2018.
- Training eligibility and purging must respect complete label information,
  including any required right anchor after the nominal forecast endpoint.

Still to agree:
Validation and final-evaluation periods, training-row frequency,
endpoint freshness and exclusion-boundary handling.
<!-- risk-interpretation-and-targets-2026-11-05:end -->

<!-- notebook_05_targets_v2:start -->
## Notebook 05 — target design
- Agreed: examples every 10 seconds; future mid changes over 30 and 60 seconds.
- Reference: latest real quote known at issue time; retain its timestamp and age.
- Future prices: eligible real and reconstructed observations.
- Reject intervals crossing exclusions or disruptions. Keep each horizon independent.
- Preserve valid synthetic outcomes before a dead zone.
- Label availability: latest information time across the whole interval.
- Purge labels not fully available before their assigned split ends.
- Train: 2018. Validate: January–June 2019. Final evaluation: 2026 through October 16.
- Notebook 06 will output both predicted pips and expected EURUSD rates.
- No model or trading rule is fitted here.
<!-- notebook_05_targets_v2:end -->

<!-- notebook_06_design_v1:start -->
## Notebook 06 — initial regression experiment
- Agreed: two LightGBM models, predicting 30- and 60-second price changes.
- MAE training loss; conditional median point forecasts. No P&L prediction.
- MAE in pips is primary; RMSE in pips is secondary.
- Six groups: past changes, displayed sizes, spread, MA distances, SampEn and CE.
- Past changes: 10/30/60 seconds and 3/5/15 minutes.
- All ten completed-candle SMA/EMA distance columns: 15/30/60/100/200.
- Both entropy families use 3/5/15/30/60-minute scores.
- Retain valid target examples with missing features; native missing handling.
- Current mid/spread/sizes use the last known real quote.
- Historical reconstructed prices enter only after their information is available.
- Past changes need fresh continuous history after actual data gaps.
- Quotes in excluded clock hours can provide history; target exclusions still apply.
- Preserve MA history and the established entropy restart rules.
- Fit on 2018; choose tree count using January-March 2019.
- Early-stopping label information must be available before April 1.
- Compare the fixed models on April-June 2019. Keep 2026 untouched.
- Conservative single parameter configuration; no tuning search or blending.
- Benchmarks: zero change, trailing horizon change and ridge.
- Trailing baseline falls back to zero change when its history is unavailable.
- Ridge uses training-only mean imputation and scaling; source features stay missing.
- Forecast timestamps cover both fitting and early-stopping information.
- April 1 model readiness is a research clock assumption, not measured fit latency.
- Zero modeled inference latency is provisional until replay design.
- Export forecasts without future outcomes or future eligibility flags.
- Later group-removal tests require a separate agreed diagnostic experiment.
- After April-June informs changes, assess the revised specification on a later development block.
- Feature gain is descriptive; it is not a statistical-significance test.
<!-- notebook_06_design_v1:end -->

<!-- short_ma_variation_v1:start -->
## Short-MA input variation
- Raymond Fendi requested testing the shorter MA input combination after weak baseline gains.
- Keep SMA 15 and EMA 15 distance inputs; remove the eight longer-MA distances.
- Other feature groups, labels, 2018 training and January-March stopping rules remain unchanged.
- Retrain only the two reduced models, with the original parameters and budget.
- April-June is development evidence; July-September is fresh validation.
- Require fresh-validation label information before October 1, 2019.
- The reduced model's information cutoff includes the April-June-informed input choice.
- Its research-clock availability begins July 1; April-June predictions are retrospective.
- Keep original models and upstream data; use separate versioned result files.
- Compare matched observations using MAE primary and RMSE secondary.
- Fewer inputs do not by themselves establish better forecasting or prove overfitting.
- No significance test or automatic model promotion is performed.
- Further changes informed by July-September require a later fresh development block.
- Keep 2026 reserved.
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
