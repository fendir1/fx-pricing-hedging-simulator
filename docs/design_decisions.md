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
