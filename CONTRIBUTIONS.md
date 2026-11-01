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
