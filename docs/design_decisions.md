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
