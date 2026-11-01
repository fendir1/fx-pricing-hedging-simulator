# Inventory-Aware FX Pricing and Hedging Simulator

A notebook-first research simulator investigating how market conditions,
forecasts, synthetic client flow, pricing, inventory, hedging, and risk exits
interact during historical EURUSD market replay.

## Data

Dukascopy/JForex EURUSD Level 1 tick exports are held locally in data/.
The accepted full-history ETL run covers January 4, 2016 through
October 16, 2026 and combines original and replacement exports.

Source and research timestamps use Europe/Athens. Stored integer timestamps
represent UTC Unix nanoseconds for actual elapsed-time calculations.
Raymond confirmed displayed-size units as millions of EUR.
Displayed sizes are not executed trade volumes.
One feed does not represent the entire FX market.

## Development workflow

Discuss → specify → build → execute → review.

Raymond executes the notebooks. Important choices are agreed immediately
before the relevant component is built.

## Notebook order

00_data_audit.ipynb
01_etl_and_clean_quotes.ipynb
01a_output_validation.ipynb
02_sampling_candles_and_indicators.ipynb
03_multiscale_volatility.ipynb
04_entropy.ipynb
05_targets_and_validation_design.ipynb
06_gradient_boosted_regression.ipynb
07_market_replay_and_accounting.ipynb
08_client_flow_and_pricing.ipynb
09_position_accumulation.ipynb
10_risk_exits_and_hedging.ipynb
11_simulation_and_evaluation.ipynb
12_results_and_documentation.ipynb

## Current methodology

Notebook 00 audits the original full history. Notebook 01 combines original
and downloaded EURUSD quotes, preserving source provenance.

The observed dataset remains separate from the interpolated analysis dataset.
Eligible real-quote gaps are at most 300 seconds. Interior points are spaced
10 seconds from the left anchor and become available at the right anchor.
Longer gaps remain unfilled and break analysis continuity.

Research exclusions and continuity identifiers are stored explicitly.
Raymond accepted Notebook 01 outputs for the agreed market analysis. Notebook 01a records the review and limitations.

## Environment

Use the existing Jupyter Python environment.
Initial libraries: pandas, NumPy, PyArrow, DuckDB, Matplotlib.
Each audit run records the actual versions used.

Clear saved notebook outputs before committing if they contain local
paths or raw-data examples.

## Results

Results pending execution and validation.

## Limitations

This is a research simulator.
Client behavior is synthetic unless actual client data is supplied.
Predictive accuracy does not establish trading profitability.
Simulated profitability depends on the modeling assumptions.

## Development approach

Development approach: This is an AI-assisted research project. Raymond Fendi defined the business objectives and provided FX domain direction. OpenAI Codex assisted with implementation, validation code, and documentation. CONTRIBUTIONS.md records completed design decisions, reviews, and validation.

<!-- notebook-02-sampling_02_v1:start -->
## Notebook 02 progress

- All-hours quote reconstruction, ten-second sampling, three-minute candles,
  SMA/EMA and close-to-average distances are implemented.
- Execution scope: full available history.
- Processed-prefix machine checks passed; human output review is pending.
- Notebook 01's accepted files remain unchanged.
- Interpolated observations are research inputs, not execution liquidity.
- Moving-average distance is a measurement, not a trading instruction.

Results pending execution and validation.
<!-- notebook-02-sampling_02_v1:end -->

<!-- notebook-03-volatility_03_v1:start -->
## Notebook 03 progress

Multiscale RMS movement, population SD and supporting diagnostics are implemented.
Execution scope: full available history.
Machine checks passed; human output review remains pending.
Measurements retain missing-data, reconstruction and availability information.
No execution policy or forecasting model is implemented by this notebook.

Results pending execution and validation.
<!-- notebook-03-volatility_03_v1:end -->

<!-- notebook-04-entropy_04_v1:start -->
## Notebook 04 progress

Conditional entropy, sample entropy and spread-to-price transfer entropy
are implemented with ten-second updates and five agreed lookbacks.

Execution scope: prefix through 2016-01-31.
Machine checks passed for the processed prefix.
Codex completed technical review of the January 4 saved outputs.
Expanded January outputs await review.

Short-window sample entropy can lack matching patterns or reach its upper
bound with limited support. These states are retained explicitly.

Results pending execution and validation.
<!-- notebook-04-entropy_04_v1:end -->
