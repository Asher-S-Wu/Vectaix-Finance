<div align="center">

# Vectaix Finance

Independent US, A-share and Hong Kong stock models, with trained weights and historical research results.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ridge](https://img.shields.io/badge/Model-Ridge-546E7A)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-00866F)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![GitHub stars](https://img.shields.io/github/stars/Asher-S-Wu/Vectaix-Finance)](https://github.com/Asher-S-Wu/Vectaix-Finance/stargazers)

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

</div>

Vectaix Finance has separate research pipelines for mainland A shares in CNY and Hong Kong equities in HKD. The A-share release includes trained Ridge, factor-score and LightGBM models. The Hong Kong section documents a monthly stock-selection strategy and its trade-by-trade replay.

[Independent US model](#us) · [A-share model](#ashare) · [Hong Kong backtest](#hong-kong) · [Run locally](#run-locally)

<a id="us"></a>

## Independent US-equity model: Experiment 2

**Recovered history and trained ranking weights improve the reused recent comparison, but a reliable out-of-sample advantage is still unproven.** The frozen `ridge_rank_5y` has positive recent IC point estimates, yet all four 95% intervals cross zero. Probability skill is negative at every horizon; the chronological nested outer mean is only **+0.0009**. The model remains ineligible for live use.

The same OEF cohort contains **101 original share classes**, dated **2015-06-30** and published **2015-09-02**. All **12** previously unavailable identities now have bounded original-security history, adding **15,284** valid quotes. This means some historical coverage for 101/101 identities, not a complete point-in-time universe or complete terminal wealth. FOXA's source ends on **2018-03-27**, before its 2019-03-19 termination; DD begins on **2015-07-02** after an unmodeled noncash spinoff. Unsupported merger, cash-settlement and CVR outcomes remain unknown.

### Rolling development and chronological selection

![Six predeclared US v2 candidates: 2020–2024 annual common-cohort IC and equally weighted mean](docs/assets/us/v2/rolling_selection.png)

Six predeclared candidates compare the fixed factor reference, 12–1 momentum, and independently trained Ridge/LightGBM ranking models with 3- or 5-year windows. The learned rank models use a fixed 24-feature input set and centered within-date return-rank targets; the baselines retain their declared scores. There are **30 candidate-fold fits**, with training before the preceding calibration year and evaluation on common rows/dates in each 2020–2024 fold. The selected five-year Ridge mean is **+0.0271**, positive in 3/5 years; this statistic selects the winner and is not its independent test.

![Nested outer selection uses only earlier inner folds; outer ICs -0.0175, +0.0275 and -0.0075](docs/assets/us/v2/nested_selection.png)

The nested choices use 2020–2021 to select the 2022 architecture, then only earlier folds for 2023 and 2024. All choose five-year Ridge. Outer IC is **−0.0175 / +0.0275 / −0.0075**, averaging **+0.0009** with only 1/3 positive years. This is stronger chronological separation, still subject to retrospective-source and cohort limits.

### REUSED DIAGNOSTIC: 2025–2026 is not a new holdout

![US v2 frozen Ridge rank IC with 95% block-bootstrap intervals, on the reused recent diagnostic](docs/assets/us/v2/reused_diagnostic_ic.png)

The final four horizon heads train through 2023, calibrate on 2024 and stay frozen for **2025-01-01–2026-09-30**. That period was already examined in Experiment 1, so it cannot become fresh confirmation for Experiment 2.

| Horizon (sessions) | Mean rank IC | 95% interval | Brier skill | Interval coverage |
| --- | ---: | --- | ---: | ---: |
| 1 | +0.0083 | [−0.0042, +0.0211] | −0.012% | 79.40% |
| 5 | +0.0158 | [−0.0131, +0.0438] | −0.026% | 81.71% |
| 20 | +0.0242 | [−0.0255, +0.0696] | −0.286% | 82.37% |
| 60 | +0.0705 | [−0.0040, +0.1415] | −0.677% | 83.63% |

On **37,113 identical mature stock-date outcomes / 417 IC dates**, the paired 20-session IC changes from v1 **−0.0772** to v2 **+0.0233**, a **+0.1005** change (95% paired interval **[+0.0211, +0.1788]**). This common-universe value differs from +0.0242 above because the latter includes additional recovered WBA observations. Model and training-data changes are coupled; this posthoc comparison neither isolates the cause nor establishes fresh confirmation.

![US v2 probability Brier skill and q10–q90 coverage with mature-outcome denominators](docs/assets/us/v2/calibration_checks.png)

IC intervals use 2,000 circular 60-session bootstrap draws. Brier skill compares with the historical-frequency baseline; negative means higher error. Interval coverage targets 80%. Overlapping outcomes are not independent samples. The **2026-09-30 latest refit is untested out of sample**.

**Full-period portfolio performance is unavailable.** The selected strategy, cost-stress replay and 101-slot cohort retain an unresolved WBA holding after 2025-08-27, with **274 valuation-gap sessions**. Their total return, CAGR and maximum drawdown remain null. Stale-reference values are not validated returns; no wealth curve is promoted as performance.

[Methods and boundaries](docs/us/v2/README.md) · [Recovered-history chart](docs/assets/us/v2/source_recovery.png) · [Rolling CSV](docs/showcase/us/v2/rolling_selection.csv) · [Nested CSV](docs/showcase/us/v2/nested_selection.csv) · [Diagnostic CSV](docs/showcase/us/v2/reused_diagnostic.csv) · [Paired comparison](docs/showcase/us/v2/paired_recent.csv) · [Replay status](docs/showcase/us/v2/replay_status.csv) · [Hashes and definitions](docs/showcase/us/v2/summary.json)

Rebuild these five v2 charts and aggregate CSVs without data downloads or training: `python -m scripts.build_us_v2_showcase`

The initial experiment remains intact below.

<a id="us-v1"></a>

## Independent US-equity model: Experiment 1

**The frozen confirmation does not establish a ranking advantage.** Mean rank IC is negative at all four horizons; the 20- and 60-session 95% intervals are wholly below zero. Return-interval coverage is below 80% at every horizon. This model is not validated for live use.

This initial public-data run uses a separate USD model trained on a fixed historical cohort of **101 original equity share classes** from the OEF holdings dated **2015-06-30**, published **2015-09-02**. Public Yahoo Finance history spans 2014-01-02 to 2026-09-30. Limited provider-name and history-window review accepts **89 histories**; **12 original identities** remain excluded or unavailable: five unresolved lineages, five missing histories and two reused tickers. Every original identity remains in the prediction denominator. This is bounded, current-vintage retrospective research, not a survivorship-free or full-US-market estimate. AKShare was checked as a secondary audit; incomplete adjustment history and reused symbols prevent it from being the training source.

The US pipeline fits its own factor, Ridge, small LightGBM and large LightGBM candidates without reusing A-share or Hong Kong trained weights. Training uses 2016–2022, calibration uses 2023, and model choice uses 2024. Only the selected architecture is then evaluated with frozen parameters on 2025-01-01 through 2026-09-30 confirmation data.

### US frozen confirmation

![US frozen Factor score: daily rank IC and 95% 60-session block-bootstrap intervals](docs/assets/us/confirmation_ic.png)

| Horizon (sessions) | Mean daily rank IC | 95% interval | Probability Brier skill | Return-interval coverage |
| --- | ---: | --- | ---: | ---: |
| 1 | -0.0095 | [-0.0254, 0.0064] | -0.02% | 72.95% |
| 5 | -0.0271 | [-0.0562, 0.0019] | +0.08% | 73.82% |
| 20 | -0.0772 | [-0.1233, -0.0200] | +0.06% | 74.67% |
| 60 | -0.0989 | [-0.1562, -0.0426] | -2.89% | 69.57% |

Rank IC is the daily cross-sectional Spearman correlation between score and subsequent adjusted-price return. The 95% intervals use 2,000 circular 60-session block-bootstrap draws, preserving missing-day slots. Metrics are conditional on observed mature outcomes; the cohort's historical gaps and provider adjustments limit interpretation.

### US model choice and calibration checks

![Four US candidate models compared on identical 2024 stock-date rows and IC dates](docs/assets/us/model_selection.png)

The selected factor score uses preset ranking weights with probability and return-interval calibration fitted on US data.

**Factor score** was selected with 20-session mean daily rank IC of **0.0292**, on the same **232 IC dates** and **20,648 mature stock-date outcomes** for all four candidates. Confirmation data did not select or tune the model.

![US frozen-model probability Brier skill by horizon against the historical-frequency baseline](docs/assets/us/probability_skill.png)

Zero matches the historical-frequency probability baseline; a negative value means higher Brier error. Positive ranking IC does not by itself establish probability skill.

![US observed q10–q90 return-interval coverage against the nominal 80% target](docs/assets/us/interval_coverage.png)

The q10–q90 return interval targets 80% coverage. Counts include only mature, task-available outcomes; overlapping stock-date outcomes are not independent samples. The separate 2026-09-30 latest refit has **no out-of-sample evaluation**. Execution remains unvalidated and eligibility is false; these forecast diagnostics do not establish investable portfolio returns or future performance.

### Conditional US research replay

![Experiment 1 conditional adjusted-unit replay: selected factor model trails both references](docs/assets/us/conditional_replay.png)

The frozen factor strategy returned **+16.01%** in the conditional research replay, below the **+27.40%** fixed 101-slot cohort and **+30.84%** SPY adjusted-price proxy. Doubling one-way costs from 15 to 30 bps reduced the model result to **+10.91%**. This is a monthly top-10, next-session-close simulation with a 95% invested target. Missing cohort slots stay cash. Average equity exposure is 94.70% for the model and 83.83% for the cohort, so this is not an exposure-matched comparison. The USD source-adjusted units and returns are research illustrations; execution and the actual-share corporate-action ledger are unvalidated.

[Replay curves CSV](docs/showcase/us/conditional_replay.csv) · [Replay metrics CSV](docs/showcase/us/conditional_replay_summary.csv) · [Replay assumptions](backtests/us/oef2015/replay_summary.json)

[US source and method guide](docs/us/README.md) · [Source audit](backtests/us/oef2015/source_audit.json) · [Frozen candidates](models/us/oef2015/frozen/) · [Latest refit](models/us/oef2015/latest/) · [Selection CSV](docs/showcase/us/model_selection.csv) · [Confirmation CSV](docs/showcase/us/confirmation.csv) · [Chart definitions and hashes](docs/showcase/us/summary.json)

Regenerate these five charts and CSVs from aggregate evidence, without market-data downloads or training: `python -m scripts.build_us_showcase`

<a id="ashare"></a>

## A-share model

The A-share pipeline calculates 37 price, volume, capitalization and market-context factors. It compares a fixed factor score, Ridge, small LightGBM and large LightGBM on the same score-available universe. Ridge was selected using 2024 development data and mean daily rank IC at the 20-session horizon. It uses 10 preset base features plus `log_horizon`.

Fitting uses 2016 to 2022, followed by separate calibration in 2023. The selected model is frozen for confirmation from 2025 through September 30, 2026; confirmation data is not used for selection or tuning. The published confirmation results are:

![Frozen Ridge confirmation: mean daily rank IC and 95% block-bootstrap intervals at 1, 5, 20 and 60 sessions](docs/assets/cn/confirmation_ic.png)

| Horizon (trading sessions) | Mean daily rank IC | 95% interval | Probability Brier skill |
| --- | ---: | --- | ---: |
| 1 | 0.0608 | [0.0455, 0.0758] | -0.35% |
| 5 | 0.0822 | [0.0503, 0.1145] | -0.54% |
| 20 | 0.1111 | [0.0608, 0.1626] | -0.22% |
| 60 | 0.1610 | [0.0812, 0.2450] | -1.15% |

Rank IC measures the daily cross-sectional Spearman correlation between scores and subsequent returns. Intervals use a 60-session circular block bootstrap with 2,000 repetitions. These estimates describe historical ranking quality. Probability Brier skill is negative at every horizon, below the historical-frequency baseline. A separate latest Ridge refit dated September 30, 2026 supplies weights for current-input research and has not been evaluated out of sample.

Unresolved corporate actions and terminal security valuations prevent verification of portfolio returns, CAGR and maximum drawdown. The snapshot records `eligible=false` and `execution_validated=false`. Historical results do not establish future performance.

The release contains four [frozen candidate models](models/cn/universal/frozen/) and the [latest Ridge model and metadata](models/cn/universal/snapshots/cn-linear-latest-20260930-v1/). Raw vendor inputs and per-security prediction snapshots require separately authorized Tushare data. The public package has no active service snapshot, so a fresh clone supports model inspection and synthetic tests; real-stock rankings require complete local inputs.

[Public release guide (Chinese)](ASHARE_RELEASE.zh-CN.md) · [Research report (Chinese PDF)](delivery_report/ashare_training_report.zh-CN.pdf) · [Method and reproduction](docs/ashare/README.md) · [Confirmation metrics](backtests/cn/universal/confirmation_summary.json) · [A-share source](ashare_quant/)

### Model selection on the development set

![2024 common-universe 20-session IC: Ridge 0.0556, large LightGBM 0.0264, small LightGBM 0.0016, factor score -0.0128](docs/assets/cn/model_selection.png)

Ridge led the saved 2024 comparison, with mean daily rank IC of 0.0556. All four candidates use the same 222 IC dates and 1,175,479 mature stock-date outcomes. The chart shows the development statistic that selected the frozen model.

[Model comparison CSV](docs/showcase/cn/model_selection.csv) · [Saved selection](backtests/cn/universal/frozen_architecture.json)

### Probability and return-interval checks

![Frozen Ridge probability Brier skill: -0.35%, -0.54%, -0.22% and -1.15% at 1, 5, 20 and 60 sessions](docs/assets/cn/probability_skill.png)

Zero matches the historical-frequency probability baseline. Negative skill means the model has higher Brier error than that baseline, even though its ranking IC is positive.

![Observed q10–q90 return-interval coverage: 80.02%, 80.32%, 78.28% and 77.26%, against an 80% target](docs/assets/cn/interval_coverage.png)

The predicted q10–q90 return interval targets 80% coverage. Observed coverage is 80.02%, 80.32%, 78.28% and 77.26% at 1, 5, 20 and 60 sessions. Each denominator includes only mature outcomes with an available interval; overlapping stock-date outcomes are not independent samples. These are frozen-model confirmation results; the latest refit's performance remains unevaluated.

[Confirmation CSV](docs/showcase/cn/confirmation.csv) · [Chart sources and definitions](docs/showcase/cn/summary.json)

<a id="hong-kong"></a>

## Hong Kong backtest

The Hong Kong pipeline combines 37 price, volume, and market features to select 30 Hong Kong stocks each month. In the historical backtest from February 1, 2024 to September 16, 2026, the small LightGBM model returned **35.56% annualized after fees**, taking HK$1 million to **HK$2.22 million**.

![Historical backtest: model wealth, Hang Seng benchmarks, and drawdown](docs/assets/performance.png)

### Returns against the indices

The comparison covers **February 1, 2024–September 16, 2026**, or 644 trading sessions. All four models rank the same score-available universe at each month-end, select the top 30 by their 20-session score, and simulate execution at the next session's close. Initial cash is HK$1 million, fees are 0.25% per side, and trading-value participation is capped at 1%.

| Model / benchmark | Annualized return | Total return | Maximum drawdown | Profitable positions |
| --- | ---: | ---: | ---: | ---: |
| **Small LightGBM** | **35.56%** | **122.09%** | -20.10% | 49.16% |
| Large LightGBM | 32.28% | 108.30% | -15.49% | 49.21% |
| Linear | 13.37% | 38.98% | -10.36% | 45.32% |
| Factor score | 5.83% | 16.02% | -21.38% | 51.78% |
| Hang Seng Index | 19.27% | 58.77% | -19.95% | — |
| Hang Seng TECH Index | 14.02% | 41.08% | -36.32% | — |

The small model exceeded the Hang Seng Index by **16.28 percentage points annualized** and **63.33 points in total return** over this window.

The profitable-position rate counts **closed positions with positive returns after fees**: 438 of 891 for the small model, or 49.16%. At 0.50% fees per side, the model returns 31.41% annualized. Model returns deduct simulated fees; benchmarks are price indices excluding dividends and fees.

[Daily wealth curves](docs/showcase/performance_curve.csv) · [Four models, two fee scenarios](docs/showcase/model_comparison.csv) · [Dates, settings, and sources](docs/showcase/summary.json)

### What goes into the ranking

The featured model uses 37 inputs, including price deviation, momentum, volatility, trading value, market capitalization, and market breadth. It combines these inputs into a score used to rank stocks.

![Ten most important model inputs by training split gain](docs/assets/factor_importance.png)

Distance from the 252-day low and 60-day price deviation account for 16.40% and 15.81% of total training split gain. The chart describes the model's use of each input during training. The wealth curve above shows the resulting portfolio performance.

[All input importances](docs/showcase/factor_importance.csv) · [Feature calculations](hk_quant/features.py) · [Model implementation](hk_quant/models.py)

### One complete trading cycle

This historical simulation follows **all 30 positions in the first entry cohort**, from month-end selection and next-session purchases to the final exits, with 85 fills and a complete cash ledger.

| Date | Event |
| --- | --- |
| 2024-01-31 | Month-end scores select 30 stocks. |
| 2024-02-01 | Buy at the next session's close. Trading-value caps reduce some allocations; entry cost including fees is HK$716,281.52. |
| 2024-03-01 | Scheduled exit date; sales begin. |
| 2024-03-04–03-14 | Remaining positions sell in parts under trading-value caps. The last exit is March 14. |

#### The portfolio after entry

Holdings were worth HK$714,495.29, with HK$283,718.48 in cash. Account value after entry fees was HK$998,213.76; cash made up 28.42%. The chart shows the four largest allocations and groups the other 26 stocks. All positions are available in the linked table.

![Holdings and cash after entry on February 1, 2024](docs/assets/holdings.png)

[Selection ranks](docs/showcase/first_cycle_selection.csv) · [All 30 holdings and account weights](docs/showcase/first_cycle_holdings.csv)

#### Entries and exits on the candles

The candles show the first two stocks by score: 08619.HK (−59.91% after fees) and 02171.HK (+61.45%). Markers cover every simulated fill, including the two partial exits for 08619.HK.

![Adjusted candlesticks with simulated entries and exits for the top two selections](docs/assets/trade_candles.png)

| Date | Stock | Side | Adjusted units | Adjusted price | Notional (HKD) | Fee (HKD) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2024-02-01 | 08619.HK | Buy | 4,885.528685 | 2.307179 | 11,271.79 | 28.18 |
| 2024-02-01 | 02171.HK | Buy | 7,742.082702 | 4.080000 | 31,587.70 | 78.97 |
| 2024-03-01 | 08619.HK | Sell | 1,951.329001 | 1.118632 | 2,182.82 | 5.46 |
| 2024-03-01 | 02171.HK | Sell | 7,742.082702 | 6.620000 | 51,252.59 | 128.13 |
| 2024-03-04 | 08619.HK | Sell | 2,934.199683 | 0.804017 | 2,359.15 | 5.90 |

The trade table uses fractional adjusted research units. [All 85 simulated fills](docs/showcase/first_cycle_trades.csv) include dates, securities, sides, quantities, prices, notionals, fees, and cash changes for this cohort.

#### What the whole cohort earned

Of 30 positions, 18 were profitable. There were 30 buy fills and 55 sell fills. Entry cost was HK$716,281.52 and net sale proceeds were HK$781,194.89: **HK$64,913.36 in net profit**, after HK$3,744.12 in total fees. That is 9.06% on entry cost and a 6.49% contribution to the initial HK$1 million.

![Net profit and loss for all 30 positions in the first cohort](docs/assets/first_cycle_pnl.png)

[Cost, proceeds, and returns for every position](docs/showcase/first_cycle_positions.csv)

The `cohort_cash_after` ledger tracks these 30 positions from HK$1 million to HK$1,064,913.36, excluding new positions from later rebalances. The daily wealth table covers the full strategy account.

### Backtest conventions

- Each rebalance budgets 95% of available cash after sales across 30 fixed slots. Purchases are capped at 1% of both signal-date average trading value over 20 sessions and execution-day trading value. Unspent allocations stay in cash; pending sales are attempted on later sessions.
- Stock data comes from Tushare `hk_daily_adj` and `hk_adjfactor`; benchmarks come from `index_global`. Replay uses adjusted research units. It does not reconstruct historical board lots, exact dividend cash dates, or broker execution.
- Annualization compounds over elapsed calendar days. Drawdown uses daily account values. Ending equity includes reference values for unsold positions; the small model has one open position. Reference valuations can differ from executable prices, and `execution_validated` is `false`.
- The featured model was trained through December 29, 2023 and replayed with frozen parameters. This window has been used to compare candidates; it is a historical comparison rather than an independent holdout.

## Run locally

Use Python 3.12 with the pinned requirements. The repository includes source, tests, A-share trained weights and aggregate reports, plus Hong Kong figures and case-study CSVs. Vendor market data and per-security forecasts require your own authorized inputs. Hong Kong model weights and full replay artifacts are stored separately.

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS / Linux, then install:

```bash
pip install -r requirements.txt
```

Run the self-contained A-share tests in a macOS / Linux shell after installing the test dependencies:

```bash
pip install pytest==9.1.1 httpx==0.28.1
python -m pytest -q tests/test_ashare_*.py
```

The full synthetic end-to-end test is opt-in; its command is in the [release guide](ASHARE_RELEASE.zh-CN.md#安装与验证). It generates temporary data to check the pipeline and API. See the [verification record](docs/ashare/verification.md) for the broader test scope and known missing Hong Kong fixtures.

### Rebuild the A-share figures

The four A-share charts and their CSV tables use aggregate JSON files included in this repository. After installing the requirements, run:

```bash
python -m scripts.build_ashare_showcase
```

The [renderer](scripts/build_ashare_showcase.py) verifies the saved selection record and metric arithmetic, then writes to `docs/assets/cn/` and `docs/showcase/cn/`. It needs no vendor inputs and does not retrain the model.

### Rebuild the Hong Kong results

With the saved local model, forecasts, market data, and completed replay available, rebuild the figures from the repository root:

```bash
python -m scripts.build_showcase
```

Replay saved forecasts with the same corporate-action inputs, without training:

```bash
python -m hk_quant.fixed_backtest --forecast-root backtests/hk/fixed_models_20260916 --data-root data/hk/research_20260916 --source-root data/hk/universal --results-root backtests/hk/fixed_execution_20260916 --terminal-actions data/hk/universal/references/terminal_actions/cash_settlements.csv --transfers data/hk/universal/references/terminal_actions/board_transfers.csv
```

See the [replay entry point](hk_quant/fixed_backtest.py) for local input paths and saved-model records, and the [figure generator](scripts/build_showcase.py) for chart inputs.

## Repository layout

| Path | Purpose | Included |
| --- | --- | --- |
| `ashare_quant/` | A-share collection, factors, models, replay, API and portfolio advice | Yes |
| `models/cn/universal/` | Four frozen candidates and the latest Ridge weights with metadata | Yes |
| `backtests/cn/universal/` | Experiment protocol, selection records and aggregate evaluation | Summaries only |
| `docs/ashare/`, `delivery_report/` | A-share method, verification records and Chinese report | Yes |
| `hk_quant/` | Hong Kong data processing, factors, models, frozen replay, API and portfolio advice | Yes |
| `scripts/build_ashare_showcase.py` | A-share evaluation figures and tables from committed aggregate metrics | Yes |
| `scripts/build_showcase.py` | Figures and case tables from existing Hong Kong results | Yes |
| `docs/assets/`, `docs/showcase/` | A-share and Hong Kong README figures and compact evidence tables | Yes |
| `tests/` | A-share and Hong Kong research and service tests | Yes |
| `legacy/` | Earlier A-share and Hong Kong scripts and evaluation criteria | Yes |
| `data/` | Vendor inputs and generated data | No |
| `models/hk/`, `backtests/hk/` | Hong Kong weights, predictions and full replay records | No |
| `reports/`, `tmp/`, `output/` | Local reports, temporary files and exports | No |

## API and deployment

Both FastAPI services expose the endpoints below for 1, 5, 20 and 60 trading sessions. Each service uses its own market data and complete prediction snapshot.

### A-share API

Prepare authorized inputs and follow the [snapshot and disclosure steps](docs/ashare/README.md#attach-observed-probability-evidence-before-serving). Set `ASHARE_QUANT_API_KEY` securely in your environment, then start the CNY service:

```bash
python -m ashare_quant.api --host 127.0.0.1 --port 8001
```

The public weights alone are insufficient to serve real-stock rankings or portfolio advice. The completed local snapshot needs matching data, predictions and an active pointer. The API returns research outputs and order suggestions; it does not submit trades.

### Hong Kong API

Serving requires a separately published snapshot referenced by `models/hk/universal/active.json`.

```powershell
$env:HK_QUANT_API_KEY = "replace-with-a-long-random-key"
python -m hk_quant.api --host 0.0.0.0 --port 8000
```

### Shared endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Health check; no key required |
| GET | `/v1/model/status` | Published model and data date |
| GET | `/v1/rankings?horizon=20&limit=50` | Market ranking |
| GET | `/v1/stocks/{code}/forecast` | Security forecasts |
| POST | `/v1/portfolio/advice` | Advice from JSON holdings |
| POST | `/v1/portfolio/advice/csv` | Advice from CSV holdings |

Endpoints other than the health check require `X-API-Key`. Hong Kong portfolio advice uses HKD cash and returns whole-lot order suggestions for the user to assess.

### Hong Kong deployment

Deploy with the Python runtime on Zeabur Dev. Install with `pip install -r requirements.txt` and start with `python -m hk_quant.api --host 0.0.0.0 --port $PORT`. Set `HK_QUANT_API_KEY` and mount matching data and published snapshots separately. Docker is not required.

<a id="credentials"></a>

## About the author

**Shuai Wu** holds **Gold Level** in the WorldQuant Challenge.

<p align="center">
  <img src="docs/assets/credentials/worldquant-challenge-gold.png" width="420" alt="Shuai Wu — WorldQuant Challenge Gold Level">
</p>

## Star History

<p align="center">
  <a href="https://www.star-history.com/#Asher-S-Wu/Vectaix-Finance&amp;Date">
    <img src="https://api.star-history.com/svg?repos=Asher-S-Wu/Vectaix-Finance&amp;type=Date" width="800" alt="GitHub star history">
  </a>
</p>
