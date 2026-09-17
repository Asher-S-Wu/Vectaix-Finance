<div align="center">

# Vectaix Finance

Multi-factor stock selection for Hong Kong equities, from scores to a trade-by-trade review.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-00866F)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![GitHub stars](https://img.shields.io/github/stars/Asher-S-Wu/Vectaix-Finance)](https://github.com/Asher-S-Wu/Vectaix-Finance/stargazers)

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

</div>

Vectaix Finance combines 37 price, volume, and market features to select 30 Hong Kong stocks each month. In the historical backtest from February 1, 2024 to September 16, 2026, the small LightGBM model returned **35.56% annualized after fees**, taking HK$1 million to **HK$2.22 million**.

![Historical backtest: model wealth, Hang Seng benchmarks, and drawdown](docs/assets/performance.png)

## Returns against the indices

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

## What goes into the ranking

The featured model uses 37 inputs, including price deviation, momentum, volatility, trading value, market capitalization, and market breadth. It combines these inputs into a score used to rank stocks.

![Ten most important model inputs by training split gain](docs/assets/factor_importance.png)

Distance from the 252-day low and 60-day price deviation account for 16.40% and 15.81% of total training split gain. The chart describes the model's use of each input during training. The wealth curve above shows the resulting portfolio performance.

[All input importances](docs/showcase/factor_importance.csv) · [Feature calculations](hk_quant/features.py) · [Model implementation](hk_quant/models.py)

## One complete trading cycle

This historical simulation follows **all 30 positions in the first entry cohort**, from month-end selection and next-session purchases to the final exits, with 85 fills and a complete cash ledger.

| Date | Event |
| --- | --- |
| 2024-01-31 | Month-end scores select 30 stocks. |
| 2024-02-01 | Buy at the next session's close. Trading-value caps reduce some allocations; entry cost including fees is HK$716,281.52. |
| 2024-03-01 | Scheduled exit date; sales begin. |
| 2024-03-04–03-14 | Remaining positions sell in parts under trading-value caps. The last exit is March 14. |

### The portfolio after entry

Holdings were worth HK$714,495.29, with HK$283,718.48 in cash. Account value after entry fees was HK$998,213.76; cash made up 28.42%. The chart shows the four largest allocations and groups the other 26 stocks. All positions are available in the linked table.

![Holdings and cash after entry on February 1, 2024](docs/assets/holdings.png)

[Selection ranks](docs/showcase/first_cycle_selection.csv) · [All 30 holdings and account weights](docs/showcase/first_cycle_holdings.csv)

### Entries and exits on the candles

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

### What the whole cohort earned

Of 30 positions, 18 were profitable. There were 30 buy fills and 55 sell fills. Entry cost was HK$716,281.52 and net sale proceeds were HK$781,194.89: **HK$64,913.36 in net profit**, after HK$3,744.12 in total fees. That is 9.06% on entry cost and a 6.49% contribution to the initial HK$1 million.

![Net profit and loss for all 30 positions in the first cohort](docs/assets/first_cycle_pnl.png)

[Cost, proceeds, and returns for every position](docs/showcase/first_cycle_positions.csv)

The `cohort_cash_after` ledger tracks these 30 positions from HK$1 million to HK$1,064,913.36, excluding new positions from later rebalances. The daily wealth table covers the full strategy account.

## Backtest conventions

- Each rebalance budgets 95% of available cash after sales across 30 fixed slots. Purchases are capped at 1% of both signal-date average trading value over 20 sessions and execution-day trading value. Unspent allocations stay in cash; pending sales are attempted on later sessions.
- Stock data comes from Tushare `hk_daily_adj` and `hk_adjfactor`; benchmarks come from `index_global`. Replay uses adjusted research units. It does not reconstruct historical board lots, exact dividend cash dates, or broker execution.
- Annualization compounds over elapsed calendar days. Drawdown uses daily account values. Ending equity includes reference values for unsold positions; the small model has one open position. Reference valuations can differ from executable prices, and `execution_validated` is `false`.
- The featured model was trained through December 29, 2023 and replayed with frozen parameters. This window has been used to compare candidates; it is a historical comparison rather than an independent holdout.

## Run locally

The repository includes code, tests, five backtest figures, author certificates, and result and case-study CSVs ready to browse. **Full market data, model weights, predictions, and bulk replay artifacts stay local and are excluded from the current source tree.**

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS / Linux, then install:

```bash
pip install -r requirements.txt
```

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

| Path | Purpose | In the current source distribution |
| --- | --- | --- |
| `hk_quant/` | Hong Kong data processing, factors, models, frozen replay, API, and portfolio advice | Yes |
| `scripts/build_showcase.py` | Figures and case tables from existing results | Yes |
| `docs/assets/`, `docs/showcase/` | README figures and compact evidence tables | Yes |
| `tests/` | Research and service tests | Yes |
| `legacy/` | Earlier A-share and Hong Kong scripts, historical evaluation criteria | Yes |
| `data/`, `models/` | Market data, features, weights, and published snapshots | No |
| `backtests/`, `reports/` | Full replay records and generated reports | No |
| `tmp/`, `output/` | Temporary files and local exports | No |

## API and deployment

FastAPI serves rankings, forecasts, and portfolio advice for 1, 5, 20, and 60 trading sessions. Serving requires a separately published snapshot referenced by `models/hk/universal/active.json`.

```powershell
$env:HK_QUANT_API_KEY = "replace-with-a-long-random-key"
python -m hk_quant.api --host 0.0.0.0 --port 8000
```

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Health check; no key required |
| GET | `/v1/model/status` | Published model and data date |
| GET | `/v1/rankings?horizon=20&limit=50` | Market ranking |
| GET | `/v1/stocks/{code}/forecast` | Security forecasts |
| POST | `/v1/portfolio/advice` | Advice from JSON holdings |
| POST | `/v1/portfolio/advice/csv` | Advice from CSV holdings |

Endpoints other than the health check require `X-API-Key`. Portfolio advice uses HKD cash and returns whole-lot order suggestions for the user to assess.

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
