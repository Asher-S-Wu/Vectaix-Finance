<div align="center">

# Vectaix Finance

### Auditable quantitative research, forecasting, and portfolio advice for Hong Kong equities

[![GitHub stars](https://img.shields.io/github/stars/Asher-S-Wu/Vectaix-Finance?style=for-the-badge&logo=github&color=f4b942)](https://github.com/Asher-S-Wu/Vectaix-Finance/stargazers)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-2f8f46?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![Zeabur](https://img.shields.io/badge/Deploy-Zeabur-6c63ff?style=for-the-badge)](https://zeabur.com/)

**English** · [简体中文](./README.zh-CN.md) · [日本語](./README.ja.md) · [한국어](./README.ko.md)

</div>

<div align="center">

> Market data, fundamentals, security identity, corporate actions, and risk limits in one traceable path — from research signal to whole-lot order advice.

</div>

## What it is

Vectaix Finance is a quantitative research and service platform for the Hong Kong equity market. It builds features from date-aligned prices, valuations, fundamentals, and corporate actions; trains forecasts for 1, 5, 20, and 60 trading days; and exposes rankings, security forecasts, and portfolio advice through FastAPI.

A model is served only after acceptance checks, a complete as-of snapshot, and schema validation. The API reads `models/hk/universal/active.json` and never trains during a request, keeping research artifacts separate from production output.

## Capabilities

| Area | What it provides |
| --- | --- |
| Multi-horizon forecasts | Score, probability of an up move, expected return, and q10/q50/q90 intervals for four horizons. |
| Market and security queries | Ranked market views and four-horizon forecasts, with explicit unavailable statuses. |
| Portfolio advice | Whole-lot orders under cash, fees, participation, position, equity, and volatility constraints. |
| Evidence checks | Security identity, lot size, currency, FX, quote date, and corporate-action evidence. |
| Research gates | Walk-forward evaluation, IC, probability calibration, interval coverage, and publication audits. |

## Workflow

```text
Collect and verify source data
            ↓
Build date-aligned features and risk history
            ↓
Walk-forward training, forecasts, and calibration
            ↓
Model acceptance and as-of snapshot audit
            ↓
Publish an eligible snapshot (active.json)
            ↓
FastAPI: rankings · forecasts · portfolio advice
```

Candidate models include linear baselines and small/large LightGBM models. Training uses deterministic per-date sampling; scoring evaluates every eligible security. Portfolio optimization uses CLARABEL and carries cash, fees, whole-lot, and turnover-capacity rules into the final order quantities.

## Quick start

### Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Research commands

```bash
python scripts/train_backtest.py a
python scripts/predict.py a --as-of 2026-09-08
python scripts/make_report.py
```

The A-share scripts are retained for historical research and report generation. The production API uses the Hong Kong universal snapshot under `hk_quant` and serves only published forecasts.

### Start the API

```bash
# Windows PowerShell
$env:HK_QUANT_API_KEY = "replace-with-a-long-random-key"
# macOS / Linux: export HK_QUANT_API_KEY=replace-with-a-long-random-key

python -m hk_quant.api --host 0.0.0.0 --port 8000
```

`GET /health` checks the process. Every other endpoint requires the `X-API-Key` header.

## API reference

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Health check; no key required. |
| `GET` | `/v1/model/status` | Active model, data date, and prediction schema. |
| `GET` | `/v1/rankings?horizon=20&limit=50` | Market ranking for horizon `1`, `5`, `20`, or `60`. |
| `GET` | `/v1/stocks/{code}/forecast` | Four-horizon forecast for one security. |
| `POST` | `/v1/portfolio/advice` | Portfolio advice from JSON holdings and cash. |
| `POST` | `/v1/portfolio/advice/csv` | Portfolio advice from holdings CSV. |

```bash
curl "http://localhost:8000/v1/rankings?horizon=20&limit=20" \
  -H "X-API-Key: replace-with-a-long-random-key"
```

Portfolio cash is denominated in HKD. Responses include target quantity, order quantity, reference price, estimated fees, weights, exclusions, and constraint status. They are decision-support outputs, not a return promise.

## Repository layout

```text
hk_quant/                 # API, contracts, training, publication, and optimization
  api.py                  # FastAPI routes
  service.py              # Published-snapshot service layer
  training.py             # Forecast tasks, coverage, and snapshots
  portfolio.py            # Risk estimates and whole-lot optimization
  registry.py             # Publication gates and active.json
data/hk/universal/        # Date-aligned Hong Kong research data
models/hk/universal/      # Candidate models, snapshots, and acceptance records
backtests/hk/universal/   # Training, backtest, and metric outputs
scripts/                 # Data, training, prediction, and report commands
tests/                   # Service, strategy, training, and report tests
reports/                 # Human-readable research reports and charts
```

## Deploy on Zeabur

Deploy as a Python service on the Zeabur Dev plan; Docker is not required.

1. Connect this repository and choose the Python runtime.
2. Set the install command to `pip install -r requirements.txt`.
3. Set the start command to `python -m hk_quant.api --host 0.0.0.0 --port $PORT`.
4. Add `HK_QUANT_API_KEY` as an environment variable.
5. Mount or publish matching `data/hk/universal` and `models/hk/universal` artifacts with the service version.

The service depends on `models/hk/universal/active.json`. If no eligible snapshot exists, it returns “no published model” instead of silently using a candidate.

## Research boundaries

Backtests describe historical behavior under their specific data window, fee assumptions, and execution rules. The project is still in research and development; recent snapshots may contain immature labels, missing execution prices, or pending audit work. Forecasts are not investment advice.

## Contributing

Issues, evidence improvements, tests, and model-gate refinements are welcome. For changes involving data or metrics, include the as-of date and a reproducible command in the pull request.

## Star history

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=Asher-S-Wu/Vectaix-Finance&type=Date&theme=dark)](https://www.star-history.com/#Asher-S-Wu/Vectaix-Finance&Date)

</div>

