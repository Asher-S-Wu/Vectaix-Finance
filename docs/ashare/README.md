# A-share model (independent from HK)

This module trains an independent CNY model from real Tushare A-share history. Existing HK source, results and artifact paths are preserved. It is a research system; neither model readiness nor historical returns imply live execution validation.

## Public trained-model release

The public release includes unchanged trained weights, source, tests and aggregate research evidence. Vendor market inputs, per-security forecasts and replay ledgers are not distributed; there is no active service pointer. Real-data inference/API operation requires separately authorized inputs and a complete local prediction snapshot. Start with [the public release guide](../../ASHARE_RELEASE.zh-CN.md). Historical source hashes and research-ready certificates describe the original complete experiment, not the contents of this data-excluded distribution.

## Reproduce

Use Python 3.12 with the repository's pinned requirements and pytest/httpx for tests. The collection/training commands below require your own authorized data access and configured helper; they are not an offline quickstart. The original HK test distribution references several omitted `tmp/` scripts and a saved HTML fixture; these baseline failures are recorded rather than hidden.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt pytest httpx
.venv/bin/python -m ashare_quant.collect --oldest-first --max-workers 12 --requests-per-minute 60
.venv/bin/python -m ashare_quant.pipeline --skip-collection
.venv/bin/python -m pytest -q tests/test_ashare_*.py
```

The source collector calls the preconfigured credential helper, never reads a token into agent messages, and never writes a credential into this repository. Its default helper path is an execution-specific external dependency; users reproducing elsewhere must supply their own approved helper implementing the same redacted JSON protocol through `HelperClient(helper=...)`. No token is bundled.

## Namespaces

- Code: `ashare_quant/`
- Raw and normalized data: `data/cn/universal/`
- Frozen and latest models: `models/cn/universal/`
- Protocol, forecasts, confirmation and reports: `backtests/cn/universal/`
- Original HK locations remain `hk_quant/`, `data/hk/`, `models/hk/`, `backtests/hk/`

## Fixed experiment

37 price/volume/capitalization/market-context factors match the showcased methodology. The fixed-score baseline uses 4 of them; the selected Ridge uses its 10 preset base inputs plus log_horizon. The LightGBM candidates use the broader factor set. Compare fixed factor, Ridge, small and large LightGBM using the same data and calibration. Training uses deterministic maximum 256 stocks per date; inference covers every available identity. Cross-sectional preprocessing is same-date only and all training labels are purged before the next split.

2016–2022: fitting; 2023: separate calibration. Development signals and mature labels are both bounded by 2024-12-31. Freeze the model with the best 20-session daily rank IC on development; tie-break favors simpler models. Confirmation starts in 2025 and is never used to choose or tune a model. A separate refit using latest mature data supplies current predictions. The protocol is persisted before results and cannot silently change.

## Functional coverage

Rankings; individual 1/5/20/60-session forecast score, up probability, expected arithmetic return and q10/q50/q90; independent task failure states; feature explanations; JSON/CSV holdings advice with CNY cash, actual lots, T+1, price bounds, suspension, risk and turnover limits; checksummed research snapshots; immutable model-selection evidence; baseline/confirmation metrics; full trade/cash/action logs; price-index benchmarks.

Prediction tasks remain independently visible: a failed probability or interval task does not erase a valid score or expected return. Account advice intentionally retains the original conservative policy requiring every forecast task to be valid before admitting a security to the trade candidate pool. A visible expected return alone is not an account buy approval.

The financial as-of adapter preserves release versions and uses a conservative day-after-announcement visibility boundary. Financial metrics are not included in this first 37-factor baseline. HK-specific HIBOR, HKEX and Webb/REIT modules remain untouched rather than being mislabeled as mainland data.

Scope of this trained version: the showcased 37-factor method and its prediction, service, portfolio-advice and audit workflows. Mainland statement-data version certification, SHIBOR/LPR factor ingestion and issuer-verified tender-offer factors are deferred extensions; the presence of an as-of join adapter does not mean those source pipelines or factors have been trained.

## Attach observed probability evidence before serving

After the real pipeline writes `pipeline_status.json` with `status=complete`, run this explicit disclosure-only step before serving or exporting the model:

```
.venv/bin/python -m ashare_quant.evidence --results-root backtests/cn/universal --model-root models/cn/universal
```

The command resolves the frozen Ridge model identity from the completed training status, development-only selection record, protocol and frozen-model metadata. It validates the snapshot hashes and confirmation metrics, saves exact prior certificate/pointer evidence under `publication_audit/`, writes a new immutable acceptance certificate, and atomically updates the active pointer. The model pickle, forecast values and training metadata remain byte-for-byte unchanged. Repeating the same step is idempotent. `publication_status.json` records the resulting disclosure and source hashes.

This run's frozen Ridge confirmation has negative probability Brier skill at all four horizons. The latest refit has not itself been evaluated out of sample. The warning appears in the API's limitations on status, rankings, stock forecasts and portfolio advice; per-horizon Brier/IC evidence is retained in the acceptance certificate. Research readiness remains true while performance eligibility and execution validation remain false.

## Serve locally

After a real snapshot exists, set `ASHARE_QUANT_API_KEY` securely yourself and run:

```
.venv/bin/python -m ashare_quant.api --host 127.0.0.1 --port 8001
```

The A-share API has `/health`, `/v1/model/status`, `/v1/rankings`, `/v1/stocks/{code}/forecast`, `/v1/portfolio/advice`, and `/v1/portfolio/advice/csv`. All except health require `X-API-Key`. A snapshot can be structurally research-ready while performance acceptance and execution validation remain false. No trades are sent.

## Important accounting limits

Raw CNY prices, actual share counts and daily source price limits govern the execution audit. Adjustment-factor ratios are never treated as stock-split ratios. Cash/share events require separate dated evidence and record-date entitlements; unresolved held events make verified equity unavailable rather than inventing proceeds. A separate adjusted-unit method-reference replay can show price-return research performance under explicit symmetric fee/slippage stress; it is not an executable actual-share account. Delisted/suspended/missing outcomes remain in coverage reports.

See `design.md` for the full parity inventory and `backtests/cn/universal/report/report.zh-CN.md` for actual finished results when training completes.

The completed source audit found three unavailable historical BSE quote identities belonging to 2022 board transfers, not failed companies. Their missing listed-period rows remain in coverage. `board_transfer_coverage.csv` records official termination and new-board listing dates separately; the intervening periods are not filled with invented trades and current-code histories are not silently relabeled.
