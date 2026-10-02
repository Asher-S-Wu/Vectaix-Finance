# Independent A-share model: design and parity contract

## Intent and constraints

Build and actually train an independent A-share model based on the existing HK multi-factor stack. Preserve all tracked HK implementation, documentation, showcase and any existing local assets. Source revision: 2881428. New runtime namespace: `ashare_quant`; data/models/results: `data/cn/universal`, `models/cn/universal`, `backtests/cn/universal`. No remote push, deployment or trading is part of this run.

The current source distribution includes no original HK market dataset or trained weights. Preservation means keeping its implementation and supplied results intact; this work cannot manufacture or claim the absent historical HK binary model.

## Design choice

Reuse tested market-neutral factor and multi-task estimation mathematics through explicit adapters. Keep market data normalization, calendars, identifiers, CNY conventions, fees, lots, execution, portfolio advice, snapshots and command entry points separate. A wholesale HK copy would silently inherit FX, lot, corporate-action and exchange assumptions. A from-scratch estimator would unnecessarily discard calibrated existing behavior.

## Functional parity inventory

| Existing capability | A-share implementation contract |
| --- | --- |
| Exchange calendar, identity and historical universe | SSE/SZSE/BSE source calendars, L/D/P security master, dated listing bounds, no present-day-only membership filter; code mapping audited |
| Tushare ingestion, provenance and quality audit | Resumable safe-helper client, endpoint pagination, units, raw hashes, timestamps, row counts, duplicate/truncation checks |
| Rolling price/volume/market factors | Same stable mathematical factors, CNY adapter, strictly past-only rolling windows, no data filling that implies executable quotes |
| Fundamental and valuation inputs | Historical daily valuation when supplied; statement metrics only announcement-time joined with conservative availability lag. Missing or unversioned metrics remain unavailable and explicitly excluded |
| Four estimation families | Factor, Ridge, small and large LightGBM, shared horizon inputs for 1/5/20/60 trading sessions |
| Independent task availability | Scores, calibrated up probability, expected arithmetic return, q10/q50/q90, baselines, reasons, feature contributions |
| Development and confirmation | Date-only chronological train/calibration/development/untouched confirmation, label maturity purging, frozen selection before confirmation |
| Frozen execution replay | Next-session monthly top30 long-only, CNY actual-share and board-lot ledger, T+1, actual daily price bounds, suspension and 1% liquidity capacity, delayed sales and audit |
| Corporate actions, delisting and cash evidence | Source-adjusted return basis stated; exact cash/share actions require evidence. Unresolved changes are blocked/flagged rather than synthetic dividends, shares or sale proceeds |
| Market comparisons and reports | CSI300/CSI500/SSE Composite, cost-stressed reference returns, turnover, drawdown, costs, prediction metrics and coverage, full artifacts and one trading cohort |
| Daily refresh, snapshots and status | Separate CN refresh/candidate/publication paths, frozen and latest-fit artifacts, schema/lineage checks, research-readiness separate from investment-performance acceptance |
| Ranking/stock/portfolio API | Identical functional endpoint set, CNY, independent API key, JSON and CSV, whole-lot/risk/cash/participation/limit/T+1 constraints |
| HK-specific HKEX/Webb/REIT evidence modules | Retained untouched. CN identity, adjustment and cash-event evidence interfaces are separate. HK-specific scrapers and REIT identifiers are not reused for mainland securities |
| Rate and tender-offer extensions | The 37-factor showcased baseline does not use them. A-share SHIBOR/LPR and issuer-verified tender-offer factor pipelines are deferred, not implemented or represented as covered by this training run |

## Dataset and statistical protocol

Target history: 2016-01-01 through latest complete source session no later than 2026-09-30, all available ordinary A-share identities including delisted shares. Historical absence and denied endpoints are reported, never fabricated. Mainland ex-right daily pre-close is distinct from preceding raw close. Amount, volume, share and cap unit conversions are explicit. Price features use observed source-adjustment ratios; raw prices are retained for execution.

Initial planned fixed comparison: training through 2022 with a separately purged 2023 calibration, 2024 development selection, 2025 onward confirmation. The final precise split is persisted before checking candidate results. The same deterministic maximum 256 stocks/date sampling budget as HK training controls memory while inference covers the full eligible universe. No cross-date normalization; cross-sectional ranks are computed using only same-day available observations. Train-only scaler fitting and label-end-before-boundary checks remain mandatory. Selected architecture is locked before confirmation. A second latest-data refit is clearly separated from frozen-holdout results.

The model is a research artifact, not a guarantee of returns or broker-ready execution. Publication/readiness cannot override unresolved identity, execution or data issues. Show failed performance gates as failures; never cherry-pick candidates on the confirmation set.

The user clarified that the repository is mainly a showcase and delegated methodological choice. This run prioritizes actual training and an independently held-out evaluation of its 37-factor method, with forecasting, risk-advice, service and audit capabilities. It does not claim every optional HK data extension has a completed mainland equivalent. Financial statement ingestion/version certification, SHIBOR/LPR features and issuer-verified tender-offer features remain explicit extensions.

## Resource and security contract

Cloud capacity observed: 9 CPUs, approximately 9.7 GiB RAM, 30 GiB free disk, no GPU required. Bound concurrency and query rates; use existing API access, no paid subscriptions. Full daily all-market collection is roughly ten thousand endpoint/date requests before pagination and optional evidence. Probe permissions/latency first; resume cached partitions on retries.

The credential remains solely inside the preapproved query helper. Never print it, read its storage into the agent, copy it into environment/repository, or pass it to an alternate service. Code consumes only the helper's redacted JSON. All test fixtures are synthetic and explicitly distinguished from real training evidence.

## Pre-training audit refinement

Before any authentic model was fitted or evaluated, independent integration audit clarified the development selector: compare 20-day mean daily rank IC on the same contemporaneously score-available identity intersection across all four candidates. Keep each model's full-universe metrics and coverage separately. Unknown future returns never define the entry universe. Brier/accuracy/coverage are pooled security-observation metrics, while IC and fitting weights balance dates (and training horizons).

The earlier pre-training protocol record was retained as `protocol_pretraining_20260930T1835.json`; `protocol_refinement.json` records the reason and pre-training timing. Chronological boundaries did not change. Frozen selection validates its entire expected record, and frozen/latest model bytes are hashed before loading.
