# Independent US research, Experiment 1: sources, evaluation and charts

This initial public-data experiment is preserved as a research snapshot, with independently fitted US/USD artifacts, four forecast-evaluation figures and a conditional replay figure. The frozen factor-score candidate won the 2024 development comparison, but confirmation did **not** establish a ranking advantage: mean rank IC is negative at every horizon, and the 20- and 60-session 95% block-bootstrap intervals are wholly below zero. Return-interval coverage is below its nominal 80% target at every horizon. The model is not validated for live use.

- [Detailed method and limitations (Chinese)](method.zh-CN.md)
- [Sealed training protocol](../../backtests/us/oef2015/protocol.json)
- [Source coverage audit](../../backtests/us/oef2015/source_audit.json)
- [Original 2015 cohort](declared_universe.csv) and [holdings-source validation](universe_source_validation.json)
- [Limited provider identity review](identity_review.json)
- [Frozen architecture selection](../../backtests/us/oef2015/frozen_architecture.json)
- [Frozen confirmation metrics](../../backtests/us/oef2015/confirmation_summary.json)
- [Calibration metadata erratum](../../backtests/us/oef2015/method_errata.json)
- [Four local candidate artifacts](../../models/us/oef2015/frozen/) and [separate latest refit](../../models/us/oef2015/latest/)

## Historical cohort and coverage

The starting universe is the **101 original long common-equity share classes** in OEF's **2015-06-30** holdings, published **2015-09-02**. The [official historical report](https://announcements.asx.com.au/asxpdf/20150902/pdf/4311bmqvg9p1d1.pdf) predates the training period. The universe is fixed; no modern successor is silently substituted for a missing original identity.

Yahoo Finance public current-vintage history covers **2014-01-02 through 2026-09-30**, with **3,205 shared SPY calendar sessions**. Limited provider-name, instrument and history-window checks accept **89 original identities**. Twelve remain excluded or unavailable: five unresolved complex lineages, five missing histories and two reused tickers. Missing identities remain in prediction counts and failure disclosures, but they cannot contribute unobserved returns to the reported metrics. This produces survivorship/availability selection; the results are neither an unbiased whole-cohort estimate nor a full-US-market estimate. Accepted histories are not a complete permanent-identifier or corporate-action audit.

AKShare was checked as an auxiliary source. Incomplete adjustment-factor coverage and reused tickers make it unsuitable for the declared training history. It is not a feature input. Historical market capitalization and fundamentals unavailable from these inputs are not manufactured. Free access does not establish redistribution rights; vendor inputs and per-security forecasts remain separate from these aggregate charts.

## Chronology and interpretation

Training uses 2016–2022, held-out calibration uses 2023, and architecture selection uses 2024. Purged endpoint rules keep each label within its allowed window. Fixed factor score, Ridge, small LightGBM and large LightGBM are compared on identical contemporaneously score-available rows and the same eligible IC dates. The selected factor score uses preset ranking weights; its probability and return-interval calibration is fitted independently on US data. No A-share or Hong Kong fitted state is reused.

For the selected factor model, interval calibration uses additive arithmetic-return residual quantiles. One inherited metadata description incorrectly calls this log1p calibration; the linked erratum records the correction without altering the fitted weights or predictions.

The selected model is frozen before confirmation from **2025-01-01 through 2026-09-30**. The separate latest refit as of **2026-09-30** has no out-of-sample evaluation and does not inherit the frozen model's confirmation results. Eligibility and execution validation remain false. Source-adjusted prices are research return inputs, not raw exchange execution prices or a verified actual-share cash-distribution ledger.

## Four source-backed figures

| Figure | Saved evidence | Interpretation |
| --- | --- | --- |
| [Model choice](../assets/us/model_selection.png) | `development_common_universe.json`, `frozen_architecture.json` | 20-session mean daily rank IC on common rows and dates; used for selection only |
| [Confirmation ranking](../assets/us/confirmation_ic.png) | `confirmation_summary.json` | Daily cross-sectional Spearman IC; 95% circular 60-session block-bootstrap interval, 2,000 draws, seed 42, missing-day slots preserved |
| [Probability skill](../assets/us/probability_skill.png) | `confirmation_summary.json` | `1 − model Brier / historical-frequency baseline Brier`; positive is better, negative is worse |
| [Return-interval coverage](../assets/us/interval_coverage.png) | `confirmation_summary.json` | Empirical q10–q90 coverage against a nominal 80%; this is not an IC confidence interval |

Every metric is conditional on mature, task-available observed outcomes. Longer-horizon counts are smaller. Overlapping stock-date outcomes are not independent observations; counts are not effective sample sizes. Small positive Brier skill at 5 and 20 sessions does not repair the negative confirmation ranking or establish economically useful probability forecasting.

The four forecast figures do not establish portfolio returns or executable performance.

## Conditional replay: lower returns than both references

The [fifth figure](../assets/us/conditional_replay.png) shows all 437 daily valuations from 2025-01-02 through 2026-09-30, normalized to the USD 1,000,000 initial capital before first-session costs. The frozen factor strategy returned **+16.01%** after 15 bps one-way assumed costs, versus **+27.40%** for the fixed 101-slot cohort and **+30.84%** for the SPY adjusted-price proxy. A 30 bps cost stress reduced the selected model result to **+10.91%**. These are cumulative research returns, not annualized figures.

The monthly model selects ten securities and targets 95% invested, with next-shared-session-close fills, 1% trading-value-proxy participation and no leverage or shorts. The cohort reference keeps 101 fixed slots; unavailable or unfillable slots stay cash. Average daily equity/NAV is 94.70% for the model versus 83.83% for the cohort, so the references are not exposure-matched. SPY is an external adjusted-price ETF proxy using the same monthly 95% target and cost assumption, not a pure price index. Prices embed provider distributions without separate dividend cash. The saved paths have no unresolved valuation gaps, but actual-share corporate-action accounting and execution remain unvalidated. Historical source gaps still create availability/survivorship selection.

[Replay curves](../showcase/us/conditional_replay.csv) · [Exact replay metrics](../showcase/us/conditional_replay_summary.csv) · [Saved assumptions and valuation gates](../../backtests/us/oef2015/replay_summary.json)

## Chart-only regeneration

From the repository root, in an environment with the repository's Matplotlib dependency:

```bash
python -m scripts.build_us_showcase
python -m pytest tests/test_us_showcase.py -q
```

For a separate output location:

```bash
python -m scripts.build_us_showcase --destination /tmp/us-chart-review
```

The renderer reads seven aggregate JSON records (`protocol`, `frozen_architecture`, `development_common_universe`, `confirmation_summary`, `source_audit`, `training_status`, `latest_refit_summary`) plus the selected frozen model's JSON metadata. When the two saved aggregate replay inputs (`replay_summary.json`, `replay_curves.csv`) are present, it also renders the fifth figure after checking hashes, shared dates, complete observed valuations and endpoint arithmetic. It never downloads prices, reads vendor data, loads a pickle, fits a model or replays trades. It refuses unfinished training, mismatched freeze/source hashes, non-common comparison dates, inconsistent metric arithmetic and incorrectly labelled latest-refit results.

Outputs are five PNGs in `docs/assets/us`, exact-value [selection CSV](../showcase/us/model_selection.csv), [confirmation CSV](../showcase/us/confirmation.csv), replay curves and replay metric CSVs, and a [summary with ten source SHA-256 hashes](../showcase/us/summary.json). Chart regeneration is byte-identical under the recorded runtime; font or Matplotlib version changes may affect PNG bytes. The source audit additionally records upstream data and code hashes for the original training run.
