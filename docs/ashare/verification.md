# Verification record

Source baseline: `2881428` on the original main branch. Work is isolated on `feat/ashare-model-parity`. No original `hk_quant/`, showcase or README file is modified.

## Original source distribution

The unmodified repository's bare pytest run cannot collect five modules because they import omitted `tmp/` utilities:

- tests/test_endpoint_identity_review.py
- tests/test_import_reviewed_cash_actions.py
- tests/test_lot_document_identity.py
- tests/test_lot_snapshot_facts.py
- tests/test_lot_snapshot_partition_audit.py

Running all other original tracked tests yielded 659 passed and 26 failed. Those failures reference further omitted temporary scripts or saved fixtures (including HKEX search HTML); they are not silently skipped or represented as new A-share regressions. No HK implementation was changed to hide them.

## A-share checks

TDD-focused tests cover data-unit conversion, cache checksums/pagination/rate limiting, listing bounds and BJ alias collisions, missing source data, causal rolling features, label maturity, strict train/calibration/development/confirmation separation, frozen architecture choice, dataset/model hashes, all prediction task schemas, API authentication and disclosures, risk/cash/lot/T+1/price-limit advice, actual-share and reference replay, dated corporate-action evidence, reporting and candidate-only daily refresh.

A complete isolated synthetic integration fixture is explicitly labeled synthetic and kept out of the real data/model namespaces. It exercises normalization → 37-factor construction → all four actual estimators → development freeze → holdout → latest snapshot → API → both replay paths → benchmark/report. Its purpose is interface verification only; it does not establish real-data training completion or investment performance.

Fresh final-code checks on 2026-09-30: 302 A-share tests passed with one opt-in integration skip. Enabling the complete synthetic path yielded 15 passing integration tests in 88.53 seconds. The combined source suite, explicitly excluding only the five uncollectable legacy modules listed above, yielded 961 passed, 26 baseline failures and one opt-in skip. A fresh bare pytest run still encounters exactly those five collection errors. The 26 failure names and causes are unchanged from the original source distribution.

The dated security master and exchange calendar now define each identity's complete listed period, including missing suspension/pre-delist tails and listed identities without any source quotes. Such rows retain unavailable status, calendar-derived horizon dates and missing security-specific inputs/returns. Four regressions verify these cases and an entirely unavailable year's empty training partition; existing observed features and raw-file hashes remain unchanged.

The current real-data experiment is not considered complete until its source coverage, authentic training artifacts, confirmation results, execution limitations and final combined checks are recorded in the generated report.

## Authentic fitting and inference audit

All 149 original tracked files match source revision 2881428 byte-for-byte. The genuine run completed all four candidate calibration/evaluation paths, including three learned-weight estimators and the fixed-score factor baseline. Development-only paired selection chose Ridge; its frozen model and latest-data refit are distinct artifacts.

Independent streaming checks reconciled all source/data/model hashes and all saved sample/horizon/direction counts. There are 653,312 deterministic sampled observations: 256 names on each of 2,552 dates. Frozen fitting labels end 2022-12-30 and calibration labels end 2023-12-29; latest fitting labels end 2025-09-29 and its calibration ends 2026-09-30. Paired 2024 selection recomputes to 222 IC dates and Ridge IC 0.05559809440534042. The 2025–2026 20-session confirmation metric recomputes to 404 IC dates and IC 0.11112162752244786. Negative probability Brier skill at all horizons is retained as a performance limitation, not hidden by ranking results.

The current snapshot contains all 5,572 listed identities and 22,288 horizon rows. Its 80 unscorable identities, including 11 without current quotes, remain visible. Real offline API checks verified authentication, status, rankings, all four forecast horizons, unavailable-name handling, JSON/CSV advice and T+1. Regenerated latest numerical forecasts match stored values within 1e-12.

## Bounded-memory normalization recovery

The initial real normalization process exited 137 before any estimator fitting. A fresh default Arrow read of the 311 MiB 2018 frame used 3,817.7 MiB peak RSS; limiting batch/file read-ahead and metadata caching reduced that identical read to 1,134.5 MiB. A complete eleven-year read/rank probe stayed within 3,294.7 MiB. Regression checks across 300 identities preserve exact rows, column order, nullable dtypes, attributes, all normalized factors and 256-name daily samples. The retained actual 2016/2017 normalized files match the rebuilt files byte-for-byte. No source history, factor, split or selection rule was reduced. The sandbox does not expose its cgroup memory cap; stage telemetry records process memory instead.

## Authentic source continuity checks (2016–2018)

The first three completed years contain 2,201,773 normalized A-share daily quote rows. Historical aliases were verified against issuer/exchange implementation notices. Equivalent model-input rows are deduplicated; 24 factor rows with retired-alias three-decimal rounding were reconciled to the higher-precision alias only within a 0.0005 absolute bound, preserving raw rows and an explicit audit.

Three larger continuity mismatches were found in the raw price / ex-right reference / adjustment-factor relationship. A separate security-range API query confirmed the 000998.SZ August 2018 factor inconsistency; no raw-data correction was invented. Official notices show merger-review suspension/resumption, with no documented distribution explaining the factor drop/restoration. 600733.SH has documented shareholder-specific consideration and capitalization, so its quoted reference and investor entitlements cannot be equated with a generic factor ratio.

The feature builder preserves all raw data and audit evidence, resets past-only rolling history at unresolved continuity breaks, and never creates forward labels across those breaks. This is quarantine with explicit unavailability, not repaired or synthetic return data. See `continuity_evidence.csv` for primary source links. Final manifests report all detected breaks and maturity/availability coverage.

## Post-training service scalability and disclosure checks

The authentic current snapshot contains 22,288 forecast rows across 5,572 securities. Its copied-bundle API smoke exercised all four horizons, authenticated rankings, CNY JSON/CSV portfolio advice over 5,439 eligible candidates, same-day T+1 holdings, and latest-prediction regeneration from the fitted Ridge model. Regenerated numeric outputs matched the saved snapshot within 1e-12; the original149 tracked source/showcase files matched revision2881428 byte for byte.

A full-universe advice call exposed quadratic growth in the constraint matrix because the entire fee expression was repeated in each concentration cap. An explicit scalar NAV with NAV + fee_bound = 1 and vectorized caps preserves the optimization constraints while keeping the matrix sparse. The regression test first recorded22,303 matrix nonzeros for100 securities and failed its linear-sparsity bound; it passes after the change. Seven small numerical cases, including frozen positions, STAR/BSE lots, partial sellability and equal weighting, retained identical rounded orders, costs and weights.

CVXPY1.9.2 also computed Sum shapes by summing uninitialized NumPy arrays, producing a spurious overflow warning on the large universe. Equivalent inner products and sum_squares avoid that upstream shape-inference path without suppressing warnings or changing installed packages. The real bundle smoke passed in18.62seconds with RuntimeWarnings treated as errors. Equivalent solver formulations can move a near-integer rounding boundary at full scale; the two valid large-universe smoke allocations differed byCNY14.66 of target cash, and both passed the final cash/risk/participation audits. No fitted model, prediction, training or historical replay values were changed.

Forecast tasks remain independently visible. Account advice deliberately retains the inherited conservative requirement that all tasks be valid before admitting a trade candidate. Separate tests cover valid expected returns coexisting with unavailable probability/interval tasks.

The explicit post-report evidence CLI validates the completed run's frozen/latest identities, probability metrics and artifact hashes, preserves prior evidence, and atomically selects an immutable disclosure certificate. Tests cover idempotence, relative paths, interrupted pointer writes, consistent reads, and unchanged model/forecast bytes. At this checkpoint77 focused service/API/portfolio/evidence tests passed. Final packaging rechecks the real snapshot after the disclosure step; it does not treat this checkpoint as a completed historical report.

## Final cached-event correction and unresolved terminal value

The provider explicitly reports total stock distribution as zero while often omitting its breakdown. The parser now accepts that cash-only fact when any supplied component is also zero, and still rejects missing aggregate values, contradictory/nonfinite components and ambiguous share-listing dates. Raw nulls remain untouched. A cache-only rerun accepted 2,459 cash schedules, reconciled 49 alias duplicates and retained 10,485 unhandled raw rows. It issued no external queries and changed no trained model or numerical forecast. The method-reference summaries remain byte-identical.

The actual-share audit remains research-limited with five unresolved corporate-action holdings. The method-reference portfolio has an unresolved successor-security claim: 600636.SH delisted on 2026-06-29 and continued as 400297/国化5; no verified terminal valuation or executable exit is supplied. Total return, CAGR and drawdown are therefore unavailable, not zero and not a certified portfolio return. Official exchange/issuer/broker evidence is recorded in terminal_claim_evidence.json. Final source changes are reconciled in final_source_audit.json; frozen and latest model bytes and numeric predictions remain unchanged.
