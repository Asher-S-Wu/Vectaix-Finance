# Independent US training implementation plan

**Goal:** Train and evaluate a real-data independent US-equity model with reproducible aggregate evidence.
**Architecture:** A small US data/feature adapter reuses shared estimator mathematics. An immutable run protocol controls fitting, development selection and confirmation. Local raw market data is kept separate from shareable code, weights and aggregates.
**Spec:** [Design](design.md)

1. Confirm source access and freeze the exact dated universe, cutoff, splits and source limitations before fitting
2. Write failing tests for parsing, calendar gaps, split/dividend semantics, backward-only features and purged temporal boundaries
3. Implement us_quant source, features, model adapter and training runner; verify red-to-green tests
4. Audit source coverage and anomalous rows; retain missing identity/outcome evidence
5. Fit factor, Ridge, small LightGBM and large LightGBM; save each weight and metadata file
6. Evaluate 2024 on common score availability, freeze the selection, then evaluate untouched confirmation once
7. Build a next-session monthly long-only adjusted-unit replay with costs and unresolved-holding gates; test cash, turnover, timing and bounds
8. Render source-backed aggregate charts, inspect pixels, and update all language READMEs without changing legacy sections
9. Run US tests and repository regression checks; independently review methodology, evidence hashes and unchanged legacy assets

Review focus: unavailable delisted histories; ticker reuse and corporate reorganizations; incomplete recent sessions; latent raw/split-adjusted mismatch; endpoint gaps or unresolved holdings that would make returns look better than warranted.
