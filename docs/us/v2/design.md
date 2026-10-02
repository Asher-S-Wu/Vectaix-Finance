# Second experiment: recovered histories and relative-return ranking

The first experiment remains immutable, including its failed confirmation. The user explicitly requested continued training and independent recovery of missing public data after seeing that result.

The next experiment has two substantive changes: verified pretermination histories from the original cohort are restored where possible, and learned models optimize within-date relative-return ranks rather than raw returns. Shorter rolling fit windows test temporal stability without searching many parameters. A conventional positive twelve-minus-one-month momentum baseline is included; the first experiment's factor is retained unchanged, never inverted after its negative recent result.

The six candidates, fixed parameters, 2020–2024 chronological folds, 24 features, purges and selection rules are declared in proposed_protocol.json before any new fitting. This is a bounded model-selection study. Each fold has earlier fitting and a separate prior-year calibration period. Architecture/window selection uses only the five pre-2025 fold outcomes. The 2025–2026 period is now a reused diagnostic; the study has no unseen future validation period. A chosen candidate is not called live-ready merely because it improves the reused period.

Source recovery may be partial. The safest contribution is original-identity OHLCV before retirement, with independently checked splits and cash distributions. Labels crossing an unresolved terminal claim remain unknown. Acquirers, replacement tickers and post-merger securities are not silently substituted. All101 original identities remain in coverage denominators.

The first run's current-vintage Yahoo history remains available; new source rows never overwrite it. Recovered Sina raw prices are converted using explicit split/cash events under a multiplicative adjustment convention, with the adjustment formula, supported period, first/last valid quote and source citations saved per identity. Incomplete adjustment evidence blocks admission of the affected interval.

The final selected rank architecture will have separate1/5/20/60-session heads. Probability, mean-return and interval calibration use only the held-out calibration period; their adequacy is measured independently from ranking. Research replays retain the first experiment's costs, timing and unresolved-holding discipline.

Nested chronological selection is reported separately: the 2022 outer year selects from 2020–2021 inner validation, 2023 selects from 2020–2022, and 2024 from 2020–2023. Each outer year's winning architecture is fixed using strictly earlier validation results. The final architecture uses all five development years only after that selection-path audit; its five-year best-of-six score is not an unbiased outer estimate.

The positive12-minus1-month momentum baseline follows the familiar past-return construction documented by the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor.html). Here252/20 trading-session lags are an approximation; this study does not recreate the CRSP/value-weighted academic factor.
