# A-share implementation and execution plan

1. Establish branch isolation and dependency environment; run the entire HK baseline suite. Keep original HK files unchanged.
2. Test then implement `ashare_quant.collect`: helper boundary, pagination, cache integrity, rate limits, provenance; probe endpoints and collect requested historical universe.
3. Test then implement `ashare_quant.data` and `features`: units, identities/calendar, adjustments, historical valuation, announcement-time joins, causal factors and mature labels. Partition large outputs.
4. Test then implement `ashare_quant.models` and `training`: CNY wrappers preserving four HK families and task status schema, explicit split contract, frozen architecture selection and latest refit; serialize artifacts and lineage.
5. In parallel test and implement `rules`/`replay`: dates/board rules, side costs, T+1, suspension/limits/liquidity, raw-share ledger, unresolved corporate-action audit. Never credit unexplained proceeds.
6. In parallel test and implement `service`/`api`/`portfolio`: separate CN research snapshots, endpoints, whole-lot risk/cash advice, schema and readiness checks.
7. Train all four families on real provenance-backed data; evaluate development only; freeze selection; run confirmation and benchmark/net-cost comparisons. Save performance and coverage including failures.
8. Generate current full-market four-horizon forecast snapshot, reports, concise commands and parity status; verify daily candidate refresh and API tests.
9. Run the complete combined suite, compile/import checks and independent code/data/leakage review. Verify git diff preserves all HK files, scan new tracked artifacts for secrets, and deliver actual trained artifacts with honest limitations. Do not push/deploy.

Checks must distinguish synthetic unit/integration tests, authentic training, source completeness, holdout evaluation and actual execution validity. A successful smoke test never substitutes for training completion.
