import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / "tmp" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def securities():
    return pd.DataFrame([
        {"security_id": "00001.HK", "exchange_code": "00001.HK", "isin": "ISIN1",
         "list_date": "2010-01-01", "delist_date": None, "identity_valid_from": "2010-01-01",
         "identity_valid_to": None, "identity_status": "verified", "asset_type": "equity"},
        {"security_id": "80001.HK", "exchange_code": "80001.HK", "isin": "ISIN1",
         "list_date": "2023-06-19", "delist_date": None, "identity_valid_from": "2023-06-19",
         "identity_valid_to": None, "identity_status": "verified", "asset_type": "equity"},
    ])


def archive_inputs():
    listings = pd.DataFrame([
        {"IssueID": 1, "StockCode": "0001", "StockExID": 1, "FirstTradeDate": "2010-01-01",
         "FinalTradeDate": None, "DelistDate": None},
        {"IssueID": 1, "StockCode": "80001", "StockExID": 1, "FirstTradeDate": "2023-06-19",
         "FinalTradeDate": None, "DelistDate": None},
    ])
    eligible = pd.DataFrame([{"IssueID": 1, "typeID": 0}])
    hkex = pd.DataFrame([{"IssueID": 1, "StockCode": "0001", "BoardLot": 500,
                          "PriceDate": "2025-12-24", "modified": "2025-12-24"}])
    old = pd.DataFrame([{"issueID": 1, "until": "2020-06-23", "lot": 1000}])
    periods = listings.copy()
    periods["mapped_security_ids"] = [["00001.HK"], ["80001.HK"]]
    return listings, eligible, hkex, old, periods, securities()


def test_archive_price_date_and_until_do_not_prove_verified_history():
    result, _ = script("build_archive_historical_lot_facts")._build_intervals(*archive_inputs())
    assert not result.verified.any()
    assert not result.account_backtest_eligible.any()
    snapshots = result[result.source_document.str.endswith("hkexdata.parquet")]
    assert snapshots.lot_valid_from.eq(snapshots.lot_valid_to).all()


def test_archive_current_lot_does_not_spread_to_other_counter():
    result, _ = script("build_archive_historical_lot_facts")._build_intervals(*archive_inputs())
    snapshots = result[result.source_document.str.endswith("hkexdata.parquet")]
    assert not snapshots.security_id.eq("80001.HK").any()
    old = result[result.source_document.str.endswith("oldlots.parquet")]
    assert old.security_id.isna().all(), "IssueID-only lot history cannot identify either trading counter"


def test_archive_maps_reit_identity_through_its_source_counter():
    args = list(archive_inputs())
    args[0] = args[0].iloc[:1]
    args[4] = args[4].iloc[:1].copy()
    args[4]["mapped_security_ids"] = [["HKREIT:4875"]]
    args[5] = args[5].iloc[:1].assign(security_id="HKREIT:4875", asset_type="reit")
    result, _ = script("build_archive_historical_lot_facts")._build_intervals(*args)
    snapshots = result[result.source_document.str.endswith("hkexdata.parquet")]
    assert snapshots.iloc[0].security_id == "HKREIT:4875"


def test_archive_does_not_map_source_counter_to_a_different_code():
    module = script("build_archive_historical_lot_facts")
    listing = pd.Series({"FirstTradeDate": pd.Timestamp("2010-01-01"), "FinalTradeDate": pd.NaT,
                         "DelistDate": pd.NaT, "code5": "00001"})
    period = pd.Series({"FirstTradeDate": pd.Timestamp("2010-01-01"), "FinalTradeDate": pd.NaT,
                        "DelistDate": pd.NaT, "StockCode": "80001"})
    by_code = {"00001": pd.DataFrame({"security_id": ["00001.HK"]})}
    mapped, _ = module._mapped_candidates(1, listing, {1: [(period, ["00001.HK"])]}, by_code, {"00001.HK"})
    assert mapped is None


def test_archive_does_not_use_overlapping_lifecycle_from_another_listing():
    module = script("build_archive_historical_lot_facts")
    listing = pd.Series({"FirstTradeDate": pd.Timestamp("2010-01-01"), "FinalTradeDate": pd.NaT,
                         "DelistDate": pd.NaT, "code5": "00001"})
    period = pd.Series({"FirstTradeDate": pd.Timestamp("2015-01-01"), "FinalTradeDate": pd.NaT,
                        "DelistDate": pd.NaT, "StockCode": "00001"})
    by_code = {"00001": pd.DataFrame({"security_id": ["00001.HK"]})}
    mapped, _ = module._mapped_candidates(1, listing, {1: [(period, ["00001.HK"])]}, by_code, {"00001.HK"})
    assert mapped is None


def test_archive_expired_listing_does_not_receive_current_snapshot():
    args = list(archive_inputs())
    args[0].loc[0, "FinalTradeDate"] = "2020-01-01"
    args[0].loc[0, "DelistDate"] = "2021-01-01"
    result, _ = script("build_archive_historical_lot_facts")._build_intervals(*args)
    snapshots = result[result.source_document.str.endswith("hkexdata.parquet")]
    assert not snapshots.security_id.eq("00001.HK").any()


def run_current(tmp_path, monkeypatch, security_rows=None, changes=None, isin="ISIN1"):
    module = script("build_current_period_lot_candidate")
    v5 = tmp_path / "v5"
    ref = v5 / "references" / "reit_research"
    out = v5 / "references" / "historical_lot_archive"
    ref.mkdir(parents=True)
    s = securities().iloc[:1] if security_rows is None else security_rows
    s.to_parquet(v5 / "securities.parquet", index=False)
    pd.DataFrame([{"Stock Code": "00001", "Name of Securities": "One", "Category": "Equity",
                   "Sub-Category": "Ordinary Shares", "Board Lot": 500, "ISIN": isin}]).to_csv(ref / "equities_and_reits.csv", index=False)
    (ref / "sources.json").write_text(json.dumps({"source_file_updated_at": "2026-09-11",
                                                 "source_url": module.HKEX_URL}), encoding="utf-8")
    pd.DataFrame(columns=["ts_code", "trade_unit", "isin"]).to_parquet(v5 / "references" / "hk_basic.parquet", index=False)
    candidates = tmp_path / "candidates.json"
    seed_change = {"stock_codes": ["00002"], "effective_date": "2026-03-01", "old_lot": 1000,
                   "new_lot": 500, "evidence_status": "candidate_effective", "verified": False,
                   "review_reasons": [], "source_url": "https://example.org/other.pdf"}
    candidates.write_text(json.dumps({"records": [seed_change] if changes is None else changes}), encoding="utf-8")
    monkeypatch.setattr(module, "V5", v5)
    monkeypatch.setattr(module, "HKEX", ref)
    monkeypatch.setattr(module, "OUT", out)
    monkeypatch.setattr(module, "BOARD_LOT", candidates)
    module.main()
    return pd.read_parquet(out / "current_period_lot_candidate.parquet")


def test_current_list_proves_only_the_source_asof_date(tmp_path, monkeypatch):
    result = run_current(tmp_path, monkeypatch)
    result = result[result.evidence_role.eq("current_list_day_observation")]
    assert result.lot_valid_from.eq(pd.Timestamp("2026-09-11")).all()
    assert result.lot_valid_to.eq(pd.Timestamp("2026-09-11")).all()
    assert result.source_observation_date.eq(pd.Timestamp("2026-09-11")).all()
    assert result.verified.all()


def test_candidate_effective_announcement_is_not_verified(tmp_path, monkeypatch):
    result = run_current(tmp_path, monkeypatch, changes=[
        {"stock_codes": ["00001"], "effective_date": "2026-03-01", "old_lot": 1000,
         "new_lot": 500, "evidence_status": "candidate_effective", "verified": False,
         "review_reasons": [], "source_url": "https://example.org/announcement.pdf",
         "evidence_sentences": ["The board lot will change to 500 shares."]}
    ])
    announcement = result[result.source_url.eq("https://example.org/announcement.pdf")]
    assert len(announcement) == 1
    assert not announcement.verified.any()
    assert not announcement.account_backtest_eligible.any()
    assert announcement.lot_valid_to.eq(pd.Timestamp("2026-03-01")).all()


@pytest.mark.parametrize("case", ["expired", "delisting_day", "identity_expired", "identity_not_started", "wrong_isin", "ambiguous_code"])
def test_current_identity_is_scoped_to_code_isin_and_active_date(tmp_path, monkeypatch, case):
    s = securities().iloc[:1].copy()
    isin = "ISIN1"
    if case == "expired":
        s["delist_date"] = "2026-02-01"
    elif case == "delisting_day":
        s["delist_date"] = "2026-09-11"
    elif case == "identity_expired":
        s["identity_valid_to"] = "2026-08-01"
    elif case == "identity_not_started":
        s["identity_valid_from"] = "2026-10-01"
    elif case == "wrong_isin":
        isin = "OTHER_ISSUER"
    else:
        s = pd.concat([s, s.assign(security_id="00001.HK_OTHER")], ignore_index=True)
    result = run_current(tmp_path, monkeypatch, security_rows=s, isin=isin)
    assert not result.verified.any()
    assert not result.account_backtest_eligible.any()


def test_an_empty_announcement_catalog_does_not_require_a_fabricated_chain(tmp_path, monkeypatch):
    result = run_current(tmp_path, monkeypatch, changes=[])
    assert len(result) == 1
    assert result.lot_valid_from.eq(pd.Timestamp("2026-09-11")).all()


def test_merge_rejects_true_flag_on_a_historical_interval_or_unknown_date_binding():
    module = script("merge_archive_lot_facts_candidate")
    result = module.merge_verified_days(pd.DataFrame([
        fact(lot_valid_from=pd.Timestamp("2010-01-01")),
        fact(historical_value_date_verified=False),
    ]))
    assert result.empty


def fact(date="2020-01-02", **kwargs):
    row = {"security_id": "00001.HK", "lot_size": 500, "lot_valid_from": pd.Timestamp(date),
           "lot_valid_to": pd.Timestamp(date), "verified": True, "source_fact_verified": True,
           "source_observation_date": pd.Timestamp(date),
           "execution_counter_verified": True, "account_backtest_eligible": True,
           "historical_value_date_verified": True, "identity_mapping_verified": True,
           "source_url": "https://example.org/history", "source_issue_id": 1, "source_listing_id": 10,
           "source_document": "history.html", "stock_code": "00001", "source_stock_code": "00001",
           "evidence_role": "public_historical_board_lot_snapshot", "snapshot_consistent": True}
    row.update(kwargs)
    return row


def test_merge_accepts_verified_reit_identity_that_is_not_its_ticker():
    module = script("merge_archive_lot_facts_candidate")
    result = module.merge_verified_days(pd.DataFrame([
        fact(security_id="HKREIT:4875", stock_code="00405", source_stock_code="00405",
             source_issue_id=4875, source_listing_id=909),
    ]))
    assert len(result) == 1
    assert result.iloc[0].security_id == "HKREIT:4875"


def test_merge_rejects_unverified_identity_even_if_the_code_matches():
    module = script("merge_archive_lot_facts_candidate")
    result = module.merge_verified_days(pd.DataFrame([fact(identity_mapping_verified=False)]))
    assert result.empty


def test_merge_rejects_stale_observation_even_when_stored_flags_claim_true():
    module=script('merge_archive_lot_facts_candidate')
    result=module.merge_verified_days(pd.DataFrame([fact(source_observation_date=pd.Timestamp('2019-12-31'))]))
    assert result.empty


def test_archive_merge_keeps_daily_proof_and_excludes_unverified_intervals(tmp_path, monkeypatch):
    module = script("merge_archive_lot_facts_candidate")
    v5, root = tmp_path / "v5", tmp_path / "archive"
    (v5 / "bars").mkdir(parents=True)
    root.mkdir()
    pd.DataFrame([fact()]).to_parquet(v5 / "historical_lots.parquet", index=False)
    pd.DataFrame([fact(lot_valid_from=pd.Timestamp("2010-01-01"), verified=False,
                       historical_value_date_verified=False, lot_size=1000)]).to_parquet(root / "archive_historical_lot_facts.parquet", index=False)
    pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "security_id": ["00001.HK"] * 2}).to_parquet(v5 / "bars" / "one.parquet", index=False)
    monkeypatch.setattr(module, "V5", v5)
    monkeypatch.setattr(module, "ARCHIVE", root)
    monkeypatch.setattr(module, "OUT", root)
    module.main()
    result = pd.read_parquet(root / "historical_lots_formal_candidate.parquet")
    assert len(result) == 1
    assert result.lot_size.eq(500).all()
    audit = json.loads((root / "formal_candidate_audit.json").read_text(encoding="utf-8"))
    assert audit["covered_bar_rows"] == 1


def test_current_merge_counts_only_verified_nonoverlapping_day_evidence(tmp_path, monkeypatch):
    module = script("merge_current_period_lot_candidate")
    v5, root = tmp_path / "v5", tmp_path / "archive"
    (v5 / "bars").mkdir(parents=True)
    root.mkdir()
    pd.DataFrame([fact()]).to_parquet(root / "historical_lots_formal_candidate.parquet", index=False)
    pd.DataFrame([fact(), fact("2020-01-03", verified=False),
                  fact("2020-01-04", historical_value_date_verified=False)]).to_parquet(root / "current_period_lot_candidate.parquet", index=False)
    pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04"]), "security_id": ["00001.HK"] * 3}).to_parquet(v5 / "bars" / "one.parquet", index=False)
    monkeypatch.setattr(module, "V5", v5)
    monkeypatch.setattr(module, "ROOT", root)
    module.main()
    audit = json.loads((root / "formal_candidate_with_current_audit.json").read_text(encoding="utf-8"))
    assert audit["covered_bar_rows"] == 1
    assert audit["overlapped_bar_rows"] == 0
