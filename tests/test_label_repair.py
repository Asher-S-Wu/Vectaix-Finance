import numpy as np
import pandas as pd
import pytest

from hk_quant.label_repair import endpoint_label_patches, export_endpoint_label_patches


def inputs():
    signal, end = pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")
    requests = pd.DataFrame([{"date": signal, "security_id": "01036.HK", "horizon": 1, "label_end": end}])
    features = pd.DataFrame([{"date": signal, "security_id": "01036.HK", "status": "ok", "raw_close": 100.,
                              "adj_close_hkd": 100., "fx_to_hkd": 1., "factor": 123.,
                              "fwd_return_1": np.nan, "label_end_1": end}])
    bars = pd.DataFrame([{"date": pd.Timestamp(date), "security_id": "01036.HK", "raw_close": price,
                          "adj_close": price, "adj_close_hkd": price if trade else np.nan,
                          "data_valid": True, "volume": 100 if trade else np.nan,
                          "amount": price * 100 if trade else np.nan, "quote_present": trade,
                          "currency": "HKD" if trade else None, "fx_to_hkd": 1. if trade else np.nan,
                          "identity_date_verified": trade, "identity_source_issue_id": 342 if trade else None}
                         for date, price, trade in [("2019-12-31", 98., True), ("2020-01-02", 100., True),
                                                    ("2020-01-03", 110., False), ("2020-01-06", 999., True)]])
    evidence = pd.DataFrame([{"date": end, "security_id": "01036.HK", "identity_date_verified": True,
                              "identity_source_issue_id": 342, "source_issue_id": 342, "source_stock_code": "01036",
                              "source_listing_id": 63, "reference_price_matches": True, "currency": "HKD",
                              "raw_close": 110., "adj_close": 110., "data_valid": True,
                              "source_closing": 110., "source_volume": 0., "source_amount": 0.,
                              "source_suspended": 0, "source_noclose": 0, "executable_quote": False,
                              "preceding_anchor_date": pd.Timestamp("2019-12-31"),
                              "following_anchor_date": pd.Timestamp("2020-01-06"),
                              "source_url": "https://example.org/original-quotes"}])
    master = pd.DataFrame([{"security_id": "01036.HK", "exchange_code": "01036.HK",
                            "identity_source_issue_id": 342, "identity_source_currency": "HKD"}])
    calendar = pd.DatetimeIndex(["2019-12-31", "2020-01-02", "2020-01-03", "2020-01-06"])
    return [requests, features, bars, evidence, master, calendar, pd.Timestamp("2023-12-31")]


def test_patch_uses_existing_adjusted_prices_and_keeps_every_input_unchanged():
    args = inputs()
    originals = [frame.copy(deep=True) for frame in args[:5]]
    patches, decisions = endpoint_label_patches(*args)
    assert len(patches) == 1
    row = patches.iloc[0]
    assert row.new_fwd_return == pytest.approx(.1)
    assert np.isnan(row.old_fwd_return)
    assert row.label_end == pd.Timestamp("2020-01-03")
    assert decisions.iloc[0].status == "patch_created"
    for actual, original in zip(args[:5], originals):
        pd.testing.assert_frame_equal(actual, original)


@pytest.mark.parametrize("case,reason", [
    ("known_label", "existing_label_kept"),
    ("not_mature", "label_not_mature"),
    ("wrong_horizon_end", "calendar_horizon_mismatch"),
    ("wrong_feature_end", "feature_label_end_mismatch"),
    ("confirmation", "outside_development_period"),
    ("missing_signal_price", "signal_price_or_identity_unverified"),
    ("missing_endpoint_price", "endpoint_price_or_data_unverified"),
    ("changed_existing_price", "endpoint_price_or_data_unverified"),
    ("invalid_endpoint_data", "endpoint_price_or_data_unverified"),
    ("wrong_currency", "endpoint_evidence_unverified"),
    ("wrong_source_issue", "endpoint_evidence_unverified"),
    ("source_no_close", "endpoint_evidence_unverified"),
    ("only_future_anchor", "past_identity_anchor_unavailable"),
    ("past_anchor_unverified", "past_identity_anchor_unavailable"),
    ("past_anchor_currency_unknown", "past_identity_anchor_unavailable"),
])
def test_patch_preserves_missingness_and_rejects_unproven_or_unmatured_labels(case, reason):
    args = inputs()
    if case == "known_label":
        args[1]["fwd_return_1"] = .2
    elif case == "not_mature":
        args[6] = pd.Timestamp("2020-01-02")
    elif case == "wrong_horizon_end":
        args[0]["label_end"] = args[1]["label_end_1"] = pd.Timestamp("2020-01-06")
    elif case == "wrong_feature_end":
        args[1]["label_end_1"] = pd.Timestamp("2020-01-06")
    elif case == "confirmation":
        args[0]["label_end"] = pd.Timestamp("2024-01-02")
    elif case == "missing_signal_price":
        args[1]["adj_close_hkd"] = np.nan
    elif case == "missing_endpoint_price":
        args[2].loc[2, "adj_close"] = np.nan
    elif case == "changed_existing_price":
        args[2].loc[2, "raw_close"] = 111.
    elif case == "invalid_endpoint_data":
        args[2].loc[2, "data_valid"] = False
    elif case == "wrong_currency":
        args[3]["currency"] = "USD"
    elif case == "wrong_source_issue":
        args[3]["source_issue_id"] = 9
    elif case == "source_no_close":
        args[3]["source_noclose"] = 1
    elif case == "only_future_anchor":
        args[3]["preceding_anchor_date"] = pd.NaT
    elif case == "past_anchor_unverified":
        args[2].loc[0, "identity_date_verified"] = False
    elif case == "past_anchor_currency_unknown":
        args[2].loc[0, "currency"] = None
    patches, decisions = endpoint_label_patches(*args)
    assert patches.empty
    assert decisions.iloc[0].status == reason


def test_following_anchor_is_retrospective_and_cannot_change_value_or_availability():
    args = inputs()
    initial, _ = endpoint_label_patches(*args)
    args[2].loc[3, ["raw_close", "adj_close", "adj_close_hkd"]] = 900000.
    args[3]["following_anchor_date"] = pd.Timestamp("2040-01-01")
    changed, _ = endpoint_label_patches(*args)
    assert changed.iloc[0].new_fwd_return == initial.iloc[0].new_fwd_return
    assert changed.iloc[0].label_available_date == initial.iloc[0].label_available_date == pd.Timestamp("2020-01-03")
    assert not changed.iloc[0].following_anchor_used_for_label


def test_missing_endpoint_bar_is_not_created_from_evidence():
    args = inputs()
    args[2] = args[2].drop(index=2)
    patches, decisions = endpoint_label_patches(*args)
    assert patches.empty
    assert decisions.iloc[0].status == "endpoint_bar_missing"


@pytest.mark.parametrize("horizon", [1, 5, 20, 60])
def test_each_horizon_uses_exact_global_trading_session_endpoint(horizon):
    args = inputs()
    args[5] = pd.bdate_range("2019-12-31", "2020-06-30")
    signal = args[0].iloc[0].date
    end = args[5][args[5].get_loc(signal) + horizon]
    args[0]["horizon"] = horizon
    args[0]["label_end"] = end
    args[1] = args[1].rename(columns={"fwd_return_1": f"fwd_return_{horizon}", "label_end_1": f"label_end_{horizon}"})
    args[1][f"label_end_{horizon}"] = end
    args[2].loc[2, "date"] = end
    args[3]["date"] = end
    args[3]["following_anchor_date"] = end + pd.Timedelta(days=1)
    patches, _ = endpoint_label_patches(*args)
    assert len(patches) == 1
    assert patches.iloc[0].horizon == horizon
    assert patches.iloc[0].label_end == end
    assert patches.iloc[0].new_fwd_return == pytest.approx(.1)


def test_export_supports_audited_securities_without_a_three_security_limit(tmp_path):
    root, proof_root = tmp_path / "data", tmp_path / "evidence"
    (root / "features").mkdir(parents=True)
    (root / "bars").mkdir()
    (root / "references").mkdir()
    proof_root.mkdir()
    requests, bars, proofs, masters = [], [], [], []
    for number in range(1, 5):
        args = inputs()
        sid, issue = f"{number:05d}.HK", 342 + number
        for frame in args[:5]:
            frame["security_id"] = sid
        args[2]["identity_source_issue_id"] = args[2].identity_source_issue_id.where(args[2].identity_source_issue_id.isna(), issue)
        args[3]["source_issue_id"] = args[3]["identity_source_issue_id"] = issue
        args[3]["source_stock_code"] = f"{number:05d}"
        args[3]["source_listing_id"] = 63 + number
        args[4]["identity_source_issue_id"] = issue
        args[4]["exchange_code"] = sid
        for horizon in [5, 20, 60]:
            args[1][f"fwd_return_{horizon}"] = np.nan
            args[1][f"label_end_{horizon}"] = pd.NaT
        args[1].to_parquet(root / "features" / f"{sid}.parquet", index=False)
        requests.append(args[0]); bars.append(args[2]); proofs.append(args[3]); masters.append(args[4])
    pd.concat(requests).to_parquet(proof_root / "prediction_gap_review.parquet", index=False)
    pd.concat(proofs).to_parquet(proof_root / "verified_endpoint_identity.parquet", index=False)
    pd.concat(masters).to_parquet(root / "securities.parquet", index=False)
    all_bars = pd.concat(bars)
    for year, frame in all_bars.groupby(all_bars.date.dt.year):
        frame.to_parquet(root / "bars" / f"{year}.parquet", index=False)
    pd.DataFrame({"cal_date": inputs()[5], "is_open": 1}).to_parquet(root / "references/calendar.parquet", index=False)
    audit = export_endpoint_label_patches(root, proof_root)
    assert audit["patch_rows"] == 4
    assert len(pd.read_parquet(proof_root / "label_patches.parquet")) == 4


def rmb_inputs():
    args = inputs()
    for frame in args[:5]:
        frame["security_id"] = "80016.HK"
    args[1]["fx_to_hkd"] = 1.1
    args[1]["adj_close_hkd"] = 100. * 1.1
    traded = args[2].quote_present
    args[2].loc[traded, "currency"] = "CNY"
    args[2].loc[traded, "fx_to_hkd"] = 1.1
    args[2].loc[traded, "adj_close_hkd"] = args[2].loc[traded, "adj_close"] * 1.1
    args[2].loc[traded, "identity_source_issue_id"] = 34304
    args[3]["currency"] = "CNY"
    args[3]["source_issue_id"] = args[3]["identity_source_issue_id"] = 34304
    args[3]["source_stock_code"] = "80016"
    args[4]["exchange_code"] = "80016.HK"
    args[4]["identity_source_issue_id"] = 34304
    args[4]["identity_source_currency"] = "CNY"
    fx = pd.DataFrame([{"date": pd.Timestamp("2020-01-03"), "currency": "CNY", "fx_to_hkd": 1.2,
                        "observation_date": pd.Timestamp("2020-01-03"), "source_url": "https://example.org/hkma/daily-fx"}])
    return args, fx


def test_rmb_label_uses_exact_endpoint_fx_and_existing_hkd_signal_price():
    args, fx = rmb_inputs()
    patches, decisions = endpoint_label_patches(*args, fx_rates=fx)
    assert len(patches) == 1
    assert decisions.iloc[0].status == "patch_created"
    assert patches.iloc[0].endpoint_fx_to_hkd == 1.2
    assert patches.iloc[0].new_fwd_return == pytest.approx(.2)
    assert patches.iloc[0].endpoint_fx_source_url == "https://example.org/hkma/daily-fx"


@pytest.mark.parametrize("case", ["no_rates", "other_day", "missing_value", "future_observation"])
def test_rmb_label_never_substitutes_fx_from_another_date_or_one(case):
    args, fx = rmb_inputs()
    if case == "no_rates":
        fx = None
    elif case == "other_day":
        fx["date"] = fx["observation_date"] = pd.Timestamp("2020-01-02")
    elif case == "missing_value":
        fx["fx_to_hkd"] = np.nan
    else:
        fx["observation_date"] = pd.Timestamp("2020-01-06")
    patches, decisions = endpoint_label_patches(*args, fx_rates=fx)
    assert patches.empty
    assert decisions.iloc[0].status == "endpoint_fx_unavailable"
