import pandas as pd
import pytest

from tmp.review_endpoint_identity import review_endpoint_rows


def inputs():
    endpoints = pd.DataFrame([{"security_id": "01036.HK", "date": pd.Timestamp("2020-01-03"),
                               "raw_close": 1.2, "volume": float("nan"), "amount": float("nan"),
                               "quote_present": False, "data_valid": True}])
    master = pd.DataFrame([{"security_id": "01036.HK", "exchange_code": "01036.HK",
                           "identity_status": "partially_verified", "identity_source_issue_id": 342,
                           "identity_source_issuer_id": 2653, "identity_source_type": "Ordinary shares",
                           "identity_source_currency": "HKD"}])
    listings = pd.DataFrame([{"ID": 63, "IssueID": 342, "StockCode": "1036", "StockExID": 1,
                             "FirstTradeDate": pd.Timestamp("1996-11-08"), "FinalTradeDate": pd.NaT,
                             "DelistDate": pd.NaT, "2ndCtr": 0, "issuer": 2653,
                             "typeLong": "Ordinary shares", "issue_currency": "HKD"}])
    quotes = pd.DataFrame([{"issueID": 342, "date": pd.Timestamp(date), "closing": 1.2,
                           "vol": volume, "turn": volume * 1.2, "susp": 0, "noclose": 0}
                          for date, volume in [("2020-01-02", 1000), ("2020-01-03", 0), ("2020-01-06", 1000)]])
    anchors = pd.DataFrame([{"security_id": "01036.HK", "date": pd.Timestamp(date), "issueID": 342,
                            "quote_source": "quotes", "all_match": True, "quote_present": True,
                            "raw_close": 1.2, "volume": 1000, "amount": 1200, "observed_conflict": False}
                           for date in ["2020-01-02", "2020-01-06"]])
    return endpoints, master, listings, quotes, anchors, pd.Timestamp("2023-12-31")


def test_existing_zero_trade_reference_price_can_support_only_that_identity_date():
    args = inputs()
    original = args[0].copy(deep=True)
    reviewed = review_endpoint_rows(*args)
    assert len(reviewed) == 1
    assert reviewed.iloc[0].identity_date_verified
    assert reviewed.iloc[0].source_listing_id == 63
    assert reviewed.iloc[0].reference_price_matches
    assert not reviewed.iloc[0].executable_quote
    pd.testing.assert_frame_equal(args[0], original)
    assert pd.isna(reviewed.iloc[0].volume)
    assert pd.isna(reviewed.iloc[0].amount)


@pytest.mark.parametrize("change,reason", [
    ("wrong_counter", "source_listing_identity_unverified"),
    ("before_listing", "source_listing_identity_unverified"),
    ("after_last_trade", "source_listing_identity_unverified"),
    ("delisting_day", "source_listing_identity_unverified"),
    ("wrong_issuer", "source_listing_identity_unverified"),
    ("second_counter", "source_listing_identity_unverified"),
    ("duplicate_listing", "source_listing_identity_unverified"),
    ("overlapping_code_reuse", "source_listing_identity_unverified"),
    ("missing_anchor", "matching_trade_anchors_missing"),
    ("anchor_source_conflict", "matching_trade_anchors_missing"),
    ("wrong_issue_quote", "source_endpoint_quote_missing"),
    ("suspended", "source_suspended_or_no_close"),
    ("price_conflict", "source_reference_price_mismatch"),
])
def test_review_rejects_unproven_identity_or_reference_price(change, reason):
    args = list(inputs())
    if change == "wrong_counter":
        args[2]["StockCode"] = "81036"
    elif change == "before_listing":
        args[2]["FirstTradeDate"] = pd.Timestamp("2020-02-01")
    elif change == "after_last_trade":
        args[2]["FinalTradeDate"] = pd.Timestamp("2020-01-02")
    elif change == "delisting_day":
        args[2]["DelistDate"] = pd.Timestamp("2020-01-03")
    elif change == "wrong_issuer":
        args[2]["issuer"] = 999
    elif change == "second_counter":
        args[2]["2ndCtr"] = 1
    elif change == "duplicate_listing":
        args[2] = pd.concat([args[2], args[2].assign(ID=64)], ignore_index=True)
    elif change == "overlapping_code_reuse":
        args[2] = pd.concat([args[2], args[2].assign(ID=64, IssueID=99, issuer=999)], ignore_index=True)
    elif change == "missing_anchor":
        args[4] = args[4].iloc[1:]
    elif change == "anchor_source_conflict":
        args[3].loc[args[3].date.eq(pd.Timestamp("2020-01-02")), "closing"] = 1.3
    elif change == "wrong_issue_quote":
        args[3]["issueID"] = 99
    elif change == "suspended":
        args[3].loc[args[3].date.eq(pd.Timestamp("2020-01-03")), ["susp", "noclose"]] = 1
    elif change == "price_conflict":
        args[3]["closing"] = 1.3
    reviewed = review_endpoint_rows(*args)
    assert not reviewed.iloc[0].identity_date_verified
    assert reviewed.iloc[0].reason == reason


def test_review_rejects_confirmation_period_input():
    args = list(inputs())
    args[0]["date"] = pd.Timestamp("2024-01-02")
    with pytest.raises(ValueError, match="开发期"):
        review_endpoint_rows(*args)


def test_a_matching_source_reference_and_prior_anchor_do_not_require_a_future_anchor():
    args = list(inputs())
    args[4] = args[4].iloc[:1]
    reviewed = review_endpoint_rows(*args)
    assert reviewed.iloc[0].identity_date_verified
    assert reviewed.iloc[0].preceding_anchor_date == pd.Timestamp("2020-01-02")
    assert pd.isna(reviewed.iloc[0].following_anchor_date)


def test_rmb_counter_uses_its_own_issue_and_listing_instead_of_the_primary_counter():
    args = list(inputs())
    for frame in [args[0], args[1], args[4]]:
        frame["security_id"] = "80016.HK"
    args[1]["exchange_code"] = "80016.HK"
    args[1]["identity_source_issue_id"] = 34304
    args[1]["identity_source_currency"] = "CNY"
    args[2]["IssueID"] = 34304
    args[2]["StockCode"] = "80016"
    args[2]["2ndCtr"] = 1
    args[2]["issue_currency"] = "CNY"
    args[3]["issueID"] = args[4]["issueID"] = 34304
    reviewed = review_endpoint_rows(*args)
    assert reviewed.iloc[0].identity_date_verified
    assert reviewed.iloc[0].source_issue_id == 34304
    assert reviewed.iloc[0].source_stock_code == "80016"
    assert reviewed.iloc[0].currency == "CNY"
