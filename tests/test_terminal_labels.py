import pandas as pd
import pytest

from hk_quant.terminal_labels import (
    PATCH_KEYS,
    SUPPORTED_EVENT_TYPES,
    _date_literals,
    _event_dates,
    _load_signal_bars,
    build_terminal_cash_label_patches,
)


def test_compulsory_acquisition_cash_event_type_is_supported():
    assert "compulsory_acquisition_cash" in SUPPORTED_EVENT_TYPES


def test_date_literals_accept_month_first_official_announcement_format():
    literals = _date_literals("2021-07-14")
    assert "July 14, 2021" in literals
    assert "July 14th, 2021" in literals


def test_signal_bars_loader_reads_signal_and_terminal_years(tmp_path):
    root = tmp_path / "data"
    (root / "bars").mkdir(parents=True)
    columns = {
        "security_id": ["00382!AA.HK"],
        "date": [pd.Timestamp("2017-12-29")],
        "raw_close": [2.0],
        "cum_adjfactor": [1.0],
    }
    pd.DataFrame(columns).to_parquet(root / "bars" / "2017.parquet", index=False)
    pd.DataFrame({**columns, "date": [pd.Timestamp("2018-02-06")], "raw_close": [2.06]}).to_parquet(
        root / "bars" / "2018.parquet", index=False
    )

    bars = _load_signal_bars(root, "00382!AA.HK", pd.Timestamp("2017-12-29"), pd.Timestamp("2018-02-06"))

    assert set(pd.to_datetime(bars.date)) == {pd.Timestamp("2017-12-29"), pd.Timestamp("2018-02-06")}


def test_event_dates_allow_payment_before_listing_withdrawal():
    effective, withdrawal, settlement = _event_dates({
        "effective_date": "2020-09-24",
        "listing_withdrawal_date": "2020-09-25",
        "settlement_date": "2020-09-24",
    })

    assert effective == pd.Timestamp("2020-09-24")
    assert withdrawal == pd.Timestamp("2020-09-25")
    assert settlement == pd.Timestamp("2020-09-24")


def test_unverified_terminal_event_is_rejected(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    pd.DataFrame({"security_id": ["01296.HK"]}).to_parquet(root / "securities.parquet", index=False)
    events = root / "events.parquet"
    pd.DataFrame([{
        "event_id": "unverified",
        "event_type": "privatisation_cash",
        "security_id": "01296.HK",
        "verified": False,
    }]).to_parquet(events, index=False)
    queue = root / "queue.parquet"
    pd.DataFrame(columns=PATCH_KEYS + ["gap_reason"]).to_parquet(queue, index=False)

    with pytest.raises(ValueError, match="终止事件未经核验"):
        build_terminal_cash_label_patches(root, events, queue)
