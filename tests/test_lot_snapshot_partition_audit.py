import gzip
import json

import pandas as pd

from tmp import parse_webb_board_lot_snapshots as parser


def setup_data(tmp_path, monkeypatch, dates):
    data = tmp_path / "v5"
    root = data / "references/webb_board_lot_snapshots"
    (root / "raw/all_hk").mkdir(parents=True)
    (root / "parsed").mkdir()
    pd.DataFrame({"cal_date": pd.to_datetime(dates), "is_open": 1}).to_parquet(data / "references/calendar.parquet", index=False)
    monkeypatch.setattr(parser, "DATA", data)
    monkeypatch.setattr(parser, "ROOT", root)
    identity = {("00001", 1): [{"security_id": "00001.HK", "start": pd.Timestamp("1990-01-01"),
                                "end": None, "source_listing_id": 10, "identity_mapping_basis": "source_issue_isin_same_counter"}]}
    return root, identity


def raw_page(root, day):
    body = f'''<input type="date" name="d" value="{day}">
        <table><tr><th>Stock Code</th><th>Date</th><th>Board lot</th></tr>
        <tr><td><a href="str.asp?i=1">00001</a></td><td>{day}</td><td>500</td></tr></table>'''.encode()
    path = root / "raw/all_hk" / f"{day.replace('-', '')}.html.gz"
    path.write_bytes(gzip.compress(body))
    return body, path


def parsed_day(root, identity, day):
    body, path = raw_page(root, day)
    frame = parser.parse_snapshot(body, day, identity, "https://example.org/source", str(path))
    frame["parse_protocol"] = parser.PROTOCOL
    return frame


def test_one_year_update_keeps_other_partitions_and_full_2010_2026_audit(tmp_path, monkeypatch):
    root, identity = setup_data(tmp_path, monkeypatch, ["2014-01-02", "2015-01-02", "2016-01-04"])
    before = {}
    for day in ["2014-01-02", "2016-01-04"]:
        path = root / "parsed" / f"{day[:4]}.parquet"
        parsed_day(root, identity, day).to_parquet(path, index=False)
        before[path] = path.read_bytes()
    raw_page(root, "2015-01-02")
    audit = parser.parse_market("all_hk", identity, "2015-01-01", "2015-12-31")
    assert audit["start"] == "2010-01-01"
    assert audit["end"] == "2026-09-11"
    assert audit["files"] == audit["actual_dates"] == 3
    assert audit["rows"] == audit["mapped_rows"] == 3
    assert [entry["year"] for entry in audit["years"]] == [2014, 2015, 2016]
    assert all(path.read_bytes() == content for path, content in before.items())


def test_day_update_does_not_remove_unrequested_dates_in_same_year(tmp_path, monkeypatch):
    root, identity = setup_data(tmp_path, monkeypatch, ["2015-01-02", "2015-06-01"])
    parsed_day(root, identity, "2015-01-02").to_parquet(root / "parsed/2015.parquet", index=False)
    raw_page(root, "2015-06-01")
    audit = parser.parse_market("all_hk", identity, "2015-06-01", "2015-06-01")
    frame = pd.read_parquet(root / "parsed/2015.parquet")
    assert frame.date.dt.strftime("%Y-%m-%d").tolist() == ["2015-01-02", "2015-06-01"]
    assert audit["actual_dates"] == 2


def test_failures_outside_requested_year_remain_in_full_audit(tmp_path, monkeypatch):
    root, identity = setup_data(tmp_path, monkeypatch, ["2014-01-02", "2015-01-02"])
    failure = {"source_document": str((root / "raw/all_hk/20140102.html.gz").resolve()), "reason": "bad source page"}
    (root / "parsed/audit.json").write_text(json.dumps({"failures": [failure]}), encoding="utf-8")
    raw_page(root, "2015-01-02")
    audit = parser.parse_market("all_hk", identity, "2015-01-01", "2015-12-31")
    assert failure in audit["failures"]
    assert audit["missing_dates"] == ["20140102"]
    assert audit["status"] == "incomplete"


def test_frozen_inventory_ignores_pages_collected_after_rebuild_started(tmp_path, monkeypatch):
    root, identity = setup_data(tmp_path, monkeypatch, ["2015-01-02", "2016-01-04"])
    _, first = raw_page(root, "2015-01-02")
    inventory = [first]
    raw_page(root, "2016-01-04")
    audit = parser.parse_market("all_hk", identity, "2010-01-01", "2026-09-11", inventory=inventory)
    assert audit["actual_dates"] == 1
    assert not (root / "parsed/2016.parquet").exists()
    assert audit["missing_dates"] == ["20160104"]


def test_v3_staging_does_not_overwrite_existing_v2_partitions(tmp_path, monkeypatch):
    root, identity = setup_data(tmp_path, monkeypatch, ["2015-01-02"])
    old=parsed_day(root, identity, "2015-01-02")
    old["parse_protocol"]='source-issue-counter-date-v2'
    original=root/'parsed/2015.parquet'
    old.to_parquet(original,index=False)
    original_bytes=original.read_bytes()
    staging=root/'v3_stage/all_hk'
    parser.parse_market('all_hk',identity,'2010-01-01','2026-09-11',
                        inventory=list((root/'raw/all_hk').glob('*.html.gz')),output=staging)
    assert original.read_bytes()==original_bytes
    assert pd.read_parquet(staging/'2015.parquet').parse_protocol.eq('source-issue-counter-date-v3').all()
