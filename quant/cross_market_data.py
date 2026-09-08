from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import research_calendar
from .general_data import load_general_dataset
from .cross_market_identity import load_cross_market_identity


ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path):
    envelope = json.loads(Path(path).read_text())
    if envelope["code"] != 0 or envelope["data"]["has_more"] is not False:
        raise ValueError(f"未成功或未证明完整的数据源：{path}")
    d = envelope["data"]
    if len(d["fields"]) != len(set(d["fields"])) or any(len(row) != len(d["fields"]) for row in d["items"]):
        raise ValueError(f"来源表格字段矛盾：{path}")
    return pd.DataFrame(d["items"], columns=d["fields"])


def support():
    root = ROOT/"data/raw/cross-market-support"
    manifest = json.loads((root/"manifest.json").read_text())
    hashes = {str(root/"manifest.json"): digest(root/"manifest.json")}
    for r in manifest["requests"]:
        path = root/r["file"]
        if digest(path) != r["sha256"]:
            raise ValueError(f"港股日历/指数来源已改变：{path}")
        hashes[str(path)] = r["sha256"]
    calendars = [table(root/f"calendar_{year}.json") for year in range(2010, 2027)]
    calendar_frame = pd.concat(calendars, ignore_index=True)
    calendar_frame["date"] = pd.to_datetime(calendar_frame.cal_date, format="%Y%m%d").dt.as_unit("ns")
    expected_civil = pd.date_range("2010-01-01", "2026-08-31").as_unit("ns")
    if (calendar_frame.date.duplicated().any() or not calendar_frame.is_open.isin([0, 1]).all()
            or not pd.DatetimeIndex(calendar_frame.date.sort_values()).equals(expected_civil)):
        raise ValueError("港股来源日历未逐个自然日完整覆盖。")
    calendar = pd.DatetimeIndex(calendar_frame.loc[calendar_frame.is_open.eq(1), "date"].sort_values()).as_unit("ns")
    if not calendar.equals(research_calendar().as_unit("ns")):
        raise ValueError("港股两套独立交易日历不一致，不能自动选择。")
    benchmark = pd.concat([table(root/"HSI_20100101_20181231.json"), table(root/"HSI_20190101_20260831.json")], ignore_index=True)
    benchmark["date"] = pd.to_datetime(benchmark.trade_date, format="%Y%m%d").dt.as_unit("ns")
    benchmark = benchmark.sort_values("date", ignore_index=True)
    if (not benchmark.ts_code.eq("HSI").all() or not pd.DatetimeIndex(benchmark.date).equals(calendar)
            or not np.isfinite(benchmark.close.to_numpy(float)).all() or not benchmark.close.gt(0).all()):
        raise ValueError("恒生价格指数未完整覆盖港股日历或收盘价无效。")
    return calendar, benchmark, hashes


def load_cross_market_dataset():
    a = load_general_dataset(ROOT/"data/general-tushare")
    calendar, benchmark, hashes = support()
    source = ROOT/"data/cross-market-hk"
    manifest = json.loads((source/"manifest.json").read_text())
    if manifest["collectionComplete"] is not True or manifest["start"] != "20100101" or manifest["end"] != "20260831":
        raise ValueError("港股固定采集尚未全部完成或日期范围错误。")
    if digest(source/"specification.json") != manifest["specificationSha256"]:
        raise ValueError("港股采集设计与冻结指纹不一致。")
    specification = json.loads((source/"specification.json").read_text())
    if manifest["stocks"] != specification["stocks"]:
        raise ValueError("港股实际股票池与预先冻结设计不同。")
    hashes[str(source/"manifest.json")] = digest(source/"manifest.json")
    source_audit = json.loads((source/"collection-audit.json").read_text())
    hashes[str(source/"collection-audit.json")] = digest(source/"collection-audit.json")
    for name, expected in manifest["fileHashes"].items():
        path = source/name
        if not path.resolve().is_relative_to(source.resolve()) or digest(path) != expected:
            raise ValueError(f"港股采集文件指纹或路径错误：{name}")
        hashes[str(path)] = expected
    requests = {(r["request"]["ts_code"], r["request"]["mode"]): r for r in manifest["requests"]}
    expected_requests = {(s["ts_code"], mode) for s in specification["stocks"] for mode in ("raw", "qfq-factor", "hfq-factor")}
    if set(requests) != expected_requests or len(requests) != len(manifest["requests"]):
        raise ValueError("港股每个预定证券必须完整保留三条来源结果。")
    hk_stocks = pd.DataFrame(manifest["stocks"])
    hk_stocks["symbol"] = hk_stocks.ts_code
    if hk_stocks.symbol.duplicated().any():
        raise ValueError("港股候选身份出现重复。")
    identity = load_cross_market_identity(a["stocks"].copy(), hk_stocks.copy())
    identity_frame = identity["stocks"]
    identity_columns = ["symbol", "issuer_id", "identity_status", "identity_resolution", "ordinary_share_status", "identity_reasons", "split"]
    a_stocks = a["stocks"].drop(columns=["split"]).rename(columns={"market": "listing_board"})
    a_stocks = a_stocks.merge(identity_frame.loc[identity_frame.market.eq("A"), identity_columns], on="symbol", validate="one_to_one")
    hk_stocks = hk_stocks.rename(columns={"market": "listing_board"}).merge(
        identity_frame.loc[identity_frame.market.eq("HK"), identity_columns], on="symbol", validate="one_to_one")
    for field in ("list_date", "delist_date"):
        hk_stocks[field] = pd.to_datetime(hk_stocks[field], format="%Y%m%d", errors="raise").dt.as_unit("ns")
    frames, audits = [], []
    for stock in hk_stocks.to_dict("records"):
        symbol = stock["symbol"]
        raw, factor = requests[(symbol, "raw")], requests[(symbol, "qfq-factor")]
        audit = {"symbol": symbol, "rawStatus": raw["status"], "factorStatus": factor["status"],
                 "identityStatus": stock["identity_status"], "includedPriceRows": 0}
        if raw["status"] != "ok" or factor["status"] != "ok":
            audit["dataStatus"] = "source_unavailable"; audits.append(audit); continue
        observed = pd.read_csv(source/raw["file"], float_precision="round_trip")
        events = pd.read_csv(source/factor["file"], float_precision="round_trip")
        required = {"date", "open", "high", "low", "close", "volume", "amount"}
        if not required.issubset(observed.columns) or set(events.columns) != {"date", "qfq_factor"}:
            audit["dataStatus"] = "required_source_fields_unavailable"; audits.append(audit); continue
        observed["date"] = pd.to_datetime(observed.date, format="%Y-%m-%d", errors="raise").dt.as_unit("ns")
        events["date"] = pd.to_datetime(events.date, format="%Y-%m-%d", errors="raise").dt.as_unit("ns")
        if observed.date.duplicated().any() or events.date.duplicated().any():
            audit["dataStatus"] = "duplicate_source_dates"; audits.append(audit); continue
        if (not np.isfinite(events.qfq_factor.to_numpy(float)).all() or not events.qfq_factor.gt(0).all()):
            audit["dataStatus"] = "nonpositive_or_nonfinite_adjustment_table"; audits.append(audit); continue
        observed = observed.sort_values("date", ignore_index=True)
        in_period = observed.date.between("2010-01-01", "2026-08-31")
        audit["outsideResearchPeriodRows"] = int((~in_period).sum())
        audit["nonSessionDates"] = observed.loc[in_period & ~observed.date.isin(calendar), "date"].dt.strftime("%Y-%m-%d").tolist()
        observed = observed.loc[in_period & observed.date.isin(calendar)].copy()
        events = events.sort_values("date", ignore_index=True).rename(columns={"date": "adjustment_event_date", "qfq_factor": "adj_factor"})
        # Match a documented effective factor regime, never fill missing market prices.
        observed = pd.merge_asof(observed, events, left_on="date", right_on="adjustment_event_date", direction="backward")
        if (observed.adjustment_event_date > observed.date).any():
            raise ValueError(f"{symbol} 调整事件使用了未来生效日。")
        observed["symbol"] = symbol
        observed = observed.rename(columns={"amount": "turnover"})
        audit.update(dataStatus="loaded_with_observation_level_validation", includedPriceRows=len(observed),
                     missingAdjustmentRows=int(observed.adj_factor.isna().sum()),
                     factorEventRows=len(events),
                     beforeListingRows=int(observed.date.lt(stock["list_date"]).sum()),
                     atOrAfterDelistingRows=int(observed.date.ge(stock["delist_date"]).sum()) if pd.notna(stock["delist_date"]) else 0)
        if audit["beforeListingRows"] or audit["atOrAfterDelistingRows"]:
            selected = hk_stocks.symbol.eq(symbol)
            hk_stocks.loc[selected, "identity_status"] = "unknown"
            hk_stocks.loc[selected, "identity_reasons"] = hk_stocks.loc[selected, "identity_reasons"] + "|source_price_outside_supplied_identity_lifecycle"
            audit["identityStatus"] = "unknown"
            audit["identityPriceConflict"] = True
        frames.append(observed); audits.append(audit)
    if not frames:
        raise ValueError("港股无可载入真实行情。")
    hk = {"calendar": calendar, "stocks": hk_stocks, "prices": pd.concat(frames, ignore_index=True),
          "benchmark": benchmark, "suspensions": pd.DataFrame(columns=["symbol", "date", "suspend_type", "suspend_timing"]),
          "provenance": {"source": "Sina raw and separate qfq factor via AKShare 1.18.94 definitions",
                         "suspensionRecordsAvailable": False, "calendarVerified": True, "stocks": audits}}
    a["stocks"] = a_stocks
    for market, ds in (("A", a), ("HK", hk)):
        ds["stocks"]["market"] = market
        ds["training_symbols"] = ds["stocks"].loc[ds["stocks"].split.eq("train"), "symbol"].tolist()
        ds["holdout_symbols"] = ds["stocks"].loc[ds["stocks"].split.eq("heldout"), "symbol"].tolist()
    for name, expected in a["provenance"]["fileHashes"].items():
        namespace, relative = name.split("/", 1)
        hashes[str(Path(a["provenance"]["sourceRoots"][namespace])/relative)] = expected
    identity_root = ROOT/"data/raw/cross-market-identity"
    for path in identity_root.rglob("*"):
        if path.is_file(): hashes[str(path)] = digest(path)
    groups = identity_frame.groupby("issuer_id").split.nunique()
    if groups.gt(1).any():
        raise ValueError("已知同公司身份跨越训练和留出。")
    return {"markets": {"A": a, "HK": hk}, "provenance": {
        "inputFileHashes": hashes, "identity": identity["provenance"],
        "candidateCounts": {"A": len(a_stocks), "HK": len(hk_stocks)},
        "identityAdmissionCounts": {m: {str(k): int(v) for k, v in ds["stocks"].identity_status.value_counts().items()} for m, ds in (("A", a), ("HK", hk))},
        "hkSourceAudit": audits,
        "hkAdjustmentDiagnostic": source_audit["sameSourceAdjustmentDiagnostic"],
        "limitations": ["Historical results are exploratory: the market periods were previously examined.",
                        "Known issuer links are grouped, but complete historical issuer isolation is not established.",
                        "Historical HK identity and delisted coverage remain incomplete; exclusions are retained in coverage.",
                        "Current-vintage supplier adjustments are not proof of the data version available historically.",
                        "Same-source qfq/HFQ corporate-action consistency diagnostics contain unresolved differences; the supplied qfq is used as a research-price convention, not certified economic return.",
                        "HK has no independently complete suspension-event feed in this dataset.",
                        "Adjusted-price direction is not executable net investment return; no transaction-cost strategy is validated.",
                        "No prospective model-issuance outcomes yet exist for this frozen cross-market model."],
        "sourcePolicy": "A=Tushare frozen dataset; HK=Sina raw*separate qfq factor, no source substitution, missing-price fill, or favourable-stock replacement",
    }}
