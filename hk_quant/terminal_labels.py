import json
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from pypdf import PdfReader


PATCH_KEYS = ["date", "security_id", "horizon", "label_end"]
GAP_REASON = "listing_exit_requires_verified_terminal_or_transfer_outcome"
SUPPORTED_HORIZONS = (1, 5, 20, 60)
SUPPORTED_EVENT_TYPES = ("privatisation_cash", "merger_cash", "compulsory_acquisition_cash")


def _source_file(value):
    path = Path(value).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"源文件缺失或为空: {path}")
    return path


def _compact(value):
    return re.sub(r"\s+", "", str(value)).lower()


def _date_literal(value):
    stamp = pd.Timestamp(value)
    return f"{stamp.day} {stamp:%B %Y}"


def _date_literals(value):
    stamp = pd.Timestamp(value)
    day = stamp.day
    suffix = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return (
        _date_literal(stamp),
        f"{day}{suffix} {stamp:%B %Y}",
        f"{day} {stamp:%B}, {stamp:%Y}",
        f"{day}{suffix} {stamp:%B}, {stamp:%Y}",
        f"{stamp:%B} {day}, {stamp:%Y}",
        f"{stamp:%B} {day}{suffix}, {stamp:%Y}",
    )


def _finite_positive(value):
    return pd.notna(value) and np.isfinite(float(value)) and float(value) > 0


def _event_dates(event):
    effective = pd.Timestamp(event["effective_date"])
    withdrawal = pd.Timestamp(event.get("listing_withdrawal_date", event["effective_date"]))
    settlement = pd.Timestamp(event["settlement_date"])
    if pd.isna(effective) or pd.isna(withdrawal) or pd.isna(settlement):
        raise ValueError("终止事件日期无效")
    if withdrawal < effective:
        raise ValueError("撤牌日期早于终止生效日期")
    return effective, withdrawal, settlement


def _validate_source(event, role):
    url = event[f"{role}_url"]
    document = _source_file(event[f"{role}_document"])
    catalog_file = _source_file(event[f"{role}_catalog"])
    stamp = pd.Timestamp(event[f"{role}_published_at"])
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError(f"{role}来源URL无效")
    if Path(urlparse(url).path).name != document.name:
        raise ValueError(f"{role}文件名与URL不匹配")
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{role}公告时间必须包含时区")
    catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    matches = [row for row in catalog.get("records", []) if row.get("pdf_url") == url]
    if len(matches) != 1 or pd.Timestamp(matches[0].get("published_at")) != stamp:
        raise ValueError(f"{role}公告时间与目录不匹配")
    page_number = event[f"{role}_evidence_page"]
    if int(page_number) != page_number or int(page_number) < 1:
        raise ValueError(f"{role}证据页码无效")
    pages = PdfReader(document).pages
    if int(page_number) > len(pages):
        raise ValueError(f"{role}证据页码超出文件范围")
    excerpt = event[f"{role}_evidence_excerpt"]
    if not isinstance(excerpt, str) or not excerpt.strip():
        raise ValueError(f"{role}缺少页码证据原文")
    actual = _compact(pages[int(page_number) - 1].extract_text() or "")
    if _compact(excerpt) not in actual:
        raise ValueError(f"{role}页码证据与原文不符")
    return actual


def _validate_event(event, master, root):
    if event.get("event_type") not in SUPPORTED_EVENT_TYPES or event.get("verified") is not True:
        raise ValueError("终止事件未经核验")
    sid = event["security_id"]
    if sid not in set(master.security_id):
        raise ValueError("终止事件证券不在主表")
    identity = master.loc[master.security_id.eq(sid)]
    if len(identity) != 1 or identity.iloc[0].identity_status != "verified":
        raise ValueError("终止事件证券身份不唯一或未核验")
    identity = identity.iloc[0]
    for field in ("effective_date", "last_trading_date", "settlement_date"):
        if pd.isna(pd.Timestamp(event[field])):
            raise ValueError(f"{field}无效")
    effective, withdrawal, settlement = _event_dates(event)
    last_trade = pd.Timestamp(event["last_trading_date"])
    if not last_trade < effective <= settlement:
        raise ValueError("终止、最后交易和派付日期顺序不成立")
    if settlement > pd.Timestamp("2023-12-31"):
        raise ValueError("派付日期不属于开发期成熟范围")
    if event.get("cash_currency") != "HKD" or not _finite_positive(event.get("cash_per_share")):
        raise ValueError("终止现金金额或币种无效")
    if pd.Timestamp(identity.identity_valid_from) != pd.Timestamp(event["listing_start"]):
        raise ValueError("身份开始日期不匹配")
    if pd.Timestamp(identity.identity_valid_to) != withdrawal:
        raise ValueError("身份结束日期不匹配")
    identity_file = _source_file(event["source_identity_file"])
    listings = pd.read_parquet(identity_file)
    for field in ("FirstTradeDate", "FinalTradeDate", "DelistDate"):
        listings[field] = pd.to_datetime(listings[field])
    code = str(event["source_stock_code"]).zfill(5)
    matches = listings.loc[
        listings.StockCode.astype(str).str.zfill(5).eq(code)
        & listings.FirstTradeDate.eq(pd.Timestamp(event["listing_start"]))
        & listings.DelistDate.eq(withdrawal)
    ]
    if len(matches) != 1:
        raise ValueError("归档证券上市期间不唯一")
    listing = matches.iloc[0]
    if int(listing.ID) != int(event["source_listing_id"]):
        raise ValueError("归档上市记录ID不匹配")
    if int(listing.IssueID) != int(event["source_issue_id"]):
        raise ValueError("归档IssueID不匹配")
    if pd.Timestamp(listing.FinalTradeDate) != last_trade:
        raise ValueError("最后交易日与归档记录不匹配")
    _validate_source(event, "composite")
    composite_document = _source_file(event["composite_document"])
    composite_full_text = _compact(" ".join(
        page.extract_text() or "" for page in PdfReader(composite_document).pages
    ))
    _validate_source(event, "implementation")
    implementation_document = _source_file(event["implementation_document"])
    implementation_full_text = _compact(" ".join(
        page.extract_text() or "" for page in PdfReader(implementation_document).pages
    ))
    event_type = event.get("event_type")
    if event_type == "compulsory_acquisition_cash":
        terms_token = (
            "compulsoryacquisition" in composite_full_text
            and "offerprice" in composite_full_text
            and "cash" in composite_full_text
        )
    else:
        terms_token = (
            "cancellationprice" in composite_full_text
            or "schemeconsideration" in composite_full_text
            or "schemeshareconsideration" in composite_full_text
            or "cancellationconsideration" in composite_full_text
        ) and "cash" in composite_full_text
    if not terms_token:
        raise ValueError("合并文件没有现金取消价证据")
    amount_matches = re.findall(r"hk\$([0-9]+(?:\.[0-9]+)?)", composite_full_text)
    if not any(np.isclose(float(amount), float(event["cash_per_share"]), rtol=1e-9, atol=1e-9)
               for amount in amount_matches):
        raise ValueError("合并文件金额与事件金额不匹配")
    if event_type == "compulsory_acquisition_cash":
        effective_token = (
            "completedthecompulsoryacquisition" in implementation_full_text
            or "completionofthecompulsoryacquisition" in implementation_full_text
            or ("compulsoryacquisition" in implementation_full_text
                and ("completion" in implementation_full_text or "completed" in implementation_full_text))
        )
    else:
        effective_token = (
            "mergerhasbecomeeffective" in implementation_full_text
            or "schemebecameeffective" in implementation_full_text
            or "schemehasbecomeeffective" in implementation_full_text
            or "becameeffective" in implementation_full_text
            or ("implementationofthemerger" in implementation_full_text
                and "hasbecomeunconditional" in implementation_full_text
                and "willbeimplemented" in implementation_full_text)
        )
    withdrawn_token = "withdrawn" in implementation_full_text or "withdrawal" in implementation_full_text
    payment_token = (
        "cheques" in implementation_full_text
        or ("payment" in implementation_full_text
            and ("despatch" in implementation_full_text or "despatched" in implementation_full_text))
    )
    if not effective_token or not withdrawn_token or "listing" not in implementation_full_text or not payment_token:
        raise ValueError("最终实施公告缺少生效、撤牌或派付证据")
    if not any(_compact(literal) in implementation_full_text for literal in _date_literals(settlement)):
        raise ValueError("最终实施公告缺少最终派付日期证据")
    return listing


def _load_signal_bars(root, security_id, signal_start, terminal_date):
    """读取信号日至最终交易日覆盖的全部年份，支持跨年预测期限。"""
    start = pd.Timestamp(signal_start)
    end = pd.Timestamp(terminal_date)
    years = range(min(start, end).year, max(start, end).year + 1)
    frames = []
    for year in years:
        path = _source_file(Path(root) / "bars" / f"{year}.parquet")
        frame = pd.read_parquet(path, filters=[("security_id", "==", security_id)])
        frame["date"] = pd.to_datetime(frame["date"])
        frames.append(frame)
    if not frames:
        raise ValueError("终止事件缺少信号行情年份")
    bars = pd.concat(frames, ignore_index=True)
    if bars.duplicated(["security_id", "date"]).any():
        raise ValueError("信号行情日期重复")
    return bars


def build_terminal_cash_label_patches(data_root, events_path, workqueue_path, cutoff="2023-12-31"):
    """只为已有缺失标签生成终止现金补丁，绝不覆盖源行情或已有标签。"""
    root = Path(data_root).resolve()
    events_path, workqueue_path = _source_file(events_path), _source_file(workqueue_path)
    master = pd.read_parquet(_source_file(root / "securities.parquet"))
    events = pd.read_parquet(events_path)
    queue = pd.read_parquet(workqueue_path)
    queue["date"] = pd.to_datetime(queue.date)
    queue["label_end"] = pd.to_datetime(queue.label_end)
    if events.event_id.duplicated().any() or events.security_id.duplicated().any():
        raise ValueError("终止事件重复")
    if queue.duplicated(PATCH_KEYS).any():
        raise ValueError("标签缺口键重复")
    patches, exclusions = [], []
    for event in events.to_dict("records"):
        listing = _validate_event(event, master, root)
        sid = event["security_id"]
        effective, withdrawal, settlement = _event_dates(event)
        last_trade = pd.Timestamp(event["last_trading_date"])
        start = pd.Timestamp(event["listing_start"])
        subset = queue.loc[queue.security_id.eq(sid) & queue.gap_reason.eq(GAP_REASON)]
        features_path = _source_file(root / "features" / f"{sid}.parquet")
        features = pd.read_parquet(features_path)
        features["date"] = pd.to_datetime(features.date)
        signal_start = pd.to_datetime(subset.date).min() if not subset.empty else last_trade
        bars = _load_signal_bars(root, sid, signal_start, last_trade)
        terminal_rows = bars.loc[bars.date.eq(last_trade)]
        if len(terminal_rows) != 1:
            raise ValueError("最后交易日缺少唯一行情")
        terminal = terminal_rows.iloc[0]
        if not _finite_positive(terminal.cum_adjfactor) or terminal.currency != "HKD":
            raise ValueError("最后交易日调整因子或币种无效")
        terminal_value = float(event["cash_per_share"]) * float(terminal.cum_adjfactor)
        for gap in subset.to_dict("records"):
            day, end, horizon = pd.Timestamp(gap["date"]), pd.Timestamp(gap["label_end"]), int(gap["horizon"])

            def exclude(reason):
                exclusions.append(dict(**{key: gap[key] for key in PATCH_KEYS}, event_id=event["event_id"], reason=reason))

            if horizon not in SUPPORTED_HORIZONS or pd.isna(end) or end > pd.Timestamp(cutoff):
                exclude("outside_development_or_horizon")
                continue
            if day < start or day > last_trade or end < effective:
                exclude("terminal_event_not_the_label_endpoint")
                continue
            feature = features.loc[features.security_id.eq(sid) & features.date.eq(day)]
            if len(feature) != 1:
                exclude("source_feature_absent")
                continue
            feature = feature.iloc[0]
            label, label_end = f"fwd_return_{horizon}", f"label_end_{horizon}"
            if pd.notna(feature[label]):
                exclude("existing_label_present")
                continue
            if pd.Timestamp(feature[label_end]) != end:
                exclude("source_label_end_mismatch")
                continue
            signal = bars.loc[bars.date.eq(day)]
            if len(signal) != 1:
                exclude("source_signal_absent")
                continue
            signal = signal.iloc[0]
            if (signal.data_valid != True or signal.currency != "HKD"
                    or not _finite_positive(signal.raw_close)
                    or not _finite_positive(signal.cum_adjfactor)
                    or not _finite_positive(feature.adj_close_hkd)
                    or not np.isclose(float(feature.adj_close_hkd), float(signal.raw_close) * float(signal.cum_adjfactor), rtol=1e-7, atol=1e-10)):
                exclude("source_signal_adjustment_unverified")
                continue
            value = terminal_value / float(feature.adj_close_hkd) - 1.0
            if not np.isfinite(value) or value < -1:
                exclude("terminal_return_invalid")
                continue
            patches.append(dict(
                date=day, security_id=sid, horizon=horizon, label_end=end,
                old_fwd_return=np.nan, new_fwd_return=float(value),
                reason="verified_terminal_cash_settlement", source_url=event["implementation_url"],
                event_id=event["event_id"], event_type=event["event_type"],
                source_issue_id=int(event["source_issue_id"]), source_listing_id=int(event["source_listing_id"]),
                last_trading_date=last_trade, effective_date=effective,
                listing_withdrawal_date=withdrawal, settlement_date=settlement,
                cash_per_share=float(event["cash_per_share"]), cash_currency=event["cash_currency"],
                terminal_adjustment_factor=float(terminal.cum_adjfactor), terminal_value_adjusted_hkd=terminal_value,
                source_signal_adj_close_hkd=float(feature.adj_close_hkd),
                source_signal_raw_close=float(signal.raw_close), source_signal_factor=float(signal.cum_adjfactor),
                payment_basis="issuer_final_schedule_simulated",
                composite_url=event["composite_url"], composite_document=event["composite_document"],
                composite_published_at=event["composite_published_at"], composite_evidence_page=int(event["composite_evidence_page"]),
                implementation_url=event["implementation_url"], implementation_document=event["implementation_document"],
                implementation_published_at=event["implementation_published_at"], implementation_evidence_page=int(event["implementation_evidence_page"]),
                source_identity_file=event["source_identity_file"], source_features_file=str(features_path),
                source_signal_bars_file=str(root / "bars"), source_workqueue_file=str(workqueue_path),
            ))
    result = pd.DataFrame(patches)
    if not result.empty and result.duplicated(PATCH_KEYS).any():
        raise ValueError("生成的终止补丁重复")
    audit = {
        "status": "verified_terminal_cash_patches",
        "events": int(len(events)),
        "examined_rows": int(sum((queue.security_id.eq(e.security_id) & queue.gap_reason.eq(GAP_REASON)).sum() for e in events.itertuples())),
        "patches": int(len(result)), "excluded_rows": int(len(exclusions)), "excluded": exclusions,
        "source_data_modified": False, "patches_applied": False,
        "payment_basis": "issuer_final_schedule_simulated",
        "return_calculation": "terminal cash per share times final-trade adjustment factor divided by existing signal adjusted HKD price minus one",
        "note": "终止现金金额和最终派付日期均来自官方公告；实际到账不由公告证明，回测按用户批准的最终公告派付日模拟。",
    }
    return result, audit
