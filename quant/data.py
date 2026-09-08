from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import exchange_calendars as exchange
import numpy as np
import pandas as pd


PROVIDER = "Qveris / FMP"
CALENDAR_VERSION = "4.13.2"
DATA_START = "2010-01-01"
DATA_END = "2026-08-31"
OFFICIAL_CLOSURES = {
    "2023-09-01": "https://www.hkex.com.hk/News/Market-Communications/2023/2309012news?sc_lang=en",
    "2023-09-08": "https://www.hkex.com.hk/News/Market-Communications/2023/2309083news?sc_lang=en",
}
SOURCE_URLS = [
    {"title": "FMP 官方行情接口与字段说明", "url": "https://site.financialmodelingprep.com/developer/docs/stable"},
    {"title": "港交所交易日历", "url": "https://www.hkex.com.hk/Services/Trading/Securities/Overview/Trading-Calendar-and-Holiday-Schedule?sc_lang=en"},
    {"title": "2023 年 9 月 1 日台风全日休市", "url": OFFICIAL_CLOSURES["2023-09-01"]},
    {"title": "2023 年 9 月 8 日黑雨全日休市", "url": OFFICIAL_CLOSURES["2023-09-08"]},
    {"title": "XHKG 独立交易日历源代码", "url": "https://github.com/gerrymanoim/exchange_calendars/blob/v4.13.2/exchange_calendars/exchange_calendar_xhkg.py"},
    {"title": "香港股票印花税历史税率", "url": "https://www.ird.gov.hk/eng/pdf/sd_stock_rates.pdf"},
    {"title": "港交所交易费用", "url": "https://www.hkex.com.hk/Services/Rules-and-Forms-and-Fees/Fees/Securities-%28Hong-Kong%29/Trading/Transaction?sc_lang=en"},
    {"title": "2025 年股份交收费调整", "url": "https://www.hkex.com.hk/-/media/HKEX-Market/Services/Circulars-and-Notices/Participant-and-Members-Circulars/HKSCC/2025/ce_HKSCC_SET_022_2025.pdf"},
    {"title": "ETF 印花税豁免", "url": "https://www.ird.gov.hk/eng/faq/ETFs.htm"},
    {"title": "药明康德 2020 年报：2019 和 2020 转增股本与分红", "url": "https://officialsite-static.wuxiapptec.com/upload/d8/20210519/2021-04-20-2020%20Annual%20Report.pdf"},
    {"title": "药明康德 2021 年股东通函与除权安排", "url": "https://static.wuxiapptec.com/c3/20210519/c3437c5d038c1d98.pdf"},
    {"title": "回测过拟合研究", "url": "https://carmamaths.org/resources/jon/backtest2.pdf"},
]


def research_calendar() -> pd.DatetimeIndex:
    if exchange.__version__ != CALENDAR_VERSION:
        raise ValueError(f"本轮封存研究要求 exchange-calendars {CALENDAR_VERSION}，当前为 {exchange.__version__}。")
    sessions = exchange.get_calendar("XHKG", start=DATA_START, end=DATA_END).sessions
    return sessions.difference(pd.DatetimeIndex(OFFICIAL_CLOSURES))


def complete_month_ends(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if calendar.empty:
        raise ValueError("不能从空交易日历生成月末信号。")
    last_month_end = calendar[-1] + pd.offsets.MonthEnd(0)
    full_sessions = exchange.get_calendar("XHKG", start=DATA_START, end=last_month_end).sessions
    full_sessions = full_sessions.difference(pd.DatetimeIndex(OFFICIAL_CLOSURES))
    official = pd.Series(full_sessions, index=full_sessions).groupby(full_sessions.to_period("M")).max()
    return pd.DatetimeIndex(official[(official >= calendar[0]) & (official <= calendar[-1])])


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _calendar_digest(calendar: pd.DatetimeIndex) -> str:
    return _digest("\n".join(calendar.strftime("%Y-%m-%d")).encode())


def _snapshot_digest(prices_hash: str, universe_hash: str, review_hash: str, calendar_hash: str) -> str:
    return _digest("\n".join((prices_hash, universe_hash, review_hash, calendar_hash)).encode())


def _read_review(data_dir: Path) -> tuple[dict, str]:
    raw = (data_dir / "review.json").read_bytes()
    review = json.loads(raw)
    if type(review["approved"]) is not bool or review["provider"] != PROVIDER:
        raise ValueError("数据审查必须明确给出布尔 approved，并对应 Qveris / FMP 来源。")
    return review, _digest(raw)


def _validate_universe(universe: dict) -> list[dict]:
    securities = universe["stocks"] + [universe["benchmark"]]
    symbols = [item["symbol"] for item in securities]
    if len(symbols) != len(set(symbols)):
        raise ValueError("股票池及基准包含重复证券代码。")
    for item in securities:
        if not isinstance(item["symbol"], str) or re.fullmatch(r"\d{5}", item["symbol"]) is None:
            raise ValueError("港股证券代码必须为五位数字字符串。")
        for field in ("name", "sector", "industry", "issuerType"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f"{item['symbol']} 的 {field} 必须是非空字符串。")
        if not isinstance(item["tags"], list) or any(not isinstance(tag, str) or not tag.strip() for tag in item["tags"]):
            raise ValueError(f"{item['symbol']} 的 tags 必须为字符串列表。")
        exempt = item["stampDutyExemptFrom"]
        if exempt is not None:
            if not isinstance(exempt, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", exempt) is None:
                raise ValueError(f"{item['symbol']} 的印花税豁免日期格式错误。")
            pd.to_datetime(exempt, format="%Y-%m-%d", errors="raise")
    for action in universe["corporateActions"]:
        if action["symbol"] not in symbols:
            raise ValueError("公司行动指向股票池以外的证券。")
    return securities


def _read_series(source: Path, symbol: str, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    raw = source.read_bytes()
    envelope = json.loads(raw)
    if envelope["success"] is not True or envelope["result"]["status_code"] != 200:
        raise ValueError(f"{symbol} 的 {source.parent.name} 不是成功的 Qveris 行情响应。")
    records = envelope["result"]["data"]
    if not isinstance(records, list) or len(records) < 300:
        raise ValueError(f"{symbol} 的 {source.parent.name} 历史行情不足 300 条。")
    frame = pd.DataFrame(records)
    columns = ["symbol", "date", "adjOpen", "adjHigh", "adjLow", "adjClose", "volume"]
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"{symbol} 的 {source.parent.name} 缺少行情字段。")
    frame = frame[columns].copy()
    if frame.isna().any().any():
        raise ValueError(f"{symbol} 的 {source.parent.name} 行情包含空值。")
    if set(frame.symbol) != {f"{symbol[1:]}.HK"}:
        raise ValueError(f"{symbol} 的 {source.parent.name} 响应证券代码不一致。")
    frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
    if frame.date.duplicated().any():
        raise ValueError(f"{symbol} 的 {source.parent.name} 存在重复日期。")
    if ((frame.date < pd.Timestamp(DATA_START)) | (frame.date > pd.Timestamp(DATA_END))).any():
        raise ValueError(f"{symbol} 的 {source.parent.name} 超出本轮封存日期范围。")
    for column in columns[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy()).all():
            raise ValueError(f"{symbol} 的 {source.parent.name}/{column} 包含非有限数字。")
    price_columns = ["adjOpen", "adjHigh", "adjLow", "adjClose"]
    if (frame[price_columns] <= 0).any().any() or (frame.volume < 0).any():
        raise ValueError(f"{symbol} 的 {source.parent.name} 包含非正价格或负成交量。")
    invalid = (
        (frame.adjHigh < frame[["adjOpen", "adjClose", "adjLow"]].max(axis=1) - 1e-8)
        | (frame.adjLow > frame[["adjOpen", "adjClose", "adjHigh"]].min(axis=1) + 1e-8)
    )
    if invalid.any():
        invalid_dates = frame.loc[invalid, "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{symbol} 的 {source.parent.name} 开高低收矛盾：{invalid_dates[:10]}")
    frame = frame.sort_values("date").reset_index(drop=True)
    non_sessions = frame.loc[~frame.date.isin(calendar), "date"].dt.strftime("%Y-%m-%d").tolist()
    frame = frame.loc[frame.date.isin(calendar)].copy()
    if len(frame) < 300:
        raise ValueError(f"{symbol} 的 {source.parent.name} 排除休市日后不足 300 条行情。")
    provenance = {
        "executionId": envelope["execution_id"],
        "sha256": _digest(raw),
        "sourceRows": len(records),
        "excludedNonSessions": len(non_sessions),
        "excludedNonSessionDates": non_sessions,
    }
    return frame, provenance


def prepare_dataset(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    review, review_hash = _read_review(data_dir)
    universe_raw = (data_dir / "universe.json").read_bytes()
    universe = json.loads(universe_raw)
    securities = _validate_universe(universe)
    calendar = research_calendar()
    frames: list[pd.DataFrame] = []
    audit: list[dict] = []
    for security in securities:
        symbol = security["symbol"]
        raw_frame, raw_info = _read_series(data_dir / "raw" / "fmp-raw" / f"{symbol}.json", symbol, calendar)
        adjusted_frame, adjusted_info = _read_series(data_dir / "raw" / "fmp-adjusted" / f"{symbol}.json", symbol, calendar)
        raw_dates = pd.DatetimeIndex(raw_frame.date)
        adjusted_dates = pd.DatetimeIndex(adjusted_frame.date)
        if not raw_dates.equals(adjusted_dates):
            only_raw = raw_dates.difference(adjusted_dates).strftime("%Y-%m-%d").tolist()
            only_adjusted = adjusted_dates.difference(raw_dates).strftime("%Y-%m-%d").tolist()
            raise ValueError(f"{symbol} 原始与复权交易日不能一一对应；仅原始：{only_raw[:15]}；仅复权：{only_adjusted[:15]}。")
        frame = raw_frame.rename(columns={"adjOpen": "open", "adjHigh": "high", "adjLow": "low", "adjClose": "close"})
        frame["symbol"] = symbol
        # Sorted session dates must be identical; neither inner joins nor
        # inferred prices are permitted to conceal missing observations.
        frame["adjusted_open"] = adjusted_frame.adjOpen.to_numpy()
        frame["adjusted_close"] = adjusted_frame.adjClose.to_numpy()
        frame["turnover"] = frame.close * frame.volume
        frame = frame[["date", "symbol", "open", "high", "low", "close", "adjusted_open", "adjusted_close", "volume", "turnover"]]
        if not np.isfinite(frame.turnover.to_numpy()).all():
            raise ValueError(f"{symbol} 的估算成交额非有限。")
        expected = calendar[(calendar >= raw_dates[0]) & (calendar <= raw_dates[-1])]
        missing = expected.difference(raw_dates)
        if symbol == universe["benchmark"]["symbol"]:
            missing_benchmark = calendar.difference(raw_dates)
            if not missing_benchmark.empty:
                raise ValueError(f"基准 {symbol} 缺少 {len(missing_benchmark)} 个 XHKG 正式交易日：{missing_benchmark.strftime('%Y-%m-%d').tolist()[:30]}。不能据此改变日历。")
        returns = frame.adjusted_close.pct_change(fill_method=None)
        jumps = frame.loc[returns.abs() > 0.35, ["date", "adjusted_close"]]
        excluded_dates = sorted(set(raw_info["excludedNonSessionDates"]) | set(adjusted_info["excludedNonSessionDates"]))
        audit.append({
            **security,
            "rows": len(frame),
            "firstDate": raw_dates[0].date().isoformat(),
            "lastDate": raw_dates[-1].date().isoformat(),
            "zeroVolumeRows": int((frame.volume == 0).sum()),
            "zeroVolumeDates": frame.loc[frame.volume == 0, "date"].dt.strftime("%Y-%m-%d").tolist(),
            "missingSessions": len(missing),
            "missingSessionDates": missing.strftime("%Y-%m-%d").tolist(),
            "coverage": float(len(raw_dates) / len(expected)),
            "excludedNonSessions": len(excluded_dates),
            "excludedNonSessionDates": excluded_dates,
            "largeMoves": [{"date": row.date.date().isoformat(), "return": float(returns.loc[index])} for index, row in jumps.iterrows()],
            "raw": raw_info,
            "adjusted": adjusted_info,
        })
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"])
    if combined.duplicated(["date", "symbol"]).any() or combined.isna().any().any():
        raise ValueError("合并行情出现重复键或空值。")
    csv_data = combined.to_csv(index=False, date_format="%Y-%m-%d", float_format="%.17g")
    prices_hash = _digest(csv_data.encode())
    universe_hash = _digest(universe_raw)
    calendar_hash = _calendar_digest(calendar)
    snapshot_hash = _snapshot_digest(prices_hash, universe_hash, review_hash, calendar_hash)
    manifest = {
        "provider": PROVIDER,
        "currency": "HKD",
        "frequency": "日线",
        "adjustment": "FMP 实际历史未复权 OHLC 与含拆股、股息的全复权 OHLC；复权研究单位记账",
        "researchReady": review["approved"],
        "qualityStatus": "已通过格式校验和显式来源审查" if review["approved"] else "等待显式来源审查批准",
        "preparedAt": datetime.now(timezone.utc).isoformat(),
        "stockCount": len(universe["stocks"]),
        "securityCount": len(securities),
        "rowCount": len(combined),
        "tradingSessions": len(calendar),
        "startDate": combined.date.min().date().isoformat(),
        "endDate": combined.date.max().date().isoformat(),
        "sha256": snapshot_hash,
        "pricesSha256": prices_hash,
        "universeSha256": universe_hash,
        "reviewSha256": review_hash,
        "calendarSha256": calendar_hash,
        "calendar": {"name": "XHKG", "package": "exchange-calendars", "version": CALENDAR_VERSION, "start": DATA_START, "end": DATA_END, "sessions": len(calendar), "officialClosures": OFFICIAL_CLOSURES},
        "universe": universe["name"],
        "selectionDate": universe["selectionDate"],
        "selectionMethod": universe["selectionMethod"],
        "survivorshipBias": True,
        "pointInTimeFundamentals": False,
        "securities": audit,
        "sources": SOURCE_URLS,
        "corporateActions": universe["corporateActions"],
        "review": review,
        "excludedNonSessions": sum(item["excludedNonSessions"] for item in audit),
        "notes": [
            "正式行情唯一来源为 Qveris / FMP；分别保留实际未复权和全复权两个接口的原始响应、调用编号与内容指纹。",
            "以独立 XHKG 日历识别正式交易日；来源中的休市日记录仅按日历剔除，逐证券保留被剔除日期。基准缺少正式交易日时直接停止准备。",
            "原始与复权价格在正式交易日必须严格一一对应，缺失或矛盾直接报错；没有填补行情，也不根据一个价格推算另一个价格。",
            "复权开盘价直接读取全复权接口；未复权成交量采用当时实际股数，估算成交额为同口径原始收盘价乘成交量，仍不是逐笔成交额。",
            "零成交量及缺失记录均不可成交；仅靠这两类日线信息不能武断认定全部属于交易所停牌。",
            "本轮仅研究用户指定四只股票；依据当前持仓回看历史存在事后选样偏差，不能推断全港股表现。",
            "普通股票按历史日期收取印花税；盈富基金自 2015-02-13 起免股票印花税。",
            "复权包含派息影响，按复权单位研究股息再投资，不重复派息；未模拟每个账户实际股息到账时间、整手股数及历史盘口。",
            "显式数据审查批准仅允许本轮研究继续，并不证明样本无偏、未来收益可靠或模型达到实盘要求。",
        ],
    }
    serialized_manifest = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    (data_dir / "prices.csv").write_text(csv_data, encoding="utf-8")
    (data_dir / "manifest.json").write_text(serialized_manifest, encoding="utf-8")
    return manifest


def verify_snapshot(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    review, review_hash = _read_review(data_dir)
    if manifest["provider"] != PROVIDER or manifest["researchReady"] is not True or review["approved"] is not True:
        raise ValueError("Qveris / FMP 行情尚未获得明确审查批准，不能进行正式研究与验收。")
    prices_hash = _digest((data_dir / "prices.csv").read_bytes())
    universe_raw = (data_dir / "universe.json").read_bytes()
    universe_hash = _digest(universe_raw)
    _validate_universe(json.loads(universe_raw))
    calendar_hash = _calendar_digest(research_calendar())
    current_hashes = {"pricesSha256": prices_hash, "universeSha256": universe_hash, "reviewSha256": review_hash, "calendarSha256": calendar_hash}
    for key, actual in current_hashes.items():
        if manifest[key] != actual:
            raise ValueError(f"封存内容 {key} 已改变，必须重新进行数据准备和审查。")
    if manifest["sha256"] != _snapshot_digest(prices_hash, universe_hash, review_hash, calendar_hash):
        raise ValueError("研究快照综合指纹不一致。")
    return manifest
