"""HKEXnews 公告目录采集、证据导入及每手单位历史链校验。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html import unescape
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .paths import DATA


SEARCH_URL = "https://www1.hkexnews.hk/search/titlesearch.xhtml?lang=en"
AJAX_URL = "https://www1.hkexnews.hk/search/titleSearchServlet.do"
HK_TZ = timezone(timedelta(hours=8))
CATEGORIES = {
    "all": ("", "", ""),
    "board_lot": ("10000", "8", "18160"),
    "dividend_form": ("10000", "3", "13251"),
    "dividend": ("10000", "3", "13250"),
    "payment_change": ("10000", "8", "18200"),
    "privatisation": ("10000", "7", "17600"),
}
BOARD_LOT_COLUMNS = (
    "security_id", "announced_at", "effective_date", "old_lot", "new_lot",
    "verified", "source_url",
)
ACTION_COLUMNS = (
    "security_id", "announced_at", "effective_date", "payment_date",
    "cash_currency", "cash_per_share", "action_type", "share_multiplier",
    "verified", "payment_basis", "source_url",
)


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value)).strip()


def _stock_codes(value: str, *, allow_unassigned: bool = False) -> list[str]:
    text = _clean_text(re.sub(r'<[^>]*>', ' ', unescape(value)))
    tokens = text.split()
    if not tokens and allow_unassigned:return []
    if not tokens or any(not re.fullmatch(r'\d{1,5}', token) for token in tokens):
        raise ValueError('公告索引证券代码无法明确拆分: '+text)
    return list(dict.fromkeys(token.zfill(5) for token in tokens))


def _select_security(records: list[dict[str, Any]], security: str) -> list[dict[str, Any]]:
    codes = _stock_codes(security.split('.')[0])
    if len(codes)!=1:raise ValueError('一次只能筛选一个证券代码')
    return [row for row in records if codes[0] in row['stock_codes']]


def _published_at(value: str) -> str:
    parsed = datetime.strptime(_clean_text(value), "%d/%m/%Y %H:%M")
    return parsed.replace(tzinfo=HK_TZ).isoformat()


class _SearchTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_row = False
        self.in_cell = False
        self.in_link = False
        self.cells: list[list[str]] = []
        self.cell_parts: list[str] = []
        self.link_parts: list[str] = []
        self.link_href = ""
        self.rows: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "tr":
            self.in_row = True
            self.cells = []
            self.link_href = ""
            self.link_parts = []
        elif self.in_row and tag == "td":
            self.in_cell = True
            self.cell_parts = []
        elif self.in_cell and tag == "a":
            href = attributes.get("href") or ""
            if "/listedco/" in href:
                self.in_link = True
                self.link_href = href
                self.link_parts = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_parts.append(data)
        if self.in_link:
            self.link_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.in_link:
            self.in_link = False
        elif tag == "td" and self.in_cell:
            self.cells.append(self.cell_parts)
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            self.in_row = False
            if len(self.cells) >= 4 and self.link_href:
                published = _clean_text(" ".join(self.cells[0])).replace("Release Time: ", "")
                code = re.sub(r'^Stock Code:\s*','',_clean_text(" ".join(self.cells[1])))
                name = _clean_text(" ".join(self.cells[2])).replace("Stock Short Name: ", "")
                try:
                    published_iso = _published_at(published)
                except ValueError:
                    return
                self.rows.append({
                    "published_at": published_iso,
                    "stock_codes": _stock_codes(code,allow_unassigned=True),
                    "stock_name": name,
                    "title": _clean_text(" ".join(self.link_parts)),
                    "pdf_url": urllib.parse.urljoin("https://www1.hkexnews.hk", self.link_href),
                })


def parse_search_html(html: str) -> dict[str, Any]:
    """解析HKEXnews搜索HTML，并明确标记页面是否包含全部报告记录。"""
    count_match = re.search(r"Total records found:\s*([\d,]+)", html, re.I)
    if not count_match:
        raise ValueError("搜索响应缺少 reported count，不能视为成功")
    reported_count = int(count_match.group(1).replace(",", ""))
    parser = _SearchTableParser()
    parser.feed(html)
    records = _deduplicate_records(parser.rows)
    if reported_count and not records:
        raise ValueError("搜索报告有记录但页面响应为空，不能视为成功")
    return {
        "reported_count": reported_count,
        "records": records,
        "parsed_count": len(parser.rows),
        "security_associations": sum(len(row["stock_codes"]) for row in records),
        "unassigned_announcements": sum(not row['stock_codes'] for row in records),
        "complete": len(parser.rows) == reported_count,
    }


def _deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (record['published_at'], record['pdf_url'])
        codes = record['stock_codes']
        if not isinstance(codes,list) or any(not re.fullmatch(r'\d{5}',code) for code in codes):
            raise ValueError('公告必须使用明确的stock_codes数组')
        if key in indexed:
            indexed[key]['stock_codes'] = list(dict.fromkeys(indexed[key]['stock_codes']+codes))
        else:
            item={**record,'stock_codes':list(dict.fromkeys(codes))}
            indexed[key]=item;output.append(item)
    for item in output:item['stock_code_status']='indexed' if item['stock_codes'] else 'unassigned'
    return output


def _parse_ajax_response(payload: str) -> dict[str, Any]:
    if not payload.strip():
        raise ValueError("分页响应为空，不能视为成功")
    outer = json.loads(payload)
    if "result" not in outer:
        raise ValueError("分页响应缺少result")
    rows = json.loads(outer["result"])
    records = []
    for row in rows:
        href = row.get("FILE_LINK") or ""
        if not href:
            continue
        records.append({
            "published_at": _published_at(row["DATE_TIME"]),
            "stock_codes": _stock_codes(str(row["STOCK_CODE"]),allow_unassigned=True),
            "stock_name": _clean_text(str(row.get("STOCK_NAME", ""))),
            "title": _clean_text(str(row.get("TITLE", ""))),
            "pdf_url": urllib.parse.urljoin("https://www1.hkexnews.hk", href),
            "headline": _clean_text(str(row.get("SHORT_TEXT", ""))),
            "file_type": row.get("FILE_TYPE"),
            "file_info": row.get("FILE_INFO"),
        })
    reported = int(outer.get("recordCnt", rows[0].get("TOTAL_COUNT", len(rows)) if rows else 0))
    parsed_count = len(records)
    records = _deduplicate_records(records)
    return {
        "reported_count": reported,
        "records": records,
        "parsed_count": parsed_count,
        "security_associations": sum(len(row["stock_codes"]) for row in records),
        "unassigned_announcements": sum(not row['stock_codes'] for row in records),
        "complete": parsed_count == reported,
    }


def _english_date(value: str) -> str | None:
    value = _clean_text(value).replace(",", "")
    for pattern in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            continue
    return None


def parse_board_lot_text(text: str, published_at: str, source_url: str) -> dict[str, Any]:
    """只从明确绑定每手单位的句子/标题提取候选；冲突或提案留待复核。"""
    base = {
        "announced_at": published_at,
        "effective_date": None,
        "old_lot": None,
        "new_lot": None,
        "verified": False,
        "evidence_status": "needs_review",
        "source_url": source_url,
        "evidence_sentences": [],
        "review_reasons": [],
    }
    normalized = _clean_text(text)
    number = r"([1-9][0-9]{0,2}(?:,[0-9]{3})+|[1-9][0-9]*)"
    shares = r"(?:(?:existing|consolidated|new|adjusted|ordinary|H|A|B)\s+)*(?:shares?|units?)\b(?:\s+each)?"
    lot_pattern = re.compile(r"\bfrom\s+" + number + r"\s+" + shares
                             + r"\s+to\s+" + number + r"\s+" + shares, re.I)
    date_pattern = re.compile(
        r"\b(?:(?:became|becomes|becoming|has become|will become)\s+effective\s+(?:on|from)|with\s+effect\s+from)\s+"
        r"(?:[0-9]{1,2}\s*:\s*[0-9]{2}\s*(?:a\.m\.|p\.m\.|am|pm)\s*(?:on\s+)?)?"
        r"(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+)?"
        r"([0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4}|[A-Za-z]+\s+[0-9]{1,2},?\s+[0-9]{4})\b", re.I)
    # 每个事项在句号/分号处结束；另一个股本事项不能借用每手单位的主语。
    other_subject = re.compile(
        r"\b(?:(?:share )?consolidation|capital (?:reduction|reorgani[sz]ation)|(?:share )?sub[- ]?division|"
        r"adjustments? to (?:the )?share options?|appointment|resignation|change of company name|board lot(?: size)?)\b", re.I)
    dates, lots = set(), set()
    for sentence in re.split(r"(?<!a\.m\.)(?<!p\.m\.)(?<=[.;!?])\s+", normalized, flags=re.I):
        for subject in re.finditer(r"\bboard lot(?: size)?\b", sentence, re.I):
            prefix = sentence[:subject.start()]
            scope = other_subject.split(sentence[subject.end():], maxsplit=1)[0]
            excerpt = sentence[subject.start():subject.end()+len(scope)].strip()
            date_matches=list(date_pattern.finditer(scope))
            lot_matches=list(lot_pattern.finditer(scope))
            if not date_matches and not lot_matches:
                continue
            if excerpt not in base['evidence_sentences']:base['evidence_sentences'].append(excerpt)
            # 括号中替代的新股份名称不改变已明确的主要每手单位；其他条件必须保留待复核。
            certainty_scope=re.split(r'\(or\s+',scope,maxsplit=1,flags=re.I)[0]
            if re.search(r"\b(?:proposed|proposes?|proposal|intends?)\b", prefix+certainty_scope, re.I):
                base['review_reasons'].append('每手变更包含提案表述');continue
            if re.search(r"\b(?:subject to|conditional (?:on|upon)|upon (?:the )?completion|may or may not)\b", certainty_scope, re.I):
                base['review_reasons'].append('每手变更仍有条件');continue
            for match in date_matches:
                date_value=match.group(1).replace(',','')
                date=None
                for pattern in ('%d %B %Y','%d %b %Y','%B %d %Y','%b %d %Y'):
                    try:
                        date=datetime.strptime(date_value,pattern).date().isoformat();break
                    except ValueError:continue
                if date is None:
                    base['review_reasons'].append('生效日期无法解析');continue
                dates.add(date)
            for match in lot_matches:
                lots.add((int(match.group(1).replace(',', '')), int(match.group(2).replace(',', ''))))
    base['review_reasons']=list(dict.fromkeys(base['review_reasons']))
    if base['review_reasons'] or len(dates) != 1 or len(lots) != 1:
        if len(dates)!=1:base['review_reasons'].append('缺少唯一明确的每手变更生效日')
        if len(lots)!=1:base['review_reasons'].append('缺少唯一明确的新旧股份或单位数')
        return base
    old_lot, new_lot = next(iter(lots))
    return {**base, "effective_date": next(iter(dates)), "old_lot": old_lot,
            "new_lot": new_lot, "evidence_status": "candidate_effective"}


def parse_dividend_form_text(text: str, published_at: str, source_url: str) -> dict[str, Any]:
    """提取发行人最终派付日候选，并明确其仅用于回测模拟。"""
    normalized = _clean_text(text)
    base = {
        "announced_at": published_at,
        "payment_date": None,
        "cash_currency": None,
        "cash_per_share": None,
        "verified": False,
        "payment_basis": "issuer_final_schedule_simulated",
        "evidence_status": "needs_review",
        "source_url": source_url,
    }
    lower = normalized.lower()
    if "payment date to be announced" in lower or "conditional" in lower or "proposed payment date" in lower:
        return base
    payment_match = re.search(
        r"payment date\s+([0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4})", normalized, re.I,
    )
    cash_match = re.search(
        r"(?:dividend declared|final (?:cash )?dividend|interim (?:cash )?dividend)\s+"
        r"(HKD|USD|RMB|CNY)\s*([0-9]+(?:\.[0-9]+)?)\s+per share",
        normalized,
        re.I,
    )
    if not payment_match or not cash_match:
        return base
    payment_date = _english_date(payment_match.group(1))
    if not payment_date:
        return base
    status_is_final = any(marker in lower for marker in (
        "status updated", "status final", "final cash dividend", "interim cash dividend",
    ))
    if not status_is_final:
        return base
    return {
        **base,
        "payment_date": payment_date,
        "cash_currency": cash_match.group(1).upper().replace("RMB", "CNY"),
        "cash_per_share": float(cash_match.group(2)),
        "evidence_status": "candidate_final_schedule",
    }


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    extra = [column for column in frame.columns if column not in required]
    if missing or extra:
        raise ValueError(f"{label}字段不符；缺少={missing}，多余={extra}")


def _strict_bool(series: pd.Series, label: str) -> pd.Series:
    mapping = {True: True, False: False, "True": True, "False": False, "true": True, "false": False}
    if not series.isin(mapping).all():
        raise ValueError(f"{label}必须明确为true或false")
    return series.map(mapping).astype(bool)


def _validate_announced(series: pd.Series) -> pd.Series:
    strings = series.astype(str)
    if series.isna().any() or (~strings.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True)).any():
        raise ValueError("announced_at必须包含真实时区")
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError("announced_at无效")
    return parsed


def _date_values(series: pd.Series, label: str, allow_blank: bool = False) -> pd.Series:
    blank = series.isna() | series.astype(str).str.strip().eq("")
    parsed = pd.to_datetime(series.where(~blank), format="%Y-%m-%d", errors="coerce")
    if ((~blank) & parsed.isna()).any() or (blank.any() and not allow_blank):
        raise ValueError(f"{label}必须为YYYY-MM-DD且不能为空")
    return parsed


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = frame.replace({np.nan: None, pd.NaT: None})
    result = []
    for row in cleaned.to_dict("records"):
        result.append({
            key: (value.isoformat() if isinstance(value, (pd.Timestamp, datetime)) else value)
            for key, value in row.items()
        })
    return result


def import_board_lot_csv(path: str | Path) -> dict[str, Any]:
    frame = pd.read_csv(path, dtype={"security_id": str, "source_url": str})
    _require_columns(frame, BOARD_LOT_COLUMNS, "board lot CSV")
    if frame.empty:
        raise ValueError("board lot CSV不能为空")
    frame["verified"] = _strict_bool(frame["verified"], "verified")
    if frame[["security_id", "source_url"]].isna().any().any() or (
        frame[["security_id", "source_url"]].astype(str).apply(lambda col: col.str.strip().eq("")).any().any()
    ):
        raise ValueError("security_id和source_url不能为空")
    _validate_announced(frame["announced_at"])
    _date_values(frame["effective_date"], "effective_date")
    for column in ("old_lot", "new_lot"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if (~np.isfinite(numeric)).any() or (numeric <= 0).any() or (numeric % 1 != 0).any():
            raise ValueError(f"{column}必须为正整数")
        frame[column] = numeric.astype(int)
    if frame.duplicated(["security_id", "effective_date", "source_url"]).any():
        raise ValueError("board lot CSV含重复事件")
    events = frame.loc[frame["verified"]].copy()
    gaps = frame.loc[~frame["verified"]].copy()
    return {"status": "complete" if gaps.empty else "incomplete", "events": _records(events), "gaps": _records(gaps)}


def import_actions_csv(path: str | Path) -> dict[str, Any]:
    frame = pd.read_csv(path, dtype={"security_id": str, "source_url": str, "cash_currency": str})
    _require_columns(frame, ACTION_COLUMNS, "actions CSV")
    if frame.empty:
        raise ValueError("actions CSV不能为空")
    frame["verified"] = _strict_bool(frame["verified"], "verified")
    if frame[["security_id", "action_type", "source_url"]].isna().any().any() or (
        frame[["security_id", "action_type", "source_url"]].astype(str)
        .apply(lambda col: col.str.strip().eq("")).any().any()
    ):
        raise ValueError("security_id、action_type和source_url不能为空")
    _validate_announced(frame["announced_at"])
    _date_values(frame["effective_date"], "effective_date", allow_blank=True)
    _date_values(frame["payment_date"], "payment_date", allow_blank=True)
    if frame.duplicated(["security_id", "action_type", "effective_date", "payment_date", "source_url"]).any():
        raise ValueError("actions CSV含重复事件")
    verified_cash = frame["verified"] & frame["action_type"].isin({"cash_dividend", "privatisation_cash"})
    if frame.loc[verified_cash, "payment_date"].isna().any() or (
        frame.loc[verified_cash, "payment_date"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("已核验现金事件必须有确定payment_date")
    if (frame.loc[verified_cash, "payment_basis"] != "issuer_final_schedule_simulated").any():
        raise ValueError("payment_basis必须是issuer_final_schedule_simulated，不能冒充actual receipt")
    cash = pd.to_numeric(frame["cash_per_share"], errors="coerce")
    if ((verified_cash & ((~np.isfinite(cash)) | (cash < 0))).any()
            or frame.loc[verified_cash, "cash_currency"].isna().any()):
        raise ValueError("已核验现金事件必须有非负每股金额和币种")
    multiplier = pd.to_numeric(frame["share_multiplier"], errors="coerce")
    verified_share = frame["verified"] & frame["action_type"].isin({"split", "consolidation"})
    if (verified_share & ((~np.isfinite(multiplier)) | (multiplier <= 0))).any():
        raise ValueError("已核验拆并股事件必须有正share_multiplier")
    frame["cash_per_share"] = cash
    frame["share_multiplier"] = multiplier
    events = frame.loc[frame["verified"]].copy()
    gaps = frame.loc[~frame["verified"]].copy()
    return {"status": "complete" if gaps.empty else "incomplete", "events": _records(events), "gaps": _records(gaps)}


def infer_lot_history(
    current_anchors: pd.DataFrame,
    verified_changes: pd.DataFrame,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    """仅在公告覆盖完整且变更链闭合时，从当前每手单位向过去逆推。"""
    _require_columns(current_anchors, ("security_id", "current_lot"), "current anchors")
    required_changes = ("security_id", "effective_date", "old_lot", "new_lot", "verified")
    missing = [column for column in required_changes if column not in verified_changes.columns]
    if missing:
        raise ValueError(f"verified changes缺少字段: {missing}")
    histories: dict[str, list[dict[str, Any]]] = {}
    gaps = []
    for anchor in current_anchors.to_dict("records"):
        security_id = str(anchor["security_id"])
        current_lot = anchor["current_lot"]
        histories[security_id] = []
        evidence = coverage.get(security_id) if isinstance(coverage, dict) else None
        if not isinstance(evidence, dict) or evidence.get("complete") is not True:
            gaps.append({"security_id": security_id, "reason": "公告覆盖未证明完整"})
            continue
        if evidence.get("unresolved_count") != 0:
            gaps.append({"security_id": security_id, "reason": "存在未解决的每手单位公告"})
            continue
        if not isinstance(current_lot, (int, float, np.integer, np.floating)) or not np.isfinite(current_lot) or current_lot <= 0:
            gaps.append({"security_id": security_id, "reason": "当前每手单位锚点无效"})
            continue
        changes = verified_changes.loc[
            (verified_changes["security_id"].astype(str) == security_id)
            & (verified_changes["verified"] == True)  # noqa: E712
        ].copy()
        changes["effective_date"] = pd.to_datetime(changes["effective_date"], errors="coerce")
        if changes["effective_date"].isna().any() or changes["effective_date"].duplicated().any():
            gaps.append({"security_id": security_id, "reason": "变更日期无效或重复，链无法确定"})
            continue
        changes = changes.sort_values("effective_date", ascending=False)
        cursor = int(current_lot)
        reverse_periods = [{"effective_from": changes["effective_date"].max().date().isoformat() if not changes.empty else None,
                            "lot_size": cursor}]
        chain_ok = True
        for change in changes.to_dict("records"):
            new_lot = int(change["new_lot"])
            old_lot = int(change["old_lot"])
            if new_lot != cursor:
                chain_ok = False
                break
            reverse_periods.append({
                "effective_until": (pd.Timestamp(change["effective_date"]) - pd.Timedelta(days=1)).date().isoformat(),
                "lot_size": old_lot,
            })
            cursor = old_lot
        if not chain_ok:
            gaps.append({"security_id": security_id, "reason": "old/new与当前锚点的变更链不闭合"})
            continue
        histories[security_id] = list(reversed(reverse_periods))
    return {"status": "complete" if not gaps else "incomplete", "histories": histories, "gaps": gaps}


@dataclass
class _Response:
    body: bytes
    url: str


class HKEXSession:
    """带Cookie、有限重试及最少一秒请求间隔的单线程网页会话。"""
    def __init__(self, retries: int = 3, min_interval: float = 1.0) -> None:
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
        self.retries = retries
        self.min_interval = min_interval
        self.last_request_started = 0.0

    def request(self, url: str, data: dict[str, str] | None = None, referer: str | None = None) -> _Response:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            wait = self.min_interval - (time.monotonic() - self.last_request_started)
            if wait > 0:
                time.sleep(wait)
            self.last_request_started = time.monotonic()
            encoded = urllib.parse.urlencode(data).encode() if data is not None else None
            headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/html,application/json"}
            if referer:
                headers["Referer"] = referer
                headers["X-Requested-With"] = "XMLHttpRequest"
            request = urllib.request.Request(url, data=encoded, headers=headers)
            try:
                with self.opener.open(request, timeout=60) as response:
                    body = response.read()
                    if not body:
                        raise ValueError("HKEX响应为空")
                    return _Response(body=body, url=response.geturl())
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.min_interval)
        raise RuntimeError(f"HKEX请求在{self.retries}次有限重试后失败: {last_error}")


def _annual_windows(start: date, end: date) -> list[tuple[date, date]]:
    windows = []
    cursor = start
    while cursor <= end:
        window_end = min(end, date(cursor.year, 12, 31))
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return windows


def _split_window(start: date, end: date) -> list[tuple[date, date]]:
    if start >= end:
        return [(start, end)]
    middle = start + (end - start) // 2
    return [(start, middle), (middle + timedelta(days=1), end)]


def _window_key(kind: str, status: str, start: date, end: date, security: str | None, stock_id: str | None = None) -> str:
    suffix = f"_{security}" if security else ""
    if stock_id is not None:suffix+=f'_stockid_{stock_id}'
    return f"{kind}_{status}_{start:%Y%m%d}_{end:%Y%m%d}{suffix}"


def _search_form(kind: str, start: date, end: date, status: str, stock_id: str | None = None) -> dict[str, str]:
    t1, group, t2 = CATEGORIES[kind]
    return {
        "lang": "EN", "category": "0" if status == "current" else "1",
        "market": "SEHK", "searchType": "1", "documentType": "",
        "t1code": t1, "t2Gcode": group, "t2code": t2, "stockId": stock_id if stock_id is not None else "",
        "from": start.strftime("%Y%m%d"), "to": end.strftime("%Y%m%d"), "title": "",
    }


def _discover_window(
    session: HKEXSession,
    kind: str,
    start: date,
    end: date,
    status: str,
    root: Path,
    security: str | None,
    stock_id: str | None = None,
) -> dict[str, Any]:
    key = _window_key(kind, status, start, end, security, stock_id)
    raw_dir = root / "raw" / kind / status
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_path = raw_dir / f"{key}.html"
    state_path = raw_dir / f"{key}.status.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "success" and html_path.exists():
            parsed = parse_search_html(html_path.read_text(encoding='utf-8'))
            if not parsed['complete']:
                parsed = _parse_ajax_response((raw_dir/f'{key}.json').read_text(encoding='utf-8'))
            if not parsed['complete'] or parsed['reported_count']!=state['reported_count']:
                raise ValueError('已缓存原始公告行数不完整或不匹配')
            return {**state,'records':_select_security(parsed['records'],security) if security else parsed['records'],
                    'source_row_count':parsed['parsed_count']}
    state: dict[str, Any] = {
        "status": "failed", "kind": kind, "security_status": status,
        "start": start.isoformat(), "end": end.isoformat(), "security": security,
        "records": [], "reported_count": None,
    }
    try:
        session.request(SEARCH_URL)
        response = session.request(SEARCH_URL, data=_search_form(kind, start, end, status, stock_id))
        html = response.body.decode("utf-8", errors="strict")
        html_path.write_text(html, encoding="utf-8")
        parsed = parse_search_html(html)
        state["reported_count"] = parsed["reported_count"]
        if parsed["reported_count"] >= 1000:
            state.update(status="split_required", reason="reported_count达到1000，必须拆短窗口")
        elif parsed["complete"]:
            state.update(status="success", records=parsed["records"])
        else:
            t1, group, t2 = CATEGORIES[kind]
            query = urllib.parse.urlencode({
                "sortDir": "0", "sortByOptions": "DateTime",
                "category": "0" if status == "current" else "1", "market": "SEHK",
                "stockId": stock_id if stock_id is not None else "", "documentType": "", "fromDate": start.strftime("%Y%m%d"),
                "toDate": end.strftime("%Y%m%d"), "title": "", "searchType": "1",
                "t1code": t1, "t2Gcode": group, "t2code": t2,
                "rowRange": str(parsed["reported_count"]), "lang": "en",
            })
            ajax = session.request(f"{AJAX_URL}?{query}", referer=response.url)
            ajax_path = raw_dir / f"{key}.json"
            ajax_text = ajax.body.decode("utf-8", errors="strict")
            ajax_path.write_text(ajax_text, encoding="utf-8")
            expanded = _parse_ajax_response(ajax_text)
            if expanded["reported_count"] != parsed["reported_count"] or not expanded["complete"]:
                raise ValueError(
                    f"页面/分页条数不符: html={parsed['reported_count']} ajax={expanded['parsed_count']}"
                )
            state.update(status="success", records=expanded["records"])
        if state["status"] == "success" and security:
            if stock_id is not None:
                expected=security.split('.')[0]
                if any(row['stock_codes'] and expected not in row['stock_codes'] for row in state['records']):
                    raise ValueError('官方stockId检索返回其他证券，必须核实身份关联')
            state["records"] = _select_security(state["records"], security)
        state["retrieved_at"] = datetime.now(HK_TZ).isoformat()
    except Exception as exc:  # failure is persisted for audit and surfaced to the caller
        state.update(status="failed", records=[], reason=str(exc), retrieved_at=datetime.now(HK_TZ).isoformat())
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return state


def discover(
    kind: str,
    start: str,
    end: str,
    security: str | None = None,
    root: str | Path = DATA / "hkex",
    session: HKEXSession | None = None,
    stock_id: str | None = None,
) -> dict[str, Any]:
    """发现一个公告类别；任何缺页或条数不一致都会使结果明确失败。"""
    if kind not in CATEGORIES:
        raise ValueError(f"kind必须是{sorted(CATEGORIES)}之一")
    if stock_id is not None:
        if not isinstance(stock_id,str) or not re.fullmatch(r'[1-9]\d*',stock_id) or not security:
            raise ValueError('stock_id必须为已知官方正整数标识，并同时提供证券代码')
    start_date = datetime.strptime(start, "%Y%m%d").date()
    end_date = datetime.strptime(end, "%Y%m%d").date()
    if start_date > end_date:
        raise ValueError("start不能晚于end")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    session = session or HKEXSession()
    queue = [(window_start, window_end, status)
             for window_start, window_end in _annual_windows(start_date, end_date)
             for status in ("current", "delisted")]
    windows = []
    records = []
    while queue:
        window_start, window_end, status = queue.pop(0)
        result = _discover_window(session, kind, window_start, window_end, status, root, security, stock_id)
        if result["status"] == "split_required" and window_start < window_end:
            queue[0:0] = [(a, b, status) for a, b in _split_window(window_start, window_end)]
            continue
        windows.append({key: value for key, value in result.items() if key != "records"})
        records.extend(result.get("records", []))
    records = _deduplicate_records(records)
    catalog_dir = root / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = catalog_dir / f"{kind}.json"
    existing = []
    if catalog_path.exists():
        existing = json.loads(catalog_path.read_text(encoding="utf-8")).get("records", [])
    merged = _deduplicate_records(existing + records)
    catalog = {
        "kind": kind,
        "updated_at": datetime.now(HK_TZ).isoformat(),
        "records": merged,
        "announcement_versions": len(merged),
        "unique_documents": len({row['pdf_url'] for row in merged}),
        "security_associations": sum(len(row['stock_codes']) for row in merged),
        "unassigned_announcements": sum(not row['stock_codes'] for row in merged),
    }
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures = [window for window in windows if window["status"] != "success"]
    run = {
        "status": "complete" if not failures else "failed",
        "kind": kind, "start": start, "end": end, "security": security,
        "stock_id": stock_id,
        "discovered_records": len(records), "catalog_records": len(merged),
        "windows": windows, "failures": failures, "catalog_path": str(catalog_path),
    }
    run_dir = root / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    identity_suffix=f'_stockid_{stock_id}' if stock_id is not None else ''
    run_path = run_dir / f"discover_{kind}_{start}_{end}_{security or 'all'}{identity_suffix}.json"
    run_path.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run


def document_path(root, kind, url):
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    filename = Path(urllib.parse.urlparse(url).path).name
    suffix = Path(filename).suffix.lower()
    if suffix not in ('.pdf','.doc'):raise ValueError('尚未支持的公告原始格式: '+suffix)
    folder = 'pdf' if suffix=='.pdf' else 'documents'
    return Path(root)/folder/kind/f'{digest}_{filename}'


def download(
    kind: str,
    security: str | None = None,
    root: str | Path = DATA / "hkex",
    session: HKEXSession | None = None,
) -> dict[str, Any]:
    """按真实PDF/Word格式保存公告原件；成功缓存不会重复请求。"""
    if kind not in CATEGORIES:
        raise ValueError(f"kind必须是{sorted(CATEGORIES)}之一")
    root = Path(root)
    catalog_path = root / "catalog" / f"{kind}.json"
    if not catalog_path.exists():
        raise ValueError("尚无该类别目录，请先运行discover")
    records = json.loads(catalog_path.read_text(encoding="utf-8"))["records"]
    if security:
        records = _select_security(records, security)
    session = session or HKEXSession()
    downloaded = cached = 0
    failures = []
    for record in records:
        target = document_path(root,kind,record['pdf_url'])
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.stat().st_size > 0:
            cached += 1
            continue
        try:
            response = session.request(record["pdf_url"])
            signature = b'%PDF' if target.suffix.lower()=='.pdf' else bytes.fromhex('d0cf11e0a1b11ae1')
            if not response.body.startswith(signature):
                raise ValueError('响应与公告声明的原始文件格式不符')
            target.write_bytes(response.body)
            downloaded += 1
        except Exception as exc:
            failures.append({"pdf_url": record["pdf_url"], "reason": str(exc)})
    return {
        "status": "complete" if not failures else "failed",
        "kind": kind, "security": security, "selected": len(records),
        "downloaded": downloaded, "cached": cached, "failures": failures,
    }


def extract_cached(kind: str, root: str | Path = DATA / 'hkex') -> dict[str, Any]:
    """只提取已经下载的原文与候选事实；候选绝不自动变成核验证据。"""
    from pypdf import PdfReader
    root = Path(root)
    catalog = json.loads((root/'catalog'/f'{kind}.json').read_text(encoding='utf-8'))['records']
    output = root/'extracted'/kind
    output.mkdir(parents=True,exist_ok=True)
    extracted, pending, failures = [], [], []
    for row in catalog:
        pdf_path = document_path(root,kind,row['pdf_url'])
        if not pdf_path.is_file():
            pending.append(row['pdf_url'])
            continue
        text_path = output/f'{pdf_path.stem}.txt'
        try:
            if text_path.is_file():
                original = text_path.read_text(encoding='utf-8')
            else:
                if pdf_path.suffix.lower()=='.doc':
                    raise ValueError('Word原件已保存，正文尚待文档解析工具提取')
                reader = PdfReader(pdf_path)
                original = '\n\n'.join(f'[PAGE {number+1}]\n{page.extract_text() or ""}' for number,page in enumerate(reader.pages))
                text_path.write_text(original,encoding='utf-8')
            if kind=='board_lot':
                candidate = parse_board_lot_text(original,row['published_at'],row['pdf_url'])
            elif kind in ('dividend_form','dividend','payment_change'):
                candidate = parse_dividend_form_text(original,row['published_at'],row['pdf_url'])
            else:
                candidate = {'verified':False,'evidence_status':'needs_review','source_url':row['pdf_url']}
            candidate.update(stock_codes=list(row['stock_codes']),index_association_only=True,title=row['title'],
                             source_text=str(text_path.resolve()),source_document=str(pdf_path.resolve()))
            if len(row['stock_codes'])>1:
                candidate.update(evidence_status='needs_review',association_note='多证券索引只表示公告关联，提取事实未分配给任何一个代码')
            elif not row['stock_codes']:
                candidate.update(evidence_status='needs_review',association_note='公告索引未提供证券代码，提取事实未归属证券')
            extracted.append(candidate)
        except Exception as exc:
            failures.append({'source_url':row['pdf_url'],'reason':str(exc)})
    result = {'status':'extracted_pending_review','catalog_records':len(catalog),'extracted_records':len(extracted),
              'pending_download_records':len(pending),'extraction_failures':failures,
              'catalog_security_associations':sum(len(row['stock_codes']) for row in catalog),
              'extracted_security_associations':sum(len(row['stock_codes']) for row in extracted),
              'candidate_effective_records':sum(row['evidence_status']=='candidate_effective' for row in extracted),
              'verified_records':0,'review_required':True,'records':extracted}
    (output/'candidates.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    candidate_frame=pd.DataFrame(extracted)
    if not candidate_frame.empty:candidate_frame['stock_codes']=candidate_frame.stock_codes.map(lambda codes:json.dumps(codes,ensure_ascii=False))
    candidate_frame.to_csv(output/'candidates.csv',index=False)
    return {key:value for key,value in result.items() if key!='records'}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HKEXnews公告证据采集")
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover_parser = subparsers.add_parser("discover")
    discover_parser.add_argument("--kind", choices=sorted(CATEGORIES), required=True)
    discover_parser.add_argument("--start", required=True)
    discover_parser.add_argument("--end", required=True)
    discover_parser.add_argument("--security")
    discover_parser.add_argument("--stock-id")
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--kind", choices=sorted(CATEGORIES), required=True)
    download_parser.add_argument("--security")
    extract_parser = subparsers.add_parser('extract')
    extract_parser.add_argument('--kind',choices=sorted(CATEGORIES),required=True)
    args = parser.parse_args(argv)
    if args.command=='discover':result=discover(args.kind,args.start,args.end,args.security,stock_id=args.stock_id)
    elif args.command=='download':result=download(args.kind,args.security)
    else:result=extract_cached(args.kind)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in ('complete','extracted_pending_review') else 1


if __name__ == "__main__":
    raise SystemExit(main())
