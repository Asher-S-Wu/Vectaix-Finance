from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree
from zipfile import ZipFile

import pandas as pd
from pypdf import PdfReader


HOLDOUT_PREFIX = "vectaix-cross-holdout-v1:"
SOURCE_ROOT = Path(__file__).resolve().parent.parent / "data/raw/cross-market-identity"
SOURCE_FILES = (
    "hkex-ah-sse-2018.pdf", "hkex-ah-szse-2018.pdf",
    "hkex-delisted-stock-list.html", "hkex-current-securities.xlsx",
)
ORDINARY_SUBCATEGORIES = {"Equity Securities (Main Board)", "Equity Securities (GEM)"}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _StockRows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attributes):
        if tag == "tr":
            self.row = []
        elif tag == "td" and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def _delisted(path: Path) -> pd.DataFrame:
    parser = _StockRows()
    parser.feed(path.read_text(encoding="utf-8-sig"))
    rows = [{"hk_code": row[0], "source_name": row[1], "source_row": index + 1}
            for index, row in enumerate(parser.rows) if len(row) == 2 and re.fullmatch(r"\d{5}", row[0])]
    if not rows or "Delisted companies" not in path.read_text(encoding="utf-8-sig"):
        raise ValueError("港交所退市证券页面为空或不是预期名单。")
    return pd.DataFrame(rows)


def _ah_pairs(path: Path, exchange: str) -> list[dict]:
    pages = [page.extract_text(extraction_mode="layout") for page in PdfReader(path).pages]
    text = "\n".join(pages)
    effective = re.search(r"Effective Date\s*:\s*(\d{2}/\d{2}/\d{4})", text)
    count = re.search(r"No\. of companies\s*:\s*(\d+)", text)
    if (not effective or not count or "Dually Listed" not in text
            or exchange not in text or "SEHK Listed Shares" not in text):
        raise ValueError(f"{path.name} 缺少同公司双重上市名单、有效日期或公司总数。")
    observed_on = pd.to_datetime(effective.group(1), format="%d/%m/%Y").date().isoformat()
    pattern = r"(?<!\d)(\d{5})\s+(\d{2}/\d{2}/\d{4})\s+(\d{6})\s+(\d{2}/\d{2}/\d{4})(?!\d)"
    records = []
    suffix = {"SSE": "SH", "SZSE": "SZ"}[exchange]
    for page_number, page in enumerate(pages, 1):
        for match in re.finditer(pattern, page):
            hk, hdate, mainland, adate = match.groups()
            line_start = page.rfind("\n", 0, match.start()) + 1
            line_end = page.find("\n", match.end())
            if line_end == -1:
                line_end = len(page)
            evidence = " ".join(page[line_start:line_end].split())
            name = re.sub(r"^\d+\s*", "", page[line_start:match.start()].strip())
            records.append({
                "a_symbol": f"{mainland}.{suffix}", "hk_symbol": f"{hk}.HK",
                "source_company_name": name, "relationship_observed_on": observed_on,
                "a_listing_date": pd.to_datetime(adate, format="%d/%m/%Y").date().isoformat(),
                "hk_listing_date": pd.to_datetime(hdate, format="%d/%m/%Y").date().isoformat(),
                "source_file": path.name, "source_page": page_number, "source_evidence": evidence,
                "relationship_status": "official_historical_snapshot",
                "continuous_historical_identity_proven": False,
            })
    if len(records) != int(count.group(1)):
        raise ValueError(f"{path.name} 提取双代码行数与原文公司总数不一致。")
    return records


def _current_securities(path: Path) -> tuple[pd.DataFrame, str]:
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        strings = ["".join(item.itertext()) for item in ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall(".//s:row", namespace):
            values = {}
            for cell in row.findall("s:c", namespace):
                value = cell.find("s:v", namespace)
                if value is not None and value.text is not None:
                    column = re.sub(r"\d", "", cell.attrib["r"])
                    values[column] = strings[int(value.text)] if cell.attrib.get("t") == "s" else value.text
            rows.append(values)
    if (rows[0].get("A") != "List of Securities" or rows[2].get("A") != "Stock Code"
            or rows[2].get("C") != "Category" or rows[2].get("D") != "Sub-Category"
            or rows[2].get("F") != "ISIN" or rows[2].get("Q") != "Trading Currency"):
        raise ValueError("港交所当前证券表列头与固定解析规格不符。")
    update = re.fullmatch(r"Updated as at (\d{2}/\d{2}/\d{4})", rows[1].get("A", ""))
    if not update:
        raise ValueError("港交所当前证券表没有明确自报日期。")
    records = [{"hk_code": row["A"], "official_name": row.get("B"), "category": row.get("C"),
                "subcategory": row.get("D"), "isin": row.get("F"), "currency": row.get("Q"),
                "source_row": index + 4}
               for index, row in enumerate(rows[3:]) if re.fullmatch(r"\d{5}", row.get("A", ""))]
    result = pd.DataFrame(records)
    if result.empty or result.hk_code.duplicated().any():
        raise ValueError("港交所当前证券表代码为空或不唯一。")
    return result, pd.to_datetime(update.group(1), format="%d/%m/%Y").date().isoformat()


def load_cross_market_identity(a_stocks: pd.DataFrame, hk_stocks: pd.DataFrame,
                               source_dir: Path = SOURCE_ROOT) -> dict:
    """Resolve proven links, flag ambiguous codes, and expose provisional identities."""
    source_dir = Path(source_dir).resolve()
    hashes, receipts = {}, {}
    for filename in SOURCE_FILES:
        path = source_dir / filename
        receipt_path = source_dir / f"{filename}.receipt.json"
        receipt = json.loads(receipt_path.read_text())
        host = urlparse(receipt["finalUrl"]).hostname
        if (receipt["statusCode"] != 200 or host is None
                or not (host == "hkex.com.hk" or host.endswith(".hkex.com.hk"))
                or receipt["sha256"] != _digest(path)):
            raise ValueError(f"身份来源 {filename} 的域名、状态或指纹不符。")
        requested, received = pd.Timestamp(receipt["requestedAt"]), pd.Timestamp(receipt["receivedAt"])
        if requested.tzinfo is None or received.tzinfo is None or requested > received:
            raise ValueError(f"身份来源 {filename} 的获取时间不合法。")
        hashes[filename], hashes[receipt_path.name] = _digest(path), _digest(receipt_path)
        receipts[filename] = receipt
    pairs = pd.DataFrame(_ah_pairs(source_dir / SOURCE_FILES[0], "SSE") + _ah_pairs(source_dir / SOURCE_FILES[1], "SZSE"))
    if pairs.a_symbol.duplicated().any() or pairs.hk_symbol.duplicated().any():
        raise ValueError("两份官方历史 A/H 名单有非一一对应关系，需明确审查，不能自动合并。")
    delisted = _delisted(source_dir / "hkex-delisted-stock-list.html")
    old_names = delisted.groupby("hk_code").source_name.agg(list).to_dict()
    reused = {code: names for code, names in old_names.items() if len(set(names)) > 1}
    current, current_asof = _current_securities(source_dir / "hkex-current-securities.xlsx")
    current_by_code = current.set_index("hk_code").to_dict("index")
    links = {}
    for row in pairs.itertuples(index=False):
        issuer = f"AH:{row.a_symbol}"
        links[row.a_symbol] = (issuer, row)
        links[row.hk_symbol] = (issuer, row)
    outputs = []
    for market, universe in (("A", a_stocks), ("HK", hk_stocks)):
        if universe.ts_code.duplicated().any():
            raise ValueError(f"{market} 股票池含重复 ts_code，不能去重或替换身份。")
        for original in universe.to_dict("records"):
            symbol = original["ts_code"]
            reasons, ordinary_status = [], "unverified"
            code = symbol.split(".")[0]
            issuer_id = f"SECURITY:{symbol}"
            resolution = "provisional_security_key_no_known_cross_link"
            ah_observed = None
            if symbol in links:
                issuer_id, evidence = links[symbol]
                resolution = "official_2018_ah_link_historical_continuity_unproven"
                ah_observed = evidence.relationship_observed_on
            if market == "A":
                if not re.fullmatch(r"\d{6}\.(SH|SZ)", symbol):
                    raise ValueError("A 股原冻结池代码不是六位沪深证券代码。")
                ordinary_status = "source_universe_cny_sse_szse_equity"
            else:
                if not re.fullmatch(r"\d{5}\.HK", symbol):
                    reasons.append("nonstandard_or_reused_symbol_suffix")
                if code in reused:
                    reasons.append("official_delisted_code_has_multiple_company_names")
                official = current_by_code.get(code)
                if official is None:
                    reasons.append("ordinary_share_class_not_in_current_official_snapshot")
                else:
                    if (official["category"] != "Equity" or official["subcategory"] not in ORDINARY_SUBCATEGORIES
                            or official["currency"] != "HKD"):
                        reasons.append("not_official_hkd_main_board_or_gem_equity")
                    else:
                        ordinary_status = "official_current_hkd_main_board_or_gem_equity"
                    supplied_isin = original.get("isin")
                    if not isinstance(supplied_isin, str) or not supplied_isin or supplied_isin != official["isin"]:
                        reasons.append("current_official_isin_missing_or_mismatch")
                    if code in old_names:
                        # One old issuer plus a current security also lacks a proven lifecycle link.
                        reasons.append("code_in_both_delisted_and_current_lists_lifecycle_unresolved")
            identity_status = "unknown" if reasons else "admitted"
            split = "heldout" if int(hashlib.sha256((HOLDOUT_PREFIX + issuer_id).encode()).hexdigest(), 16) % 5 == 0 else "train"
            outputs.append({
                **original, "symbol": symbol, "source_market": original.get("market"), "market": market,
                "issuer_id": issuer_id, "identity_status": identity_status, "identity_resolution": resolution,
                "identity_reasons": "|".join(reasons), "ordinary_share_status": ordinary_status,
                "ah_relationship_observed_on": ah_observed, "split": split,
                "issuer_isolation_proven": False, "historical_identity_continuity_proven": False,
            })
    stocks = pd.DataFrame(outputs)
    if stocks.symbol.duplicated().any():
        raise ValueError("跨市场代码主键重复。")
    members = stocks.groupby("issuer_id").symbol.agg(list).to_dict()
    paired_groups = {issuer: symbols for issuer, symbols in members.items() if len(symbols) > 1}
    if (stocks.groupby("issuer_id").split.nunique() > 1).any():
        raise ValueError("同一已知关联身份跨越训练与留出组。")
    admitted = stocks.loc[stocks.identity_status.eq("admitted")]
    return {
        "stocks": stocks, "ah_pairs": pairs, "delisted_records": delisted,
        "current_securities": current,
        "provenance": {
            "sourceRoot": str(source_dir), "fileHashes": hashes, "receipts": receipts,
            "officialAhRelationships": len(pairs), "officialAhSnapshotDates": sorted(pairs.relationship_observed_on.unique()),
            "officialCurrentSecuritiesAsOf": current_asof,
            "delistedSourceRows": len(delisted), "delistedCodes": int(delisted.hk_code.nunique()),
            "multipleNameDelistedCodes": reused, "selectedKnownAhGroups": paired_groups,
            "admittedByMarket": {str(k): int(v) for k, v in admitted.groupby("market").size().items()},
            "unknownByReason": {str(k): int(v) for k, v in stocks.loc[stocks.identity_status.eq("unknown"), "identity_reasons"].str.split("|").explode().value_counts().items()},
            "splitRule": "int(sha256('vectaix-cross-holdout-v1:'+issuer_id),16)%5==0",
            "oldAStockHoldoutReused": False, "issuerIsolationProven": False,
            "unknownIdentityPolicy": "Keep planned rows; no fit, calibration or issuer-heldout claim for unknown identities. Nonstandard suffixes are never stripped to fabricate a historical code mapping.",
            "provisionalPolicy": "Standard codes with current official equity class and matching ISIN, no observed reuse conflict, may enter exploratory research under a security-level provisional key. Absence of a known link does not prove independence of issuers.",
            "ordinaryShareScope": "Current HKD Equity Securities Main Board/GEM only; excludes investment companies, trading-only, receipts, REITs, funds, warrants and debt. This is an exchange category gate, not a complete historical legal share-class proof.",
            "ahCoverageLimitation": "The two official 2018 lists prove 107 historical cross-listing observations, not all current or historical A/H links, mergers, corporate groups or issuer lifecycles. Later links may be absent; even known pairs have no complete continuous-history proof.",
            "timingLimitation": "Downloaded current-vintage identity sources are used for retrospective research partition and risk gates, never as historical predictive features. A source's printed as-of date is retained even when later than acquisition time or the price cutoff.",
        },
    }
