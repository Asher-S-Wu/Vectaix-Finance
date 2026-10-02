"""Audited A-share data acquisition through the approved credential helper.

This module never reads credentials or configures network authentication. The
separate helper is the only process that reads the locally saved Tushare token.
Raw fields retain Tushare's original units; ``normalize_units`` adds SI columns.
Every successful partition has a checksummed Parquet file and a JSON manifest.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable

import pandas as pd

HELPER = Path("/workspace/shared/tushare_persistent.py")
SOURCE = "https://api.tushare.pro"
FIELDS = {
    "stock_basic": "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date",
    "daily": "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
    "daily_basic": "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv",
    "adj_factor": "ts_code,trade_date,adj_factor",
    "stk_limit": "trade_date,ts_code,up_limit,down_limit",
    "stock_st": "",
    "trade_cal": "exchange,cal_date,is_open,pretrade_date",
    "index_daily": "ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount",
    "namechange": "ts_code,name,start_date,end_date,ann_date,change_reason",
    "bse_mapping": "",
}
PAGE_SIZES = {"stk_limit": 5800, "stock_st": 1000, "namechange": 1000}
DATE_ENDPOINTS = ("daily", "daily_basic", "adj_factor", "stk_limit")


class CollectionError(RuntimeError):
    """An incomplete, inconsistent, or corrupted partition must not be consumed."""


class PermanentAPIError(CollectionError):
    """A rejected/unsafe request requiring diagnosis, not an automatic retry."""


class TransientAPIError(CollectionError):
    """A bounded retry budget was exhausted."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_canonical(value))
    os.replace(temporary, path)


class HelperClient:
    """Subprocess-only client with a thread-shared, conservative rate budget.

    Tushare rejection codes may conflate permission and quota errors. Without an
    approved, unambiguous transient signal, API rejections are never retried.
    No helper stderr/stdout is included in raised exceptions or progress logs.
    """

    def __init__(self, helper: Path = HELPER, *, runner: Callable = subprocess.run,
                 sleep: Callable = time.sleep, requests_per_minute: float = 60,
                 retries: int = 3, python: str = sys.executable, control_path: Path | None = None):
        if requests_per_minute < 0 or requests_per_minute > 200:
            raise ValueError("rate budget must be between 0 and 200 requests/minute")
        if retries < 0 or retries > 5:
            raise ValueError("retries must be between 0 and 5")
        self.helper, self.runner, self.sleep, self.python = Path(helper), runner, sleep, python
        self.interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self.retries = retries
        self.control_path = Path(control_path) if control_path is not None else None
        self.max_requests_per_minute = requests_per_minute
        self._lock = threading.Lock()
        self._next_request = 0.0

    def _wait(self) -> None:
        with self._lock:
            while self.control_path is not None and self.control_path.exists():
                control = json.loads(self.control_path.read_text())
                if control.get("paused", False):
                    self.sleep(0.5)
                    continue
                budget = control.get("requests_per_minute", self.max_requests_per_minute)
                if not 0 < budget <= self.max_requests_per_minute:
                    raise CollectionError("control budget must be positive and cannot exceed the initial rate ceiling")
                self.interval = 60. / budget
                break
            delay = max(0.0, self._next_request - time.monotonic())
            if delay:
                self.sleep(delay)
            self._next_request = time.monotonic() + self.interval

    def query(self, endpoint: str, params: dict, fields: str) -> dict:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", endpoint):
            raise ValueError("invalid endpoint")
        argv = [self.python, str(self.helper), "query", endpoint,
                json.dumps(params, sort_keys=True, separators=(",", ":")), fields]
        last_status = "unknown"
        for attempt in range(self.retries + 1):
            self._wait()
            try:
                completed = self.runner(argv, capture_output=True, text=True, timeout=40)
            except subprocess.TimeoutExpired:
                response = {"status": "timeout"}
            else:
                try:
                    response = json.loads(completed.stdout)
                except (TypeError, ValueError):
                    raise PermanentAPIError(f"{endpoint}: invalid helper response") from None
            if not isinstance(response, dict):
                raise PermanentAPIError(f"{endpoint}: invalid helper response")
            status = response.get("status")
            if status == "success":
                if not isinstance(response.get("data"), dict):
                    raise PermanentAPIError(f"{endpoint}: invalid data envelope")
                return response
            transient = status in {"timeout", "request_failed"} or (status == "http_error" and response.get("http_status") in {408, 429, 500, 502, 503, 504})
            if not transient:
                raise PermanentAPIError(f"{endpoint}: helper status={status}, api_code={response.get('api_code')}, http_status={response.get('http_status')}")
            last_status = status
            if attempt < self.retries:
                self.sleep(min(30.0, 2.0 ** attempt))
        raise TransientAPIError(f"{endpoint}: retry budget exhausted ({last_status})")


def _identity(endpoint: str, params: dict, fields: str) -> tuple[str, str]:
    digest = hashlib.sha256(_canonical({"endpoint": endpoint, "params": params, "fields": fields})).hexdigest()[:16]
    label = str(params.get("trade_date") or params.get("list_status") or params.get("ts_code") or params.get("exchange") or "all")
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
    return label, digest


def _partition_paths(root: Path, endpoint: str, params: dict, fields: str) -> tuple[Path, Path, Path]:
    label, digest = _identity(endpoint, params, fields)
    base = Path(root) / endpoint / f"{label}--{digest}"
    return base.with_suffix(base.suffix + ".parquet"), base.with_suffix(base.suffix + ".manifest.json"), base.with_suffix(base.suffix + ".error.json")


def collect_query(client: HelperClient, endpoint: str, params: dict, fields: str,
                  root: Path, *, page_size: int = 6000, allow_empty: bool = False,
                  keys: tuple[str, ...] | None = None, max_pages: int = 10000) -> dict:
    """Collect all deterministic offset pages, fail closed on overlap/truncation.

    A short final page is required; a full last page always triggers another
    request. Paginated endpoints that ignore offsets produce a duplicate error.
    Consumers must select only partitions with complete/empty manifests.
    """
    if page_size < 1 or max_pages < 1:
        raise ValueError("positive page size and max pages required")
    if "offset" in params or "limit" in params:
        raise ValueError("pagination parameters are controlled by the collector")
    root = Path(root)
    artifact, manifest_path, error_path = _partition_paths(root, endpoint, params, fields)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") not in {"complete", "empty"}:
            raise CollectionError("cached partition status is not complete")
        if not artifact.exists() or hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest.get("sha256"):
            raise CollectionError(f"checksum mismatch: {artifact.name}")
        if manifest["status"] == "empty" and not allow_empty:
            raise CollectionError(f"{endpoint}: cached empty response")
        return manifest
    requests, rows, columns, seen = [], [], None, set()
    started = _now()
    try:
        for page in range(max_pages):
            query_params = {**params, "limit": page_size, "offset": page * page_size}
            fetched_at = _now()
            response = client.query(endpoint, query_params, fields)
            data = response.get("data", {})
            page_columns, items = data.get("fields"), data.get("items")
            if not isinstance(page_columns, list) or not isinstance(items, list):
                raise CollectionError(f"{endpoint}: invalid data schema")
            if len(set(page_columns)) != len(page_columns):
                raise CollectionError(f"{endpoint}: duplicate field names")
            requests.append({"endpoint": endpoint, "params": query_params, "fields": fields,
                             "fetched_at": fetched_at, "rows": len(items),
                             "response_sha256": hashlib.sha256(_canonical(data)).hexdigest()})
            if len(items) > page_size:
                raise CollectionError(f"{endpoint}: response exceeds requested page size")
            if columns is None:
                columns = page_columns
                if fields and page_columns != fields.split(","):
                    raise CollectionError(f"{endpoint}: returned schema differs from requested fields")
            elif columns != page_columns:
                raise CollectionError(f"{endpoint}: schema changed between pages")
            if keys is None:
                chosen_keys = (("ts_code", "trade_date", "type") if endpoint == "stock_st" and "type" in columns else (("ts_code", "trade_date") if {"ts_code", "trade_date"}.issubset(columns) else (("ts_code",) if endpoint in {"stock_basic"} else tuple(columns))))
            else:
                chosen_keys = keys
            if not set(chosen_keys).issubset(columns):
                raise CollectionError(f"{endpoint}: missing unique key fields")
            indices = [columns.index(key) for key in chosen_keys]
            for row in items:
                if not isinstance(row, list) or len(row) != len(columns):
                    raise CollectionError(f"{endpoint}: malformed row")
                key = tuple(row[i] for i in indices)
                if key in seen:
                    raise CollectionError(f"{endpoint}: duplicate keys; pagination overlap or source corruption")
                seen.add(key)
            rows.extend(items)
            if len(items) < page_size:
                break
        else:
            raise CollectionError(f"{endpoint}: pagination max_pages exceeded (possible truncation)")
        if not rows and not allow_empty:
            raise CollectionError(f"{endpoint}: empty response")
        frame = pd.DataFrame(rows, columns=columns or fields.split(","))
        if "trade_date" in params and "trade_date" in frame and not frame.empty:
            if not frame["trade_date"].astype(str).eq(str(params["trade_date"])).all():
                raise CollectionError(f"{endpoint}: returned unexpected trade_date")
        if "trade_date" in frame and not frame.empty:
            observed_dates = frame["trade_date"].astype(str)
            if (params.get("start_date") and observed_dates.lt(str(params["start_date"])).any()) or (params.get("end_date") and observed_dates.gt(str(params["end_date"])).any()):
                raise CollectionError(f"{endpoint}: returned data outside requested date range")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact.with_name(artifact.name + ".tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, artifact)
        manifest = {"format_version": 1, "source": SOURCE, "credential_transport": "approved_helper_subprocess",
                    "endpoint": endpoint, "params": params, "fields": columns,
                    "requested_fields": fields, "started_at": started, "completed_at": _now(),
                    "status": "complete" if rows else "empty", "rows": len(rows),
                    "artifact": str(artifact.relative_to(root)), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "requests": requests, "page_size": page_size, "duplicate_keys": 0}
        _write_json(manifest_path, manifest)
        return manifest
    except Exception as error:
        _write_json(error_path, {"endpoint": endpoint, "params": params, "requested_fields": fields,
                                "status": "error", "started_at": started, "failed_at": _now(),
                                "error_type": type(error).__name__, "error": str(error), "requests": requests})
        raise


def is_a_share(ts_code: str) -> bool:
    """Mainland listed equities/CDRs, including historical Beijing code prefixes."""
    return bool(re.fullmatch(r"(?:60\d{4}\.SH|68[89]\d{3}\.SH|(?:00|30)\d{4}\.SZ|(?:(?:4|8)\d{5}|92\d{4})\.BJ)", str(ts_code)))


def normalize_units(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rules = {"vol": ("volume_shares", 100.), "amount": ("amount_cny", 1000.),
             "pct_chg": ("return_fraction", .01)}
    rules.update({name: (name + "_cny", 10000.) for name in ("total_mv", "circ_mv")})
    rules.update({name: (name + "_shares", 10000.) for name in ("total_share", "float_share", "free_share")})
    rules.update({name: (name + "_fraction", .01) for name in ("turnover_rate", "turnover_rate_f", "dv_ratio", "dv_ttm")})
    for source, (target, factor) in rules.items():
        if source in result:
            result[target] = pd.to_numeric(result[source], errors="coerce") * factor
    return result


def iter_partitions(root: Path, endpoint: str, *, year: int | None = None,
                    start: str | None = None, end: str | None = None):
    """Yield (frame, manifest) after checksum verification, optionally date-filtered.

    Daily partitions outside the requested interval are skipped before reading
    their Parquet data. Range-query partitions are filtered after verification.
    """
    root = Path(root)
    start = start or (f"{year}0101" if year is not None else None)
    end = end or (f"{year}1231" if year is not None else None)
    for path in sorted((root / endpoint).glob("*.manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest["status"] not in {"complete", "empty"}:
            continue
        date = manifest["params"].get("trade_date")
        if date and ((start and str(date) < start) or (end and str(date) > end)):
            continue
        artifact = root / manifest["artifact"]
        if not artifact.exists() or hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest["sha256"]:
            raise CollectionError(f"checksum mismatch: {artifact.name}")
        frame = pd.read_parquet(artifact)
        if not date and "trade_date" in frame and (start or end):
            mask = pd.Series(True, index=frame.index)
            if start:
                mask &= frame["trade_date"].astype(str).ge(start)
            if end:
                mask &= frame["trade_date"].astype(str).le(end)
            frame = frame.loc[mask].copy()
        yield frame, manifest


def cached_partition(root: Path, endpoint: str, params: dict):
    """Return one verified exact-query partition, or None; ambiguous schemas fail."""
    matches = [(frame, manifest) for frame, manifest in iter_partitions(root, endpoint,
                start=params.get("trade_date"), end=params.get("trade_date")) if manifest["params"] == params]
    if len(matches) > 1:
        raise CollectionError(f"{endpoint}: ambiguous cached query schemas")
    return matches[0] if matches else None


def load_partitions(root: Path, endpoint: str) -> pd.DataFrame:
    """Read only checksum-verified, successfully completed raw partitions."""
    frames = [frame for frame, _ in iter_partitions(root, endpoint)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def audit_collection(root: Path, *, expected_dates: list[str] | None = None,
                     verify_checksums: bool = True) -> dict:
    """Report core endpoint readiness separately from optional-reference errors.

    This is safe during collection. Atomic manifests become visible only after
    a completed Parquet write. Set verify_checksums=False only for lightweight
    progress; normalizers must still verify each partition before consuming it.
    """
    root = Path(root)
    if expected_dates is None:
        calendars = [frame for frame, manifest in iter_partitions(root, "trade_cal")
                     if manifest["params"].get("exchange") == "SSE"]
        if not calendars:
            raise CollectionError("SSE calendar required for collection coverage audit")
        calendar = pd.concat(calendars, ignore_index=True)
        expected_dates = sorted(set(calendar.loc[pd.to_numeric(calendar["is_open"]).eq(1), "cal_date"].astype(str)))
    expected = set(expected_dates)
    endpoints = {}
    for endpoint in DATE_ENDPOINTS:
        observed, rows, errors = set(), 0, []
        for path in sorted((root / endpoint).glob("*.manifest.json")):
            manifest = json.loads(path.read_text())
            date = manifest.get("params", {}).get("trade_date")
            if manifest.get("status") != "complete" or date not in expected:
                continue
            artifact = root / manifest["artifact"]
            if not artifact.exists() or (verify_checksums and hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest["sha256"]):
                errors.append({"artifact": manifest["artifact"], "error": "checksum mismatch or missing file"})
                continue
            if date in observed:
                errors.append({"date": date, "error": "duplicate partition schemas"})
            observed.add(date)
            rows += manifest["rows"]
        missing = sorted(expected - observed)
        endpoints[endpoint] = {"status": "complete" if not missing and not errors else "incomplete",
                               "completed_dates": len(observed), "expected_dates": len(expected),
                               "missing_dates": missing, "rows": rows, "errors": errors}
    optional_errors, resolved_optional_errors = [], []
    for path in root.glob("*/*.error.json"):
        if path.parent.name not in DATE_ENDPOINTS:
            error = json.loads(path.read_text())
            sibling = path.with_name(path.name.removesuffix(".error.json") + ".manifest.json")
            recovered = False
            if sibling.exists():
                saved = json.loads(sibling.read_text())
                artifact = root / saved["artifact"]
                recovered = saved.get("status") in {"complete", "empty"} and artifact.exists()
                if recovered and verify_checksums:
                    recovered = hashlib.sha256(artifact.read_bytes()).hexdigest() == saved.get("sha256")
                if recovered and error.get("failed_at"):
                    recovered = saved.get("completed_at", "") >= error["failed_at"]
            (resolved_optional_errors if recovered else optional_errors).append(error)
    core_ready = all(state["status"] == "complete" for state in endpoints.values())
    return {"audited_at": _now(), "core_status": "complete" if core_ready else "incomplete",
            "overall_status": "complete" if core_ready and not optional_errors else "partial",
            "checksums_verified": verify_checksums, "endpoints": endpoints,
            "optional_errors": optional_errors, "resolved_optional_errors": resolved_optional_errors,
            "calendar_basis": "SSE trading sessions; BSE calendar not independently verified"}


def monthly_ranges(start: str, end: str) -> list[dict]:
    """Inclusive month-clipped ranges for endpoints supporting date intervals."""
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first > last:
        raise ValueError("start must not exceed end")
    result = []
    cursor = first
    while cursor <= last:
        month_end = min(cursor + pd.offsets.MonthEnd(0), last)
        result.append({"start_date": cursor.strftime("%Y%m%d"), "end_date": month_end.strftime("%Y%m%d")})
        cursor = month_end + pd.Timedelta(days=1)
    return result


def collect_market(*, root: Path, start: str, end: str, max_workers: int = 8,
                   requests_per_minute: float = 60, endpoints: tuple[str, ...] = DATE_ENDPOINTS,
                   newest_first: bool = True, max_days: int | None = None, include_references: bool = True) -> dict:
    """Run a resumable stock/date collection, stopping an endpoint on rejection."""
    if not 1 <= max_workers <= 12:
        raise ValueError("max_workers must be in 1..12")
    root = Path(root)
    client = HelperClient(requests_per_minute=requests_per_minute, control_path=root / "collection_control.json")
    for state in ("L", "D", "P"):
        collect_query(client, "stock_basic", {"list_status": state}, FIELDS["stock_basic"], root, allow_empty=state == "P")
    calendar = collect_query(client, "trade_cal", {"exchange": "SSE", "start_date": start, "end_date": end}, FIELDS["trade_cal"], root)
    cal = pd.read_parquet(root / calendar["artifact"])
    dates = sorted(cal.loc[pd.to_numeric(cal["is_open"]).eq(1), "cal_date"].astype(str), reverse=newest_first)
    if max_days is not None:
        dates = dates[:max_days]
    for code in ("000300.SH", "000905.SH", "000001.SH"):
        collect_query(client, "index_daily", {"ts_code": code, "start_date": start, "end_date": end}, FIELDS["index_daily"], root)
    stopped, failures, completed = set(), [], []
    lock = threading.Lock()
    def job(endpoint, params):
        date = params.get("trade_date") or params.get("start_date") or "all"
        with lock:
            if endpoint in stopped:
                return None
        try:
            result = collect_query(client, endpoint, params, FIELDS[endpoint], root, page_size=PAGE_SIZES.get(endpoint, 6000), allow_empty=endpoint == "stock_st")
            print(json.dumps({"endpoint": endpoint, "date": date, "status": result["status"], "rows": result["rows"]}), flush=True)
            return result
        except PermanentAPIError:
            with lock:
                stopped.add(endpoint)
            raise
    # Keep only a small bounded set in flight. A rejected endpoint is stopped
    # before future dates are submitted, while independent endpoints continue.
    def scheduled_queries():
        if include_references:
            _, _, prior_error = _partition_paths(root, "namechange", {}, FIELDS["namechange"])
            if not prior_error.exists():
                yield "namechange", {}
        years = sorted({date[:4] for date in dates}, reverse=newest_first)
        ranges = monthly_ranges(start, end) if include_references else []
        for year in years:
            for params in ranges:
                if params["start_date"].startswith(year):
                    yield "stock_st", params
            for date in dates:
                if date.startswith(year):
                    for endpoint in endpoints:
                        yield endpoint, {"trade_date": date}
    schedule = iter(scheduled_queries())
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending = {}
        def replenish():
            while len(pending) < max_workers:
                try:
                    endpoint, params = next(schedule)
                except StopIteration:
                    return
                if endpoint not in stopped:
                    pending[pool.submit(job, endpoint, params)] = (endpoint, params)
        replenish()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                endpoint, params = pending.pop(future)
                date = params.get("trade_date") or params.get("start_date") or "all"
                try:
                    result = future.result()
                    if result:
                        completed.append({"endpoint": endpoint, "date": date, "rows": result["rows"]})
                except CollectionError as error:
                    failures.append({"endpoint": endpoint, "date": date, "error": str(error)})
                    print(json.dumps({"endpoint": endpoint, "date": date, "status": "error", "error": str(error)}), flush=True)
            _write_json(root / "collection_progress.json", {"updated_at": _now(), "start": start, "end": end, "requested_dates": len(dates), "completed_partitions": len(completed), "stopped_endpoints": sorted(stopped), "failures": failures})
            replenish()
    report = {"updated_at": _now(), "start": start, "end": end, "requested_dates": len(dates), "completed_partitions": len(completed), "stopped_endpoints": sorted(stopped), "failures": failures, "status": "complete" if not failures and not stopped else "incomplete"}
    _write_json(root / "collection_run.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="20160101")
    parser.add_argument("--end", default="20260930")
    parser.add_argument("--raw-root", type=Path, default=Path("data/cn/universal/raw"))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--requests-per-minute", type=float, default=60)
    parser.add_argument("--endpoints", default=",".join(DATE_ENDPOINTS))
    parser.add_argument("--oldest-first", action="store_true")
    parser.add_argument("--max-days", type=int)
    parser.add_argument("--skip-references", action="store_true")
    args = parser.parse_args(argv)
    report = collect_market(root=args.raw_root, start=args.start, end=args.end, max_workers=args.max_workers, requests_per_minute=args.requests_per_minute, endpoints=tuple(args.endpoints.split(",")), newest_first=not args.oldest_first, max_days=args.max_days, include_references=not args.skip_references)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
