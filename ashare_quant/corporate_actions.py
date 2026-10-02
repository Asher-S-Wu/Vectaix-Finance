"""Selected-security cash-dividend evidence, never adjusted-factor share inference.

``verified`` means a provider final cash-only schedule passed the checks here.
It DOES NOT mean issuer-verified, historically complete point-in-time evidence,
or actual investor net/accounting cash. Every accepted row explicitly carries
``provider_final_schedule_simulated``, ``issuer_verified=False`` and the
conservative 20% cash tax-reserve convention used by ``replay.run_replay``.
Bonus/conversion shares, rights and identifiable differential distributions
remain unhandled. Tushare's dividend schema cannot rule out unflagged buyback-
excluded distributions or omitted historical revisions; consult issuer filings
before treating this output as an actual account reconstruction.

Only call the collector after the model/selection is frozen and core collection
has stopped. The helper subprocess owns authentication; this module neither
opens credential files nor logs helper responses. All tests are offline.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import pandas as pd

from .collect import (CollectionError, HelperClient, PermanentAPIError,
                      collect_query, is_a_share)

SOURCE_URL = 'https://tushare.pro/document/2?doc_id=103'
SOURCE_ENDPOINT = 'tushare.dividend'
DIVIDEND_FIELDS = ('ts_code,end_date,ann_date,div_proc,stk_div,stk_bo_rate,'
                   'stk_co_rate,cash_div,cash_div_tax,record_date,ex_date,'
                   'pay_date,div_listdate,imp_ann_date,base_date,base_share')
MAX_SELECTED_SECURITIES = 700
MAX_SOURCE_CODES = 1000
EVIDENCE_BASIS = 'provider_final_schedule_simulated'
LIMITATIONS = [
    'Provider history fetched after model selection is not proven point-in-time complete.',
    'Issuer filings have not independently verified these schedules.',
    'cash_div is provider after-tax cash, not investor-specific net cash.',
    'A 20% gross-cash reserve is a research assumption, not actual broker withholding.',
    'Bonus/conversion, rights and identified differential distributions are unhandled.',
    'The dividend endpoint cannot rule out unflagged buyback-excluded distributions, rights, or omitted revisions.',
    'Date-only final announcements become usable no earlier than the next calendar day.',
    'Empty, failed or capped history is incomplete evidence, never proof of no actions.',
]
ACTION_COLUMNS = [
    'action_id', 'security_id', 'ann_date', 'record_date', 'ex_date', 'pay_date',
    'share_list_date', 'cash_per_share_pre_tax', 'share_multiplier', 'verified',
    'source_url', 'source_endpoint', 'payment_basis', 'source_ann_date',
    'available_from', 'provider_cash_per_share_after_tax', 'tax_reserve_rate',
    'issuer_verified', 'investor_net_cash_verified', 'point_in_time_complete',
    'evidence_basis', 'publication_time_granularity', 'source_ts_code',
    'source_ts_codes_json', 'raw_records_json', 'version_resolution',
]
UNVERIFIED_COLUMNS = [
    'security_id', 'source_ts_code', 'ex_date', 'imp_ann_date', 'verified',
    'issuer_verified', 'source_endpoint', 'rejection_reasons', 'raw_record_json',
]


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in
                              {'', 'none', 'null', 'nan', 'nat', '<na>'}) or bool(pd.isna(value))


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if _missing(value):
        return None
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _json(value: Any) -> str:
    return json.dumps(_clean(value), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def _date(value: Any) -> str | None:
    if _missing(value):
        return None
    if isinstance(value, (date, datetime, pd.Timestamp)):
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is not None or stamp != stamp.normalize():
            raise ValueError('expected date-only source field')
        return stamp.date().isoformat()
    text = str(value).strip()
    if not (re.fullmatch(r'\d{8}', text) or re.fullmatch(r'\d{4}-\d{2}-\d{2}', text)):
        raise ValueError('invalid source date format')
    try:
        return datetime.strptime(text, '%Y%m%d' if len(text) == 8 else '%Y-%m-%d').date().isoformat()
    except ValueError:
        raise ValueError('invalid source date') from None


def _number(value: Any) -> float | None:
    if _missing(value) or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _alias_map(mapping: Mapping[str, str] | None) -> dict[str, str]:
    aliases = {str(old): str(new) for old, new in (mapping or {}).items()}
    resolved = {}
    for code in aliases:
        cursor, seen = code, set()
        while cursor in aliases and aliases[cursor] != cursor:
            if cursor in seen:
                raise ValueError('cyclic corporate-action alias mapping')
            seen.add(cursor)
            cursor = aliases[cursor]
        if not is_a_share(code) or not is_a_share(cursor):
            raise ValueError('alias mapping contains a non-A-share identity')
        resolved[code] = cursor
    return resolved


def _unsupported(row: dict) -> bool:
    for key in ('rights_ratio', 'rights_issue', 'differential_dividend',
                'is_differential', 'buyback_excluded', 'excluded_shares'):
        value = row.get(key)
        if _missing(value):
            continue
        if str(value).strip().lower() not in {'false', '0', '0.0', 'no', 'n'}:
            return True
    for key in ('distribution_note', 'notes', 'description', 'payment_basis', 'action_type'):
        value = row.get(key)
        if not _missing(value) and re.search(r'差异化|回购|配股|rights|differential|buyback|excluded', str(value), re.I):
            return True
    return False


def _candidate(row: dict, aliases: dict[str, str], asof: str | None) -> dict:
    source = str(row.get('ts_code', ''))
    item = dict(raw=row, source_ts_code=source,
                security_id=aliases.get(source, source), reasons=[], dates={})
    reasons, dates = item['reasons'], item['dates']
    if not is_a_share(source):
        reasons.append('invalid_security_id')
    if str(row.get('div_proc', '')).strip() != '实施':
        reasons.append('not_implemented')
    for field in ('end_date', 'ann_date', 'imp_ann_date', 'record_date', 'ex_date',
                  'pay_date', 'div_listdate', 'base_date'):
        try:
            dates[field] = _date(row.get(field))
        except (TypeError, ValueError, OverflowError):
            dates[field] = None
            reasons.append('invalid_' + field)
    for field, reason in [('imp_ann_date', 'missing_implementation_publication'),
                          ('record_date', 'missing_record_date'),
                          ('ex_date', 'missing_ex_date'),
                          ('pay_date', 'missing_payment_date')]:
        if dates[field] is None and 'invalid_' + field not in reasons:
            reasons.append(reason)
    publication, record, ex, pay = (dates[key] for key in
                                   ('imp_ann_date', 'record_date', 'ex_date', 'pay_date'))
    if publication and record and publication >= record:
        reasons.append('publication_not_before_record')
    if (record and ex and record >= ex) or (pay and ex and pay < ex):
        reasons.append('invalid_event_dates')
    if dates['ann_date'] and publication and dates['ann_date'] > publication:
        reasons.append('proposal_after_implementation_publication')
    if dates['div_listdate'] and ex and dates['div_listdate'] < ex:
        reasons.append('invalid_event_dates')
    try:
        available = (date.fromisoformat(publication) + timedelta(days=1)).isoformat() if publication else None
    except OverflowError:
        available = None
        reasons.append('invalid_imp_ann_date')
    item['available_from'] = available
    if asof and available and available > asof:
        reasons.append('not_yet_visible')
    gross, net = _number(row.get('cash_div_tax')), _number(row.get('cash_div'))
    item.update(gross=gross, net=net)
    if gross is None or gross < 0:
        reasons.append('invalid_gross_cash')
    elif gross == 0:
        reasons.append('no_cash_distribution')
    if not _missing(row.get('cash_div')) and (net is None or net < 0 or (gross is not None and net > gross + 1e-10)):
        reasons.append('inconsistent_provider_after_tax_cash')
    stock = [_number(row.get(key)) for key in ('stk_div', 'stk_bo_rate', 'stk_co_rate')]
    if stock[0] == 0.:
        # Tushare defines stk_div as the total bonus + conversion per share.
        # An explicit finite zero proves cash-only even when its decomposition
        # is omitted, but every supplied component must independently be zero.
        # Canonicalize only the economic signature; raw source nulls survive.
        if any(not _missing(row.get(key)) and value != 0.
               for key, value in zip(('stk_bo_rate', 'stk_co_rate'), stock[1:])):
            reasons.append('inconsistent_stock_distribution')
        else:
            stock = [0., 0., 0.]
        if dates['div_listdate']:
            reasons.append('ambiguous_stock_distribution')
    elif any(value is None for value in stock):
        reasons.append('missing_stock_distribution_fields')
    elif any(value < 0 for value in stock) or not math.isclose(stock[0], stock[1] + stock[2], abs_tol=1e-10, rel_tol=1e-8):
        reasons.append('inconsistent_stock_distribution')
    elif any(value != 0 for value in stock):
        reasons.append('unsupported_stock_distribution')
    elif dates['div_listdate']:
        reasons.append('ambiguous_stock_distribution')
    basis = _number(row.get('base_share'))
    if ((not _missing(row.get('base_share')) and (basis is None or basis <= 0)) or
            (dates['base_date'] and record and dates['base_date'] > record)):
        reasons.append('invalid_distribution_basis')
    if _unsupported(row):
        reasons.append('unsupported_rights_or_differential_event')
    # All economic/schedule fields are compared, including the provider's net
    # field and distribution share basis. Proposal dates are retained, not used
    # to pretend a final implementation was knowable at proposal time.
    item['signature'] = _json(dict(dates={k: v for k, v in dates.items() if k != 'ann_date'},
                                   gross=gross, net=net, stock=stock,
                                   base_share=row.get('base_share')))
    return item


def normalize_dividends(raw: pd.DataFrame, *, aliases: Mapping[str, str] | None = None,
                        asof_date: Any = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (accepted simulated schedules, retained unhandled rows, summary).

    The latest distinct implementation-publication version may supersede an
    earlier one only before entitlement and within the same fiscal event.
    Conflicting same-publication records, missing revision dates and late
    corrections fail closed. An optional date-asof cut uses next-day visibility;
    it cannot establish the vendor preserved every historical revision.
    """
    aliases = _alias_map(aliases)
    asof = _date(asof_date)
    rows = raw.to_dict('records') if isinstance(raw, pd.DataFrame) else [dict(r) for r in raw]
    groups, rejected, actions = defaultdict(list), [], []
    duplicates = 0

    def reject(item: dict, extra: str | None = None) -> None:
        reasons = list(dict.fromkeys(item['reasons'] + ([extra] if extra else [])))
        rejected.append(dict(security_id=item['security_id'], source_ts_code=item['source_ts_code'],
                             ex_date=item['dates'].get('ex_date'),
                             imp_ann_date=item['dates'].get('imp_ann_date'),
                             verified=False, issuer_verified=False, source_endpoint=SOURCE_ENDPOINT,
                             rejection_reasons=';'.join(reasons), raw_record_json=_json(item['raw'])))

    candidates = [_candidate(row, aliases, asof) for row in rows]
    periods = defaultdict(set)
    for item in candidates:
        if not {'not_implemented', 'not_yet_visible'}.intersection(item['reasons']):
            if item['dates']['end_date']:
                periods[(item['security_id'], item['dates']['end_date'])].add(item['dates']['ex_date'])
    for item in candidates:
        # Without a stable provider event ID, two ex-dates for one fiscal period
        # could be a revised schedule. Do not manufacture two entitlements.
        if len(periods.get((item['security_id'], item['dates']['end_date']), set())) > 1:
            item['reasons'].append('ambiguous_period_schedule')
        if 'not_implemented' in item['reasons'] or 'not_yet_visible' in item['reasons'] or not item['dates']['ex_date']:
            reject(item)
        else:
            groups[(item['security_id'], item['dates']['ex_date'])].append(item)

    for (sid, ex), group in sorted(groups.items()):
        # Any undated implementation version makes ordering unknowable.
        publications = {item['dates']['imp_ann_date'] for item in group}
        conflict = len(group) > 1 and (None in publications or any(
            'publication_not_before_record' in item['reasons'] for item in group))
        if len({item['dates']['end_date'] for item in group}) > 1:
            conflict = True
        latest = max((value for value in publications if value), default=None)
        if latest and any(item['dates']['record_date'] and
                          latest >= item['dates']['record_date'] for item in group):
            conflict = conflict or len(group) > 1
        chosen = [item for item in group if item['dates']['imp_ann_date'] == latest]
        if len({item['signature'] for item in chosen}) > 1:
            conflict = True
        if conflict:
            for item in group:
                reject(item, 'conflicting_final_versions')
            continue
        # Never choose a passing alias over an invalid final record, even when
        # invalid numeric values normalized to the same missing representation.
        if any(item['reasons'] for item in chosen):
            for item in group:
                reject(item, 'invalid_final_version' if not item['reasons'] else None)
            continue
        for item in group:
            if item['dates']['imp_ann_date'] != latest:
                reject(item, 'superseded_provider_version')
        duplicates += len(chosen) - 1
        preferred = sorted(chosen, key=lambda item: (item['source_ts_code'] != sid, item['source_ts_code']))[0]
        dates, source = preferred['dates'], preferred['raw']
        action_id = hashlib.sha256(f'{sid}|{ex}|{preferred["signature"]}'.encode()).hexdigest()[:24]
        actions.append(dict(
            action_id=action_id, security_id=sid, ann_date=dates['imp_ann_date'],
            record_date=dates['record_date'], ex_date=ex, pay_date=dates['pay_date'],
            share_list_date=None, cash_per_share_pre_tax=preferred['gross'], share_multiplier=1.,
            verified=True, source_url=SOURCE_URL, source_endpoint=SOURCE_ENDPOINT,
            payment_basis='gross_cash_with_20pct_conservative_tax_reserve',
            source_ann_date=source.get('ann_date'), available_from=preferred['available_from'],
            provider_cash_per_share_after_tax=preferred['net'], tax_reserve_rate=.20,
            issuer_verified=False, investor_net_cash_verified=False, point_in_time_complete=False,
            evidence_basis=EVIDENCE_BASIS, publication_time_granularity='date_only',
            source_ts_code=preferred['source_ts_code'],
            source_ts_codes_json=_json(sorted({item['source_ts_code'] for item in chosen})),
            raw_records_json=_json([item['raw'] for item in chosen]),
            version_resolution='latest_dated_pre_record_version' if len(publications) > 1 else 'single_equivalent_final_schedule'))
    reasons = Counter(reason for row in rejected for reason in row['rejection_reasons'].split(';') if reason)
    summary = dict(raw_rows=len(rows), accepted_events=len(actions), unhandled_rows=len(rejected),
                   equivalent_duplicate_rows=duplicates, rejection_counts=dict(sorted(reasons.items())),
                   evidence_basis=EVIDENCE_BASIS, issuer_verified=False,
                   actual_accounting_verified=False, point_in_time_complete=False,
                   asof_date=asof, limitations=list(LIMITATIONS))
    return (pd.DataFrame(actions, columns=ACTION_COLUMNS),
            pd.DataFrame(rejected, columns=UNVERIFIED_COLUMNS), summary)


def _load_aliases(root: Path) -> tuple[dict[str, str], list[dict]]:
    aliases, references = {}, []
    for name in ('verified_code_changes', 'bse_mapping'):
        path = root / 'references' / f'{name}.parquet'
        if not path.exists() and name == 'verified_code_changes':
            path = Path(__file__).resolve().parents[1] / 'docs' / 'ashare' / 'verified_code_changes.csv'
        if not path.exists():
            references.append(dict(name=name, status='absent'))
            continue
        frame = pd.read_csv(path) if path.suffix == '.csv' else pd.read_parquet(path)
        if not {'o_code', 'n_code'}.issubset(frame):
            raise ValueError(f'{name}: missing alias columns')
        if name == 'verified_code_changes':
            if 'verified' not in frame or not frame['verified'].eq(True).all():
                raise ValueError('non-BSE code-change references require verified identities')
        for row in frame.to_dict('records'):
            old, new = str(row['o_code']), str(row['n_code'])
            if old in aliases and aliases[old] != new:
                raise ValueError('ambiguous corporate-action alias mapping')
            aliases[old] = new
        references.append(dict(name=name, status='loaded', rows=len(frame), format=path.suffix[1:],
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return _alias_map(aliases), references


def _save_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + '.tmp')
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def collect_selected_dividends(data_root: Path, selected_security_ids: Iterable[str], *,
                               client: HelperClient | None = None,
                               model_frozen: bool = False,
                               core_collection_stopped: bool = False,
                               asof_date: Any = None,
                               max_selected_securities: int = MAX_SELECTED_SECURITIES) -> dict:
    """Collect frozen canonical identities plus aliases under explicit scope caps.

    Defaults to 700 canonical identities, configurable up to the hard maximum
    of 1000 source codes including aliases. Oversize selections are never trimmed.
    Default transport uses the approved helper with 30 requests/minute. A client
    may be injected for offline testing. Per-code history queries use only the
    documented ts_code filter; the existing collector audits offset pagination
    at the endpoint's 2000-row cap and requires a short terminal page. Permission
    failures, corruption and empty responses are recorded without blocking the
    already-frozen model. Returned ``collection_status`` covers transport only;
    unhandled rows and evidence limitations remain separate and explicit.
    """
    if model_frozen is not True:
        raise ValueError('model selection must be frozen before dividend collection')
    if core_collection_stopped is not True:
        raise ValueError('core collection must be stopped before dividend collection')
    _date(asof_date)  # Reject an invalid visibility cut before any requests.
    if not isinstance(max_selected_securities, int) or not 1 <= max_selected_securities <= MAX_SOURCE_CODES:
        raise ValueError('max_selected_securities must be between 1 and 1000')
    root = Path(data_root)
    selected = sorted(set(str(s) for s in selected_security_ids))
    if not selected or any(not is_a_share(s) for s in selected):
        raise ValueError('a nonempty explicit A-share selection is required')
    aliases, references = _load_aliases(root)
    selected = sorted({aliases.get(s, s) for s in selected})
    sources = sorted(set(selected) | {old for old, new in aliases.items() if new in selected})
    scope_blocker = ('selected_identity_limit' if len(selected) > max_selected_securities else
                     'source_code_hard_limit' if len(sources) > MAX_SOURCE_CODES else None)
    client = client if client is not None else HelperClient(requests_per_minute=30)
    if isinstance(client, HelperClient) and client.interval < 2.:
        raise ValueError('dividend HelperClient budget must be at most 30 requests/minute')
    frames, manifests, errors, empty, affected = [], [], [], [], set()
    unattempted = list(sources) if scope_blocker else []
    if scope_blocker:
        affected.update(selected)
    raw_root = root / 'raw'
    for index, code in enumerate([] if scope_blocker else sources):
        sid = aliases.get(code, code)
        try:
            manifest = collect_query(client, 'dividend', {'ts_code': code}, DIVIDEND_FIELDS,
                                     raw_root, page_size=2000, allow_empty=True,
                                     keys=tuple(DIVIDEND_FIELDS.split(',')), max_pages=20)
            manifests.append(manifest)
            frame = pd.read_parquet(raw_root / manifest['artifact'])
            returned = set(frame['ts_code'].astype(str))
            if any(aliases.get(value, value) != sid for value in returned):
                raise CollectionError('dividend: returned an unrelated security identity')
            if frame.empty:
                empty.append(code)
                affected.add(sid)
            else:
                frame['query_ts_code'] = code
                frame['source_manifest_artifact'] = manifest['artifact']
                frame['source_fetched_at'] = manifest['completed_at']
                frames.append(frame)
        except (CollectionError, ValueError, OSError) as error:
            # Do not expose arbitrary transport exception text or helper output.
            errors.append(dict(source_ts_code=code, security_id=sid,
                               error_type=type(error).__name__, status='incomplete'))
            affected.add(sid)
            if isinstance(error, PermanentAPIError):
                unattempted = sources[index + 1:]
                affected.update(aliases.get(value, value) for value in unattempted)
                break
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    actions, unhandled, summary = normalize_dividends(raw, aliases=aliases, asof_date=asof_date)
    refs = root / 'references'
    refs.mkdir(parents=True, exist_ok=True)
    _save_parquet(refs / 'verified_corporate_actions.parquet', actions)
    _save_parquet(refs / 'unverified_corporate_actions.parquet', unhandled)
    summary.update(format_version=1, collected_at=datetime.now(timezone.utc).isoformat(),
                   collection_status='incomplete' if errors or empty or scope_blocker else 'complete',
                   scope_blocker=scope_blocker, max_selected_securities=max_selected_securities,
                   max_source_codes=MAX_SOURCE_CODES, unattempted_source_codes=unattempted,
                   selected_security_ids=selected, selected_canonical_count=len(selected),
                   requested_source_codes=sources,
                   queried_source_codes=[code for code in sources if code not in set(unattempted)],
                   empty_source_codes=empty,
                   partition_errors=errors, affected_security_ids=sorted(affected),
                   requests_per_minute=30, endpoint_page_cap=2000,
                   model_frozen=True, core_collection_stopped=True,
                   source_endpoint=SOURCE_ENDPOINT, source_url=SOURCE_URL,
                   alias_references=references, partitions=manifests,
                   output_sha256={name: hashlib.sha256((refs / name).read_bytes()).hexdigest()
                                  for name in ('verified_corporate_actions.parquet', 'unverified_corporate_actions.parquet')},
                   verified_actions_path=str(refs / 'verified_corporate_actions.parquet'),
                   unverified_actions_path=str(refs / 'unverified_corporate_actions.parquet'))
    path = refs / 'corporate_actions_provenance.json'
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(_json(summary) + '\n', encoding='utf-8')
    os.replace(temporary, path)
    return summary
