from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CORE_METRICS = ('revenue', 'net_income_attributable', 'basic_eps')
FUNDAMENTAL_FEATURE_NAMES = {
    'revenue_growth': '当期营收与同公告上年同期营收的对称增长',
    'profit_growth': '当期归母净利润与同公告上年同期归母净利润的对称增长',
    'eps_growth': '当期基本每股收益与同公告上年同期基本每股收益的对称增长',
    'profit_margin_change': '当期归母净利润与营业收入比例相对同公告上年同期的变化',
}
FUNDAMENTAL_KEYS = tuple(FUNDAMENTAL_FEATURE_NAMES)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_document_metrics(document: dict) -> dict:
    selected = {}
    for name in CORE_METRICS:
        for comparison in ('current', 'prior'):
            matches = [m for m in document['metrics'] if m['metric'] == name and m['comparison'] == comparison]
            if len(matches) != 1:
                raise ValueError(f'{document["eventId"]} 财务字段 {name}/{comparison} 不唯一。')
            metric = matches[0]
            if (metric['status'] != 'verified' or metric['scope'] != 'total_operations'
                    or metric['sourceEventId'] != document['eventId']
                    or metric['sourcePublishedAt'] != document['publishedAt']
                    or metric['comparisonBasis'] != 'as_presented_in_this_publication'
                    or not isinstance(metric['page'], int) or not 1 <= metric['page'] <= document['pages']
                    or not metric['evidence'] or not metric['periodAndUnitEvidence']):
                raise ValueError(f'{document["eventId"]} 核心财务字段未经核验或证据不足。')
            value, multiplier = metric['value'], metric['unitMultiplier']
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or multiplier not in (0.01, 1, 1000, 1000000, 100000000)):
                raise ValueError('财务数值或单位倍数无效。')
            if name == 'net_income_attributable' and metric['attribution'] != 'equity_holders_of_parent':
                raise ValueError('利润必须明确归属母公司股东。')
            period = metric['period']
            start, end = pd.Timestamp(period['start']), pd.Timestamp(period['end'])
            months = 12 if document['kind'] == 'annual' else 6
            if (period['durationMonths'] != months or end < start or start.month != 1 or start.day != 1
                    or end.year != start.year or end.month != months or end.day != (31 if months == 12 else 30)
                    or pd.Timestamp(document['publishedAt']).tz_localize(None) <= end):
                raise ValueError('财务期间或公布时间无效。')
            selected[name, comparison] = metric
        current, prior = selected[name, 'current'], selected[name, 'prior']
        if (current['currency'] != prior['currency']
                or pd.Timestamp(current['period']['end']) - pd.DateOffset(years=1) != pd.Timestamp(prior['period']['end'])):
            raise ValueError('财务同比期间或币种不可比。')
    for comparison in ('current', 'prior'):
        records = [selected[name, comparison] for name in CORE_METRICS]
        if any(m['period'] != records[0]['period'] for m in records):
            raise ValueError('同一公告核心财务期间不一致。')
        if selected['revenue', comparison]['currency'] != selected['net_income_attributable', comparison]['currency']:
            raise ValueError('利润率的营收和利润币种不同。')
    return selected


def add_fundamental_factors(dataset: dict, data_dir: Path) -> dict:
    """Four-stock dataset: only the latest publicly known result is admitted."""
    data_dir = Path(data_dir)
    targets = dataset['target_symbols']
    event_bytes = (data_dir / 'events.json').read_bytes()
    events = json.loads(event_bytes)
    hashes = dict(dataset['provenance']['fileHashes'])
    if hashes['events.json'] != _digest(event_bytes):
        raise ValueError('财务整合需要已验证且未变更的业绩事件数据集。')
    # Existing event alignment already verifies source HTML, actual close, and release times.
    if 'resultsEventId' not in dataset['factors']:
        raise ValueError('财务因子必须在已验证的公告时间对齐后构造。')
    by_event = {event['eventId']: event for security in events['stocks'] for event in security['events']}
    rows, audits = [], []
    for symbol in targets:
        filename = f'fundamentals/{symbol}.json'
        raw = (data_dir / filename).read_bytes()
        hashes[filename] = _digest(raw)
        manifest = json.loads(raw)
        if (manifest['schemaVersion'] != 1 or manifest['symbol'] != symbol
                or manifest['source'] != 'HKEXnews' or manifest['researchReady'] is not True
                or manifest['sourceEventsSha256'] != hashes['events.json']
                or not set(CORE_METRICS).issubset(manifest['approvedMetrics'])):
            raise ValueError(f'{symbol} 财务封存或准入状态无效。')
        expected = {key for key, event in by_event.items() if event['symbol'] == symbol}
        documents = manifest['documents']
        if len(documents) != len(expected) or {d['eventId'] for d in documents} != expected:
            raise ValueError(f'{symbol} 财务公告覆盖不完整或重复。')
        excluded, admitted = [], 0
        for document in documents:
            event = by_event[document['eventId']]
            if any(document[key] != event[key] for key in ('symbol', 'publishedAt', 'kind', 'url', 'title')):
                raise ValueError('财务原公告身份与已验证事件不一致。')
            for file_key, hash_key in (('pdfFile', 'pdfSha256'), ('textFile', 'textSha256')):
                name = document[file_key]
                if not name.startswith(f'raw/fundamentals/{symbol}/') or '..' in Path(name).parts:
                    raise ValueError('财务证据路径不正确。')
                content = (data_dir / name).read_bytes()
                hashes[name] = _digest(content)
                if hashes[name] != document[hash_key]:
                    raise ValueError('财务原PDF或文本指纹不一致。')
            metrics = _read_document_metrics(document)
            values = {(name, comparison): m['value'] * m['unitMultiplier'] for (name, comparison), m in metrics.items()}
            denominators = {name: abs(values[name, 'current']) + abs(values[name, 'prior']) for name in CORE_METRICS}
            if any(value == 0 for value in denominators.values()) or any(values['revenue', c] <= 0 for c in ('current', 'prior')):
                excluded.append({'eventId': event['eventId'], 'reason': '同比双零或营收非正，数值因子无法按固定定义计算'})
                continue
            growth = {name: 2 * (values[name, 'current'] - values[name, 'prior']) / denominators[name] for name in CORE_METRICS}
            rows.append({
                'symbol': symbol, 'resultsEventId': event['eventId'],
                'revenue_growth': growth['revenue'], 'profit_growth': growth['net_income_attributable'],
                'eps_growth': growth['basic_eps'],
                'profit_margin_change': (values['net_income_attributable', 'current'] / values['revenue', 'current']
                                         - values['net_income_attributable', 'prior'] / values['revenue', 'prior']),
            })
            admitted += 1
        audits.append({'symbol': symbol, 'documents': len(documents), 'admittedDocuments': admitted, 'excludedDocuments': excluded})
    observations = pd.DataFrame(rows)
    factors = dataset['factors'].loc[dataset['factors'].symbol.isin(targets)].merge(
        observations, on=['symbol', 'resultsEventId'], how='inner', validate='many_to_one')
    labels = dataset['labels'].loc[dataset['labels'].symbol.isin(targets)].merge(
        observations, on=['symbol', 'resultsEventId'], how='inner', validate='many_to_one')
    if set(factors.symbol) != set(targets) or not np.isfinite(factors[list(FUNDAMENTAL_KEYS)].to_numpy()).all():
        raise ValueError('四股财务因子覆盖或数值不完整。')
    names = {**dataset['feature_names'], **FUNDAMENTAL_FEATURE_NAMES}
    groups = {**dataset['feature_groups'], 'fundamental_compact': (*dataset['feature_groups']['event_compact'], *FUNDAMENTAL_KEYS)}
    provenance = {**dataset['provenance'], 'fileHashes': hashes,
                  'sha256': _digest(json.dumps(hashes, sort_keys=True).encode()),
                  'peerSymbols': [], 'sourcePeerSymbols': dataset['peer_symbols'],
                  'fundamentalDocuments': audits, 'factorRows': len(factors), 'labelRows': len(labels),
                  'featureCount': len(names), 'featureGroupCounts': {key: len(value) for key, value in groups.items()},
                  'fundamentalRule': '只使用当前已公布公告的当期和该公告同期比较列；对称增长=2*(当期-同期)/(绝对当期+绝对同期)。最新公告无有效值则排除，不沿用更早公告。',
                  'trainingUniverseRule': '财务研究只含四只已有核验原公告的目标股票；不补同业财务。'}
    return {**dataset, 'factors': factors, 'labels': labels, 'peer_symbols': [],
            'feature_names': names, 'feature_groups': groups, 'provenance': provenance}
