from __future__ import annotations

import hashlib
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import exchange_calendars as exchange
import numpy as np
import pandas as pd

from .data import CALENDAR_VERSION, DATA_END, DATA_START, research_calendar
from .lab_data import SECTOR_ETFS


EVENT_FEATURE_NAMES = {
    'result_age': '距最近已公布业绩的交易日数，封顶126日后归一化',
    'recent_results_5': '最近5个交易日内公布业绩',
    'recent_results_21': '最近21个交易日内公布业绩',
    'annual_results_21': '最近21个交易日内公布年度业绩',
    'interim_results_21': '最近21个交易日内公布中期业绩',
    'recent_results_reaction': '业绩有效日前一收盘至当前收盘的21日窗口内累计收益',
}
EVENT_KEYS = tuple(EVENT_FEATURE_NAMES)
EVENT_METADATA = ('resultsPublishedAt', 'resultsEffectiveDate', 'resultsSignalAt', 'resultsEventId')
CATEGORY_CODES = {'annual': '13300', 'interim': '13400'}
OFFICIAL_HOSTS = {'www.hkexnews.hk', 'www1.hkexnews.hk', 'www2.hkexnews.hk'}


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _text(value: str) -> str:
    return ' '.join(html.unescape(re.sub(r'<[^>]*>', ' ', value)).split())


class _Element:
    def __init__(self, tag: str, attributes: dict):
        self.tag = tag
        self.attributes = attributes
        self.children: list[_Element | str] = []

    def text(self) -> str:
        return ' '.join(' '.join(child.text() if isinstance(child, _Element) else child
                                 for child in self.children).split())

    def elements(self, tag: str, class_name: str | None = None) -> list[_Element]:
        found = []
        for child in self.children:
            if isinstance(child, _Element):
                if child.tag == tag and (class_name is None or class_name in child.attributes.get('class', '').split()):
                    found.append(child)
                found.extend(child.elements(tag, class_name))
        return found

    def one(self, tag: str, class_name: str) -> _Element:
        found = self.elements(tag, class_name)
        if len(found) != 1:
            raise ValueError(f'港交所原响应中 {class_name} 结构不唯一或缺失。')
        return found[0]


class _ResponseParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Element('document', {})
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs):
        element = _Element(tag, dict(attrs))
        self.stack[-1].children.append(element)
        if tag not in {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}:
            self.stack.append(element)

    def handle_endtag(self, tag: str):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str):
        self.stack[-1].children.append(data)


def _official_url(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError('公告来源链接必须是字符串。')
    parsed = urlparse(value)
    if parsed.scheme != 'https' or parsed.hostname not in OFFICIAL_HOSTS or parsed.username or parsed.password:
        raise ValueError('公告来源必须是港交所官方 HTTPS 链接。')
    return parsed.path


def _parse_response(raw: bytes, query: dict, symbol: str) -> list[dict]:
    parser = _ResponseParser()
    parser.feed(raw.decode('utf-8'))
    parser.close()
    page = parser.root
    inputs = {}
    for element in page.elements('input'):
        attributes = element.attributes
        if 'id' in attributes and 'value' in attributes:
            inputs[attributes['id']] = attributes['value']
    expected = {'startDate': query['from'], 'endDate': query['to'], 'searchTypeInt': '1',
                'tierOneId': '10000', 'tierTwoId': query['officialCategoryCode'], 'tierTwoGpId': '3'}
    if any(inputs.get(key) != value for key, value in expected.items()) or not inputs.get('stockCode', '').startswith(symbol + ' '):
        raise ValueError(f'{symbol} 公告原响应的证券、时间窗口或查询类别不一致。')
    reported = re.findall(r'顯示\s*([\d,]+)\s*紀錄，\s*共有\s*([\d,]+)\s*紀錄', page.text())
    if not reported or any((int(a.replace(',', '')), int(b.replace(',', ''))) != (query['reportedCount'], query['totalCount']) for a, b in reported):
        raise ValueError(f'{symbol} 公告原响应未证实完整的返回数量。')
    rows = []
    for tr in page.elements('tr'):
        if not tr.elements('td', 'release-time'):
            continue
        timestamp = re.findall(r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}', tr.one('td', 'release-time').text())
        codes = re.findall(r'(?<!\d)\d{5}(?!\d)', tr.one('td', 'stock-short-code').text())
        categories = re.findall(r'\[([^\[\]]*)\]', tr.one('div', 'headline').text())
        links = tr.one('div', 'doc-link').elements('a')
        if len(timestamp) != 1 or symbol not in codes or len(categories) != 1 or len(links) != 1:
            raise ValueError(f'{symbol} 公告原响应结果行缺少唯一时间、类别或文档链接。')
        published = pd.Timestamp(pd.to_datetime(timestamp[0], format='%d/%m/%Y %H:%M')).tz_localize('Asia/Hong_Kong')
        path = links[0].attributes['href']
        if not path.startswith('/listedco/listconews/sehk/') or not path.lower().endswith('.pdf'):
            raise ValueError(f'{symbol} 公告文档不是原响应的港股公告 PDF。')
        rows.append({'publishedAt': published, 'title': links[0].text(), 'path': path,
                     'officialCategories': [part.strip() for part in categories[0].split(' / ')]})
    if len(rows) != query['parsedRows'] or query['parsedRows'] != query['reportedCount'] or query['reportedCount'] != query['totalCount']:
        raise ValueError(f'{symbol} 公告原响应未被完整解析。')
    return rows


def _read_events(data_dir: Path, symbols: list[str]) -> tuple[dict, dict[str, pd.DataFrame], dict[str, str]]:
    manifest_raw = (data_dir / 'events.json').read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest['schemaVersion'] != 1 or manifest['researchReady'] is not True:
        raise ValueError('公告清单尚未通过研究审查。')
    if manifest['start'] != DATA_START or manifest['end'] != DATA_END:
        raise ValueError('公告源窗口与封存研究行情不一致。')
    stocks = manifest['stocks']
    if len(stocks) != 12 or {s['symbol'] for s in stocks} != set(symbols):
        raise ValueError('公告清单必须完整且唯一覆盖十二只研究股票。')
    hashes, raw_files = {'events.json': _digest(manifest_raw)}, {}
    if not isinstance(manifest['rawFiles'], dict) or not manifest['rawFiles']:
        raise ValueError('公告清单没有可核验的原始响应。')
    for name, expected_hash in manifest['rawFiles'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.parts[:2] != ('raw', 'events'):
            raise ValueError('公告原始文件必须位于 data/raw/events 内。')
        if not isinstance(expected_hash, str) or re.fullmatch(r'[0-9a-f]{64}', expected_hash) is None:
            raise ValueError(f'{name} 没有有效 SHA256 指纹。')
        raw_files[name] = (data_dir / path).read_bytes()
        hashes[name] = _digest(raw_files[name])
        if hashes[name] != expected_hash:
            raise ValueError(f'{name} 公告原响应与封存指纹不一致。')
    by_symbol = {}
    for stock in stocks:
        symbol, coverage = stock['symbol'], stock['coverage']
        if coverage['complete'] is not True or coverage['requestedStart'] != DATA_START or coverage['requestedEnd'] != DATA_END:
            raise ValueError(f'{symbol} 公告历史覆盖尚未证实完整。')
        queries, responses = coverage['queries'], {}
        if not queries:
            raise ValueError(f'{symbol} 公告历史没有完整查询记录。')
        for query in queries:
            kind = query['kind']
            if kind not in CATEGORY_CODES or query['officialCategoryCode'] != CATEGORY_CODES[kind] or query['complete'] is not True:
                raise ValueError(f'{symbol} 公告查询类别或完整性不一致。')
            if any(type(query[key]) is not int or query[key] < 0 for key in ('parsedRows', 'reportedCount', 'totalCount')):
                raise ValueError(f'{symbol} 公告查询数量必须是非负整数。')
            _official_url(query['sourceURL'])
            name = query['rawFile']
            if name not in raw_files or name in responses:
                raise ValueError(f'{symbol} 公告查询原响应未封存或重复。')
            responses[name] = (query, _parse_response(raw_files[name], query, symbol))
        for kind in CATEGORY_CODES:
            cursor = pd.Timestamp(DATA_START)
            windows = sorted((pd.Timestamp(q['from']), pd.Timestamp(q['to'])) for q in queries if q['kind'] == kind)
            for start, end in windows:
                if start != cursor or end < start or end > pd.Timestamp(DATA_END):
                    raise ValueError(f'{symbol} {kind} 公告查询存在窗口缺口、重叠或越界。')
                cursor = end + pd.Timedelta(days=1)
            if cursor != pd.Timestamp(DATA_END) + pd.Timedelta(days=1):
                raise ValueError(f'{symbol} {kind} 公告查询未覆盖全部研究日期。')
        records, identities, source_rows = [], set(), set()
        admitted_ids = {event['eventId'] for event in stock['events']}
        original_ids = admitted_ids | {event['eventId'] for event in stock['excludedEvents']
                                       if event['reason'] not in {'duplicate_cross_category', 'title_replacement_same_document'}}
        classified = [(event, True) for event in stock['events']] + [(event, False) for event in stock['excludedEvents']]
        for event, admitted in classified:
            kind = event['kind']
            if kind not in CATEGORY_CODES:
                raise ValueError(f'{symbol} 公告事件只能为年度或中期业绩。')
            if 'symbol' in event and event['symbol'] != symbol:
                raise ValueError(f'{symbol} 公告事件证券代码不一致。')
            if not isinstance(event['publishedAt'], str) or not event['publishedAt'].endswith('+08:00'):
                raise ValueError(f'{symbol} 公告必须提供明确香港时区的实际发放时间。')
            published = pd.Timestamp(event['publishedAt'])
            if pd.isna(published) or published.date().isoformat() < DATA_START or published.date().isoformat() > DATA_END:
                raise ValueError(f'{symbol} 公告发放时间超出源窗口。')
            event_id = event['eventId']
            if not isinstance(event_id, str) or not event_id.strip() or (admitted and event_id in identities):
                raise ValueError(f'{symbol} 公告身份缺失或重复。')
            name, row = event['sourceFile'], event['sourceRow']
            if name not in responses or type(row) is not int or not 0 <= row < len(responses[name][1]) or (name, row) in source_rows:
                raise ValueError(f'{symbol} 公告原响应行缺失或重复。')
            query, parsed_rows = responses[name]
            parsed = parsed_rows[row]
            path = _official_url(event['url'])
            if ((admitted and query['kind'] != kind) or published != parsed['publishedAt'] or _text(event['title']) != parsed['title']
                    or path != parsed['path'] or event['officialCategories'] != parsed['officialCategories']):
                raise ValueError(f'{symbol} 公告内容与对应原响应行不一致。')
            if not pd.Timestamp(query['from']).date() <= published.date() <= pd.Timestamp(query['to']).date():
                raise ValueError(f'{symbol} 公告发放日期超出对应原响应的查询窗口。')
            source_rows.add((name, row))
            if admitted:
                identities.add(event_id)
                records.append({'publishedAt': published, 'kind': kind, 'eventId': event_id, 'documentPath': path})
            else:
                if not isinstance(event['reason'], str) or not event['reason'].strip():
                    raise ValueError(f'{symbol} 排除的原公告行必须有明确原因。')
                if event['reason'] == 'duplicate_cross_category' and event['canonicalEventId'] not in original_ids:
                    raise ValueError(f'{symbol} 公告重复来源没有关联已明确分类的原始记录。')
                if event['reason'] == 'title_replacement_same_document' and event['canonicalEventId'] not in admitted_ids:
                    raise ValueError(f'{symbol} 公告标题修订没有关联原始入选事件。')
        expected_rows = {(name, row) for name, (_, rows) in responses.items() for row in range(len(rows))}
        if source_rows != expected_rows:
            raise ValueError(f'{symbol} 原公告结果没有逐行归入正式事件或明确排除记录。')
        if not records:
            raise ValueError(f'{symbol} 没有可用于研究的已公布业绩事件。')
        events = pd.DataFrame(records).sort_values(['publishedAt', 'eventId'], kind='stable').reset_index(drop=True)
        if events.documentPath.duplicated().any():
            raise ValueError(f'{symbol} 公告文档重复。')
        by_symbol[symbol] = events
    return manifest, by_symbol, hashes


def add_event_factors(dataset: dict, data_dir: Path) -> dict:
    """Add six fixed, publication-time-aware event factors to a validated lab dataset."""
    data_dir = Path(data_dir)
    symbols = list(dataset['target_symbols']) + list(dataset['peer_symbols'])
    if len(symbols) != 12 or set(symbols) != set(SECTOR_ETFS):
        raise ValueError('业绩事件因子仅适用于本轮固定十二股研究池。')
    calendar = pd.DatetimeIndex(dataset['calendar'])
    if exchange.__version__ != CALENDAR_VERSION or not calendar.equals(research_calendar()):
        raise ValueError('业绩事件因子必须沿用封存的完整港股交易日历。')
    original_factors, original_labels = dataset['factors'], dataset['labels']
    if any(key in original_factors.columns or key in dataset['feature_names'] for key in EVENT_KEYS):
        raise ValueError('不能重复加入业绩事件因子。')
    for name, frame in (('因子', original_factors), ('标签', original_labels)):
        if frame.duplicated(['date', 'symbol']).any() or not frame.date.isin(calendar).all() or set(frame.symbol) != set(symbols):
            raise ValueError(f'原研究{name}必须包含十二股唯一的正式交易日记录。')
    manifest, events_by_symbol, event_hashes = _read_events(data_dir, symbols)
    trading_calendar = exchange.get_calendar('XHKG', start=DATA_START, end=DATA_END)
    signals = pd.DatetimeIndex(trading_calendar.schedule.loc[calendar, 'close']) + pd.Timedelta(minutes=10)
    signals = signals.tz_convert('Asia/Hong_Kong').as_unit('ns')
    frames, audits = [], []
    for symbol in symbols:
        events = events_by_symbol[symbol]
        published = pd.DatetimeIndex(events.publishedAt).as_unit('ns')
        effective = np.searchsorted(signals.asi8, published.asi8, side='right')
        latest = np.searchsorted(published.asi8, signals.asi8, side='left') - 1
        positions = np.flatnonzero(latest >= 0)
        selected = latest[positions]
        ages = positions - effective[selected]
        if (ages < 0).any() or not (published.asi8[selected] < signals.asi8[positions]).all():
            raise ValueError(f'{symbol} 业绩信息尚未在信号时点之前公布。')
        values = pd.DataFrame({'date': calendar[positions], 'symbol': symbol})
        values['result_age'] = np.minimum(ages, 126) / 126
        values['recent_results_5'] = (ages < 5).astype(float)
        in_window = ages < 21
        values['recent_results_21'] = in_window.astype(float)
        for kind, key in (('annual', 'annual_results_21'), ('interim', 'interim_results_21')):
            indices = np.flatnonzero(events.kind.to_numpy() == kind)
            known = np.searchsorted(published.asi8[indices], signals.asi8[positions], side='left') - 1
            present = known >= 0
            active = np.zeros(len(positions), dtype=bool)
            active[present] = positions[present] - effective[indices[known[present]]] < 21
            values[key] = active.astype(float)
        price_rows = dataset['prices'].loc[dataset['prices'].symbol == symbol]
        if price_rows.date.duplicated().any():
            raise ValueError(f'{symbol} 业绩价格来源日期重复。')
        prices_on_calendar = price_rows.set_index('date').reindex(calendar)
        close = pd.to_numeric(prices_on_calendar.research_close, errors='raise').to_numpy(dtype=float)
        volume = pd.to_numeric(prices_on_calendar.volume, errors='raise').to_numpy(dtype=float)
        baseline_positions = effective[selected] - 1
        valid = np.ones(len(positions), dtype=bool)
        valid[in_window] = baseline_positions[in_window] >= 0
        usable = in_window & valid
        valid[usable] &= (np.isfinite(close[positions[usable]]) & (close[positions[usable]] > 0)
                          & np.isfinite(close[baseline_positions[usable]]) & (close[baseline_positions[usable]] > 0)
                          & np.isfinite(volume[positions[usable]]) & (volume[positions[usable]] > 0)
                          & np.isfinite(volume[baseline_positions[usable]]) & (volume[baseline_positions[usable]] > 0))
        reaction = np.full(len(positions), np.nan)
        reaction[~in_window] = 0.0
        usable = in_window & valid
        reaction[usable] = close[positions[usable]] / close[baseline_positions[usable]] - 1
        values['recent_results_reaction'] = reaction
        values['resultsPublishedAt'] = [timestamp.isoformat() for timestamp in published[selected]]
        values['resultsEffectiveDate'] = calendar[effective[selected]].strftime('%Y-%m-%d')
        values['resultsSignalAt'] = [timestamp.isoformat() for timestamp in signals[positions]]
        values['resultsEventId'] = events.eventId.to_numpy()[selected]
        factor_dates = pd.DatetimeIndex(original_factors.loc[original_factors.symbol == symbol, 'date'])
        invalid_dates = set(calendar[positions[~valid]])
        no_prior_dates = set(calendar[latest < 0])
        audits.append({
            'symbol': symbol, 'eventCount': len(events),
            'annualEventCount': int((events.kind == 'annual').sum()), 'interimEventCount': int((events.kind == 'interim').sum()),
            'firstPublishedAt': published[0].isoformat(), 'lastPublishedAt': published[-1].isoformat(),
            'eventsAfterLastSignal': int((effective == len(calendar)).sum()),
            'originalFactorRows': len(factor_dates),
            'excludedNoPriorEvent': int(factor_dates.isin(no_prior_dates).sum()),
            'excludedInvalidReactionPrice': int(factor_dates.isin(invalid_dates).sum()),
            'excludedNoPriorEventDates': factor_dates[factor_dates.isin(no_prior_dates)].strftime('%Y-%m-%d').tolist(),
            'excludedInvalidReactionPriceDates': factor_dates[factor_dates.isin(invalid_dates)].strftime('%Y-%m-%d').tolist(),
        })
        frames.append(values.loc[valid])
    additions = pd.concat(frames, ignore_index=True)
    factors = original_factors.merge(additions, on=['date', 'symbol'], validate='one_to_one').sort_values(['symbol', 'date']).reset_index(drop=True)
    if not np.isfinite(factors[list(EVENT_KEYS)].to_numpy()).all() or set(factors.symbol) != set(symbols):
        raise ValueError('真实可计算业绩事件因子无法完整覆盖十二只研究股票。')
    labels = original_labels.merge(factors[['date', 'symbol', *EVENT_KEYS, *EVENT_METADATA]],
                                   on=['date', 'symbol'], validate='one_to_one').sort_values(['symbol', 'date']).reset_index(drop=True)
    if set(labels.symbol) != set(symbols):
        raise ValueError('加入公告事件后的有效标签无法覆盖十二只研究股票。')
    groups = {**dataset['feature_groups'],
              'event_compact': (*dataset['feature_groups']['compact'], *EVENT_KEYS),
              'event_drivers': (*dataset['feature_groups']['drivers'], *EVENT_KEYS)}
    names = {**dataset['feature_names'], **EVENT_FEATURE_NAMES}
    hashes = dict(dataset['provenance']['fileHashes'])
    for filename, digest in event_hashes.items():
        if filename in hashes and hashes[filename] != digest:
            raise ValueError(f'{filename} 与已有研究数据指纹冲突。')
        hashes[filename] = digest
    provenance = {
        **dataset['provenance'], 'sha256': _digest(json.dumps(hashes, sort_keys=True).encode()), 'fileHashes': hashes,
        'eventsManifestSha256': event_hashes['events.json'],
        'eventsDatasetSha256': _digest(json.dumps(event_hashes, sort_keys=True).encode()),
        'factorRows': len(factors), 'labelRows': len(labels), 'featureCount': len(names),
        'featureGroupCounts': {name: len(keys) for name, keys in groups.items()},
        'events': {
            'source': 'HKEXnews 官方公告发放时间与类别', 'researchReady': True,
            'start': manifest['start'], 'end': manifest['end'], 'manifestFile': 'events.json',
            'rawFileHashes': manifest['rawFiles'],
            'coverage': {stock['symbol']: stock['coverage'] for stock in manifest['stocks']},
            'excludedSourceRows': {stock['symbol']: len(stock['excludedEvents']) for stock in manifest['stocks']},
            'signalTimeRule': '实际 XHKG session_close 后10分钟；半日市按实际收盘；公告发放时间必须严格早于信号时间，等时或之后到下一信号日生效。',
            'windowRule': '事件有效信号日为第0日，5日窗口为 age<5，21日窗口为 age<21；年龄按正式港股交易日计算，result_age=min(age,126)/126。',
            'reactionRule': '最近已知事件有效日前一正式交易日复权收盘至当前复权收盘收益，仅 age<21 时使用；窗口外按因子定义为0。盘后但在信号前发布的公告，其第0日收益可包含公告前行情，不等于纯公告后反应。',
            'missingRule': '无此前已公布业绩的因子日剔除；窗口内起点或当前复权收盘缺失、无效或当日成交量不大于0同样剔除；不补缺价、停牌价或未知公告。',
            'excludedFactorRows': len(original_factors) - len(factors),
            'excludedLabelRows': len(original_labels) - len(labels), 'securities': audits,
            'featureNames': EVENT_FEATURE_NAMES, 'independentValidation': False,
        },
    }
    return {**dataset, 'factors': factors, 'labels': labels, 'feature_groups': groups,
            'feature_names': names, 'provenance': provenance}
