from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


START = '20100101'
END = '20260831'
SAMPLE_PREFIX = 'vectaix-general-v1:'
HOLDOUT_PREFIX = 'vectaix-general-holdout-v1:'
FIELDS = {
    'stock_basic': 'ts_code,symbol,name,market,exchange,curr_type,list_status,list_date,delist_date',
    'trade_cal': 'exchange,cal_date,is_open,pretrade_date',
    'index_daily': 'ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount',
    'daily': 'ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount',
    'adj_factor': 'ts_code,trade_date,adj_factor',
    'daily_basic': 'ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,total_share,float_share,free_share,total_mv,circ_mv',
    'stk_limit': 'ts_code,trade_date,pre_close,up_limit,down_limit',
    'suspend_d': 'ts_code,trade_date,suspend_timing,suspend_type',
}
PRICE_NUMBERS = ('open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'vol', 'amount')


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _invalid_constant(value: str):
    raise ValueError(f'来源 JSON 含非标准数值：{value}。')


def _json(raw: bytes) -> dict:
    result = json.loads(raw, parse_constant=_invalid_constant)
    if not isinstance(result, dict):
        raise ValueError('来源不是 JSON 对象。')
    return result


def _dates(values: pd.Series, name: str, *, nullable: bool = False) -> pd.Series:
    missing = values.isna() | values.eq('')
    text = values.astype('string')
    if ((missing.any() and not nullable)
            or not text.loc[~missing].str.fullmatch(r'\d{8}').all()):
        raise ValueError(f'{name} 包含缺失或格式错误的日期。')
    return pd.to_datetime(text.mask(missing), format='%Y%m%d', errors='raise').dt.as_unit('ns')


def _date_list(values) -> list[str]:
    return pd.DatetimeIndex(values).sort_values().strftime('%Y-%m-%d').tolist()


def _numbers(frame: pd.DataFrame, columns, label: str) -> None:
    for column in columns:
        if frame[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f'{label}.{column} 把布尔值用作数值。')
        frame[column] = pd.to_numeric(frame[column], errors='raise').astype(float)
        if np.isinf(frame[column].to_numpy()).any():
            raise ValueError(f'{label}.{column} 包含无穷值。')


class _Snapshot:
    def __init__(self, root: Path, namespace: str):
        self.root = root.resolve()
        self.namespace = namespace
        manifest_path = self.root / 'manifest.json'
        if not manifest_path.is_file():
            raise ValueError(f'{namespace} 尚无完成采集清单，不能加载部分下载结果。')
        raw = manifest_path.read_bytes()
        self.manifest = _json(raw)
        m = self.manifest
        if (m['schemaVersion'] != 1 or m['downloadComplete'] is not True
                or m['start'] != START or m['end'] != END):
            raise ValueError(f'{namespace} 采集未完成或研究区间不符。')
        self.hashes = {f'{namespace}/manifest.json': _sha(raw)}
        for filename, expected in m['fileHashes'].items():
            if not isinstance(expected, str) or re.fullmatch(r'[0-9a-f]{64}', expected) is None:
                raise ValueError(f'{namespace} 文件指纹格式无效。')
            actual = _sha(self.path(filename).read_bytes())
            if actual != expected:
                raise ValueError(f'{namespace}/{filename} 与冻结指纹不符。')
            self.hashes[f'{namespace}/{filename}'] = actual
        requests = m['requests']
        if not requests or any(item['status'] != 'ok' for item in requests):
            raise ValueError(f'{namespace} 有缺失或失败的采集请求。')
        self.requests = {item['file']: item for item in requests}
        if len(self.requests) != len(requests):
            raise ValueError(f'{namespace} 请求清单出现重复原文件。')

    def path(self, filename: str) -> Path:
        relative = Path(filename)
        path = self.root / relative
        if (relative.is_absolute() or '..' in relative.parts
                or not path.resolve().is_relative_to(self.root)):
            raise ValueError(f'{self.namespace} 来源路径越界。')
        return path

    def read(self, filename: str) -> dict:
        raw = self.path(filename).read_bytes()
        expected = self.manifest['fileHashes'].get(filename)
        if expected is None or _sha(raw) != expected:
            raise ValueError(f'{self.namespace}/{filename} 未封存或加载时已改变。')
        return _json(raw)

    def table(self, api: str, params: dict, key: str) -> tuple[pd.DataFrame, dict]:
        filename, receipt_file = f'raw/{key}.json', f'receipts/{key}.json'
        expected = {'api': api, 'params': params, 'fields': FIELDS[api]}
        receipt = self.read(receipt_file)
        if (receipt != self.requests.get(filename) or receipt['request'] != expected
                or receipt['file'] != filename
                or receipt['sha256'] != self.manifest['fileHashes'].get(filename)):
            raise ValueError(f'{self.namespace}/{key} 的请求、回执与清单不一致。')
        requested, received = pd.Timestamp(receipt['requestedAt']), pd.Timestamp(receipt['receivedAt'])
        if requested.tzinfo is None or received.tzinfo is None or requested > received:
            raise ValueError(f'{self.namespace}/{key} 获取时间无时区或顺序错误。')
        payload = self.read(filename)
        if payload['code'] != 0 or not isinstance(payload['data'], dict):
            raise ValueError(f'{self.namespace}/{key} 原响应未成功。')
        data = payload['data']
        fields, rows = data['fields'], data['items']
        if (data.get('has_more') is not False or len(fields) != len(set(fields))
                or set(fields) != set(FIELDS[api].split(','))
                or len(rows) != receipt['rows'] or len(rows) >= params['limit']
                or any(len(row) != len(fields) for row in rows)):
            raise ValueError(f'{self.namespace}/{key} 存在未证明完整的响应或字段错误。')
        if 'responseRequestId' in receipt and receipt['responseRequestId'] != payload.get('request_id'):
            raise ValueError(f'{self.namespace}/{key} 来源请求 ID 不符。')
        return pd.DataFrame(rows, columns=fields), {
            'status': 'ok' if rows else 'ok_empty', 'rows': len(rows),
            'rawFile': f'{self.namespace}/{filename}',
            'receiptFile': f'{self.namespace}/{receipt_file}',
            'hasMore': False, 'providerCount': data.get('count'),
            'request': expected, 'requestedAt': receipt['requestedAt'], 'receivedAt': receipt['receivedAt'],
        }


def _market_table(frame: pd.DataFrame, symbol: str, api: str, info: dict) -> pd.DataFrame:
    if not frame.ts_code.eq(symbol).all():
        raise ValueError(f'{api}/{symbol} 包含其他证券。')
    frame = frame.rename(columns={'ts_code': 'symbol', 'trade_date': 'date'}).copy()
    frame['date'] = _dates(frame['date'], f'{api}/{symbol}')
    if not frame.date.between(pd.Timestamp(START), pd.Timestamp(END)).all():
        raise ValueError(f'{api}/{symbol} 返回了请求范围外的日期。')
    if api == 'suspend_d':
        if (not frame.suspend_type.isin(['S', 'R']).all()
                or not frame.suspend_timing.map(lambda value: pd.isna(value) or isinstance(value, str)).all()
                or frame.duplicated(['symbol', 'date', 'suspend_type', 'suspend_timing']).any()):
            raise ValueError(f'{api}/{symbol} 停复牌类型、时段或重复原行异常。')
    else:
        if frame.duplicated(['symbol', 'date']).any():
            raise ValueError(f'{api}/{symbol} 存在重复证券日期，不能自动去重。')
        _numbers(frame, [column for column in frame if column not in ('symbol', 'date')], f'{api}/{symbol}')
    frame = frame.sort_values('date', ignore_index=True)
    info.update(firstDate=None if frame.empty else frame.date.iloc[0].date().isoformat(),
                lastDate=None if frame.empty else frame.date.iloc[-1].date().isoformat(),
                nullCounts={column: int(frame[column].isna().sum()) for column in frame if column not in ('symbol', 'date')})
    return frame


def _calendar(source: _Snapshot) -> tuple[pd.DatetimeIndex, dict]:
    calendars, audit = [], {}
    expected = pd.date_range(START, END).as_unit('ns')
    for exchange in ('SSE', 'SZSE'):
        frame, info = source.table('trade_cal', {
            'exchange': exchange, 'start_date': START, 'end_date': END, 'limit': 7000,
        }, f'trade_cal/{exchange}')
        if not frame.exchange.eq(exchange).all() or not frame.is_open.isin([0, 1]).all():
            raise ValueError(f'{exchange} 官方日历的交易所或开市标记错误。')
        frame['cal_date'] = _dates(frame.cal_date, f'{exchange}.cal_date')
        frame['pretrade_date'] = _dates(frame.pretrade_date, f'{exchange}.pretrade_date')
        frame = frame.sort_values('cal_date', ignore_index=True)
        if not pd.DatetimeIndex(frame.cal_date).equals(expected):
            raise ValueError(f'{exchange} 官方日历未逐日覆盖整个自然日区间。')
        previous = None
        for row in frame.itertuples(index=False):
            if row.pretrade_date >= row.cal_date or (previous is not None and row.pretrade_date != previous):
                raise ValueError(f'{exchange} 官方日历前一交易日不一致。')
            if row.is_open == 1:
                previous = row.cal_date
        calendars.append(frame[['cal_date', 'is_open', 'pretrade_date']])
        audit[exchange] = {**info, 'civilDays': len(frame), 'openDays': int(frame.is_open.sum())}
    if not calendars[0].equals(calendars[1]):
        raise ValueError('沪深官方交易日历不一致，不能隐式选用其中一个。')
    return pd.DatetimeIndex(calendars[0].loc[calendars[0].is_open == 1, 'cal_date'], name='date'), audit


def _universe(source: _Snapshot, specification: dict) -> tuple[pd.DataFrame, dict]:
    if (specification['schemaVersion'] != 1 or specification['source'] != 'Tushare Pro HTTPS'
            or specification['start'] != START or specification['end'] != END
            or specification['selectionUsesOutcomes'] is not False
            or specification['unknownMissingValuesFilled'] is not False
            or specification['fields'] != {k: v for k, v in FIELDS.items() if k != 'suspend_d'}
            or specification['samplingRule'] != "int(sha256('vectaix-general-v1:'+ts_code),16)%10==0"
            or specification['holdoutRule'] != "int(sha256('vectaix-general-holdout-v1:'+ts_code),16)%5==0"):
        raise ValueError('通用研究的原始范围、抽样或留出规格不符。')
    all_rows, audit = [], {}
    for status in ('L', 'D', 'P'):
        frame, info = source.table('stock_basic', {'list_status': status, 'limit': 6000}, f'stock_basic/{status}')
        if not frame.list_status.eq(status).all():
            raise ValueError('股票原始列表的上市状态与请求不符。')
        all_rows.extend(frame.to_dict('records'))
        audit[status] = info
    if len(all_rows) != specification['universeRows'] or len({row['ts_code'] for row in all_rows}) != len(all_rows):
        raise ValueError('L/D/P 全体来源名单的数量或证券唯一性错误。')
    selected = []
    for row in all_rows:
        if (row['exchange'] not in ('SSE', 'SZSE') or row['curr_type'] != 'CNY'
                or not row['list_date'] or row['list_date'] > END
                or (row['delist_date'] and row['delist_date'] < START)):
            continue
        if int(_sha((SAMPLE_PREFIX + row['ts_code']).encode()), 16) % 10 == 0:
            split = 'heldout' if int(_sha((HOLDOUT_PREFIX + row['ts_code']).encode()), 16) % 5 == 0 else 'train'
            selected.append({**row, 'split': split})
    selected.sort(key=lambda row: row['ts_code'])
    if (selected != specification['stocks'] or selected != source.manifest['stocks']
            or len(selected) != source.manifest['stockCount'] or not 300 <= len(selected) <= 800):
        raise ValueError('冻结股票池或永久留出组不符合来源与预先指定的哈希抽样。')
    stocks = pd.DataFrame(selected).rename(columns={'symbol': 'exchange_symbol'})
    stocks['symbol'] = stocks.ts_code
    if not stocks.symbol.str.fullmatch(r'\d{6}\.(SH|SZ)').all():
        raise ValueError('冻结股票代码格式错误。')
    stocks['list_date'] = _dates(stocks.list_date, 'stocks.list_date')
    stocks['delist_date'] = _dates(stocks.delist_date, 'stocks.delist_date', nullable=True)
    if (stocks.delist_date.notna() & (stocks.delist_date <= stocks.list_date)).any():
        raise ValueError('股票上市与退市日期顺序错误。')
    return stocks, audit


def _price_anomalies(frame: pd.DataFrame) -> dict[str, list[str]]:
    prices = frame[['open', 'high', 'low', 'close']]
    checks = {
        'missingPrice': prices.isna().any(axis=1),
        'nonPositivePrice': prices.le(0).any(axis=1),
        'invalidOHLC': (frame.high + 1e-8 < prices.max(axis=1)) | (frame.low - 1e-8 > prices.min(axis=1)),
        'missingVolumeOrAmount': frame[['vol', 'amount']].isna().any(axis=1),
        'negativeVolumeOrAmount': frame[['vol', 'amount']].lt(0).any(axis=1),
        'zeroVolume': frame.vol.eq(0),
    }
    return {name: _date_list(frame.loc[mask, 'date']) for name, mask in checks.items()}


def load_general_dataset(data_dir: Path) -> dict:
    """核验两份完整采集快照，保留原始观测及所有未解释缺口供训练端准入。"""
    main = _Snapshot(Path(data_dir), 'main')
    suspensions = _Snapshot(Path(data_dir).parent / 'general-tushare-suspensions', 'suspensions')
    spec = main.read('specification.json')
    if (suspensions.manifest['universeSpecificationSha256'] != main.manifest['fileHashes']['specification.json']
            or suspensions.manifest['stockCount'] != main.manifest['stockCount']):
        raise ValueError('停复牌来源没有绑定同一份冻结股票池。')
    stocks, universe_audit = _universe(main, spec)
    symbols = stocks.symbol.tolist()
    required_main = ({f'raw/stock_basic/{status}.json' for status in ('L', 'D', 'P')}
                     | {f'raw/trade_cal/{exchange}.json' for exchange in ('SSE', 'SZSE')}
                     | {'raw/index_daily/000300.SH.json'}
                     | {f'raw/{api}/{symbol}.json' for symbol in symbols for api in ('daily', 'adj_factor', 'daily_basic', 'stk_limit')})
    if (set(main.requests) != required_main
            or set(suspensions.requests) != {f'raw/{symbol}.json' for symbol in symbols}):
        raise ValueError('完整采集清单缺少必要的证券/API 查询或包含非规格查询。')
    calendar, calendar_audit = _calendar(main)
    benchmark, benchmark_info = main.table('index_daily', {
        'ts_code': '000300.SH', 'start_date': START, 'end_date': END, 'limit': 6000,
    }, 'index_daily/000300.SH')
    benchmark = _market_table(benchmark, '000300.SH', 'index_daily', benchmark_info)
    if not pd.DatetimeIndex(benchmark.date).equals(calendar):
        raise ValueError('沪深 300 基准行情未逐日完整覆盖官方交易日历。')
    price_frames, basic_frames, limit_frames, suspension_frames = [], [], [], []
    coverage, issues = {}, []
    for stock in stocks.to_dict('records'):
        symbol = stock['symbol']
        frames, tables = {}, {}
        for api in ('daily', 'adj_factor', 'daily_basic', 'stk_limit'):
            frame, info = main.table(api, {
                'ts_code': symbol, 'start_date': START, 'end_date': END, 'limit': 5800,
            }, f'{api}/{symbol}')
            frames[api] = _market_table(frame, symbol, api, info)
            tables[api] = info
        suspended, info = suspensions.table('suspend_d', {
            'ts_code': symbol, 'start_date': START, 'end_date': END, 'limit': 6000,
        }, symbol)
        suspended = _market_table(suspended, symbol, 'suspend_d', info)
        tables['suspend_d'] = info
        daily, adjustment, basic, limits = [frames[api] for api in ('daily', 'adj_factor', 'daily_basic', 'stk_limit')]
        dates = pd.DatetimeIndex(daily.date)
        expected = calendar[calendar >= stock['list_date']]
        if pd.notna(stock['delist_date']):
            expected = expected[expected < stock['delist_date']]
        missing = expected.difference(dates)
        empty_timing = suspended.suspend_timing.isna() | suspended.suspend_timing.eq('')
        full_s = pd.DatetimeIndex(suspended.loc[suspended.suspend_type.eq('S') & empty_timing, 'date']).unique()
        intraday_s = pd.DatetimeIndex(suspended.loc[suspended.suspend_type.eq('S') & ~empty_timing, 'date']).unique()
        anomalies = _price_anomalies(daily)
        anomalies.update({
            'nonSessionPrice': _date_list(dates.difference(calendar)),
            'outsideListingLifecycle': _date_list(dates.difference(expected)),
            'unexplainedMissingPrice': _date_list(missing.difference(full_s)),
            'fullDaySuspensionWithPrice': _date_list(full_s.intersection(dates)),
            'missingAdjustmentObservation': _date_list(dates.difference(pd.DatetimeIndex(adjustment.date))),
            'invalidAdjustment': _date_list(adjustment.loc[adjustment.adj_factor.isna() | adjustment.adj_factor.le(0), 'date']),
            'missingDailyBasicObservation': _date_list(dates.difference(pd.DatetimeIndex(basic.date))),
            'missingLimitObservation': _date_list(dates.difference(pd.DatetimeIndex(limits.date))),
            'invalidLimitValues': _date_list(limits.loc[limits[['up_limit', 'down_limit']].isna().any(axis=1)
                | limits[['up_limit', 'down_limit']].le(0).any(axis=1) | limits.up_limit.lt(limits.down_limit), 'date']),
        })
        for kind, affected in anomalies.items():
            if affected:
                issues.append({'symbol': symbol, 'kind': kind, 'count': len(affected), 'dates': affected})
        coverage[symbol] = {
            'split': stock['split'], 'listDate': stock['list_date'].date().isoformat(),
            'delistDate': None if pd.isna(stock['delist_date']) else stock['delist_date'].date().isoformat(),
            'tables': tables, 'expectedLifecycleSessions': len(expected),
            'missingPriceDates': _date_list(missing),
            'missingPriceExplainedByFullDaySuspension': _date_list(missing.intersection(full_s)),
            'unexplainedMissingPriceDates': anomalies['unexplainedMissingPrice'],
            'intradaySuspensionDates': _date_list(intraday_s),
            'adjustmentWithoutPriceDates': _date_list(pd.DatetimeIndex(adjustment.date).difference(dates)),
            'dailyBasicWithoutPriceDates': _date_list(pd.DatetimeIndex(basic.date).difference(dates)),
            'limitsWithoutPriceDates': _date_list(pd.DatetimeIndex(limits.date).difference(dates)),
            'anomalies': anomalies,
        }
        daily = daily.merge(adjustment, on=['symbol', 'date'], how='left', validate='one_to_one')
        daily['volume'] = daily.vol * 100.0
        daily['turnover'] = daily.amount * 1000.0
        price_frames.append(daily)
        basic_frames.append(basic)
        limit_frames.append(limits)
        suspension_frames.append(suspended)
    benchmark_anomalies = _price_anomalies(benchmark)
    for kind, affected in benchmark_anomalies.items():
        if affected:
            issues.append({'symbol': '000300.SH', 'kind': kind, 'count': len(affected), 'dates': affected})
    benchmark['volume'] = benchmark.vol * 100.0
    benchmark['turnover'] = benchmark.amount * 1000.0
    def concatenate(frames):
        return pd.concat(frames, ignore_index=True).sort_values(['date', 'symbol'], ignore_index=True)
    return {
        'prices': concatenate(price_frames), 'calendar': calendar, 'stocks': stocks,
        'benchmark': benchmark, 'daily_basic': concatenate(basic_frames),
        'limits': concatenate(limit_frames), 'suspensions': concatenate(suspension_frames),
        'training_symbols': stocks.loc[stocks.split.eq('train'), 'symbol'].tolist(),
        'holdout_symbols': stocks.loc[stocks.split.eq('heldout'), 'symbol'].tolist(),
        'provenance': {
            'sourceRoots': {'main': str(main.root), 'suspensions': str(suspensions.root)},
            'fileHashes': {**main.hashes, **suspensions.hashes}, 'coverage': coverage, 'issues': issues,
            'stockCount': len(stocks), 'calendarName': 'XSHG', 'calendarSource': 'Tushare trade_cal SSE and SZSE',
            'calendarAudit': calendar_audit, 'universeAudit': universe_audit,
            'benchmarkAudit': {**benchmark_info, 'anomalies': benchmark_anomalies},
            'calendarSha256': _sha('\n'.join(calendar.strftime('%Y-%m-%d')).encode()),
            'unknownMissingValuesFilled': False, 'futureExecutionAvailabilityFiltersApplied': False,
            'metadataUsage': '股票身份、上市/退市生命周期及固定训练分组；名称、当前状态、行业均不进入预测特征。',
            'units': {'price': 'CNY per share', 'vol': '100 shares', 'amount': '1000 CNY',
                      'volume': 'shares', 'turnover': 'CNY', 'adj_factor': 'dimensionless',
                      'daily_basic.shareFields': '10000 shares', 'daily_basic.marketValueFields': '10000 CNY',
                      'benchmark.price': 'index points'},
            'coverageInterpretation': '成功空表与未完成采集严格区分；缺行情只以原始全日S记录解释，否则保持未解释。无S记录不是无停牌证明。退市日之前是待核验生命周期窗口，不把退市日期当作最后成交日。',
            'priceInterpretation': '保留全部来源行情和异常；adj_factor按真实证券日期左连接。pre_close是除权昨收，不要求等于前一行原价。OHLC比较绝对容差1e-8，没有修改数值。',
            'timingInterpretation': '原始停复牌、限价和缺行情仅供对应执行日核验，不按未来可成交性提前删除预测。',
        },
    }
