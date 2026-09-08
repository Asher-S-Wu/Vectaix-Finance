from __future__ import annotations

import hashlib
import json
from pathlib import Path

import exchange_calendars as exchange
import numpy as np
import pandas as pd

from .data import CALENDAR_VERSION, DATA_END, DATA_START, research_calendar
from .engine import _read_inputs
from .factors import FACTOR_KEYS, LABEL_HORIZON
from .macro_data import read_macro_snapshot
from .trend_factors import TREND_FACTOR_NAMES, build_trend_factors


CONTEXT_SYMBOLS = ('USO', 'XLE', 'XBI', 'XLU', 'XLF', 'FXI', 'SPY')
SECTOR_ETFS = {
    '00883': 'XLE', '00857': 'XLE', '00386': 'XLE',
    '02359': 'XBI', '01093': 'XBI', '01177': 'XBI',
    '00003': 'XLU', '00384': 'XLU', '02688': 'XLU',
    '00939': 'XLF', '01398': 'XLF', '03988': 'XLF',
}
SECTOR_NAMES = {'XLE': '能源', 'XBI': '医疗', 'XLU': '公用事业', 'XLF': '金融'}
SECTOR_FEATURES = {'XLE': 'sector_energy', 'XBI': 'sector_healthcare', 'XLU': 'sector_utilities', 'XLF': 'sector_financials'}
COMPACT_KEYS = (
    'momentum_12_1', 'reversal_1', 'low_volatility', 'low_beta',
    'relative_momentum_3', 'distance_high', 'volatility_ratio', 'signed_turnover',
    'overnight_gap', 'market_momentum_1', 'market_volatility', 'bond_momentum_1',
)
PRICE_COLUMNS = ('open', 'high', 'low', 'close', 'adjusted_open', 'adjusted_close', 'volume', 'turnover')
QUOTE_COLUMNS = ('adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'volume')


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_manifest(data_dir: Path, name: str, hashes: dict[str, str]) -> dict:
    raw = (data_dir / name).read_bytes()
    hashes[name] = _digest(raw)
    manifest = json.loads(raw)
    if manifest['schemaVersion'] != 1 or manifest['researchReady'] is not True or manifest['source'] != 'Qveris / FMP':
        raise ValueError(f'{name} 未通过对应来源的研究审查。')
    if manifest['requestedStart'] != DATA_START or manifest['requestedEnd'] != DATA_END:
        raise ValueError(f'{name} 日期范围与本轮封存行情不一致。')
    return manifest


def _verify_calendar(info: dict, name: str, sessions: pd.DatetimeIndex) -> None:
    if info['name'] != name or info['version'] != CALENDAR_VERSION or exchange.__version__ != CALENDAR_VERSION:
        raise ValueError(f'{name} 行情日历或运行版本与封存记录不一致。')
    if info['start'] != DATA_START or info['end'] != DATA_END or info['sessions'] != len(sessions):
        raise ValueError(f'{name} 日历范围或交易日数与封存记录不一致。')
    if info['sha256'] != _digest('\n'.join(sessions.strftime('%Y-%m-%d')).encode()):
        raise ValueError(f'{name} 交易日历指纹不一致。')


def _read_quote(
    data_dir: Path, item: dict, provider_symbol: str, expected_file: str,
    calendar: pd.DatetimeIndex, hashes: dict[str, str], *, exclude_non_sessions: bool,
    allow_zero_volume: bool,
) -> pd.DataFrame:
    if item['researchReady'] is not True or item['file'] != expected_file:
        raise ValueError(f'{provider_symbol} 原始行情来源尚未通过审查或文件位置不一致。')
    raw = (data_dir / expected_file).read_bytes()
    digest = _digest(raw)
    if digest != item['sha256']:
        raise ValueError(f'{provider_symbol} 原始行情内容与封存指纹不一致。')
    hashes[expected_file] = digest
    envelope = json.loads(raw)
    if envelope['success'] is not True or envelope['result']['status_code'] != 200 or envelope['execution_id'] != item['executionId']:
        raise ValueError(f'{provider_symbol} 不是封存的成功 Qveris 响应。')
    rows = pd.DataFrame(envelope['result']['data'])
    required = {'date', 'symbol', *QUOTE_COLUMNS}
    if not required.issubset(rows.columns) or rows[list(required)].isna().any().any():
        raise ValueError(f'{provider_symbol} 原始行情缺少必要字段或包含空值。')
    if set(rows.symbol) != {provider_symbol}:
        raise ValueError(f'{provider_symbol} 原始行情证券代码不一致。')
    rows['date'] = pd.to_datetime(rows.date, format='%Y-%m-%d', errors='raise').dt.as_unit('ns')
    rows = rows.sort_values('date').reset_index(drop=True)
    if rows.date.duplicated().any():
        raise ValueError(f'{provider_symbol} 原始行情日期重复。')
    rows[list(QUOTE_COLUMNS)] = rows[list(QUOTE_COLUMNS)].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(rows[list(QUOTE_COLUMNS)].to_numpy()).all():
        raise ValueError(f'{provider_symbol} 原始行情包含非有限数字。')
    if (rows[list(QUOTE_COLUMNS[:-1])] <= 0).any().any() or (rows.volume < 0).any():
        raise ValueError(f'{provider_symbol} 原始行情包含非正价格或负成交量。')
    if not allow_zero_volume and (rows.volume == 0).any():
        raise ValueError(f'{provider_symbol} 美国环境行情包含零成交量，不能形成连续有效因子。')
    if ((rows.adjHigh < rows[['adjOpen', 'adjClose', 'adjLow']].max(axis=1) - 1e-8)
            | (rows.adjLow > rows[['adjOpen', 'adjClose', 'adjHigh']].min(axis=1) + 1e-8)).any():
        raise ValueError(f'{provider_symbol} 原始行情开高低收矛盾。')
    if ((rows.date < pd.Timestamp(DATA_START)) | (rows.date > pd.Timestamp(DATA_END))).any():
        raise ValueError(f'{provider_symbol} 原始行情超出封存日期范围。')
    if exclude_non_sessions:
        rows = rows.loc[rows.date.isin(calendar)].reset_index(drop=True)
    if not pd.DatetimeIndex(rows.date).equals(calendar):
        raise ValueError(f'{provider_symbol} 未完整覆盖正式交易日；不填补缺失行情。')
    if item['start'] != calendar[0].date().isoformat() or item['end'] != calendar[-1].date().isoformat():
        raise ValueError(f'{provider_symbol} 原始行情起止记录与封存元信息不一致。')
    return rows


def _read_peers(data_dir: Path, targets: list[str], calendar: pd.DatetimeIndex, hashes: dict[str, str]):
    manifest = _read_manifest(data_dir, 'peers.json', hashes)
    _verify_calendar(manifest['calendar'], 'XHKG', calendar)
    if manifest['qualityReady'] is not True or manifest['selection']['originalSymbols'] != targets:
        raise ValueError('同行数据审查或目标股票名单不一致。')
    policy = manifest['zeroVolumePolicy']
    if (policy['labelHorizonSessions'] != LABEL_HORIZON or policy['fillMissingPrices'] is not False
            or policy['fillZeroVolumePrices'] is not False or policy['fillInvalidFeatures'] is not False):
        raise ValueError('同行样本必须执行明确的零量及缺价排除规则。')
    securities = manifest['securities']
    symbols = [s['symbol'] for s in securities]
    if len(symbols) != 8 or len(set(symbols)) != 8 or set(symbols) != set(SECTOR_ETFS) - set(targets):
        raise ValueError('同行股票必须是本轮指定的八只证券。')
    if manifest['pricesFile'] != 'peer-prices.csv':
        raise ValueError('同行合并行情文件位置与本轮约定不一致。')
    raw_csv = (data_dir / 'peer-prices.csv').read_bytes()
    hashes['peer-prices.csv'] = _digest(raw_csv)
    if hashes['peer-prices.csv'] != manifest['pricesSha256']:
        raise ValueError('同行合并行情内容与封存指纹不一致。')
    prices = pd.read_csv(data_dir / 'peer-prices.csv', dtype={'symbol': str}, float_precision='round_trip')
    required = {'date', 'symbol', *PRICE_COLUMNS}
    if not required.issubset(prices.columns) or prices[list(required)].isna().any().any():
        raise ValueError('同行合并行情缺少必要字段或包含空值。')
    prices['date'] = pd.to_datetime(prices.date, format='%Y-%m-%d', errors='raise').dt.as_unit('ns')
    if prices.duplicated(['symbol', 'date']).any() or set(prices.symbol) != set(symbols) or len(prices) != manifest['rowCount']:
        raise ValueError('同行合并行情的证券、行数或唯一日期不一致。')
    prices[list(PRICE_COLUMNS)] = prices[list(PRICE_COLUMNS)].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(prices[list(PRICE_COLUMNS)].to_numpy()).all():
        raise ValueError('同行合并行情包含非有限数字。')
    for security in securities:
        symbol = security['symbol']
        if security['researchReady'] is not True or security['includedInCsv'] is not True:
            raise ValueError(f'{symbol} 同行数据没有明确获准参加研究。')
        if (security['sector'] != SECTOR_NAMES[SECTOR_ETFS[symbol]]
                or security['groupTargetSymbol'] not in targets
                or SECTOR_ETFS[security['groupTargetSymbol']] != SECTOR_ETFS[symbol]):
            raise ValueError(f'{symbol} 同行行业或对应目标股不一致。')
        _verify_calendar(security['calendar'], 'XHKG', calendar)
        raw = _read_quote(data_dir, security['raw'], security['providerSymbol'], f'raw/peers/raw/{symbol}.json',
                          calendar, hashes, exclude_non_sessions=True, allow_zero_volume=True)
        adjusted = _read_quote(data_dir, security['adjusted'], security['providerSymbol'], f'raw/peers/adjusted/{symbol}.json',
                               calendar, hashes, exclude_non_sessions=True, allow_zero_volume=True)
        current = prices.loc[prices.symbol == symbol].sort_values('date')
        if not pd.DatetimeIndex(current.date).equals(calendar) or len(current) != security['csvRows']:
            raise ValueError(f'{symbol} 同行合并行情没有完整覆盖正式交易日。')
        expected = np.column_stack([
            raw.adjOpen, raw.adjHigh, raw.adjLow, raw.adjClose,
            adjusted.adjOpen, adjusted.adjClose, raw.volume, raw.adjClose * raw.volume,
        ])
        if not np.isclose(current[list(PRICE_COLUMNS)].to_numpy(), expected, rtol=1e-12, atol=1e-10).all():
            raise ValueError(f'{symbol} 合并行情与真实原始、复权响应不一致。')
        if not np.array_equal(raw.volume.to_numpy() == 0, adjusted.volume.to_numpy() == 0):
            raise ValueError(f'{symbol} 原始与复权数据的零成交量日期不一致。')
        zero_dates = raw.loc[raw.volume == 0, 'date'].dt.strftime('%Y-%m-%d').tolist()
        if zero_dates != security['sourceZeroVolume']['dates']:
            raise ValueError(f'{symbol} 零成交量记录与显式审查不一致。')
    prices['research_open'] = prices.adjusted_open
    prices['research_close'] = prices.adjusted_close
    return prices, securities, manifest


def _context_factors(data_dir: Path, calendar: pd.DatetimeIndex, hashes: dict[str, str]):
    manifest = _read_manifest(data_dir, 'context.json', hashes)
    securities = manifest['securities']
    if len(securities) != len(CONTEXT_SYMBOLS) or {s['symbol'] for s in securities} != set(CONTEXT_SYMBOLS):
        raise ValueError('市场环境数据必须完整包含本轮指定的七只美国 ETF。')
    us_calendar = exchange.get_calendar('XNYS', start=DATA_START, end=DATA_END).sessions
    aligned = pd.DataFrame({'date': calendar})
    names, drivers, source_dates = {}, [], {}
    by_symbol = {s['symbol']: s for s in securities}
    for symbol in CONTEXT_SYMBOLS:
        item = by_symbol[symbol]
        if item['source'] != 'Qveris / FMP':
            raise ValueError(f'{symbol} 市场环境行情来源不一致。')
        _verify_calendar(item['calendar'], 'XNYS', us_calendar)
        rows = _read_quote(data_dir, item, symbol, f'raw/context/{symbol}.json', us_calendar, hashes,
                           exclude_non_sessions=False, allow_zero_volume=False).set_index('date')
        prefix = symbol.lower()
        close = rows.adjClose
        source_column = f'{prefix}_sourceDate'
        values = pd.DataFrame({
            f'{prefix}_momentum_1': close / close.shift(21) - 1,
            f'{prefix}_momentum_3': close / close.shift(63) - 1,
            f'{prefix}_volatility': close.pct_change(fill_method=None).rolling(21).std(),
            f'{prefix}_trend': close / close.rolling(126).mean() - 1,
        }).rename_axis(source_column).reset_index()
        for suffix, label in (('momentum_1', '月度动量'), ('momentum_3', '季度动量'), ('volatility', '月度波动'), ('trend', '半年趋势')):
            names[f'{prefix}_{suffix}'] = f'{item["name"]}{label}'
        drivers.extend(f'{prefix}_{suffix}' for suffix in ('momentum_1', 'momentum_3', 'volatility'))
        current = pd.merge_asof(pd.DataFrame({'date': calendar}), values,
                                left_on='date', right_on=source_column, direction='backward', allow_exact_matches=False)
        current[f'{prefix}_sourceAgeDays'] = (current.date - current[source_column]).dt.days
        aligned = aligned.merge(current, on='date', validate='one_to_one')
        source_dates[symbol] = source_column
    return aligned, names, tuple(drivers), source_dates, manifest


def _labels_with_complete_windows(factors: pd.DataFrame, prices: pd.DataFrame, symbols: list[str],
                                  benchmark: str, calendar: pd.DatetimeIndex):
    price_keys = ['open', 'high', 'low', 'close', 'adjusted_open', 'adjusted_close']
    market = prices.loc[prices.symbol == benchmark].set_index('date').reindex(calendar)
    market_valid = (market.volume > 0) & market[price_keys].notna().all(axis=1) & (market[price_keys] > 0).all(axis=1)
    horizon_prices = LABEL_HORIZON + 1
    market_window = market_valid.astype(int).rolling(horizon_prices).sum().shift(-horizon_prices) == horizon_prices
    market_target = market.research_open.shift(-horizon_prices) / market.research_open.shift(-1) - 1
    dates = pd.Series(calendar, index=calendar)
    frames, audit = [], []
    for symbol in symbols:
        rows = prices.loc[prices.symbol == symbol].set_index('date').reindex(calendar)
        valid_day = (rows.volume > 0) & rows[price_keys].notna().all(axis=1) & (rows[price_keys] > 0).all(axis=1)
        complete_window = valid_day.astype(int).rolling(horizon_prices).sum().shift(-horizon_prices) == horizon_prices
        candidate = pd.DataFrame({
            'label_start': dates.shift(-1), 'label_end': dates.shift(-horizon_prices),
            'target': rows.research_open.shift(-horizon_prices) / rows.research_open.shift(-1) - 1,
            'market_target': market_target,
        })
        feature_dates = pd.DatetimeIndex(factors.loc[factors.symbol == symbol, 'date'])
        matured = candidate.label_end.notna() & candidate.index.isin(feature_dates)
        admitted = matured & complete_window & market_window & candidate[['target', 'market_target']].notna().all(axis=1)
        selected = candidate.loc[admitted].copy()
        selected['symbol'] = symbol
        frames.append(selected.rename_axis('date').reset_index())
        audit.append({
            'symbol': symbol, 'priceRows': int(rows.close.notna().sum()),
            'zeroVolumeRows': int((rows.volume == 0).sum()), 'factorRows': len(feature_dates),
            'maturedFeatureDates': int(matured.sum()), 'labelRows': int(admitted.sum()),
            'excludedLabelWindows': int((matured & ~admitted).sum()),
            'featureStart': feature_dates.min().date().isoformat(), 'featureEnd': feature_dates.max().date().isoformat(),
        })
    labels = factors.merge(pd.concat(frames, ignore_index=True), on=['date', 'symbol'], validate='one_to_one')
    if not np.isfinite(labels[['target', 'market_target']].to_numpy()).all():
        raise ValueError('成熟标签包含无效收益。')
    return labels.sort_values(['symbol', 'date']).reset_index(drop=True), audit


def build_lab_dataset(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    prices, universe, manifest = _read_inputs(data_dir)
    targets = [s['symbol'] for s in universe['stocks']]
    if set(targets) != {'00883', '02359', '00003', '00939'} or universe['benchmark']['symbol'] != '02800':
        raise ValueError('本轮共享学习仅面向原四只股票及盈富基金基准。')
    hashes = {name: _digest((data_dir / name).read_bytes()) for name in ('prices.csv', 'universe.json', 'review.json', 'manifest.json')}
    for security in manifest['securities']:
        for kind, directory in (('raw', 'fmp-raw'), ('adjusted', 'fmp-adjusted')):
            filename = f'raw/{directory}/{security["symbol"]}.json'
            digest = _digest((data_dir / filename).read_bytes())
            if digest != security[kind]['sha256']:
                raise ValueError(f'{security["symbol"]} 原股票池原始响应与封存指纹不一致。')
            hashes[filename] = digest
    calendar = research_calendar()
    peers, peer_securities, peer_manifest = _read_peers(data_dir, targets, calendar, hashes)
    peer_symbols = [s['symbol'] for s in peer_securities]
    symbols = targets + peer_symbols
    prices = pd.concat([prices, peers], ignore_index=True).sort_values(['symbol', 'date']).reset_index(drop=True)
    if prices.duplicated(['symbol', 'date']).any() or set(prices.symbol) != set(symbols) | {'02800'}:
        raise ValueError('合并后的股票池或行情唯一键不一致。')
    macro, macro_manifest = read_macro_snapshot(data_dir)
    hashes['macro.json'] = macro_manifest['manifestSha256']
    hashes[macro_manifest['file']] = macro_manifest['sha256']
    factors, calendar = build_trend_factors(prices, symbols, '02800', macro)
    context, context_names, driver_keys, source_dates, context_manifest = _context_factors(data_dir, calendar, hashes)
    factors = factors.merge(context, on='date', validate='many_to_one')
    factors['sectorETF'] = factors.symbol.map(SECTOR_ETFS)
    factors['sector'] = factors.sectorETF.map(SECTOR_NAMES)
    factors['sector_momentum_1'] = np.nan
    names = {**TREND_FACTOR_NAMES, **context_names}
    for etf, key in SECTOR_FEATURES.items():
        mask = factors.sectorETF == etf
        factors.loc[mask, 'sector_momentum_1'] = factors.loc[mask, f'{etf.lower()}_momentum_1']
        factors[key] = mask.astype(float)
    factors['sector_relative_1'] = -factors.reversal_1 - factors.sector_momentum_1
    names.update({'sector_relative_1': '本股相对对应行业 ETF 的月度强弱', 'sector_momentum_1': '对应行业 ETF 的月度动量'})
    names.update({key: f'{SECTOR_NAMES[etf]}行业标识' for etf, key in SECTOR_FEATURES.items()})
    before_context = len(factors)
    factors = factors.replace([np.inf, -np.inf], np.nan).dropna(subset=list(names))
    source_dates = {'IEF': 'sourceDate', **source_dates}
    for symbol, column in source_dates.items():
        if factors[column].isna().any() or not (factors[column] < factors.date).all():
            raise ValueError(f'{symbol} 美国环境信息尚未在对应香港信号日之前收盘。')
    if set(factors.symbol) != set(symbols):
        raise ValueError('有效因子无法覆盖十二只研究股票。')
    factors = factors.sort_values(['symbol', 'date']).reset_index(drop=True)
    labels, audit = _labels_with_complete_windows(factors, prices, symbols, '02800', calendar)
    groups = {
        'core': FACTOR_KEYS, 'compact': COMPACT_KEYS,
        'drivers': (*COMPACT_KEYS, *driver_keys, 'sector_relative_1', 'sector_momentum_1', *SECTOR_FEATURES.values()),
    }
    peer_metadata = [{key: security[key] for key in ('symbol', 'name', 'sector', 'industry', 'issuerType', 'tags', 'groupTargetSymbol', 'trainingRole')}
                     for security in peer_securities]
    provenance = {
        'sha256': _digest(json.dumps(hashes, sort_keys=True).encode()),
        'source': 'Qveris / FMP', 'baseDatasetSha256': manifest['sha256'],
        'fileHashes': hashes, 'peerPricesSha256': peer_manifest['pricesSha256'],
        'macroDatasetSha256': macro_manifest['sha256'], 'macroManifestSha256': macro_manifest['manifestSha256'],
        'peerManifestSha256': hashes['peers.json'], 'contextManifestSha256': hashes['context.json'],
        'dataStart': DATA_START, 'dataEnd': DATA_END, 'calendarSessions': len(calendar),
        'targetSymbols': targets, 'peerSymbols': peer_symbols, 'contextSymbols': list(CONTEXT_SYMBOLS),
        'contextSourceDateColumns': source_dates,
        'priceRows': len(prices), 'factorRows': len(factors), 'labelRows': len(labels),
        'featureCount': len(names), 'featureGroupCounts': {key: len(values) for key, values in groups.items()},
        'excludedForContextFeatures': before_context - len(factors), 'securities': audit,
        'zeroVolumePolicy': peer_manifest['zeroVolumePolicy'],
        'labelRule': '下一交易日开盘至第22个交易日开盘，计算21个交易日复权收益；本股及基准在含首尾的22个价格日期内，任何零量或缺价均剔除整个标签。',
        'featureRule': '只使用当时已观测的真实可计算因子；不补行情、零成交量或失效因子。',
        'contextRule': 'ETF在美国原生交易日窗口计算；对齐严格早于香港信号日期的最近真实美国收盘。',
        'sectorRule': '美国行业ETF用于环境与相对强弱特征，不代表港股行业指数；两地月度窗口各按本市场21个交易日计算。',
        'selectionNote': peer_manifest['selection'], 'independentValidation': False,
        'contextExecutionIds': {s['symbol']: s['executionId'] for s in context_manifest['securities']},
    }
    return {
        'prices': prices, 'universe': {**universe, 'peerStocks': peer_metadata},
        'target_symbols': targets, 'peer_symbols': peer_symbols, 'factors': factors,
        'labels': labels, 'calendar': calendar, 'feature_groups': groups,
        'feature_names': names, 'provenance': provenance,
    }
