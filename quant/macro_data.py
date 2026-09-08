from __future__ import annotations

import hashlib
import json
from pathlib import Path

import exchange_calendars as exchange
import numpy as np
import pandas as pd


def read_macro_snapshot(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    manifest_raw = (data_dir / 'macro.json').read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest['schemaVersion'] != 1 or manifest['researchReady'] is not True or manifest['source'] != 'Qveris / FMP':
        raise ValueError('宏观行情未通过来源审查。')
    if manifest['symbol'] != 'IEF' or manifest['file'] != 'raw/macro/IEF.json':
        raise ValueError('本轮宏观因子只接受已封存的IEF美债基金。')
    raw = (data_dir / manifest['file']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['sha256']:
        raise ValueError('宏观原始行情与封存指纹不一致。')
    envelope = json.loads(raw)
    if envelope['success'] is not True or envelope['result']['status_code'] != 200 or envelope['execution_id'] != manifest['executionId']:
        raise ValueError('宏观行情不是对应的成功Qveris响应。')
    rows = pd.DataFrame(envelope['result']['data'])
    if rows[['symbol', 'date', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'volume']].isna().any().any():
        raise ValueError('IEF行情缺少值。')
    if set(rows.symbol) != {'IEF'}:
        raise ValueError('IEF响应证券代码不一致。')
    rows['date'] = pd.to_datetime(rows.date, format='%Y-%m-%d', errors='raise').dt.as_unit('ns')
    rows = rows.sort_values('date').reset_index(drop=True)
    if rows.date.duplicated().any():
        raise ValueError('IEF行情日期重复。')
    numeric = ['adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'volume']
    rows[numeric] = rows[numeric].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(rows[numeric]).all().all() or (rows[numeric] <= 0).any().any():
        raise ValueError('IEF行情含无效价格或成交量。')
    if (rows.adjHigh < rows[['adjOpen', 'adjLow', 'adjClose']].max(axis=1)).any() or (rows.adjLow > rows[['adjOpen', 'adjHigh', 'adjClose']].min(axis=1)).any():
        raise ValueError('IEF开高低收不一致。')
    calendar_info = manifest['calendar']
    if calendar_info['name'] != 'XNYS' or exchange.__version__ != calendar_info['version']:
        raise ValueError('IEF交易日历版本不一致。')
    sessions = exchange.get_calendar('XNYS', start=calendar_info['start'], end=calendar_info['end']).sessions
    if not pd.DatetimeIndex(rows.date).equals(sessions):
        raise ValueError('IEF行情没有完整覆盖美国正式交易日，不填补缺失行情。')
    calendar_hash = hashlib.sha256('\n'.join(sessions.strftime('%Y-%m-%d')).encode()).hexdigest()
    if calendar_hash != calendar_info['sha256']:
        raise ValueError('IEF日历与封存指纹不一致。')
    return rows, {**manifest, 'manifestSha256': hashlib.sha256(manifest_raw).hexdigest()}
