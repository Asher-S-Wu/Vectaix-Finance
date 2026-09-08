from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_END, DATA_START, research_calendar


RATE_FIELDS = ('ir_overnight', 'ir_1w', 'ir_1m', 'ir_3m', 'ir_6m', 'ir_12m')
HKAB_FIELDS = ('Overnight', '1 Week', '1 Month', '3 Months', '6 Months', '12 Months')
SOURCE_SPLIT = '2014-01-21'
RATE_FEATURE_NAMES = {
    'hibor_1m': '上一已公布定盘日的一月港元拆息，年利率小数',
    'hibor_term_spread': '上一已公布定盘日的十二月减一月拆息，年利率小数',
    'hibor_change_21': '一月拆息相对21个真实定盘日前的变化，年利率小数',
    'hibor_change_63': '一月拆息相对63个真实定盘日前的变化，年利率小数',
}
RATE_KEYS = tuple(RATE_FEATURE_NAMES)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_hibor_snapshot(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Read the frozen official fixing series; reject gaps rather than fill rates."""
    data_dir = Path(data_dir)
    raw = (data_dir / 'hkma.json').read_bytes()
    manifest = json.loads(raw)
    if (manifest['schemaVersion'] != 1 or manifest['researchReady'] is not True
            or manifest['source'] != 'Qveris / HKMA / HKAB'
            or manifest['segment'] != 'hibor.fixing'
            or manifest['dataStart'] != DATA_START or manifest['dataEnd'] != DATA_END
            or manifest['rateFields'] != list(RATE_FIELDS)):
        raise ValueError('港元拆息封存规格与本轮研究不一致。')
    hashes = {'hkma.json': _digest(raw)}
    for filename, expected_hash in manifest['evidenceFiles'].items():
        if not filename.startswith('raw/hkma/') or '..' in Path(filename).parts:
            raise ValueError('拆息政策证据路径不正确。')
        hashes[filename] = _digest((data_dir / filename).read_bytes())
        if hashes[filename] != expected_hash:
            raise ValueError('拆息公布接口或交易日历差异证据已改变。')
    def read_verified(filename, expected_hash):
        content = (data_dir / filename).read_bytes()
        hashes[filename] = _digest(content)
        if hashes[filename] != expected_hash:
            raise ValueError(f'拆息原始来源指纹改变：{filename}')
        return json.loads(content)
    # Two fixed, disjoint source intervals. No request failure changes this definition.
    first = manifest['hkmaInterval']
    if first['file'] != 'raw/hkma/hibor-0000.json' or first['end'] != SOURCE_SPLIT:
        raise ValueError('金管局历史区间规格不一致。')
    envelope = read_verified(first['file'], first['sha256'])
    payload = envelope['result']['data']
    if (envelope['success'] is not True or envelope['result']['status_code'] != 200
            or payload['header']['success'] is not True or payload['header']['err_code'] != '0000'):
        raise ValueError('金管局区间不是成功的官方响应。')
    records = list(payload['result']['records'])
    if len(records) != 1000 or payload['result']['datasize'] != 1000 or records[-1]['end_of_day'] != SOURCE_SPLIT:
        raise ValueError('金管局固定历史区间不完整。')
    second = manifest['hkabInterval']
    if second['file'] != 'hkab-collection.json' or second['start'] != '2014-01-22' or second['end'] != DATA_END:
        raise ValueError('银行公会历史区间规格不一致。')
    collection = read_verified(second['file'], second['sha256'])
    if (collection['schemaVersion'] != 1 or collection['source'] != 'HKAB'
            or collection['researchReady'] is not True or collection['coverageComplete'] is not True
            or collection['start'] != second['start'] or collection['end'] != second['end']
            or collection['failures']):
        raise ValueError('银行公会采集未完整通过准入。')
    weekdays = pd.bdate_range(second['start'], second['end']).strftime('%Y-%m-%d').tolist()
    if collection['expectedWeekdays'] != len(weekdays) or [r['date'] for r in collection['records']] != weekdays:
        raise ValueError('银行公会必须明确覆盖区间内每一个周一至周五，包括非营业日。')
    def read_hkab(item):
        expected = f'raw/hkab/{item["date"]}.json'
        if item['file'] != expected:
            raise ValueError('银行公会原始日响应路径不一致。')
        response = read_verified(expected, item['sha256'])
        if type(response['isHoliday']) is not bool:
            raise ValueError('银行公会工作日状态无效。')
        return response
    stock_sessions = set(research_calendar().strftime('%Y-%m-%d'))
    weather_no_fixing = []
    for item in collection['records']:
        response = read_hkab(item)
        if response['isHoliday'] != item['isHoliday'] or response['time'] != item['time']:
            raise ValueError('银行公会定盘时间或非营业日标志与清单不一致。')
        if response['isHoliday']:
            if item['sourceStatus'] != 'holiday':
                raise ValueError('银行公会非营业日分类不一致。')
            continue
        no_fixing = (response['ds_is_fb'] == 'Y'
                     and all(response[key] is None for key in ('year', 'month', 'day', 'date', *HKAB_FIELDS))
                     and isinstance(response['ds_fb_msg_1_en'], str)
                     and all(phrase in response['ds_fb_msg_1_en'] for phrase in
                             ('adverse weather', 'banks will close to the public', 'next business day')))
        if no_fixing:
            if item['sourceStatus'] != 'weather_closure_no_fixing':
                raise ValueError('天气无定盘记录的明确分类未审查。')
            weather_no_fixing.append(item['date'])
            continue
        if item['sourceStatus'] != 'fixing':
            raise ValueError('银行公会有效定盘分类不一致。')
        if pd.Timestamp(response['date']).strftime('%Y-%m-%d') != item['date'] or not response['time']:
            raise ValueError('银行公会响应不是所查询日期的定盘。')
        values = [response[key] for key in HKAB_FIELDS]
        if any(isinstance(v, bool) or not isinstance(v, (int, float, str)) for v in values):
            raise ValueError('银行公会六个期限包含无效数值。')
        records.append({'end_of_day': item['date'], **{key: float(value) for key, value in zip(RATE_FIELDS, values, strict=True)}})
    if weather_no_fixing != manifest['weatherNoFixingDates']:
        raise ValueError('天气无定盘的明确排除日期与封存审查不一致。')
    weather_stock_dates = [day for day in weather_no_fixing if day in stock_sessions]
    if weather_stock_dates != manifest['weatherNoFixingStockSessionDates']:
        raise ValueError('股票开市而官方未发布拆息的明确日期与封存审查不一致。')
    old_by_date = {row['end_of_day']: row for row in payload['result']['records']}
    overlap_dates = ['2012-01-03', '2013-06-03', '2014-01-21']
    if collection['overlapDates'] != overlap_dates or [r['date'] for r in collection['overlap']] != overlap_dates:
        raise ValueError('两个官方来源的交叉核验日期不完整。')
    for item in collection['overlap']:
        response = read_hkab(item)
        if response['isHoliday'] or pd.Timestamp(response['date']).strftime('%Y-%m-%d') != item['date']:
            raise ValueError('交叉核验不是有效定盘日。')
        for hkma_key, hkab_key in zip(RATE_FIELDS, HKAB_FIELDS, strict=True):
            if Decimal(str(response[hkab_key])) != Decimal(str(old_by_date[item['date']][hkma_key])):
                raise ValueError(f'金管局与银行公会历史定盘不一致：{item["date"]}/{hkma_key}')
    frame = pd.DataFrame(records)[['end_of_day', *RATE_FIELDS]]
    frame['end_of_day'] = pd.to_datetime(frame.end_of_day, format='%Y-%m-%d', errors='raise').dt.as_unit('ns')
    if frame.empty or frame.end_of_day.duplicated().any() or not frame.end_of_day.is_monotonic_increasing:
        raise ValueError('拆息日期必须完整、唯一、严格递增。')
    if not np.isfinite(frame[list(RATE_FIELDS)].to_numpy(dtype=float)).all():
        raise ValueError('六个持续发布的拆息期限包含缺失或无效数值。')
    if not frame.end_of_day.between(DATA_START, DATA_END).all():
        raise ValueError('拆息响应超出封存日期。')
    calendar = research_calendar()
    observed = pd.DatetimeIndex(frame.end_of_day)
    missing = calendar.difference(observed)
    extra = observed.difference(calendar)
    if (missing.strftime('%Y-%m-%d').tolist() != weather_stock_dates
            or extra.strftime('%Y-%m-%d').tolist() != manifest['nonStockSessionDates']):
        raise ValueError(f'拆息未覆盖股票交易日或额外日期审查不一致：{missing.strftime("%Y-%m-%d").tolist()}')
    if len(frame) != manifest['rows']:
        raise ValueError('拆息总记录数与封存清单不符。')
    frame[list(RATE_FIELDS)] = frame[list(RATE_FIELDS)].astype(float) / 100.0
    return frame, {**manifest, 'fileHashes': hashes, 'manifestSha256': hashes['hkma.json']}


def build_hibor_factors(data_dir: Path, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    rates, manifest = read_hibor_snapshot(data_dir)
    native = pd.DataFrame({
        'hiborSourceDate': rates.end_of_day,
        'hibor_1m': rates.ir_1m,
        'hibor_term_spread': rates.ir_12m - rates.ir_1m,
        'hibor_change_21': rates.ir_1m - rates.ir_1m.shift(21),
        'hibor_change_63': rates.ir_1m - rates.ir_1m.shift(63),
    })
    # Previous real fixing is the specified lagged feature, not a missing-value fill.
    aligned = pd.merge_asof(pd.DataFrame({'date': calendar}), native,
                            left_on='date', right_on='hiborSourceDate',
                            direction='backward', allow_exact_matches=False)
    admitted = aligned.dropna(subset=list(RATE_KEYS)).copy()
    admitted['hiborSourceAgeCalendarDays'] = (admitted.date - admitted.hiborSourceDate).dt.days
    if admitted.empty or not (admitted.hiborSourceDate < admitted.date).all():
        raise ValueError('港元拆息因子存在时间越界或无有效窗口。')
    if not np.isfinite(admitted[list(RATE_KEYS)].to_numpy()).all():
        raise ValueError('港元拆息因子不是有限值。')
    return admitted, {**manifest, 'featureNames': RATE_FEATURE_NAMES,
                      'featureRows': len(admitted), 'excludedWarmupDates': len(aligned) - len(admitted),
                      'timingRule': '固定使用严格早于香港信号日期的最近真实定盘；变化窗口在原生定盘日期计算，不补值。',
                      'vintageCaveat': '来源为获取时的官方历史序列，未获得逐日历史版本；不能排除来源后续纠错。'}
