from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_END, DATA_START


SHIBOR_FIELDS = ('ON', '1W', '2W', '1M', '3M', '6M', '9M', '1Y')
SHIBOR_FEATURE_NAMES = {
    'shibor_1m': '上一真实定盘日的一月人民币拆息，年利率小数',
    'shibor_term_spread': '上一真实定盘日的一年减一月人民币拆息，年利率小数',
    'shibor_change_21': '一月人民币拆息相对21个原生定盘日前的变化，年利率小数',
    'shibor_change_63': '一月人民币拆息相对63个原生定盘日前的变化，年利率小数',
}
SHIBOR_KEYS = tuple(SHIBOR_FEATURE_NAMES)
COLLECTION_FILE = 'raw/rmb-rates-research/collection.json'


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_shibor_snapshot(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    data_dir = Path(data_dir)
    raw = (data_dir / COLLECTION_FILE).read_bytes()
    manifest = json.loads(raw)
    if (manifest['schemaVersion'] != 1 or manifest['source'] != 'CFETS / Shibor'
            or manifest['researchReady'] is not True
            or manifest['dataStart'] != DATA_START or manifest['dataEnd'] != DATA_END
            or manifest['rateFields'] != list(SHIBOR_FIELDS)
            or manifest['sourceUnit'] != 'percent' or manifest['decimalMultiplier'] != 0.01
            or manifest['currency'] != 'CNY'):
        raise ValueError('人民币拆息来源或封存规格未通过准入。')
    hashes = {COLLECTION_FILE: _digest(raw)}
    for filename, expected_hash in manifest['fileHashes'].items():
        if not filename.startswith('raw/rmb-rates-research/') or '..' in Path(filename).parts:
            raise ValueError('人民币拆息准入证据路径无效。')
        hashes[filename] = _digest((data_dir / filename).read_bytes())
        if hashes[filename] != expected_hash:
            raise ValueError('人民币拆息单位、时点、样本或请求证据已改变。')
    def read_verified(item):
        filename = item['file']
        if not filename.startswith('raw/rmb-rates-research/') or '..' in Path(filename).parts:
            raise ValueError('人民币拆息来源路径无效。')
        content = (data_dir / filename).read_bytes()
        hashes[filename] = _digest(content)
        if hashes[filename] != item['sha256']:
            raise ValueError('人民币拆息或银行间日历原响应指纹已改变。')
        response = json.loads(content)
        if response['head']['rep_code'] != '200':
            raise ValueError('人民币拆息或银行间日历原响应非成功状态。')
        return response
    holidays_by_year = {}
    for source in manifest['calendarSources']:
        response = read_verified(source)
        selected = source['selectedYear']
        if response['data']['year'] != str(selected):
            raise ValueError('银行间日历响应不是所查询年份。')
        holidays = pd.to_datetime(response['data']['currentHoliday'], format='%d/%m/%Y', errors='raise').as_unit('ns')
        source_years = {selected - 1, selected, selected + 1}
        study_years = set(range(pd.Timestamp(DATA_START).year, pd.Timestamp(DATA_END).year + 1))
        if (holidays.duplicated().any() or set(holidays.year) != source_years
                or source['coveredYears'] != sorted(source_years & study_years)):
            raise ValueError('银行间日历休市日期重复或覆盖年份不正确。')
        for year in source['coveredYears']:
            observed = pd.DatetimeIndex(holidays[holidays.year == year]).sort_values()
            if year in holidays_by_year and not holidays_by_year[year].equals(observed):
                raise ValueError('不同官方日历响应的重叠年份存在差异。')
            holidays_by_year[year] = observed
    expected_years = list(range(pd.Timestamp(DATA_START).year, pd.Timestamp(DATA_END).year + 1))
    if [item['year'] for item in manifest['annualWindows']] != expected_years:
        raise ValueError('人民币拆息年度窗口缺失、重复或顺序不正确。')
    frames, audits = [], []
    for item in manifest['annualWindows']:
        year = item['year']
        if year not in holidays_by_year:
            raise ValueError('缺少独立的人民币银行间休市日历。')
        response = read_verified(item)
        start = max(pd.Timestamp(DATA_START), pd.Timestamp(year=year, month=1, day=1))
        end = min(pd.Timestamp(DATA_END), pd.Timestamp(year=year, month=12, day=31))
        if (response['data']['startDateCN'] != start.strftime('%Y-%m-%d')
                or response['data']['endDateCN'] != end.strftime('%Y-%m-%d')
                or response['data']['message'] != '' or response['data']['messageEn'] != ''):
            raise ValueError('人民币拆息查询被拒绝或返回区间不一致，不能把空响应当休市。')
        frame = pd.DataFrame(response['records'])[['showDateCN', *SHIBOR_FIELDS]].copy()
        frame['showDateCN'] = pd.to_datetime(frame.showDateCN, format='%Y-%m-%d', errors='raise').dt.as_unit('ns')
        if frame.empty or frame.showDateCN.duplicated().any() or not frame.showDateCN.is_monotonic_decreasing:
            raise ValueError('人民币拆息官方年度响应为空、重复或排序不正确。')
        for field in SHIBOR_FIELDS:
            if not frame[field].map(lambda value: isinstance(value, str) and re.fullmatch(r'-?\d+(?:\.\d+)?', value) is not None).all():
                raise ValueError('人民币拆息期限存在未知空值或非数值原文。')
            frame[field] = frame[field].astype(float) / 100.0
        observed = pd.DatetimeIndex(frame.showDateCN).sort_values()
        expected = pd.date_range(start, end).as_unit('ns').difference(holidays_by_year[year])
        if not observed.equals(expected):
            raise ValueError(f'{year} 人民币拆息与独立银行间日历不符，缺失{expected.difference(observed).strftime("%Y-%m-%d").tolist()}，额外{observed.difference(expected).strftime("%Y-%m-%d").tolist()}。')
        if not np.isfinite(frame[list(SHIBOR_FIELDS)].to_numpy()).all():
            raise ValueError('人民币拆息包含非有限值。')
        frames.append(frame)
        audits.append({'year': year, 'rows': len(frame), 'officialWorkingWeekendRows': int((observed.dayofweek >= 5).sum())})
    result = pd.concat(frames, ignore_index=True).sort_values('showDateCN').rename(columns={'showDateCN': 'shiborSourceDate'}).reset_index(drop=True)
    if result.shiborSourceDate.duplicated().any() or len(result) != manifest['rows']:
        raise ValueError('人民币拆息合并后的总数或唯一性不正确。')
    dates_hash = _digest('\n'.join(result.shiborSourceDate.dt.strftime('%Y-%m-%d')).encode())
    if dates_hash != manifest['dateSha256']:
        raise ValueError('人民币拆息真实定盘日期指纹不一致。')
    return result, {**manifest, 'fileHashes': hashes, 'calendarAudit': audits, 'manifestSha256': hashes[COLLECTION_FILE]}


def build_shibor_factors(data_dir: Path, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    rates, manifest = read_shibor_snapshot(data_dir)
    native = pd.DataFrame({
        'shiborSourceDate': rates.shiborSourceDate,
        'shibor_1m': rates['1M'], 'shibor_term_spread': rates['1Y'] - rates['1M'],
        'shibor_change_21': rates['1M'] - rates['1M'].shift(21),
        'shibor_change_63': rates['1M'] - rates['1M'].shift(63),
    })
    aligned = pd.merge_asof(pd.DataFrame({'date': calendar}), native,
                            left_on='date', right_on='shiborSourceDate', direction='backward', allow_exact_matches=False)
    admitted = aligned.dropna(subset=list(SHIBOR_KEYS)).copy()
    admitted['shiborSourceAgeCalendarDays'] = (admitted.date - admitted.shiborSourceDate).dt.days
    if admitted.empty or not (admitted.shiborSourceDate < admitted.date).all():
        raise ValueError('人民币拆息因子存在时间越界或没有足够历史。')
    if not np.isfinite(admitted[list(SHIBOR_KEYS)].to_numpy()).all():
        raise ValueError('人民币拆息因子不是有限值。')
    return admitted, {**manifest, 'featureNames': SHIBOR_FEATURE_NAMES, 'factorRows': len(admitted),
                      'excludedWarmupDates': len(aligned) - len(admitted),
                      'timingRule': '严格早于香港信号日的最近真实人民币定盘；21/63阶差在银行间原生定盘序列计算，保留实际补班周末，不生成未公布利率。',
                      'vintageCaveat': '当前取得的官方历史版本无逐日历史公布时刻或修订日志；响应时间和配置时间不是过去的公布时间。'}
