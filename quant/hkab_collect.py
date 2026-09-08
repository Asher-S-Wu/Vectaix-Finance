from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.parse import urlencode, urlparse

from .data import research_calendar


START = '2014-01-22'
END = '2026-08-31'
OVERLAP_DATES = ('2012-01-03', '2013-06-03', '2014-01-21')
SAMPLE_DATES = (*OVERLAP_DATES, START, '2014-01-23', '2014-01-31')
MAX_REQUESTS = 3500
CONCURRENCY = 3
MIN_START_INTERVAL = 0.3
TIMEOUT = 45
RATE_FIELDS = {
    'Overnight': 'ir_overnight', '1 Week': 'ir_1w', '1 Month': 'ir_1m',
    '3 Months': 'ir_3m', '6 Months': 'ir_6m', '12 Months': 'ir_12m',
}


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _weekdays() -> list[str]:
    start, end = date.fromisoformat(START), date.fromisoformat(END)
    return [(start + timedelta(days=offset)).isoformat() for offset in range((end - start).days + 1)
            if (start + timedelta(days=offset)).weekday() < 5]


def _url(day: str) -> str:
    current = date.fromisoformat(day)
    return 'https://www.hkab.org.hk/api/hibor?' + urlencode({'year': current.year, 'month': current.month, 'day': current.day})


def _normalized_time(value: str) -> str:
    match = re.fullmatch(r'(\d{1,2}):(\d{2})(am|pm)', value.lower())
    if match is None:
        raise ValueError('HKAB 非假日公布时间没有明确时分及上下午。')
    hour, minute, meridiem = int(match[1]), int(match[2]), match[3]
    if (not (0 <= hour <= 23 and 0 <= minute <= 59)
            or (hour > 12 and meridiem != 'pm') or (hour == 0 and meridiem == 'pm')):
        raise ValueError('HKAB 非假日公布时间超出有效时分范围。')
    if meridiem == 'am' and hour == 12:
        hour = 0
    elif meridiem == 'pm' and hour < 12:
        hour += 12
    return f'{hour:02d}:{minute:02d}:00'


def _weather_closure(response: dict) -> bool:
    message = response.get('ds_fb_msg_1_en')
    return (response['isHoliday'] is False and response.get('ds_is_fb') == 'Y'
            and all(response[key] is None for key in ('year', 'month', 'day', 'date'))
            and all(response[name] is None for name in RATE_FIELDS)
            and isinstance(message, str) and 'adverse weather condition' in message
            and 'banks will close to the public' in message and 'next business day' in message)


def _validate_response(raw: bytes, day: str) -> dict:
    response = json.loads(raw, parse_float=Decimal)
    expected = date.fromisoformat(day)
    if not isinstance(response, dict) or type(response['isHoliday']) is not bool:
        raise ValueError('HKAB 必须明确返回布尔 isHoliday。')
    if response['isHoliday']:
        if any(response[key] is not None for key in ('year', 'month', 'day', 'date')):
            raise ValueError('HKAB 假日响应应按实测官方结构明确返回空日期。')
        if any(response[name] is not None for name in RATE_FIELDS):
            raise ValueError('HKAB 假日响应包含非空利率。')
        if not isinstance(response['time'], str):
            raise ValueError('HKAB 假日模板时间不是字符串。')
        return response
    if _weather_closure(response):
        if not isinstance(response['time'], str):
            raise ValueError('HKAB 天气停业模板时间不是字符串。')
        return response
    if any(type(response[key]) is not int for key in ('year', 'month', 'day')):
        raise ValueError('HKAB 原响应日期字段不是整数。')
    if date(response['year'], response['month'], response['day']) != expected:
        raise ValueError('HKAB 原响应日期与请求日期不一致。')
    parts = response['date'].split('-')
    if len(parts) != 3 or date(*(int(part) for part in parts)) != expected:
        raise ValueError('HKAB date 字符串与请求日期不一致。')
    if not response['isHoliday']:
        _normalized_time(response['time'])
        for name in RATE_FIELDS:
            value = response[name]
            if isinstance(value, bool) or not isinstance(value, (int, Decimal)) or not Decimal(value).is_finite():
                raise ValueError(f'HKAB 非假日的 {name} 不是有限原始数值。')
    return response


def _record(day: str, raw: bytes, response: dict) -> dict:
    status = ('holiday' if response['isHoliday'] else
              'weather_closure_no_fixing' if _weather_closure(response) else 'fixing')
    return {'date': day, 'file': f'raw/hkab/{day}.json', 'sha256': _digest(raw),
            'isHoliday': response['isHoliday'], 'time': response['time'],
            'sourceStatus': status,
            'publishedAt': f'{day}T{_normalized_time(response["time"])}+08:00' if status == 'fixing' else None}


def _save(path: Path, manifest: dict) -> None:
    manifest['updatedAt'] = _utc_now()
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _audit(data_dir: Path, manifest: dict, records: dict[str, dict], expected: list[str]) -> None:
    manifest['records'] = [records[day] for day in expected if day in records]
    source = data_dir / 'raw/hkma/hibor-0000.json'
    source_raw = source.read_bytes()
    envelope = json.loads(source_raw, parse_float=Decimal)
    if envelope['result']['status_code'] != 200 or envelope['result']['data']['header']['success'] is not True:
        raise ValueError('HKMA 对照原响应没有成功状态。')
    hkma = {row['end_of_day']: row for row in envelope['result']['data']['result']['records']}
    overlaps = []
    for day in OVERLAP_DATES:
        if day not in records:
            continue
        item = records[day]
        raw = (data_dir / item['file']).read_bytes()
        if _digest(raw) != item['sha256']:
            raise ValueError('HKAB 对照文件在采集过程中变更。')
        response = _validate_response(raw, day)
        differences = []
        if response['isHoliday']:
            differences.append({'field': 'isHoliday', 'hkab': True, 'hkma': False})
        else:
            for hkab_field, hkma_field in RATE_FIELDS.items():
                if Decimal(response[hkab_field]) != Decimal(hkma[day][hkma_field]):
                    differences.append({'field': hkab_field, 'hkab': str(response[hkab_field]), 'hkma': str(hkma[day][hkma_field])})
        overlaps.append({**item, 'hkmaFile': 'raw/hkma/hibor-0000.json', 'hkmaSha256': _digest(source_raw),
                         'comparedFields': list(RATE_FIELDS), 'matched': not differences, 'differences': differences})
    manifest['overlap'] = overlaps
    sessions = set(research_calendar().strftime('%Y-%m-%d'))
    differences = [{'date': item['date'], 'isHoliday': item['isHoliday'],
                    'sourceStatus': item['sourceStatus'], 'xhkgSession': item['date'] in sessions}
                   for item in manifest['records'] if (item['sourceStatus'] != 'fixing') == (item['date'] in sessions)]
    classified = [item for item in differences
                  if item['sourceStatus'] == 'weather_closure_no_fixing' and item['xhkgSession']]
    discrepancies = [item for item in differences if item not in classified]
    policy_verified = not classified
    if classified and 'calendarPolicyEvidence' in manifest:
        evidence = manifest['calendarPolicyEvidence']
        if (evidence['effectiveFrom'] != '2024-09-23'
                or _digest((data_dir / evidence['file']).read_bytes()) != evidence['sha256']
                or any(item['date'] < evidence['effectiveFrom'] for item in classified)):
            raise ValueError('恶劣天气交易政策证据或适用日期不一致。')
        policy_verified = True
    coverage = len(manifest['records']) == len(expected)
    manifest['coverageComplete'] = coverage
    manifest['holidayReview'] = {
        'complete': coverage and not discrepancies and policy_verified,
        'rule': '逐个周一至周五保存原响应；明确假日及官方天气关闭、延期至次营业日定盘均单列为无定盘；按公开政策区分证券与银行营业状态，天气导致的股票开市但无定盘透明单列；未知缺文件、未知空值、半缺期限仍拒绝；不填利率。',
        'sourceHolidayDates': [item['date'] for item in manifest['records'] if item['isHoliday']],
        'weatherNoFixingDates': [item['date'] for item in manifest['records']
                                 if item['sourceStatus'] == 'weather_closure_no_fixing'],
        'calendarDiscrepancies': discrepancies,
        'classifiedCalendarDifferences': classified,
        'weatherNoFixingStockSessionDates': [item['date'] for item in classified],
        'calendarPolicyVerified': policy_verified,
    }
    manifest['nonstandardPublicationTimes'] = [
        {'date': item['date'], 'time': item['time'],
         'publishedAt': f'{item["date"]}T{_normalized_time(item["time"])}+08:00'} for item in manifest['records']
        if item['sourceStatus'] == 'fixing' and item['time'].lower() != '11:15am']
    manifest['researchReady'] = (coverage and not manifest['failures'] and manifest['holidayReview']['complete']
                                 and len(overlaps) == len(OVERLAP_DATES) and all(item['matched'] for item in overlaps))


def collect(data_dir: Path, *, sample: bool = False) -> dict:
    data_dir = Path(data_dir)
    folder = data_dir / 'raw/hkab'
    folder.mkdir(parents=True, exist_ok=True)
    path = data_dir / 'hkab-collection.json'
    expected = _weekdays()
    all_dates = sorted(set(expected) | set(OVERLAP_DATES))
    with (folder / '.collection.lock').open('a') as process_lock:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            manifest = json.loads(path.read_bytes())
            if (manifest['schemaVersion'] != 1 or manifest['source'] != 'HKAB'
                    or manifest['start'] != START or manifest['end'] != END
                    or manifest['expectedWeekdays'] != len(expected) or manifest['overlapDates'] != list(OVERLAP_DATES)):
                raise ValueError('已有 HKAB 采集进度与固定窗口不一致。')
        else:
            manifest = {'schemaVersion': 1, 'source': 'HKAB', 'start': START, 'end': END,
                        'researchReady': False, 'expectedWeekdays': len(expected), 'records': [], 'failures': [],
                        'overlapDates': list(OVERLAP_DATES), 'overlap': [], 'requests': [],
                        'createdAt': _utc_now(),
                        'requestPolicy': {'maxRequests': MAX_REQUESTS, 'concurrency': CONCURRENCY,
                                          'minimumStartIntervalSeconds': MIN_START_INTERVAL, 'timeoutSeconds': TIMEOUT,
                                          'automaticRetry': False},
                        'endpointEvidence': {'file': 'raw/hkma/hkab-request.js',
                                             'sha256': _digest((data_dir / 'raw/hkma/hkab-request.js').read_bytes())}}
        endpoint = manifest['endpointEvidence']
        if (endpoint['file'] != 'raw/hkma/hkab-request.js'
                or _digest((data_dir / endpoint['file']).read_bytes()) != endpoint['sha256']):
            raise ValueError('HKAB 官方接口页面脚本证据指纹不同。')
        known = {item['date']: item for item in manifest['records'] + manifest['overlap']}
        if len(known) != len(manifest['records']) + len(manifest['overlap']):
            raise ValueError('HKAB 已有进度日期重复。')
        failures = {item['date']: item for item in manifest['failures']}
        attempted = {item['date'] for item in manifest['requests']}
        saved_attempts = {item['date']: item for item in manifest['requests'] if 'file' in item}
        records = {}
        for day in all_dates:
            source = folder / f'{day}.json'
            if source.exists():
                raw = source.read_bytes()
                if day in failures:
                    attempt = saved_attempts[day]
                    if attempt['file'] != f'raw/hkab/{day}.json' or _digest(raw) != attempt['sha256']:
                        raise ValueError(f'{day} 已保存失败响应与原请求指纹不同。')
                    continue
                if day in known and _digest(raw) != known[day]['sha256']:
                    raise ValueError(f'{day} 已存 HKAB 文件与进度指纹不同。')
                response = _validate_response(raw, day)
                record = _record(day, raw, response)
                if day in known and any(record[key] != known[day][key] for key in ('date', 'file', 'sha256', 'isHoliday', 'time')):
                    raise ValueError(f'{day} 已存 HKAB 文件与进度元数据不同。')
                if day in known and 'reusedFromFile' in known[day]:
                    original = known[day]
                    if _digest((data_dir / original['reusedFromFile']).read_bytes()) != original['reusedFromSha256']:
                        raise ValueError(f'{day} 已批准复用的原始来源指纹不同。')
                    record.update({key: original[key] for key in
                                   ('reusedFromFile', 'reusedFromSha256', 'reusedFromURL')})
                records[day] = record
            elif day in known:
                raise ValueError(f'{day} 进度已有记录但原响应文件缺失。')
            elif day in attempted and day not in failures:
                failures[day] = {'date': day, 'url': _url(day), 'attemptedAt': None,
                                 'error': '此前请求已开始但未留下完整响应，不自动重试。', 'httpStatus': None}
        manifest['failures'] = sorted(failures.values(), key=lambda item: item['date'])
        _audit(data_dir, manifest, records, expected)
        _save(path, manifest)
        selected_dates = sorted(SAMPLE_DATES) if sample else all_dates
        pending = [day for day in selected_dates if day not in records and day not in attempted]
        if len(manifest['requests']) + len(pending) > MAX_REQUESTS:
            raise ValueError('剩余日期将超过本轮3500次免费HTTP授权上限。')
        lock = threading.Lock()
        last_start = time.monotonic()
        last_wall = datetime.fromisoformat(manifest['requests'][-1]['startedAt']) if manifest['requests'] else None
        if last_wall is not None:
            elapsed = max(0.0, (datetime.now(timezone.utc) - last_wall).total_seconds())
            last_start -= elapsed
        else:
            last_start -= MIN_START_INTERVAL

        def acquire(day: str) -> dict:
            nonlocal last_start
            url = _url(day)
            with lock:
                delay = MIN_START_INTERVAL - (time.monotonic() - last_start)
                if delay > 0:
                    time.sleep(delay)
                request_entry = {'date': day, 'url': url, 'startedAt': _utc_now(), 'status': 'started'}
                manifest['requests'].append(request_entry)
                _save(path, manifest)
                last_start = time.monotonic()
            raw, status, final_url, error = None, None, None, None
            try:
                temporary = folder / f'{day}.response.part'
                completed = subprocess.run([
                    '/usr/bin/curl', '--silent', '--show-error', '--connect-timeout', '15',
                    '--max-time', str(TIMEOUT), '--user-agent', 'Mozilla/5.0',
                    '--header', 'Accept: application/json', '--output', str(temporary),
                    '--write-out', '%{http_code}\n%{url_effective}', '--url', url,
                ], capture_output=True, check=False)
                details = completed.stdout.decode('utf-8').splitlines()
                status, final_url = int(details[0]), details[1]
                if temporary.exists():
                    raw = temporary.read_bytes()
                    temporary.unlink()
                if completed.returncode:
                    raise ValueError(f'curl exit {completed.returncode}: {completed.stderr.decode("utf-8").strip()}')
                if urlparse(final_url).hostname != 'www.hkab.org.hk':
                    raise ValueError('HKAB 请求来源不是已批准的银行公会域名。')
                if status != 200:
                    raise ValueError(f'HTTP {status}')
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            if raw is not None:
                target = folder / f'{day}.json'
                temporary = target.with_suffix('.json.part')
                temporary.write_bytes(raw)
                temporary.replace(target)
            record = None
            if error is None:
                try:
                    record = _record(day, raw, _validate_response(raw, day))
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
            with lock:
                request_entry.update({'finishedAt': _utc_now(), 'httpStatus': status, 'finalURL': final_url,
                                      'status': 'failed' if error else 'complete'})
                if raw is not None:
                    request_entry.update({'file': f'raw/hkab/{day}.json', 'sha256': _digest(raw), 'bytes': len(raw)})
                if error:
                    failures[day] = {'date': day, 'url': url, 'attemptedAt': request_entry['startedAt'],
                                     'error': error, 'httpStatus': status}
                    manifest['failures'] = sorted(failures.values(), key=lambda item: item['date'])
                else:
                    records[day] = record
                manifest['records'] = [records[current] for current in expected if current in records]
                manifest['overlap'] = [records[current] for current in OVERLAP_DATES if current in records]
                manifest['researchReady'] = False
                _save(path, manifest)
            return {'date': day, 'success': error is None, 'error': error}

        print(json.dumps({'stage': 'started', 'pendingRequests': len(pending), 'existingValidated': len(records),
                          'previousRequests': len(manifest['requests']), 'sample': sample}), flush=True)
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = [pool.submit(acquire, day) for day in pending]
            for number, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if number % 100 == 0 or not result['success'] or sample:
                    print(json.dumps({'completedThisRun': number, 'totalThisRun': len(pending), **result}), flush=True)
        _audit(data_dir, manifest, records, expected)
        _save(path, manifest)
        result = {'researchReady': manifest['researchReady'], 'records': len(manifest['records']),
                  'expectedWeekdays': len(expected), 'requests': len(manifest['requests']),
                  'failures': len(manifest['failures']), 'overlap': manifest['overlap'],
                  'calendarDiscrepancies': manifest['holidayReview']['calendarDiscrepancies']}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()
    collect(args.data_dir, sample=args.sample)


if __name__ == '__main__':
    main()
