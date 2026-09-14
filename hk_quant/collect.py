"""按月份采集全港股原始数据，不以当前股票名单截断历史。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import gzip
import json
from pathlib import Path
import sys

import pandas as pd

from .paths import DATA

FIELDS = {
    'hk_daily_adj': 'ts_code,trade_date,close,open,high,low,vol,amount,vwap,adj_factor,turnover_ratio,free_share,total_share,free_mv,total_mv',
    'hk_adjfactor': 'ts_code,trade_date,cum_adjfactor,close_price',
}
PAGE_SIZE = 6000


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def validate_page(response, api, start, end):
    if response['code'] != 0 or response['data']['fields'] != FIELDS[api].split(','):
        raise ValueError(f'{api}: 返回状态或字段不符')
    frame = pd.DataFrame(response['data']['items'], columns=response['data']['fields'])
    if not frame.empty:
        if not frame.trade_date.between(start, end).all():
            raise ValueError(f'{api}: 返回日期超出请求区间')
        if frame.duplicated(['ts_code', 'trade_date']).any():
            raise ValueError(f'{api}: 页面有重复证券日期')
    return frame


def collect_month(client, api, start, end, data_root=DATA):
    key = start[:6]
    output = data_root / 'source' / api / f'{key}.parquet'
    metadata_file = output.with_suffix('.json')
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
        if metadata['end_date'] == end and metadata['status'] == 'complete':
            pd.read_parquet(output, columns=['ts_code', 'trade_date'])
            return metadata
    frames, requests, observed = [], [], set()
    offset = 0
    while True:
        params = {'start_date': start, 'end_date': end, 'limit': PAGE_SIZE, 'offset': offset}
        cached = data_root / 'raw' / api / f'{start}_{end}' / f'{offset:08d}.json.gz'
        if cached.exists():
            with gzip.open(cached, 'rt', encoding='utf-8') as stream:
                record = json.load(stream)
        else:
            response = client.query(api, params, FIELDS[api])
            record = {'api_name': api, 'params': params, 'fields': FIELDS[api],
                      'http_status': 200, 'queried_at': datetime.now().astimezone().isoformat(),
                      'response': response}
            validate_page(response, api, start, end)
            cached.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(cached, 'wt', encoding='utf-8') as stream:
                json.dump(record, stream, ensure_ascii=False, allow_nan=False)
        frame = validate_page(record['response'], api, start, end)
        keys = set(zip(frame.ts_code, frame.trade_date))
        if observed.intersection(keys):
            raise ValueError(f'{api}/{key}: 分页重复，不能接受截断或重叠数据')
        observed.update(keys)
        frames.append(frame)
        requests.append({'params': params, 'rows': len(frame), 'http_status': 200,
                         'business_code': record['response']['code'],
                         'source': record['response'].get('source'), 'queried_at': record['queried_at']})
        if len(frame) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    combined = pd.concat(frames, ignore_index=True)
    combined['trade_date'] = pd.to_datetime(combined.trade_date, format='%Y%m%d')
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.sort_values(['trade_date', 'ts_code']).to_parquet(output, index=False)
    metadata = {'status': 'complete', 'api_name': api, 'start_date': start, 'end_date': end,
                'rows': len(combined), 'securities': int(combined.ts_code.nunique()),
                'fields': FIELDS[api], 'requests': requests, 'file': str(output)}
    write_json(metadata_file, metadata)
    return metadata


def collect_references(client, start, end, data_root=DATA):
    references = data_root / 'references'
    basics = []
    for status in ('L', 'D', 'P'):
        params = {'list_status': status}
        response = client.query('hk_basic', params)
        write_json(references / f'hk_basic_{status}.json', {
            'api_name': 'hk_basic', 'params': params, 'http_status': 200,
            'queried_at': datetime.now().astimezone().isoformat(), 'response': response})
        basics.append(pd.DataFrame(response['data']['items'], columns=response['data']['fields']))
    pd.concat(basics, ignore_index=True).to_parquet(references / 'hk_basic.parquet', index=False)
    calendars = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        params = {'start_date': max(start, f'{year}0101'), 'end_date': min(end, f'{year}1231')}
        response = client.query('hk_tradecal', params, 'cal_date,is_open,pretrade_date')
        if len(response['data']['items']) == 2000:
            raise ValueError('交易日历被截断')
        write_json(references / f'hk_tradecal_{year}.json', {
            'api_name': 'hk_tradecal', 'params': params, 'http_status': 200,
            'queried_at': datetime.now().astimezone().isoformat(), 'response': response})
        calendars.append(pd.DataFrame(response['data']['items'], columns=response['data']['fields']))
    calendar = pd.concat(calendars, ignore_index=True)
    calendar['cal_date'] = pd.to_datetime(calendar.cal_date, format='%Y%m%d')
    if calendar.cal_date.duplicated().any():
        raise ValueError('交易日历重复')
    calendar.sort_values('cal_date').to_parquet(references / 'calendar.parquet', index=False)


def main():
    parser = argparse.ArgumentParser(description='采集全港股长期行情与精确复权数据')
    parser.add_argument('--skill-root', type=Path, required=True)
    parser.add_argument('--start', default='20100101')
    parser.add_argument('--end', required=True)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--references-only', action='store_true')
    parser.add_argument('--skip-references', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.skill_root / 'scripts'))
    from tushare_client import RelayClient, load_settings
    settings = load_settings(root=args.skill_root)
    client = RelayClient(settings.api_key, settings.base_url, timeout=90,
                         max_retries=2, interval_seconds=1.6)
    if not args.skip_references:
        collect_references(client, args.start, args.end, args.data_root)
    if args.references_only:
        return
    periods = pd.period_range(pd.Timestamp(args.start), pd.Timestamp(args.end), freq='M')
    tasks = [(api, max(args.start, period.start_time.strftime('%Y%m%d')),
              min(args.end, period.end_time.strftime('%Y%m%d'))) for period in periods for api in FIELDS]
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = {executor.submit(collect_month, client, api, start, end, args.data_root): (api, start, end)
                for api, start, end in tasks}
        for future in as_completed(jobs):
            api, start, end = jobs[future]
            try:
                result = future.result()
                results.append(result)
                print(f'{len(results)}/{len(tasks)} {api} {start[:6]} {result["rows"]} rows', flush=True)
            except Exception as error:
                errors.append({'api_name': api, 'start': start, 'end': end, 'error': str(error)})
                print(f'ERROR {api} {start}: {error}', flush=True)
            write_json(args.data_root / 'collection_status.json', {
                'status': 'collecting', 'requested_start': args.start, 'requested_end': args.end,
                'completed_month_apis': len(results), 'total_month_apis': len(tasks),
                'rows': sum(r['rows'] for r in results), 'errors': errors})
    write_json(args.data_root / 'collection_status.json', {
        'status': 'complete' if not errors else 'incomplete',
        'requested_start': args.start, 'requested_end': args.end,
        'completed_month_apis': len(results), 'total_month_apis': len(tasks),
        'rows': sum(r['rows'] for r in results), 'errors': errors})
    if errors:
        raise RuntimeError(f'{len(errors)}个数据分区未完成')


if __name__ == '__main__':
    main()
