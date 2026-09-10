"""导入 QVeris 恒生综合指数历史快照，更新候选股票主表。"""
import argparse
import json
from pathlib import Path
import pandas as pd
from project_paths import data_dir


def import_snapshot(path, as_of):
    response = json.loads(path.read_text(encoding='utf-8-sig'))
    rows = pd.DataFrame(response['data']['data']['rows'])
    if len(rows) != response['_qveris_pagination']['total_count'] or rows.empty:
        raise ValueError('指数名单响应不完整')
    if not rows.indexcode.eq('HSCI.HK').all():
        raise ValueError('响应不是恒生综合指数成分股')
    if pd.to_datetime(rows.inclusiondate).gt(as_of).any():
        raise ValueError('名单包含快照日之后才纳入的股票')
    rows = rows.rename(columns={'stockcode': 'code', 'stockname': 'name',
                                'indexcode': 'index_code', 'inclusiondate': 'inclusion_date'})
    rows['as_of'] = as_of
    rows['market_cap_hkd'] = rows.totalmarketval * 100_000_000
    rows['source_file'] = path.name
    reference = data_dir('hk2') / 'reference'
    history_file = reference / 'hsci_memberships.csv'
    history = pd.read_csv(history_file, parse_dates=['as_of'])
    columns = ['as_of', 'code', 'name', 'market_cap_hkd', 'index_code', 'inclusion_date', 'source_file']
    history = pd.concat([history[history.as_of.ne(as_of)], rows[columns]], ignore_index=True)
    history.sort_values(['as_of', 'code']).to_csv(history_file, index=False)
    names = history.dropna(subset=['code', 'name']).sort_values('as_of').drop_duplicates('code', keep='last')
    names[['code', 'name']].rename(columns={'code': 'windcode'}).sort_values('windcode').to_csv(
        reference / 'universe.csv', index=False)
    print(f'已导入 {as_of.date()}：{len(rows)} 条成分记录；历史股票合计 {len(names)} 只')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='导入按指定历史日期查询的 QVeris 完整指数响应')
    parser.add_argument('--snapshot-file', type=Path, required=True)
    parser.add_argument('--as-of', type=pd.Timestamp, required=True)
    args = parser.parse_args()
    import_snapshot(args.snapshot_file, args.as_of)
