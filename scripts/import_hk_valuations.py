"""导入 QVeris 港股滚动市盈率和市净率历史数据。"""
import argparse
import json
from pathlib import Path
import pandas as pd

from project_paths import data_dir


def import_valuations(source):
    response = json.loads(Path(source).read_text(encoding='utf-8-sig'))
    rows = response['data']['data']['rows']
    if not rows or len(rows) != response['_qveris_pagination']['total_count']:
        raise ValueError('估值响应分页不完整，不能导入')
    frame = pd.DataFrame(rows)
    frame = frame.rename(columns={'tradingday': 'date', 'pettm': '市盈率(TTM)', 'pb': '市净率'})
    frame['总市值'] = frame.totalmarketcap / 100_000_000
    root = data_dir('hk2') / 'raw/valuations'
    for code, group in frame.groupby('stockcode'):
        symbol = code.split('.')[0].zfill(5)
        if group.date.duplicated().any():
            raise ValueError(f'{code} 存在重复估值日期')
        group[['date', '总市值', '市盈率(TTM)', '市净率']].sort_values('date').to_csv(root / f'{symbol}.csv', index=False)
    print(f'导入 {frame.stockcode.nunique()} 只股票，{len(frame)} 行估值')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    args = parser.parse_args()
    import_valuations(args.source)
