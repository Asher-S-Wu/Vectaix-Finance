"""导入 QVeris iFinD 的前复权港股行情响应（cps=2，fill=Blank）。"""
import argparse
import json
from pathlib import Path
import pandas as pd


def import_quotes(source, output):
    batches = json.loads(Path(source).read_text(encoding='utf-8-sig'))
    frame = pd.DataFrame([row for batch in batches for row in batch])
    frame = frame.rename(columns={'time': 'trade_date', 'thscode': 'wind_code', 'amount': 'amt'})
    frame['wind_code'] = frame.wind_code.str.split('.').str[0].str.zfill(5) + '.HK'
    if frame.duplicated(['trade_date', 'wind_code']).any():
        raise ValueError('行情响应包含重复股票日期')
    frame[['trade_date', 'wind_code', 'close', 'volume', 'amt']].to_csv(output, index=False)
    print(f'{frame.wind_code.nunique()} 只股票，{len(frame)} 行')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('output')
    args = parser.parse_args()
    import_quotes(args.source, args.output)
