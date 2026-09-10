"""将 QVeris 港股日行情响应转为复权行情与真实成交额。"""
import argparse
import json
from pathlib import Path
import pandas as pd


def import_quotes(source, output):
    response = json.loads(Path(source).read_text(encoding='utf-8-sig'))
    rows = response['data']['data']['rows']
    total = response['_qveris_pagination']['total_count']
    if not rows or len(rows) != total:
        raise ValueError(f'行情响应不完整: {len(rows)}/{total}')
    frame = pd.DataFrame(rows)
    if frame.currency.ne('港元').any():
        raise ValueError('行情包含非港元交易柜台')
    frame['wind_code'] = frame.secucode.str.zfill(5) + '.HK'
    frame = frame.rename(columns={'tradingday': 'trade_date', 'close': 'raw_close',
                                  'befadjcloseprice': 'close', 'amount': 'amt'})
    frame['mktcap_hkd_yi'] = frame.marketcap / 10_000
    if frame.duplicated(['wind_code', 'trade_date']).any():
        raise ValueError('行情响应包含重复股票日期')
    columns = ['trade_date', 'wind_code', 'close', 'raw_close', 'volume', 'amt', 'mktcap_hkd_yi']
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    frame[columns].sort_values(['trade_date', 'wind_code']).to_csv(output, index=False)
    print(f'{len(frame)} 行，{frame.wind_code.nunique()} 只股票 -> {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('output')
    args = parser.parse_args()
    import_quotes(args.source, args.output)
