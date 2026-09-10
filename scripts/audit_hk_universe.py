"""导出当前季度候选池、数据缺口及历史名单覆盖情况。"""
import argparse
import json
import pandas as pd

from hk_universe import load_memberships, quarterly_pool, complete_factors
from factor_lib import compute_factors
from project_paths import data_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--as-of', type=pd.Timestamp)
    args = parser.parse_args()
    root = data_dir('hk2')
    px = pd.read_pickle(root / 'processed/prices.pkl')
    memberships = load_memberships()
    date = args.as_of if args.as_of is not None else px['close'].index[-1]
    audit = quarterly_pool(px['close'], px['amt'], px['volume'], memberships, date)
    val = pd.read_pickle(root / 'processed/valuations.pkl')
    factors = compute_factors(px['close'].loc[:date], px['amt'].loc[:date], px['volume'].loc[:date],
                              val_wide={key: values.loc[:date] for key, values in val.items()}, eval_dates=[date])
    features = [c for c in factors if c not in ('date', 'code')]
    complete_codes = factors.loc[complete_factors(factors, features), 'code']
    audit['factors_complete'] = audit.code.isin(complete_codes)
    trading = px['close'].loc[date].gt(0) & px['amt'].loc[date].gt(0) & px['volume'].loc[date].gt(0)
    audit['trading_at_signal'] = audit.code.isin(trading.index[trading])
    audit['ready_for_scoring'] = audit.selected & audit.factors_complete & audit.trading_at_signal
    audit.to_csv(root / 'reference/universe_audit.csv', index=False)
    audit.loc[audit.ready_for_scoring].to_csv(root / 'reference/universe_eligible.csv', index=False)
    latest = memberships[memberships.as_of.eq(audit.as_of.iloc[0])]
    latest[['code', 'name', 'market_cap_hkd']].rename(columns={'code': 'windcode'}).to_csv(
        root / 'reference/universe_candidates.csv', index=False)
    summary = {
        'as_of': str(date.date()), 'review_date': str(audit.as_of.iloc[0].date()),
        'target_size': 500, 'index_members': len(audit),
        'selected_with_available_data': int(audit.selected.sum()),
        'ready_for_scoring': int(audit.ready_for_scoring.sum()),
        'missing_price_columns': int((~audit.data_available).sum()),
        'insufficient_history': int((audit.data_available & ~audit.history_ok).sum()),
        'insufficient_liquidity_or_missing_amount': int((~audit.liquidity_ok).sum()),
        'historical_snapshot_dates': [str(x.date()) for x in sorted(memberships.as_of.unique())],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
