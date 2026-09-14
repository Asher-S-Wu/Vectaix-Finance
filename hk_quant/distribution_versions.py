"""按实际公开时间读取分派版本，供历史数据构建使用。"""
import argparse
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd


# Only facts belonging to the original version; present-day review/supersession
# annotations are deliberately absent from a historical information set.
VERSION_FIELDS = (
    'security_id', 'issueID', 'event_id', 'event_type', 'published_at',
    'source_url', 'source_document', 'ex_date', 'record_date', 'payment_date',
    'payment_date_type', 'payment_date_literal', 'cash_per_unit_decimal',
    'cash_currency', 'conditional_cash_amount', 'cash_amount_status',
    'reporting_period_end', 'financial_year_end', 'dividend_type',
    'announcement_status', 'rights_ratio', 'rights_subscription_price',
    'rights_currency', 'scrip_optional', 'scrip_default_option',
    'rights_new_units', 'rights_old_units', 'rights_terms_public',
    'scrip_price', 'scrip_currency', 'other_currency_per_unit_decimal',
    'other_currency', 'other_currency_status',
)


def distributions_asof(versions, cutoff):
    """保留当时最新原公告，包括条件和撤销；不认证现金到账或数据覆盖。"""
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError('截止时间必须包含时区')
    required = ['security_id', 'event_id', 'published_at']
    if not set(required).issubset(versions.columns):
        raise ValueError('分派版本缺少身份、事件或公开时间')
    frame = versions.loc[:, [c for c in VERSION_FIELDS if c in versions]].copy()
    if frame[['security_id', 'event_id']].isna().any().any():
        raise ValueError('分派版本缺少证券或事件身份')
    times = []
    for value in frame.published_at:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError('公开时间必须已知且包含时区')
        times.append(stamp.tz_convert('UTC'))
    frame['published_at'] = pd.to_datetime(times, utc=True)
    if frame.duplicated(required).any():
        raise ValueError('同一事件的公开时间重复，必须先核实版本')
    visible = frame.loc[frame.published_at.le(cutoff)]
    return (visible.sort_values('published_at')
            .drop_duplicates(['security_id', 'event_id'], keep='last')
            .sort_values(['security_id', 'event_id']).reset_index(drop=True))


def versioned_cash_actions(versions, cutoff):
    """将已审阅公告导入现金回放；公开信息选择和事后来源审核分别保留。"""
    selected = distributions_asof(versions, cutoff)
    reviewed = versions.copy()
    reviewed['published_at'] = pd.to_datetime(reviewed.published_at, utc=True)
    review_columns = ['security_id', 'event_id', 'published_at', 'verified',
                      'record_date_source_conflict', 'final_schedule_usable']
    selected = selected.merge(reviewed[review_columns],
                              on=review_columns[:3], validate='one_to_one')
    actions, gaps = [], []
    for row in selected.to_dict('records'):
        reason = None
        if row['verified'] != True or row['record_date_source_conflict'] != False:
            reason = 'unresolved_source_evidence'
        elif row['conditional_cash_amount'] != False or row['final_schedule_usable'] != True:
            reason = 'cash_schedule_not_final'
        elif row['payment_date_type'] != 'specified_payment_date':
            reason = 'payment_date_not_exact'
        elif row['scrip_optional'] and row['scrip_default_option'] != 'Cash':
            reason = 'cash_election_not_established'
        ex = pd.Timestamp(row['ex_date'])
        payment = pd.Timestamp(row['payment_date'])
        if pd.isna(ex) or pd.isna(payment) or payment < ex:
            reason = 'invalid_entitlement_or_payment_date'
        elif reason is None:
            # Capture units using a date that was already public before ex-date.
            before = distributions_asof(versions,
                ex.tz_localize('Asia/Hong_Kong') - pd.Timedelta(nanoseconds=1))
            before = before.loc[before.security_id.eq(row['security_id']) & before.event_id.eq(row['event_id'])]
            if len(before) != 1 or pd.Timestamp(before.iloc[0].ex_date) != ex:
                reason = 'ex_date_not_known_before_entitlement'
        if reason:
            gaps.append(dict(security_id=row['security_id'], event_id=row['event_id'],
                             reason=reason, source_url=row['source_url']))
            continue
        amount = Decimal(row['cash_per_unit_decimal'])
        if not amount.is_finite() or amount < 0 or row['cash_currency'] not in ('HKD', 'CNY', 'USD'):
            raise ValueError('分派金额或币种无效')
        actions.append(dict(security_id=row['security_id'], event_id=row['event_id'],
            action_type='cash_dividend', effective_date=ex, payment_date=payment,
            amount_known_at=row['published_at'], cash_per_share=float(amount),
            cash_per_share_decimal=str(amount), cash_currency=row['cash_currency'],
            verified=True, source_url=row['source_url'],
            payment_basis='issuer_final_schedule_simulated'))
    result = pd.DataFrame(actions)
    result.attrs['coverage_complete'] = False
    return result, pd.DataFrame(gaps)


def build_daily_versions(source, output, start, end):
    """生成19:00香港时间信息集；日历天文件不代表交易日或完整事件覆盖。"""
    versions = pd.read_parquet(source)
    snapshots = []
    for day in pd.date_range(start, end):
        cutoff = day.tz_localize('Asia/Hong_Kong') + pd.Timedelta(hours=19)
        snapshot = distributions_asof(versions, cutoff)
        snapshot['as_of'] = cutoff
        snapshots.append(snapshot)
    if not snapshots:
        raise ValueError('开始日期不能晚于结束日期')
    result = pd.concat(snapshots, ignore_index=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    audit = dict(source=str(source), rows=len(result), start=start, end=end,
                 cutoff='19:00 Asia/Hong_Kong', source_versions=len(versions),
                 coverage_complete=False, approved_for_training=False,
                 cash_receipts_simulated=False,
                 purpose='Historical public information sets; not adjusted prices or payment ledger')
    output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    print(json.dumps(build_daily_versions(**vars(parser.parse_args()))))
