"""以原始成交逐日证据限定历史证券身份。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCOPE = 'explicit_matching_observation_dates'
ASSET_TYPES = {'Ordinary shares': 'equity', 'A-share': 'equity',
               'B-share': 'equity', 'H-share': 'equity'}


def identity_row_mask(bars, master):
    status = bars.security_id.map(master.identity_status)
    start = bars.security_id.map(master.identity_valid_from)
    end = bars.security_id.map(master.identity_valid_to)
    valid = status.eq('verified') & start.notna() & bars.date.ge(start) & (end.isna() | bars.date.lt(end))
    partial = status.eq('partially_verified')
    if partial.any() and {'identity_date_verified', 'identity_source_issue_id'} <= set(bars.columns):
        proof = (bars.identity_date_verified.eq(True)
                 & bars.identity_source_issue_id.notna()
                 & bars.identity_source_issue_id.eq(bars.security_id.map(master.identity_source_issue_id))
                 & bars.security_id.map(master.identity_evidence_scope).eq(SCOPE))
        valid |= partial & proof
    return valid


def validate_evidence(review, candidates, evidence):
    if review.security_id.duplicated().any() or candidates.candidate_id.duplicated().any():
        raise ValueError('身份审查或候选记录重复')
    if evidence.duplicated(['security_id', 'date']).any():
        raise ValueError('逐日身份证据重复')
    if not (evidence.all_match.eq(True) & evidence.quote_source.eq('quotes')
            & evidence.quote_present.eq(True) & evidence.volume.gt(0)
            & evidence.close_match.eq(True) & evidence.volume_match.eq(True)
            & evidence.amount_match.eq(True) & evidence.observed_conflict.eq(False)).all():
        raise ValueError('身份证据必须是完全匹配的原始正成交记录')
    records = review.set_index('security_id')
    source = candidates.set_index('candidate_id')
    for security, rows in evidence.groupby('security_id'):
        record = records.loc[security]
        if rows.issueID.nunique() != 1 or rows.issueID.iloc[0] != record.IssueID:
            raise ValueError('证券IssueID不一致')
        if record.review_status != 'unique_historical_candidate_all_overlap_matched':
            raise ValueError('证券未通过唯一候选审查')
        if record.source_type not in ASSET_TYPES or record.source_currency not in ('HKD', 'USD', 'CNY'):
            raise ValueError('证券类别或币种缺明确支持')
        matched = source.loc[rows.candidate_id]
        if not (matched.security_id.eq(security).all() and matched.IssueID.eq(record.IssueID).all()
                and matched.issuer.eq(record.issuer).all() and matched.typeLong.eq(record.source_type).all()
                and matched.issue_currency.eq(record.source_currency).all()):
            raise ValueError('候选证券属性与逐日证据不一致')
    return records.loc[evidence.security_id.unique()]


def apply_bar_evidence(bars, evidence, records, fx):
    if bars.duplicated(['security_id', 'date']).any():
        raise ValueError('行情日期重复')
    out = bars.copy()
    out['identity_date_verified'] = False
    out['identity_source_issue_id'] = pd.Series(pd.NA, index=out.index, dtype='Int64')
    keys = pd.MultiIndex.from_frame(out[['security_id', 'date']])
    proof = evidence.set_index(['security_id', 'date'])
    matched = keys.isin(proof.index)
    selected = proof.reindex(keys[matched])
    for column in ('raw_close', 'volume', 'amount'):
        actual = out.loc[matched, column].to_numpy()
        expected = selected[column].to_numpy()
        if pd.isna(actual).any() or pd.isna(expected).any() or not np.equal(actual, expected).all():
            raise ValueError(f'行情与核验快照不一致: {column}')
    out.loc[matched, 'identity_date_verified'] = True
    out.loc[matched, 'identity_source_issue_id'] = selected.issueID.to_numpy(dtype='int64')
    currencies = out.loc[matched, 'security_id'].map(records.source_currency)
    out.loc[matched, 'currency'] = currencies
    rate = pd.Series(np.nan, index=currencies.index)
    rate.loc[currencies.eq('HKD')] = 1.
    for currency, field in [('USD', 'usd'), ('CNY', 'cny')]:
        indices = currencies.index[currencies.eq(currency)]
        rate.loc[indices] = out.loc[indices, 'date'].map(fx[field])
    out.loc[matched, 'fx_to_hkd'] = rate
    # adj_close already uses the source's known cumulative adjustment factor.
    out.loc[matched, 'adj_close_hkd'] = out.loc[matched, 'adj_close'] * rate
    out.loc[matched, 'amount_hkd'] = out.loc[matched, 'amount'] * rate
    return out


def import_dated_identity(data_root, evidence_root):
    data_root, evidence_root = Path(data_root), Path(evidence_root)
    master = pd.read_parquet(data_root / 'securities.parquet')
    review = pd.read_csv(evidence_root / 'identity_review.csv')
    candidates = pd.read_csv(evidence_root / 'identity_candidates.csv')
    evidence = pd.read_parquet(evidence_root / 'identity_supported_observation_dates.parquet')
    records = validate_evidence(review, candidates, evidence)
    targeted = master.security_id.isin(records.index)
    if not master.loc[targeted, 'identity_status'].isin(['unresolved', 'partially_verified']).all():
        raise ValueError('导入仅适用于尚未整体核验的身份')
    if set(records.index) - set(master.security_id):
        raise ValueError('证据证券不在主表')
    fx = pd.DataFrame(json.loads((data_root / 'references/fx_research/hkma_usd_cny_hkd_2010_present.json').read_text(encoding='utf-8'))['records'])
    fx.index = pd.to_datetime(fx.end_of_day)
    if fx.index.duplicated().any():
        raise ValueError('汇率日期重复')
    paths = sorted((data_root / 'bars').glob('*.parquet'))
    # Validate every partition before writing any changes.
    supported = 0
    for path in paths:
        bars = apply_bar_evidence(pd.read_parquet(path), evidence, records, fx)
        supported += int(bars.identity_date_verified.sum())
    if supported != len(evidence):
        raise ValueError('逐日证据未完整对应行情分区')
    missing_fx = remaining = unresolved_rows = 0
    unresolved_ids = set(master.loc[master.identity_status.eq('unresolved'), 'security_id']) - set(records.index)
    for path in paths:
        bars = apply_bar_evidence(pd.read_parquet(path), evidence, records, fx)
        missing_fx += int((bars.identity_date_verified & bars.fx_to_hkd.isna()).sum())
        remaining += int((bars.security_id.isin(records.index) & ~bars.identity_date_verified).sum())
        unresolved_rows += int(bars.security_id.isin(unresolved_ids).sum())
        bars.to_parquet(path, index=False)
    master.loc[targeted, 'identity_status'] = 'partially_verified'
    master.loc[targeted, 'identity_period_status'] = 'partially_verified'
    master.loc[targeted, 'identity_evidence_scope'] = SCOPE
    for source, target in [('IssueID','identity_source_issue_id'), ('issuer','identity_source_issuer_id'),
                           ('source_type','identity_source_type'), ('source_currency','identity_source_currency'),
                           ('source_isin','identity_source_isin')]:
        master[target] = master.security_id.map(records[source])
    master.loc[targeted, 'asset_type'] = master.loc[targeted, 'security_id'].map(records.source_type.map(ASSET_TYPES))
    master.loc[targeted, 'asset_type_evidence_status'] = 'dated_source_issue_type'
    master.to_parquet(data_root / 'securities.parquet', index=False)
    audit = dict(evidence_scope=SCOPE, supported_securities=len(records), supported_rows=supported,
                 supported_types=records.source_type.value_counts().to_dict(),
                 supported_rows_by_type=evidence.security_id.map(records.source_type).value_counts().to_dict(),
                 remaining_unverified_rows_in_partial_identities=remaining,
                 remaining_unverified_rows=remaining+unresolved_rows,
                 remaining_fully_unresolved_securities=int(master.identity_status.eq('unresolved').sum()),
                 supported_rows_missing_fx=missing_fx, whole_history_verified=False, approved_for_training=False)
    (data_root / 'dated_identity_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--evidence-root', required=True)
    args = parser.parse_args()
    print(json.dumps(import_dated_identity(args.data_root, args.evidence_root), ensure_ascii=False, indent=2))
