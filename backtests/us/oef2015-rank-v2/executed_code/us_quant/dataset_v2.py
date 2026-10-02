"""Immutable recovered-data panel; original sources and first experiment preserved."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import write_json,write_immutable_json,sha256
from .features import security_features,add_market_context,normalize_features
from .pipeline import absent_features,validate_calendar
from .recovery_v2 import reconstruct_original_bars

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/us/oef2015'
DATA=ROOT/'data/us/oef2015_v2'


def apply_gap_fills(raw,fills):
    raw=raw.copy();fills=fills.copy();raw['date']=pd.to_datetime(raw.date);fills['date']=pd.to_datetime(fills.date)
    if raw.date.isin(fills.date).any():raise ValueError('Explicit gap fill cannot overwrite an existing source row')
    return pd.concat([raw,fills],ignore_index=True).sort_values('date').reset_index(drop=True)


def assemble_panel(bars_by_identity,cohort,calendar):
    calendar=validate_calendar(calendar);features=[];prices=[];audit=[]
    for identity in cohort:
        bars=bars_by_identity.get(identity)
        if bars is None:
            frame=absent_features(identity,calendar,'source_history_unavailable')
            p=pd.DataFrame(dict(date=calendar,security_id=identity,adj_close=np.nan,quote_present=False,data_valid=False,dollar_volume_proxy=np.nan,adv20_amount=np.nan))
            audit.append(dict(security_id=identity,valid_quote_rows=0,eligible_rows=0,source_status='unavailable'))
        else:
            frame=security_features(bars,calendar)
            p=bars.set_index('date').reindex(calendar).reset_index(names='date');p['security_id']=identity
            for col in ['quote_present','data_valid']:p[col]=p[col].fillna(False).astype(bool)
            p=p.merge(frame[['date','adv20_amount']],on='date',validate='one_to_one')
            valid=bars.quote_present&bars.data_valid
            audit.append(dict(security_id=identity,valid_quote_rows=int(valid.sum()),eligible_rows=int(frame.status.eq('ok').sum()),
                first_valid_quote=str(bars.loc[valid,'date'].min().date()) if valid.any() else None,last_valid_quote=str(bars.loc[valid,'date'].max().date()) if valid.any() else None,source_status='observed_original_identity_segment',
                mature_endpoint_rows={str(h):int(frame[f'fwd_return_{h}'].notna().sum()) for h in [1,5,20,60]}))
        features.append(frame);prices.append(p)
    features=add_market_context(pd.concat(features,ignore_index=True),calendar)
    return features,normalize_features(features),pd.concat(prices,ignore_index=True),audit


def build(data=DATA):
    data=Path(data);output=data/'source_audit.json'
    if output.exists() or any((data/f'{name}.parquet').exists() for name in ['features','normalized','prices']):raise FileExistsError('Derived v2 dataset already exists; use a new version')
    plan=json.loads((data/'sources/admissions.json').read_text());universe=pd.read_csv(ROOT/'docs/us/declared_universe.csv')
    base_audit=json.loads((ROOT/'backtests/us/oef2015/source_audit.json').read_text())
    base_review=json.loads((ROOT/'docs/us/identity_review.json').read_text())
    calendar=validate_calendar(pd.read_csv(BASE/'calendar.csv',parse_dates=['date']).date)
    if sha256(BASE/'calendar.csv')!=base_audit['calendar_sha256']:raise ValueError('Original calendar changed')
    bars={}
    for identity in base_review['accepted_original_identities']:
        p=BASE/'bars'/f'{identity}.parquet'
        if sha256(p)!=base_audit['bar_sha256'][p.name]:raise ValueError('Original source bar changed')
        bars[identity]=pd.read_parquet(p)
    recovered=[]
    for record in plan['recovered']:
        identity=record['security_id']
        if identity in bars or identity not in set(universe.ticker_2015):raise ValueError('Recovered identity must be an originally missing cohort member')
        raw_path=data/'sources'/record['raw_file'];event_path=data/'sources'/record['actions_file']
        raw=pd.read_csv(raw_path,parse_dates=['date']);events=pd.read_csv(event_path,parse_dates=['ex_date'])
        if record.get('gap_fills_file'):
            gaps=pd.read_csv(data/'sources'/record['gap_fills_file']);gaps=gaps.loc[gaps.original_symbol.eq(identity)].copy()
            if len(gaps):raw=apply_gap_fills(raw,gaps)
        frame=reconstruct_original_bars(raw,events,calendar,identity,record['valid_start'],record['valid_end'])
        bars[identity]=frame
        observed=frame.loc[frame.quote_present&frame.data_valid]
        (data/'recovered_bars').mkdir(parents=True,exist_ok=True);frame.to_parquet(data/'recovered_bars'/f'{identity}.parquet',index=False)
        recovered.append(dict(security_id=identity,raw_export_sha256=sha256(raw_path),events_export_sha256=sha256(event_path),valid_start=record['valid_start'],supported_data_end=record['valid_end'],last_original_trade_date=record.get('last_original_trade_date'),
            valid_quote_rows=len(observed),unsupported_terminal_labels=True,event_verification_scope=record['event_verification_scope'],source_notes=record['notes']))
    features,normalized,prices,coverage=assemble_panel(bars,list(universe.ticker_2015),calendar)
    features.to_parquet(data/'features.parquet',index=False);normalized.to_parquet(data/'normalized.parquet',index=False);prices.to_parquet(data/'prices.parquet',index=False)
    pd.DataFrame({'date':calendar}).to_csv(data/'calendar.csv',index=False)
    sourcefiles=[p for p in (data/'sources').rglob('*') if p.is_file()]
    result=dict(dataset_version='oef2015-recovered-v2',declared_cohort=101,identities_with_some_history=len(bars),fully_unavailable_identities=101-len(bars),
        recovered_identity_count=len(recovered),recovered=recovered,rows=len(features),calendar_sessions=len(calendar),coverage=coverage,
        first_experiment_preserved=True,original_source_audit_sha256=sha256(ROOT/'backtests/us/oef2015/source_audit.json'),universe_sha256=sha256(ROOT/'docs/us/declared_universe.csv'),
        source_files_sha256={str(p.relative_to(data)):sha256(p) for p in sourcefiles},normalized_sha256=sha256(data/'normalized.parquet'),features_sha256=sha256(data/'features.parquet'),prices_sha256=sha256(data/'prices.parquet'),calendar_sha256=sha256(data/'calendar.csv'),
        vendor_vintages_point_in_time=False,survivorship_free=False,terminal_outcomes_complete=False,
        boundary='All original identities remain in every date denominator. Recovered price histories do not establish complete post-merger entitlement/wealth paths.')
    write_immutable_json(output,result);print(json.dumps({k:result[k] for k in ['identities_with_some_history','recovered_identity_count','rows','calendar_sessions']},indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=DATA);args=p.parse_args();build(args.data_root)
