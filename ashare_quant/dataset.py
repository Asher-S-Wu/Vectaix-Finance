"""Memory-bounded normalized and feature datasets with explicit source identity."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import HORIZONS,FORECAST_RETURN_BASIS
from .paths import DATA
from .data import normalize_daily,normalize_securities
from .features import security_features,feature_columns,cross_sectional_inputs,sample_observations


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def default(value):
        if isinstance(value,(pd.Timestamp,np.datetime64)):return pd.Timestamp(value).isoformat()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    import os
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=default,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def canonicalize_codes(frame,mapping,*,allow_equivalent=False,used_columns=None,prefer_dated_factor=False):
    out=frame.copy();out['source_security_id']=out.security_id
    if mapping.empty:return out
    if mapping.o_code.duplicated().any() or mapping.n_code.duplicated().any():raise ValueError('ambiguous BSE alias mapping')
    aliases=mapping.set_index('o_code').n_code
    out['security_id']=out.security_id.map(aliases).fillna(out.security_id)
    duplicated=out.duplicated(['security_id','date'],keep=False)
    removed=0;precision_reconciliations=[];dated_factor_choices=[]
    if duplicated.any():
        if not allow_equivalent:raise ValueError('overlapping alias quotes require manual source review')
        columns=used_columns if used_columns is not None else [c for c in out if c not in ['security_id','date','source_security_id']]
        chosen=[]
        for (_,date),group in out.loc[duplicated].groupby(['security_id','date'],sort=False):
            different=group[columns].nunique(dropna=False).gt(1)
            precision_choice=None
            if different.any():
                # Some retired aliases round the same cumulative factor to
                # three decimals while the canonical source keeps four. Only
                # that bounded precision case is reconciled, never OHLC/caps.
                if list(different.index[different])!=['adj_factor']:raise ValueError('conflicting model-input values across verified aliases')
                values=pd.to_numeric(group.adj_factor,errors='coerce').to_numpy(float)
                rounded=np.isclose(values*1000,np.round(values*1000),atol=1e-8,rtol=0)
                valid_values=np.isfinite(values).all() and (values>0).all()
                bounded=valid_values and rounded.any() and not rounded.all() and np.ptp(values)<=.0005000001
                if bounded:
                    precision_choice=group.loc[~rounded].iloc[[0]]
                    precision_reconciliations.append(dict(security_id=str(group.security_id.iloc[0]),date=str(pd.Timestamp(date).date()),
                        values=values.tolist(),selected=float(precision_choice.adj_factor.iloc[0]),policy='bounded three-decimal rounding; retain higher precision; all raw records preserved'))
                elif prefer_dated_factor and valid_values:
                    evidence=mapping.loc[mapping.n_code.eq(group.security_id.iloc[0])]
                    legal=(len(evidence)==1 and 'verified' in evidence and evidence.verified.eq(True).all() and 'effective_date' in evidence and evidence.effective_date.notna().all() and 'source_url' in evidence and evidence.source_url.str.startswith('https://').all())
                    if not legal:raise ValueError('conflicting adjustment factors lack verified date-active source')
                    e=evidence.iloc[0];active=e.o_code if pd.Timestamp(date)<pd.Timestamp(e.effective_date) else e.n_code
                    available=group.loc[group.source_security_id.eq(active)]
                    if len(available)!=1:raise ValueError('conflicting adjustment factors have no unique date-active source')
                    precision_choice=available.iloc[[0]]
                    dated_factor_choices.append(dict(security_id=str(e.n_code),date=str(pd.Timestamp(date).date()),source_security_id=str(active),
                        values=values.tolist(),selected=float(precision_choice.adj_factor.iloc[0]),source_url=str(e.source_url),
                        policy='documented date-active source code; alternative retained in raw audit'))
                else:raise ValueError('conflicting adjustment-factor values across verified aliases')
            sid=group.security_id.iloc[0]
            prefer=sid
            mapping_row=mapping.loc[mapping.n_code.eq(sid)]
            if 'effective_date' in mapping_row and mapping_row.effective_date.notna().any():
                if pd.Timestamp(date)<pd.Timestamp(mapping_row.effective_date.iloc[0]):prefer=mapping_row.o_code.iloc[0]
            match=group.loc[group.source_security_id.eq(prefer)]
            chosen.append(precision_choice if precision_choice is not None else (match if len(match) else group.sort_values('source_security_id')).iloc[[0]])
            removed+=len(group)-1
        out=pd.concat([out.loc[~duplicated],*chosen],ignore_index=True)
    out.attrs['equivalent_alias_rows_removed']=removed
    out.attrs['precision_reconciliations']=precision_reconciliations
    out.attrs['dated_factor_choices']=dated_factor_choices
    return out


def save_normalized_year(data_root,year,daily,basic,factors,limits,securities,mapping=None,st=None):
    root=Path(data_root)
    mapping=pd.DataFrame() if mapping is None else mapping
    def canonical_source(frame):
        if frame.empty:return frame.copy()
        renamed=frame.rename(columns={'ts_code':'security_id','trade_date':'date'})
        columns=[c for c in ['open','high','low','close','pre_close','vol','amount','adj_factor','total_mv','circ_mv','total_share','float_share','free_share','turnover_rate','up_limit','down_limit'] if c in renamed]
        return canonicalize_codes(renamed,mapping,allow_equivalent=True,used_columns=columns,prefer_dated_factor=True).rename(columns={'security_id':'ts_code','date':'trade_date'})
    normalized_sources=[canonical_source(f) for f in [daily,basic,factors,limits]]
    bars,audit=normalize_daily(*normalized_sources)
    audit['raw_quote_rows_before_alias_dedup']=len(daily)
    audit['dated_alias_factor_choices']={name:f.attrs.get('dated_factor_choices',[]) for name,f in zip(['daily','daily_basic','adj_factor','stk_limit'],normalized_sources)}
    audit['alias_precision_reconciliations']={name:f.attrs.get('precision_reconciliations',[]) for name,f in zip(['daily','daily_basic','adj_factor','stk_limit'],normalized_sources)}
    audit['equivalent_alias_rows_removed']={name:int(f.attrs.get('equivalent_alias_rows_removed',0)) for name,f in zip(['daily','daily_basic','adj_factor','stk_limit'],normalized_sources)}
    original=normalized_sources[0][['ts_code','trade_date','source_security_id']].rename(columns={'ts_code':'security_id','trade_date':'date'})
    original['date']=pd.to_datetime(original.date,format='%Y%m%d')
    bars=bars.merge(original,on=['security_id','date'],validate='one_to_one')
    master=securities.set_index('security_id')
    bars['listed']=bars.security_id.isin(master.index)
    known=bars.security_id.map(master.list_date)
    end=bars.security_id.map(master.delist_date)
    valid=bars.listed & known.notna() & bars.date.ge(known) & (end.isna()|bars.date.lt(end))
    audit['raw_rows']=audit['rows']
    audit['normalized_rows']=int(valid.sum())
    audit['outside_master_or_listing_period']=int((~valid).sum())
    audit['outside_master_identifiers']=sorted(set(bars.loc[~bars.listed,'security_id']))
    bars=bars.loc[valid].drop(columns='listed').copy()
    audit['normalized_missing_limit_rows']=int(bars[['up_limit','down_limit']].isna().any(axis=1).sum())
    audit['normalized_missing_market_cap_rows']=int(bars.total_mv.isna().sum())
    if st is not None and not st.empty:
        status=st.rename(columns={'ts_code':'security_id','trade_date':'date'})[['security_id','date']].copy()
        status['date']=pd.to_datetime(status.date,format='%Y%m%d')
        if not mapping.empty:
            aliases=mapping.set_index('o_code').n_code
            status['security_id']=status.security_id.map(aliases).fillna(status.security_id)
        # Membership is a boolean union of active source warning types, not quotes.
        status=status.drop_duplicates(['security_id','date'])
        marked=pd.MultiIndex.from_frame(status[['date','security_id']])
        bars['is_st']=pd.MultiIndex.from_frame(bars[['date','security_id']]).isin(marked)
    else:bars['is_st']=pd.NA
    (root/'bars').mkdir(parents=True,exist_ok=True)
    bars.to_parquet(root/'bars'/f'{year}.parquet',index=False,row_group_size=100000)
    for sid,group in bars.groupby('security_id',sort=True):
        directory=root/'bars_by_security'/sid;directory.mkdir(parents=True,exist_ok=True)
        group.to_parquet(directory/f'{year}.parquet',index=False)
    write_json(root/'audits'/f'normalization-{year}.json',audit)
    return audit


def _dated_master_features(frame,calendar,master,security_id):
    """Keep known listing membership even when source quotes are unavailable.

    Reindex only the feature table: absent rows carry no invented prices,
    factors or returns. Horizon dates are exchange-calendar facts, including
    when the corresponding outcome is missing or follows a delisting.
    """
    dates=calendar[(calendar>=master.list_date)&(pd.isna(master.delist_date)|(calendar<master.delist_date))]
    result=frame.set_index('date').reindex(dates)
    result['date']=dates;result['security_id']=security_id
    result['status']=result.status.fillna('no_trade_quote')
    global_dates=pd.Series(calendar,index=calendar)
    for horizon in HORIZONS:
        result[f'label_end_{horizon}']=global_dates.shift(-horizon).reindex(dates).to_numpy()
    return result.reset_index(drop=True)


def read_feature_window(path,start,end,columns=None):
    """Read an exact calendar window with bounded Arrow IO concurrency.

    Default dataset read-ahead across thousands of security fragments can use
    many times the final frame's memory. Keep the same rows/order/schema while
    limiting buffered batches and discarding per-scan metadata caches.
    """
    import gc
    import pyarrow as pa
    import pyarrow.dataset as ds
    dataset=ds.dataset(str(path),format='parquet')
    expression=(ds.field('date')>=pa.scalar(pd.Timestamp(start))) & (ds.field('date')<pa.scalar(pd.Timestamp(end)))
    scanner=dataset.scanner(columns=columns,filter=expression,batch_size=32768,
        batch_readahead=1,fragment_readahead=1,use_threads=False,cache_metadata=False)
    table=scanner.to_table()
    result=table.to_pandas()
    if table.schema.metadata and b'PANDAS_ATTRS' in table.schema.metadata:
        result.attrs=json.loads(table.schema.metadata[b'PANDAS_ATTRS'])
    del table,scanner,dataset
    gc.collect();pa.default_memory_pool().release_unused()
    return result


def build_features(data_root=DATA):
    import gc
    import pyarrow as pa
    root=Path(data_root)
    audit_path=root/'data_audit.json'
    if audit_path.exists() and json.loads(audit_path.read_text()).get('status')!='complete':raise ValueError('Source dataset incomplete; cannot certify features')
    calendar_frame=pd.read_parquet(root/'calendar.parquet')
    calendar=pd.DatetimeIndex(pd.to_datetime(calendar_frame.loc[calendar_frame.is_open.eq(1),'cal_date'])).sort_values().drop_duplicates()
    if not len(calendar):raise ValueError('empty exchange calendar')
    securities=pd.read_parquet(root/'securities.parquet').set_index('security_id')
    output=root/'features';output.mkdir(parents=True,exist_ok=True)
    rows=eligible=0;files=[];adjustment_issues=[];status_counts={}
    feature_template=None;written=set();identities_without_source_bars=0
    def save_frame(frame,sid):
        nonlocal rows,eligible
        path=output/f'{sid}.parquet';frame.to_parquet(path,index=False,row_group_size=252)
        files.append(path);written.add(sid);rows+=len(frame);eligible+=int(frame.status.eq('ok').sum())
        for state,count in frame.status.value_counts().items():status_counts[str(state)]=status_counts.get(str(state),0)+int(count)
    source_root=root/'bars_by_security'
    directories=sorted(source_root.iterdir()) if source_root.exists() else []
    for i,directory in enumerate(directories,1):
        if not directory.is_dir():continue
        sid=directory.name
        if sid not in securities.index:raise ValueError('Unregistered source identity')
        source=pd.read_parquet(directory).sort_values('date')
        source=source.loc[source.date.isin(calendar)].copy()
        if source.empty:continue
        frame=security_features(source,calendar)
        adjustment_issues.extend(frame.attrs.get('adjustment_issues',[]))
        # The dated master, not the final observed quote, defines membership.
        master=securities.loc[sid]
        feature_template=frame.iloc[:0].copy();feature_template.attrs={}
        frame=_dated_master_features(frame,calendar,master,sid)
        if frame.empty:continue
        frame['quote_present']=frame.date.map(source.set_index('date').quote_present).eq(True)
        save_frame(frame,sid)
        if i%250==0:print(f'features securities={i} rows={rows}',flush=True)
    if feature_template is not None:
        for sid,master in securities.loc[~securities.index.isin(written)].sort_index().iterrows():
            frame=_dated_master_features(feature_template,calendar,master,sid)
            if frame.empty:continue
            frame['quote_present']=False
            save_frame(frame,sid);identities_without_source_bars+=1
    if not files:raise ValueError('no source bars available')
    stats=[]
    for year in sorted(set(calendar.year)):
        minimal=read_feature_window(output,pd.Timestamp(year,1,1),pd.Timestamp(year+1,1,1),columns=['date','return_1','bias_60','quote_present'])
        good=minimal.quote_present.eq(True)
        minimal.loc[~good,['return_1','bias_60']]=np.nan
        market=minimal.groupby('date').return_1.median().rename('market_return')
        breadth=minimal.loc[minimal.bias_60.notna()].groupby('date').bias_60.agg(lambda x:float(x.gt(0).mean())).rename('market_breadth_60')
        stats.append(pd.concat([market,breadth],axis=1))
        del minimal,good
        gc.collect();pa.default_memory_pool().release_unused()
    market=pd.concat(stats).reindex(calendar)
    level=(1+market.market_return).cumprod()
    for n in [20,60,252]:market[f'market_momentum_{n}']=level/level.shift(n)-1
    market['market_volatility_60']=market.market_return.rolling(60,min_periods=30).std()*np.sqrt(252)
    market=market.drop(columns='market_return').rename_axis('date').reset_index()
    normalized_dir=root/'normalized';normalized_dir.mkdir(exist_ok=True)
    sample_dir=root/'training_samples';sample_dir.mkdir(exist_ok=True)
    latest_features=[];latest_inputs=[];risk=[];columns=None;coverage_counts=None;normal_rows=0
    label_coverage={str(h):dict(eligible_mature=0,eligible_mature_observed=0,eligible_mature_missing=0) for h in HORIZONS}
    for year in sorted(set(calendar.year)):
        frame=read_feature_window(output,pd.Timestamp(year,1,1),pd.Timestamp(year+1,1,1))
        if frame.empty:continue
        frame=frame.merge(market,on='date',validate='many_to_one')
        columns=feature_columns(frame)
        frame[columns]=frame[columns].astype(np.float32)
        counts=frame[columns].notna().sum();coverage_counts=counts if coverage_counts is None else coverage_counts.add(counts,fill_value=0)
        normal_rows+=len(frame)
        for h in HORIZONS:
            mature=frame.status.eq('ok') & pd.to_datetime(frame[f'label_end_{h}']).le(calendar[-1])
            observed=mature & frame[f'fwd_return_{h}'].notna()
            label_coverage[str(h)]['eligible_mature']+=int(mature.sum())
            label_coverage[str(h)]['eligible_mature_observed']+=int(observed.sum())
            label_coverage[str(h)]['eligible_mature_missing']+=int((mature & ~observed).sum())
        normalized=cross_sectional_inputs(frame,columns)
        normalized.to_parquet(normalized_dir/f'{year}.parquet',index=False,row_group_size=100000)
        eligible_samples=normalized.loc[normalized.status.eq('ok')]
        sampled=sample_observations(eligible_samples) if not eligible_samples.empty else eligible_samples
        sampled.to_parquet(sample_dir/f'{year}.parquet',index=False)
        latest_features.append(frame.loc[frame.date.eq(calendar[-1])])
        latest_inputs.append(normalized.loc[normalized.date.eq(calendar[-1])])
        risk.append(frame.loc[frame.date.ge(calendar[max(0,len(calendar)-300)]),['date','security_id','return_1']])
        print(f'normalized year={year} rows={len(frame)}',flush=True)
        del frame,normalized,eligible_samples,sampled,mature,observed
        gc.collect();pa.default_memory_pool().release_unused()
    pd.concat(latest_features,ignore_index=True).to_parquet(root/'latest_features.parquet',index=False)
    pd.concat(latest_inputs,ignore_index=True).to_parquet(root/'latest_inputs.parquet',index=False)
    pd.concat(risk,ignore_index=True).pivot(index='date',columns='security_id',values='return_1').reset_index().to_parquet(root/'risk_returns.parquet',index=False)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sample_dir.glob('*.parquet'))}
    manifest=dict(status='complete',market='CN',currency='CNY',start_date=str(calendar[0].date()),data_as_of=str(calendar[-1].date()),
        rows=rows,eligible_rows=eligible,securities=len(files),features=columns,horizons=list(HORIZONS),label_coverage=label_coverage,
        delisted_master_identities=int(securities.delist_date.notna().sum()),input_status_counts=status_counts,
        identities_without_source_bars=identities_without_source_bars,
        universe_policy='Dated master/calendar membership; missing quotes remain unavailable without filling prices or returns',
        adjustment_reference_issues=adjustment_issues,adjustment_policy='Separate continuity segments; past-only re-warmup and no labels across unresolved source breaks; raw records retained',
        coverage={k:float(v/normal_rows) for k,v in coverage_counts.items()},training_sample_sha256=hashes,
        return_basis=FORECAST_RETURN_BASIS,signal_cutoff='19:00 Asia/Shanghai after complete source session',
        normalization='same-date available cross-sectional ranks; historical sigma retained unranked',
        financial_features='not included in 37-factor baseline; announcement-time adapter separate',
        limitations=['Provider historical revisions and terminal outcome completeness are not independently certified.',
          'Source-adjusted return is not an actual-share-plus-cash total-return account.',
          'Rows without observed horizon-end prices have missing labels, not assumed zero or fabricated liquidation.'])
    manifest['artifact_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sorted(normalized_dir.glob('*.parquet')),root/'latest_inputs.parquet',root/'securities.parquet',root/'calendar.parquet']}
    write_json(root/'feature_manifest.json',manifest)
    return manifest


def prepare_data(data_root=DATA,through_year=None,require_complete=True):
    from .collect import iter_partitions,load_partitions
    root=Path(data_root);raw=root/'raw'
    calendars=load_partitions(raw,'trade_cal')
    if calendars.empty:raise ValueError('No verified source calendar')
    calendars=calendars.loc[calendars.exchange.eq('SSE')].drop_duplicates('cal_date').sort_values('cal_date')
    calendars['cal_date']=pd.to_datetime(calendars.cal_date,format='%Y%m%d')
    calendar=pd.DatetimeIndex(calendars.loc[calendars.is_open.eq(1),'cal_date'])
    master=normalize_securities(load_partitions(raw,'stock_basic'))
    mapping=load_partitions(raw,'bse_mapping')
    supplementary=root/'references'/'verified_code_changes.parquet'
    fallback=Path(__file__).resolve().parents[1]/'docs'/'ashare'/'verified_code_changes.csv'
    if supplementary.exists() or fallback.exists():
        extra=pd.read_parquet(supplementary) if supplementary.exists() else pd.read_csv(fallback)
        required={'o_code','n_code','source_url','effective_date','verified'}
        if not required<=set(extra) or not extra.verified.eq(True).all() or not extra.source_url.str.startswith('https://').all():raise ValueError('Unverified historical code aliases')
        mapping=pd.concat([mapping,extra],ignore_index=True)
    audits=[];missing=[]
    for year in sorted(set(calendar.year)):
        if through_year is not None and year>through_year:continue
        expected={d.strftime('%Y%m%d') for d in calendar[calendar.year==year]}
        endpoints={}
        source_hashes={'normalization_schema':'cn-v6',
            'master':hashlib.sha256(master.to_json(date_format='iso').encode()).hexdigest(),
            'aliases':hashlib.sha256(mapping.to_json(date_format='iso').encode()).hexdigest(),
            'stock_st':[m['sha256'] for _,m in iter_partitions(raw,'stock_st',year=year)]}
        for endpoint in ['daily','daily_basic','adj_factor','stk_limit']:
            parts=list(iter_partitions(raw,endpoint,year=year))
            found={str(m['params'].get('trade_date')) for _,m in parts}
            absent=sorted(expected-found)
            if absent:missing.append(dict(year=int(year),endpoint=endpoint,missing_dates=absent))
            endpoints[endpoint]=pd.concat([f for f,_ in parts],ignore_index=True) if parts else pd.DataFrame()
            source_hashes[endpoint]={m['params']['trade_date']:m['sha256'] for _,m in parts}
        if any(x['year']==year for x in missing):continue
        target=root/'audits'/f'normalization-{year}.json'
        fingerprint=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
        if target.exists():
            previous=json.loads(target.read_text())
            if previous.get('source_sha256')==fingerprint:
                audits.append(previous);continue
        stparts=[f for f,_ in iter_partitions(raw,'stock_st',year=year)]
        st=pd.concat(stparts,ignore_index=True).drop_duplicates() if stparts else pd.DataFrame()
        audit=save_normalized_year(root,year,endpoints['daily'],endpoints['daily_basic'],endpoints['adj_factor'],endpoints['stk_limit'],master,mapping,st)
        audit.update(year=int(year),source_sha256=fingerprint)
        write_json(target,audit);audits.append(audit)
        print(f'normalized source year={year} rows={audit["rows"]}',flush=True)
    if through_year is not None:
        missing.extend(dict(year=int(y),endpoint='all',reason='outside_normalized_prefix') for y in sorted(set(calendar.year)) if y>through_year)
    if require_complete and missing:raise ValueError(f'Incomplete source calendar partitions: {len(missing)} endpoint/year gaps')
    root.mkdir(parents=True,exist_ok=True)
    master.to_parquet(root/'securities.parquet',index=False)
    calendars.to_parquet(root/'calendar.parquet',index=False)
    (root/'references').mkdir(exist_ok=True)
    load_partitions(raw,'index_daily').to_parquet(root/'references'/'benchmarks.parquet',index=False)
    mapping.to_parquet(root/'references'/'bse_mapping.parquet',index=False)
    result=dict(status='complete' if not missing else 'partial',start=str(calendar[0].date()),end=str(calendar[-1].date()),
        market='CN',securities=len(master),years=audits,missing=missing,
        calendar_policy='SSE calendar used for mainland research; BSE endpoint was empty, BSE dates cross-checked by actual observed quotes',
        name_policy='Current display names/industry never feed historical model inputs',
        return_basis=FORECAST_RETURN_BASIS)
    write_json(root/'data_audit.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=DATA)
    a=p.parse_args();print(json.dumps(build_features(a.data_root),ensure_ascii=False))
