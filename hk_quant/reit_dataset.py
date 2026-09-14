"""从时点公告生成可复现的REIT逐日价量因子与分期限标签。"""
import json
from pathlib import Path

import pandas as pd
import numpy as np

from . import HORIZONS
from .features import _reit_features_for_state, write_security_features
from .reit_adjustment import adjusted_history_asof
from .distribution_versions import distributions_asof


def build_reit_feature_history(bars, calendar, event_versions, cutoff, quote_currency, output):
    """calendar须已按证券身份期间限定；缺行情日保留空记录，绝不填价格。"""
    cutoff=pd.Timestamp(cutoff)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError('数据截止时间必须包含时区')
    if bars.empty or bars.security_id.isna().any() or bars.security_id.nunique()!=1:
        raise ValueError('必须提供单一证券身份的原始数据')
    sessions=pd.DatetimeIndex(calendar)
    if sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError('交易日历必须唯一且按时间排序')
    security=bars.security_id.iloc[0]
    original=bars.copy();original['date']=pd.to_datetime(original.date)
    if original.date.duplicated().any() or not original.date.isin(sessions).all():
        raise ValueError('原始行情日期重复或不在指定交易日历')
    local=cutoff.tz_convert('Asia/Hong_Kong')
    end=local.tz_localize(None).normalize()
    if local.hour<19:end-=pd.Timedelta(days=1)
    days=sessions[(sessions>=original.date.min())&(sessions<=end)]
    if days.empty:raise ValueError('截止时点没有可构建的交易日')
    # These are explicit missing-data rows in a known identity period, not quotes.
    inputs=original.set_index('date').reindex(days)
    inputs['security_id']=security
    inputs=inputs.rename_axis('date').reset_index()
    distributions_asof(event_versions,cutoff) # Validate all source timestamps before segmentation.
    boundaries={0,len(days)}
    for row in event_versions.to_dict('records'):
        stamp=pd.Timestamp(row['published_at'])
        if stamp>cutoff:continue
        public=stamp.tz_convert('Asia/Hong_Kong')
        available=public.tz_localize(None).normalize()
        if public>public.normalize()+pd.Timedelta(hours=19):available+=pd.Timedelta(days=1)
        for change in [available,pd.to_datetime(row.get('ex_date'),errors='coerce')]:
            if pd.notna(change):boundaries.add(int(days.searchsorted(change)))
    boundaries=sorted(boundaries)
    frames=[];label_values={h:pd.Series(np.nan,index=days) for h in HORIZONS}
    for left,right in zip(boundaries,boundaries[1:]):
        segment=days[left:right]
        if segment.empty:continue
        as_of=segment[-1].tz_localize('Asia/Hong_Kong')+pd.Timedelta(hours=19)
        snapshot=adjusted_history_asof(inputs,sessions,event_versions,as_of,quote_currency)
        frame=_reit_features_for_state(snapshot,sessions)
        frames.append(frame.loc[frame.date.isin(segment)].copy())
        raw=snapshot.history.set_index('date')
        legal=(raw.data_valid.eq(True)&raw.adjustment_status.eq('resolved')&raw.fx_to_hkd.gt(0)
               &np.isfinite(raw.fx_to_hkd)&raw.adjusted_close.gt(0)&np.isfinite(raw.adjusted_close))
        values=(raw.adjusted_close*raw.fx_to_hkd).where(legal).reindex(sessions)
        end_positions=sessions.get_indexer(segment)
        for horizon in HORIZONS:
            returns=(values/values.shift(horizon)-1).reindex(segment)
            valid=end_positions>=horizon
            starts=sessions[end_positions[valid]-horizon]
            inside=starts.isin(days)
            label_values[horizon].loc[starts[inside]]=returns.to_numpy()[valid][inside]
    history=pd.concat(frames,ignore_index=True)
    label_counts={}
    for horizon in HORIZONS:
        ends=pd.Series(sessions,index=sessions).shift(-horizon).reindex(days)
        states=pd.Series('no_completed_horizon',index=days)
        states.loc[ends.notna()]='not_mature'
        mature=(ends.dt.tz_localize('Asia/Hong_Kong')+pd.Timedelta(hours=19)).le(cutoff)
        states.loc[mature]='unresolved_price_adjustment_or_fx'
        states.loc[label_values[horizon].notna()]='available_at_horizon_end'
        history[f'label_end_{horizon}']=history.date.map(ends)
        history[f'fwd_return_{horizon}']=history.date.map(label_values[horizon])
        label_counts[str(horizon)]=states.value_counts().to_dict()
    if len(history)!=len(days) or history.date.duplicated().any():
        raise ValueError('逐日因子记录不完整或重复')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=write_security_features(history,output)
    audit=dict(path=str(path),security_id=security,rows=len(history),
               source_quote_rows=len(original),explicit_missing_source_rows=int((~days.isin(original.date)).sum()),
               start=str(days[0].date()),end=str(days[-1].date()),cutoff=cutoff.isoformat(),
               status_counts=history.status.value_counts().to_dict(),label_states=label_counts,
               adjustment_state_evaluations=len(boundaries)-1,
               approved_for_training=False,coverage_complete=False,
               scope='Price factors and horizon-end public-information labels; market/financial joins and source acceptance are separate.')
    audit_dir=output.parent/(output.name+'_audit')
    audit_dir.mkdir(parents=True,exist_ok=True)
    (audit_dir/(path.stem+'.json')).write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return audit
