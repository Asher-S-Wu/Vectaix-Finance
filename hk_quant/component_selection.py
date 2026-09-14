"""仅用完整开发期的共同样本，分别选择四项预测任务的模型。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .evaluation import _ece
from .prediction_tasks import MODEL_REJECTION_STATUSES, TASK_COLUMNS, TASK_STATUS_COLUMNS, validate_task_outputs

YEARS = tuple(range(2016,2024))
HORIZONS = (1,5,20,60)
TASKS = ('score','probability_up','intervals','expected_return')
BASELINE_COLUMNS = ('baseline_probability','baseline_q10','baseline_q50','baseline_q90')
COLUMNS = ('date','security_id','horizon','fwd_return','label_end','status','score',
           'probability_up','expected_return','q10','q50','q90',*BASELINE_COLUMNS,
           *TASK_STATUS_COLUMNS.values())


def _task_metrics(frame,task):
    count=len(frame)
    if task=='score':
        correlations=[]
        for _,group in frame.groupby(level='date',sort=True):
            if len(group)>=20 and group.score.nunique()>1 and group.fwd_return.nunique()>1:
                correlation=group.score.corr(group.fwd_return,method='spearman')
                if np.isfinite(correlation):correlations.append(float(correlation))
        return dict(score_dates=len(correlations),score=float(np.mean(correlations)) if correlations else None)
    actual=frame.fwd_return.to_numpy(float)
    if task=='probability_up':
        if not count:return dict(probability_up=None,ece=None,baseline_brier=None)
        return dict(probability_up=float(np.mean((frame.probability_up.to_numpy(float)-(actual>0))**2)),
                    ece=_ece((actual>0).astype(float),frame.probability_up.to_numpy(float))[0],
                    baseline_brier=float(np.mean((frame.baseline_probability.to_numpy(float)-(actual>0))**2)))
    if task=='intervals':
        if not count:return dict(intervals=None,interval_coverage=None,baseline_pinball=None)
        losses=[];baseline_losses=[]
        for column,quantile in (('q10',.1),('q50',.5),('q90',.9)):
            residual=actual-frame[column].to_numpy(float)
            losses.append(float(np.maximum(quantile*residual,(quantile-1)*residual).mean()))
            baseline_residual=actual-frame['baseline_'+column].to_numpy(float)
            baseline_losses.append(float(np.maximum(quantile*baseline_residual,(quantile-1)*baseline_residual).mean()))
        return dict(baseline_pinball=float(np.mean(baseline_losses)),intervals=float(np.mean(losses)),
                    interval_coverage=float(np.mean((actual>=frame.q10.to_numpy()) & (actual<=frame.q90.to_numpy()))))
    if task=='expected_return':
        return dict(expected_return=float(np.mean((frame.expected_return.to_numpy(float)-actual)**2)) if count else None)
    raise ValueError('未知预测任务')


def _overall(annual,probability_samples):
    values=list(annual.values())
    samples={task:sum(v['task_samples'][task] for v in values) for task in TASKS}
    days=sum(v['score_dates'] for v in values)
    result=dict(task_samples=samples,score_dates=days,
                score=sum(v['score']*v['score_dates'] for v in values if v['score_dates'])/days if days else None)
    metric_tasks={'probability_up':'probability_up','baseline_brier':'probability_up',
                  'intervals':'intervals','interval_coverage':'intervals','baseline_pinball':'intervals',
                  'expected_return':'expected_return'}
    for metric,task in metric_tasks.items():
        result[metric]=(sum(v[metric]*v['task_samples'][task] for v in values if v['task_samples'][task])/samples[task]
                        if samples[task] else None)
    result['ece']=(_ece(np.concatenate([v[0] for v in probability_samples]),
                        np.concatenate([v[1] for v in probability_samples]))[0] if samples['probability_up'] else None)
    return result


def compare_components(results_root, as_of='2023-12-31'):
    """逐年度、逐期限读开发预测；从不读取确认目录，也不写冻结配置。"""
    root=Path(results_root)
    cutoff=pd.Timestamp(as_of)
    if pd.isna(cutoff) or cutoff>pd.Timestamp('2023-12-31'):
        raise ValueError('组件选择截止不能越过2023-12-31开发期')
    protocol=json.loads((root/'training_protocol.json').read_text(encoding='utf-8'))
    kinds=protocol['kinds']
    if not isinstance(kinds,list) or len(set(kinds))!=len(kinds) or not {'factor','linear'}.issubset(kinds):
        raise ValueError('候选必须包含factor和linear且不能重复')
    if protocol['development_years']!=list(YEARS):
        raise ValueError('组件选择必须包含2016至2023完整八年开发期')
    for kind in kinds:
        for year in YEARS:
            path=root/'development'/f'{kind}_{year}.parquet'
            if not path.is_file():
                raise ValueError(f'缺少完整年度预测: {kind} {year}')
            if set(TASK_STATUS_COLUMNS.values()).difference(pq.ParquetFile(path).schema_arrow.names):
                raise ValueError(f'{kind} {year}缺少独立任务状态，必须重新生成预测，不能推断旧的整体状态')
    result={'as_of':cutoff.date().isoformat(),'development_years':list(YEARS),'kinds':kinds,
            'diagnostic_only':True,'comparison_scope':'task_specific_common_scorable_matured_subsets',
            'scope_notice':'每个任务分别比较全部候选共同可评分且收益已成熟的子样本；不同任务使用各自分母，不代表全市场能力验收。',
            'pairing':'Per year, horizon and task, intersect all candidates with that task available and mature legal labels. Other task failures do not change this task cohort.',
            'sample_counts':'annual/overall.task_samples records separate denominators for score, probability_up, intervals and expected_return; coverage[horizon][task] records raw, scorable, mature and common observations.',
            'metric_definitions':{'score':'mean daily Spearman IC; at least 20 securities per day',
                                  'probability_up':'Brier loss; lower is better',
                                  'ece':'10 equal-frequency bins on the full paired sample; overall re-bins all years',
                                  'baseline_brier':'Brier loss of common historical label frequency',
                                  'baseline_pinball':'mean pinball loss of common historical volatility quantiles',
                                  'intervals':'mean pinball loss at 0.1, 0.5, 0.9; lower is better',
                                  'interval_coverage':'fraction within q10 and q90; nominal 80%',
                                  'expected_return':'mean squared error; lower is better'},
            'coverage':{},'horizons':{}}
    for horizon in HORIZONS:
        key=str(horizon)
        result['horizons'][key]={kind:{'annual':{}} for kind in kinds}
        probability_samples={kind:[] for kind in kinds}
        coverage={task:{'raw_rows':{kind:0 for kind in kinds},'scorable_rows':{kind:0 for kind in kinds},
                        'mature_legal_rows':{kind:0 for kind in kinds},'input_unavailable_rows':{kind:0 for kind in kinds},
                        'model_rejected_rows':{kind:0 for kind in kinds},'common_samples':0,'annual':{}}
                  for task in TASKS}
        for year in YEARS:
            frames={}
            for kind in kinds:
                path=root/'development'/f'{kind}_{year}.parquet'
                frame=pd.read_parquet(path,columns=list(COLUMNS),filters=[('horizon','=',horizon)])
                frame['date']=pd.to_datetime(frame.date,errors='coerce')
                original_end=frame.label_end
                frame['label_end']=pd.to_datetime(original_end,errors='coerce')
                if frame.date.isna().any() or frame.date.dt.year.ne(year).any() or (frame.label_end.isna() & original_end.notna()).any():
                    raise ValueError(f'{kind} {year}预测日期或标签日期无效')
                if frame.security_id.isna().any() or frame.security_id.astype(str).str.strip().eq('').any():
                    raise ValueError(f'{kind} {year}缺少证券标识')
                frame=frame.set_index(['date','security_id','horizon'])
                if frame.index.duplicated().any():raise ValueError(f'{kind} {year}预测键重复')
                numeric=['score','probability_up','expected_return','q10','q50','q90',*BASELINE_COLUMNS,'fwd_return']
                frame[numeric]=frame[numeric].apply(pd.to_numeric,errors='coerce')
                validate_task_outputs(frame)
                for other in frames.values():
                    overlap=frame.index.intersection(other.index)
                    left,right=frame.loc[overlap],other.loc[overlap]
                    same_return=left.fwd_return.eq(right.fwd_return) | (left.fwd_return.isna() & right.fwd_return.isna())
                    same_end=left.label_end.eq(right.label_end) | (left.label_end.isna() & right.label_end.isna())
                    disagreement=~same_return | ~same_end
                    if disagreement.any():raise ValueError(f'{kind} {year}同键标签冲突')
                frames[kind]=frame
                result['horizons'][key][kind]['annual'][str(year)]={'task_samples':{}}
            for task in TASKS:
                common=None
                year_counts={'raw_rows':{},'scorable_rows':{},'mature_legal_rows':{},'input_unavailable_rows':{},
                             'model_rejected_rows':{},'common_samples':0}
                for kind,frame in frames.items():
                    scores=frame[TASK_STATUS_COLUMNS[task]].eq('ok')
                    task_status=frame[TASK_STATUS_COLUMNS[task]]
                    mature=scores & np.isfinite(frame.fwd_return) & frame.fwd_return.ge(-1) & frame.label_end.le(cutoff)
                    eligible=frame.index[mature]
                    common=eligible if common is None else common.intersection(eligible)
                    for name,value in (('raw_rows',len(frame)),('scorable_rows',int(scores.sum())),('mature_legal_rows',int(mature.sum())),
                                       ('input_unavailable_rows',int(task_status.eq('insufficient_model_inputs').sum())),
                                       ('model_rejected_rows',int(task_status.isin(MODEL_REJECTION_STATUSES).sum()))):
                        year_counts[name][kind]=value;coverage[task][name][kind]+=value
                year_counts['common_samples']=len(common)
                coverage[task]['common_samples']+=len(common)
                coverage[task]['annual'][str(year)]=year_counts
                baseline_columns=[column for column in TASK_COLUMNS[task] if column in BASELINE_COLUMNS]
                reference=frames[kinds[0]].loc[common]
                for kind,frame in frames.items():
                    paired=frame.loc[common]
                    for column in baseline_columns:
                        if paired[column].ne(reference[column]).any():
                            raise ValueError(f'{kind} {year}任务{task}同键基准冲突: {column}')
                    annual=result['horizons'][key][kind]['annual'][str(year)]
                    annual['task_samples'][task]=len(paired)
                    annual.update(_task_metrics(paired,task))
                    if task=='probability_up':
                        probability_samples[kind].append((paired.fwd_return.gt(0).to_numpy(float),paired.probability_up.to_numpy(float)))
        for kind in kinds:
            value=result['horizons'][key][kind]
            value['overall']=_overall(value['annual'],probability_samples[kind])
        result['coverage'][key]=coverage
    return result


def choose_component_heads(metrics):
    """按每任务的完整八年证据选择；不代表通过发布门槛。"""
    if metrics['development_years']!=list(YEARS):raise ValueError('必须提供完整八年验证指标')
    heads={}
    for horizon in HORIZONS:
        kinds=metrics['horizons'][str(horizon)]
        for kind in metrics['kinds']:
            if set(kinds[kind]['annual'])!=set(map(str,YEARS)):
                raise ValueError(f'{kind}期限{horizon}缺少完整八年指标')
        heads[str(horizon)]={}
        for task in TASKS:
            def usable(kind):
                values=[kinds[kind]['overall'][task]]+[kinds[kind]['annual'][str(y)][task] for y in YEARS]
                return all(v is not None and np.isfinite(v) for v in values)
            def better(left,right):
                return left>right if task=='score' else left<right
            simple=[kind for kind in ('factor','linear') if usable(kind)]
            if not simple:raise ValueError(f'期限{horizon}任务{task}简单基准缺少完整八年有效证据')
            base=simple[0]
            for kind in simple[1:]:
                if better(kinds[kind]['overall'][task],kinds[base]['overall'][task]):base=kind
            contenders=list(simple)
            for kind in metrics['kinds']:
                if kind in ('factor','linear') or not usable(kind):continue
                annual_wins=sum(better(kinds[kind]['annual'][str(y)][task],kinds[base]['annual'][str(y)][task]) for y in YEARS)
                if annual_wins/len(YEARS)>=.60 and better(kinds[kind]['overall'][task],kinds[base]['overall'][task]):
                    contenders.append(kind)
            def feasible(kind):
                row=kinds[kind]['overall']
                if metrics['coverage'][str(horizon)][task]['model_rejected_rows'][kind] != 0:
                    return False
                if task=='probability_up':
                    return (all(row[name] is not None and np.isfinite(row[name]) for name in ('baseline_brier','ece'))
                            and row[task]<row['baseline_brier'] and row['ece']<=.05)
                if task=='intervals':
                    return (all(row[name] is not None and np.isfinite(row[name]) for name in ('baseline_pinball','interval_coverage'))
                            and row[task]<row['baseline_pinball'] and .75<=row['interval_coverage']<=.85)
                return True
            qualified=[kind for kind in contenders if feasible(kind)]
            if not qualified:raise ValueError(f'期限{horizon}任务{task}没有满足原验收门槛的合格模型，不能冻结')
            chosen=qualified[0]
            for kind in qualified[1:]:
                if better(kinds[kind]['overall'][task],kinds[chosen]['overall'][task]):chosen=kind
            heads[str(horizon)][task]=chosen
    return heads


def main():
    parser=argparse.ArgumentParser(description='按共同开发期样本选择分任务模型组件')
    parser.add_argument('--results-root',type=Path,required=True)
    parser.add_argument('--as-of',default='2023-12-31')
    args=parser.parse_args()
    comparison=compare_components(args.results_root,args.as_of)
    (args.results_root/'component_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    config={'heads':choose_component_heads(comparison),'as_of':comparison['as_of'],'comparison':comparison,
            'diagnostic_only':True,'comparison_scope':comparison['comparison_scope'],
            'feasibility_gates':{'probability_up':'Brier < common baseline Brier and full-period ECE <= 0.05',
                                 'intervals':'coverage in [0.75,0.85] and pinball < common historical-volatility baseline'},
            'confirmation_used':False,'selection_scope':'2016-2023 development only; no confirmation predictions read',
            'release_policy':'Selection does not confer release approval; evaluation.release_decision must independently apply all original gates',
            'complex_selection_rule':'Improve full-period task metric and at least 60% of all eight validation years over the stronger simple baseline'}
    path=args.results_root/'component_selection.json'
    path.write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(str(path))


if __name__=='__main__':main()
