"""已保存逐日预测、实际现金账户与同池约束等权基准的连接。"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .contracts import RiskProfile
from .evaluation import evaluate_portfolio
from .portfolio import advise_portfolio
from .replay import replay_account

FORECAST_INPUTS = ('date','security_id','horizon','expected_return','status','data_as_of','model_trained_as_of')
LOT_COLUMNS = ('security_id','lot_size','lot_valid_from','lot_valid_to','verified','source_url')


class SavedForecastPolicy:
    """policy(day, account) 返回以含应收总NAV为分母的security_id/target_weight。"""
    def __init__(self,forecast_dir,master,historical_lots,returns,daily_factors,profile=RiskProfile(),allocation='utility'):
        paths=sorted(Path(forecast_dir).glob('*.parquet'))
        if not paths:raise ValueError('没有已保存的预测文件')
        self.forecasts=ds.dataset([str(path) for path in paths],format='parquet')
        self.master=master.copy();self.historical_lots=historical_lots.copy()
        self.returns=returns;self.daily_factors=daily_factors
        self.profile=profile;self.allocation=allocation;self.reports=[]
        if self.master.security_id.duplicated().any():raise ValueError('证券主表存在重复身份')
        if not set(LOT_COLUMNS).issubset(self.historical_lots.columns):raise ValueError('缺少完整历史交易单位证据字段')

    def __call__(self,day,account):
        day=pd.Timestamp(day)
        # 只投影输入字段，不读取保存文件中的未来标签。
        predictions=self.forecasts.to_table(filter=ds.field('date')==day.to_datetime64(),columns=list(FORECAST_INPUTS)).to_pandas()
        if predictions.empty:raise ValueError('信号日没有已保存预测: '+day.date().isoformat())
        dates=pd.to_datetime(predictions.date,errors='coerce')
        asof=pd.to_datetime(predictions.data_as_of,errors='coerce')
        trained=pd.to_datetime(predictions.model_trained_as_of,errors='coerce')
        if not dates.eq(day).all() or not asof.eq(day).all() or trained.isna().any() or trained.gt(day).any():
            raise ValueError('预测时间证据不匹配信号日或使用了未来训练结果')
        lots=self.historical_lots
        start=pd.to_datetime(lots.lot_valid_from,errors='coerce')
        end=pd.to_datetime(lots.lot_valid_to,errors='coerce')
        eligible=(lots.verified.eq(True) & lots.source_url.notna() & lots.source_url.astype(str).str.strip().ne('')
                  & start.le(day) & (lots.lot_valid_to.isna() | (end.notna() & end.ge(day))))
        effective=lots.loc[eligible,list(LOT_COLUMNS)]
        if effective.security_id.duplicated().any():raise ValueError('信号日历史交易单位证据冲突')
        identity_columns=[name for name in self.master.columns if name not in ('lot_size','lot_valid_from','lot_valid_to','verified','source_url')]
        securities=self.master[identity_columns].merge(effective,on='security_id',how='left',validate='one_to_one')
        factors=self.daily_factors.loc[pd.to_datetime(self.daily_factors.date).eq(day),['security_id','adv20_amount']]
        bars=account['bars'].loc[:,account['bars'].columns!='adv20_amount'].copy()
        if not pd.to_datetime(bars.date).eq(day).all():raise ValueError('账户行情不是信号日行情')
        bars=bars.merge(factors,on='security_id',how='left',validate='one_to_one')
        if not np.isfinite(account['equity']) or account['equity']<=0:raise ValueError('含应收账户净值无效')
        result=advise_portfolio(predictions,account['holdings'].copy(),account['cash'],bars,securities,self.returns,
                                profile=self.profile,allocation=self.allocation)
        self.reports.append({'date':day,'allocation':self.allocation,**result})
        if result['status']!='ok':raise ValueError('账户优化未完成: '+result['reason'])
        rows=result['recommendations']
        return pd.DataFrame({'security_id':rows.security_id,'target_weight':rows.target_value_hkd/account['equity']})


def run_strategy_backtest(data_root,forecast_dir,*,start,end,initial_cash=1000000.,profile=RiskProfile()):
    """对指定日期运行两种策略和两档费用；只读文件并返回结果，不交易或写模型。"""
    root=Path(data_root);start=pd.Timestamp(start);end=pd.Timestamp(end)
    def incomplete(reason):
        return {'status':'incomplete','reason':reason,'runs':{},'policies':{},'portfolio_metrics':None}
    if pd.isna(start) or pd.isna(end) or start>end:return incomplete('回放日期无效')
    required=('execution_coverage.json','historical_lots.parquet','corporate_actions.parquet','securities.parquet',
              'risk_returns.parquet','references/calendar.parquet','bars','features')
    if not list(Path(forecast_dir).glob('*.parquet')):return incomplete('没有已保存的逐日预测文件')
    missing=[name for name in required if not (root/name).exists()]
    if missing:return incomplete('缺少回放文件或覆盖证据: '+','.join(missing))
    evidence=json.loads((root/'execution_coverage.json').read_text(encoding='utf-8'))
    fields={'verified','start_date','end_date','security_ids','source_urls','historical_lot_coverage_complete','corporate_action_cash_coverage_complete'}
    if not fields.issubset(evidence):return incomplete('执行覆盖声明缺少日期、证券范围或来源字段')
    if (evidence['verified'] is not True or evidence['historical_lot_coverage_complete'] is not True
        or evidence['corporate_action_cash_coverage_complete'] is not True):
        return incomplete('历史交易单位或公司行动现金覆盖未经明确核验')
    if (not isinstance(evidence['source_urls'],list) or not evidence['source_urls']
        or not all(isinstance(url,str) and url.startswith(('https://','http://')) for url in evidence['source_urls'])):
        return incomplete('执行覆盖声明没有明确来源')
    if pd.Timestamp(evidence['start_date'])>start or pd.Timestamp(evidence['end_date'])<end:
        return incomplete('执行覆盖证据不包含完整回放日期')
    filters=[('date','>=',start),('date','<=',end)]
    bars=pd.read_parquet(root/'bars',filters=filters)
    if bars.empty:return incomplete('回放区间无行情')
    if not isinstance(evidence['security_ids'],list) or not set(bars.security_id).issubset(evidence['security_ids']):
        return incomplete('执行覆盖证据不包含行情中的全部证券')
    master=pd.read_parquet(root/'securities.parquet')
    lots=pd.read_parquet(root/'historical_lots.parquet')
    actions=pd.read_parquet(root/'corporate_actions.parquet')
    if not set(LOT_COLUMNS).issubset(lots.columns):return incomplete('历史交易单位缺少证据字段')
    if not {'security_id','action_type','effective_date','payment_date','verified','source_url'}.issubset(actions.columns):
        return incomplete('公司行动缺少必要字段')
    for name,records in [('历史交易单位',lots),('公司行动',actions)]:
        if not records.verified.eq(True).all() or records.source_url.isna().any() or records.source_url.astype(str).str.strip().eq('').any():
            return incomplete(name+'存在未经核验或缺少来源的记录')
    # 空行动表只有在上述独立覆盖证据明确成立后，才可描述为区间内无行动。
    actions.attrs['coverage_complete']=True
    master_columns=[name for name in master.columns if name not in ('lot_size','lot_valid_from','lot_valid_to','verified','source_url')]
    replay_securities=master[master_columns].merge(lots[list(LOT_COLUMNS)],on='security_id',how='left',validate='one_to_many')
    factors=pd.read_parquet(root/'features',columns=['date','security_id','adv20_amount'],filters=filters)
    returns=pd.read_parquet(root/'risk_returns.parquet').set_index('date')
    calendar=pd.read_parquet(root/'references/calendar.parquet')
    dates=pd.to_datetime(calendar.loc[calendar.is_open.eq(1),'cal_date'])
    dates=dates.loc[dates.between(start,end)]
    if dates.empty:return incomplete('回放区间没有交易日历')
    lot_coverage_gaps=[]
    lot_start=pd.to_datetime(lots.lot_valid_from,errors='coerce')
    lot_end=pd.to_datetime(lots.lot_valid_to,errors='coerce')
    # 只有有效且明确零量、无成交额矛盾的记录无需执行手数；未知状态继续检查。
    confirmed_no_trade=(bars.data_valid.isin([True]) & bars.quote_present.isin([False])
                        & bars.volume.isin([0]) & (bars.amount.isna() | bars.amount.isin([0])))
    for security,observations in bars.loc[pd.to_datetime(bars.date).isin(dates) & ~confirmed_no_trade].groupby('security_id'):
        observed_dates=pd.to_datetime(observations.date)
        count=np.zeros(len(observed_dates),dtype=int)
        for index,row in lots.loc[lots.security_id.eq(security)].iterrows():
            if not np.isfinite(row.lot_size) or row.lot_size<=0:continue
            valid=observed_dates.ge(lot_start.loc[index])
            if pd.notna(row.lot_valid_to):valid &= observed_dates.le(lot_end.loc[index])
            count+=valid.to_numpy(int)
        bad=observed_dates.loc[count!=1]
        if not bad.empty:
            lot_coverage_gaps.append(dict(security_id=security,dates_without_unique_lot=len(bad),
                                          first_date=bad.min().isoformat(),last_date=bad.max().isoformat()))
    runs={};policies={}
    for name,allocation,fee in (('model','utility',.0025),('equal_weight','equal_weight',.0025),
                                ('stress_model','utility',.005),('stress_equal_weight','equal_weight',.005)):
        settings=replace(profile,fee_per_side=fee)
        policy=SavedForecastPolicy(forecast_dir,master,lots,returns,factors,settings,allocation)
        runs[name]=replay_account(None,bars,pd.DataFrame({'date':dates}),replay_securities,actions,settings,initial_cash,policy=policy)
        policies[name]=policy.reports
    complete=not lot_coverage_gaps and all(run['summary']['status']=='ok' for run in runs.values())
    execution_summary={'status':'ok' if complete else 'incomplete',
                       'gap_count':sum(run['summary']['gap_count'] for run in runs.values())+sum(gap['dates_without_unique_lot'] for gap in lot_coverage_gaps),
                       'corporate_action_coverage':True,'historical_lot_coverage_complete':not lot_coverage_gaps,
                       'historical_lot_gaps':lot_coverage_gaps,
                       'cash_payment_policy':'actual_account_receipt or issuer_final_schedule_simulated; receivables are not spendable cash'}
    try:
        metrics=evaluate_portfolio(runs['model']['daily'],runs['equal_weight']['daily'],runs['stress_model']['daily'],
                                   runs['stress_equal_weight']['daily'],execution_summary)
    except ValueError as exc:
        return {'status':'incomplete','reason':'账户净值无法完成评价: '+str(exc),'runs':runs,'policies':policies,'portfolio_metrics':None}
    return {'status':'ok' if complete else 'incomplete','runs':runs,'policies':policies,
            'portfolio_metrics':metrics,'execution_summary':execution_summary}
