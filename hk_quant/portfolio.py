"""基于可核查行情的现金约束组合优化。"""
import numpy as np
import pandas as pd
import cvxpy as cp
from .contracts import RiskProfile


def advise_portfolio(forecasts, holdings, cash, latest_bars, securities, returns, profile=RiskProfile(),allocation='utility'):
    excluded=[];risk_summary={}
    def error(reason, status='data_error'):
        return {'status': status, 'reason': reason, 'recommendations': pd.DataFrame(),
                'excluded': pd.DataFrame(excluded,columns=['security_id','reason']), 'summary': risk_summary}
    if allocation not in ('utility','equal_weight'):return error('未知配置方式')
    if not np.isfinite(cash) or cash < 0:
        return error('现金必须为非负有限数')
    if holdings.security_id.duplicated().any() or (not holdings.empty and (not np.isfinite(holdings.quantity).all() or (holdings.quantity < 0).any())):
        return error('持仓数量无效或重复')
    if latest_bars.empty or latest_bars.security_id.duplicated().any() or securities.security_id.duplicated().any():
        return error('市价或证券身份缺失/重复')
    bars=latest_bars.set_index('security_id'); sec=securities.set_index('security_id')
    def fx_valid(s):
        if s not in bars.index or s not in sec.index or 'currency' not in sec or 'fx_to_hkd' not in bars: return False
        currency=sec.loc[s,'currency'];fx=bars.loc[s,'fx_to_hkd']
        return pd.notna(currency) and bool(str(currency).strip()) and pd.notna(fx) and np.isfinite(fx) and fx>0 and (currency!='HKD' or fx==1)
    owned=holdings.set_index('security_id').quantity.to_dict()
    for sid,q in owned.items():
        if q and (sid not in bars.index or not np.isfinite(bars.loc[sid,'raw_close']) or bars.loc[sid,'raw_close']<=0):
            return error('持仓缺少明确市价: '+str(sid))
        if q and not fx_valid(sid): return error('持仓缺少币种或当日有效汇率: '+str(sid))
    equity=float(cash+sum(q*bars.loc[s,'raw_close']*bars.loc[s,'fx_to_hkd'] for s,q in owned.items() if q))
    if equity<=0:
        return error('账户资产必须大于零')
    asof=pd.to_datetime(latest_bars.date).max()
    f=forecasts.loc[(forecasts.horizon==profile.horizon)&(pd.to_datetime(forecasts.date)<=asof)].copy()
    if f.security_id.duplicated().any():
        return error('同一证券有重复预测')
    f=f.set_index('security_id')
    def tradable(s):
        return s in bars.index and bool(bars.loc[s,'quote_present']==True) and np.isfinite(bars.loc[s,'volume']) and bars.loc[s,'volume']>0 and np.isfinite(bars.loc[s,'raw_close']) and bars.loc[s,'raw_close']>0 and pd.Timestamp(bars.loc[s,'date'])==asof
    def lot_valid(s):
        if s not in sec.index or not {'lot_valid_from','lot_valid_to','lot_size'}.issubset(sec.columns):return False
        start=pd.to_datetime(sec.loc[s,'lot_valid_from'],errors='coerce')
        end_value=sec.loc[s,'lot_valid_to']
        end=pd.to_datetime(end_value,errors='coerce')
        return (pd.notna(start) and start<=asof and (pd.isna(end_value) or (pd.notna(end) and end>=asof))
                and np.isfinite(sec.loc[s,'lot_size']) and sec.loc[s,'lot_size']>0)
    def buyable(s):
        return tradable(s) and fx_valid(s) and s in sec.index and sec.loc[s,'identity_status']=='verified' and lot_valid(s) and np.isfinite(bars.loc[s,'adv20_amount']) and bars.loc[s,'adv20_amount']>0
    valid=f.loc[(f.status=='ok') & np.isfinite(f.expected_return)].sort_values('expected_return',ascending=False)
    candidates=[]
    for s in valid.index:
        if buyable(s):
            candidates.append(s)
        else:
            if not fx_valid(s):reason='缺少币种或当日有效汇率，不可买入'
            elif s not in sec.index or sec.loc[s,'identity_status']!='verified':reason='证券身份未核实，不可买入'
            elif not lot_valid(s):reason='缺少信号日有效交易单位及证据日期，不可买入'
            elif not tradable(s):reason='缺少当日可成交行情，不可买入'
            else:reason='缺少已知成交额，不可买入'
            excluded.append(dict(security_id=s,reason=reason))
    ids=list(dict.fromkeys(candidates+[s for s,q in owned.items() if q]))
    if not ids:
        return error('没有具备可交易证据、交易单位及币种汇率的候选证券')
    history_dates=pd.DatetimeIndex(pd.to_datetime(returns.index,errors='coerce'))
    if history_dates.isna().any() or history_dates.duplicated().any() or returns.columns.duplicated().any():
        return error('风险历史日期或证券列无效/重复')
    positions=np.flatnonzero(history_dates<asof)
    positions=positions[np.argsort(history_dates[positions])][-252:]
    history=returns.iloc[positions].copy()
    history.index=history_dates[positions]
    history=history.where(np.isfinite(history))
    market=history.median(axis=1)
    market_variance=float(market.var(ddof=1)*252)
    risk_summary=dict(risk_model='market_single_factor_ols_independent_residuals',risk_window_sessions=252,
                      market_observations=int(market.notna().sum()),market_variance_annual=market_variance,
                      risk_observations={},risk_estimates={},risk_evaluated_count=len(ids),
                      risk_limitations='市场单因子与独立个股残差估计，未覆盖全部行业、共同风险或停牌流动性风险，也不保证实际波动率',
                      risk_data_policy='仅信号日前最近252个交易日已核验估值收益；全市场可用收益中位数；个股至少60对观察；无成交参考估值不代表可成交；不填补缺失收益')
    if market.notna().sum()<60 or not np.isfinite(market_variance) or market_variance<=0:
        return error('市场风险因子缺少至少60日真实历史或方差不可估计')
    risk_summary['risk_window_start']=history.index.min().isoformat()
    risk_summary['risk_window_end']=history.index.max().isoformat()
    estimates={}
    for s in ids:
        reason=None
        if s not in history:
            count=0;reason='个股风险缺少收益历史，至少需要60对真实观察'
        else:
            paired=history[s].notna() & market.notna()
            count=int(paired.sum())
            if count<60:reason='个股风险历史不足60对真实观察'
            else:
                x=np.column_stack([np.ones(count),market.loc[paired].to_numpy(float)])
                y=history.loc[paired,s].to_numpy(float)
                coefficients,_,rank,_=np.linalg.lstsq(x,y,rcond=None)
                if rank!=2:reason='个股重合日期的市场因子无可识别变化，风险不可估计'
                else:
                    residual=y-x@coefficients
                    variance=float(residual@residual/(count-2)*252)
                    if not np.isfinite(coefficients).all() or not np.isfinite(variance):reason='个股风险估计非有限值'
                    else:estimates[s]=dict(intercept_daily=float(coefficients[0]),beta=float(coefficients[1]),residual_variance_annual=variance,observations=count)
        risk_summary['risk_observations'][s]=count
        if reason:
            excluded.append(dict(security_id=s,reason=reason))
            if owned.get(s,0):return error('实际持仓风险无法估计: '+str(s)+'；'+reason)
    risk_summary['risk_estimates']=estimates
    ids=[s for s in ids if s in estimates]
    if not ids:return error('所有候选均缺少可估计的真实风险历史')
    beta=np.array([estimates[s]['beta'] for s in ids])
    residual_variance=np.array([estimates[s]['residual_variance_annual'] for s in ids])
    prices=np.array([bars.loc[s,'raw_close']*bars.loc[s,'fx_to_hkd'] for s in ids],float)
    quantities=np.array([owned.get(s,0) for s in ids],float)
    current=quantities*prices/equity
    mu=np.array([float(f.loc[s,'expected_return']) if s in valid.index else 0. for s in ids])
    adjustable=np.array([buyable(s) and s in valid.index for s in ids])
    capacity_lots=np.zeros(len(ids));minimum_quantities=quantities.copy()
    for i,s in enumerate(ids):
        if adjustable[i]:
            lot=float(sec.loc[s,'lot_size'])
            capacity_lots[i]=np.floor(profile.max_participation*float(bars.loc[s,'adv20_amount'])*float(bars.loc[s,'fx_to_hkd'])/(prices[i]*lot)+1e-9)
            minimum_quantities[i]-=min(np.floor(quantities[i]/lot),capacity_lots[i])*lot
    minimum=minimum_quantities*prices/equity
    isin_groups={}
    if 'isin' in sec:
        for i,s in enumerate(ids):
            if s in sec.index and sec.loc[s,'identity_status']=='verified' and pd.notna(sec.loc[s,'isin']) and str(sec.loc[s,'isin']).strip():
                isin_groups.setdefault(str(sec.loc[s,'isin']),[]).append(i)
    groups=[dict(constraint='single_position',security_id=s,indices=[i],limit=profile.max_position_weight) for i,s in enumerate(ids)]
    groups.extend(dict(constraint='isin_concentration',isin=isin,indices=indices,limit=profile.max_position_weight) for isin,indices in isin_groups.items())
    groups.append(dict(constraint='total_equity',indices=list(range(len(ids))),limit=profile.max_equity_weight))
    for group in groups:group['minimum_weight']=float(minimum[group['indices']].sum())
    w=cp.Variable(len(ids));nav=cp.Variable();turnover=cp.norm1(w-current)
    fee_fraction=profile.fee_per_side*turnover
    risk=cp.sum(cp.multiply(residual_variance,cp.square(w)))+market_variance*cp.square(beta@w)
    risk_norm=cp.norm(cp.hstack([cp.multiply(np.sqrt(residual_variance),w),np.sqrt(market_variance)*(beta@w)]),2)
    hard_constraints=[w>=minimum,cp.sum(w)+fee_fraction<=1,nav+fee_fraction<=1]
    for i,s in enumerate(ids):
        if not adjustable[i]:
            hard_constraints.append(w[i]==current[i])
        else:
            hard_constraints.append(cp.abs(w[i]-current[i])<=capacity_lots[i]*float(sec.loc[s,'lot_size'])*prices[i]/equity)
    # 最低保留量跨过扣费后上限时，该组只能保留最低数量；其他组继续遵守原上限。
    # 按净值临界点预先分段，每段均为同一个 CLARABEL 凸问题。
    minimum_nav=(1-profile.fee_per_side*float(current.sum()))/(1+profile.fee_per_side)
    breakpoints=sorted({group['minimum_weight']/group['limit'] for group in groups
                        if minimum_nav<group['minimum_weight']/group['limit']<1})
    cutpoints=[minimum_nav,*breakpoints,1.]
    segments=[]
    for lower,upper in zip(cutpoints[:-1],cutpoints[1:]):
        constraints=[*hard_constraints,nav>=lower,nav<=upper]
        for group in groups:
            value=cp.sum(w[group['indices']])
            if group['minimum_weight']>group['limit']*(lower+upper)/2:
                constraints.append(value==group['minimum_weight'])
            else:
                constraints.append(value<=group['limit']*nav)
        segments.append(constraints)
    risk_trials=[]
    try:
        for constraints in segments:
            certificate=cp.Problem(cp.Minimize(risk_norm-profile.target_annual_volatility*nav),constraints)
            certificate.solve(solver='CLARABEL')
            if certificate.status=='infeasible':continue
            if certificate.status!='optimal':return error('风险可达性求解未完成: '+certificate.status,'solver_error')
            risk_trials.append(dict(constraints=constraints,gap=float(certificate.value),weights=w.value.copy()))
    except cp.error.SolverError as exc:
        return error(str(exc),'solver_error')
    if not risk_trials:return error('现金、参与率及最低保留持仓无法共同满足','infeasible')
    least_gap=min(risk_trials,key=lambda result:result['gap'])
    risk_reachable=least_gap['gap']<=1e-7
    if allocation=='utility':
        objective=cp.Maximize(mu@w-profile.fee_per_side*turnover-risk*profile.horizon/252)
    else:
        equal_target=np.full(len(ids),min(profile.max_position_weight,profile.max_equity_weight/len(ids)))
        equal_risk=float(residual_variance@(equal_target**2)+market_variance*(beta@equal_target)**2)
        if equal_risk>profile.target_annual_volatility**2:
            equal_target*=profile.target_annual_volatility/np.sqrt(equal_risk)
        objective=cp.Minimize(cp.sum_squares(w-equal_target))
    if risk_reachable:
        solutions=[]
        try:
            for trial in risk_trials:
                if trial['gap']>1e-7:continue
                problem=cp.Problem(objective,[*trial['constraints'],risk_norm<=profile.target_annual_volatility*nav])
                problem.solve(solver='CLARABEL')
                if problem.status=='infeasible':continue
                if problem.status!='optimal':return error('账户优化未完成: '+problem.status,'solver_error')
                solutions.append((float(problem.value),w.value.copy()))
        except cp.error.SolverError as exc:
            return error(str(exc),'solver_error')
        if not solutions:return error('风险可达性与账户优化结果不一致','solver_error')
        optimized=(max(solutions,key=lambda result:result[0]) if allocation=='utility' else min(solutions,key=lambda result:result[0]))[1]
    else:
        optimized=least_gap['weights']
    target=quantities.copy()
    rows=[]
    for i,s in enumerate(ids):
        lot=sec.loc[s,'lot_size'] if s in sec.index else np.nan
        if adjustable[i]:
            desired=max(0,float(optimized[i]))*equity/prices[i]
            # Orders are whole lots; inherited odd shares are preserved.
            delta=desired-quantities[i]
            if delta >= 0:
                target[i]=quantities[i]+min(np.floor(delta/lot+1e-7),capacity_lots[i])*lot
            else:
                sell_lots=min(np.ceil(abs(delta)/lot-1e-7),np.floor(quantities[i]/lot),capacity_lots[i])
                target[i]=quantities[i]-sell_lots*lot
    for i,s in enumerate(ids):
        if target[i]!=quantities[i] and (not buyable(s) or abs(target[i]-quantities[i])*prices[i]>profile.max_participation*float(bars.loc[s,'adv20_amount'])*float(bars.loc[s,'fx_to_hkd'])+1e-6):
            return error('整手订单超出交易参与率约束','rounding_infeasible')
    fees=float(np.sum(abs(target-quantities)*prices)*profile.fee_per_side)
    target_cash=float(equity-np.sum(target*prices)-fees)
    post_fee_equity=equity-fees
    target_weights=target*prices/post_fee_equity
    vol=float(np.sqrt(residual_variance@(target_weights**2)+market_variance*(beta@target_weights)**2))
    violations=[]
    for group in groups:
        actual=float(target_weights[group['indices']].sum())
        if actual>group['limit']+1e-7:
            minimum_actual=group['minimum_weight']*equity/post_fee_equity
            unavoidable=minimum_actual>group['limit']+1e-7 and actual<=minimum_actual+1e-7
            identity={name:group[name] for name in ('security_id','isin') if name in group}
            violations.append(dict(constraint=group['constraint'],**identity,actual=actual,limit=group['limit'],
                                   excess=actual-group['limit'],minimum_retained_weight=minimum_actual,unavoidable=unavoidable,
                                   reason='已无法卖出的持仓及本次费用造成超限' if unavoidable else '整手后仍有可调整仓位超限'))
    if vol>profile.target_annual_volatility+1e-7:
        violations.append(dict(constraint='annual_volatility',actual=vol,limit=profile.target_annual_volatility,
                               excess=vol-profile.target_annual_volatility,unavoidable=not risk_reachable,
                               reason='原波动目标在连续交易约束下不可达' if not risk_reachable else '连续目标可达，但整手后的风险仍超过原目标'))
    if target_cash < -1e-6:
        violations.append(dict(constraint='cash',actual=target_cash,limit=0.,excess=-target_cash,unavoidable=False,reason='整手后的现金不足'))
    for i,s in enumerate(ids):
        delta=target[i]-quantities[i]
        action='买入' if delta>0 and quantities[i]==0 else '加仓' if delta>0 else '卖出' if target[i]==0 and delta<0 else '减仓' if delta<0 else '持有'
        lot=sec.loc[s,'lot_size'] if s in sec.index else np.nan
        reason='预期收益、交易成本和账户风险联合优化' if allocation=='utility' else '同一候选池在风险与实际现金约束下等权配置'
        if s not in sec.index or sec.loc[s,'identity_status']!='verified': reason='证券身份未核实，冻结现有持仓'
        elif not tradable(s): reason='当前无法交易，保留持仓'
        elif not lot_valid(s): reason='缺少信号日有效交易单位及证据日期，冻结现有持仓，不提供订单数量'
        elif not buyable(s): reason='缺少已知成交额，冻结现有持仓'
        elif s not in valid.index: reason='预测不可用，未生成交易建议'
        rows.append(dict(security_id=s,action=action,current_quantity=quantities[i],target_quantity=target[i],order_quantity=abs(delta) if lot_valid(s) else None,target_weight=target_weights[i],reference_price=float(bars.loc[s,'raw_close']),currency=sec.loc[s,'currency'],fx_to_hkd=float(bars.loc[s,'fx_to_hkd']),reference_price_hkd=prices[i],target_value_hkd=target[i]*prices[i],estimated_fee=abs(delta)*prices[i]*profile.fee_per_side,reason=reason))
    return {'status':'constraints_unmet' if violations else 'ok','reason':'原账户目标仍有未满足项，详见逐项约束结果' if violations else None,
            'excluded':pd.DataFrame(excluded,columns=['security_id','reason']),'recommendations':pd.DataFrame(rows),
            'summary':dict(allocation=allocation,equity=equity,post_fee_equity=post_fee_equity,current_cash=float(cash),target_cash=target_cash,
                           estimated_fees=fees,annual_volatility=vol,target_equity_weight=float(target_weights.sum()),coverage_count=len(f),candidate_count=len(ids),
                           constraints_satisfied=not violations,constraint_violations=violations,
                           controllable_constraints_satisfied=all(v['unavoidable'] for v in violations),
                           volatility_target_reachable=bool(risk_reachable),minimum_volatility_constraint_shortfall=max(0.,least_gap['gap']),
                           volatility_certificate_basis='扣费前净值归一的风险金额减原波动目标乘扣费后净值；连续可行性不等于整手可行性',
                           preference_optimized=bool(risk_reachable),nav_constraint_segments=len(segments),
                           minimum_retained_quantities={s:float(minimum_quantities[i]) for i,s in enumerate(ids)},
                           **risk_summary,as_of=asof.isoformat(),solver='CLARABEL',sector_constraints='未取得可靠行业分类，未设行业约束')}
