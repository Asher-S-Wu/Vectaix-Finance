"""未复权真实股数的现金账户；所有缺口公开，不填补行情。"""
import numpy as np
import pandas as pd
from .contracts import RiskProfile


def replay_account(targets,bars,calendar,securities,corporate_actions,profile=RiskProfile(),initial_cash=1000000,policy=None):
    daily=[]; trades=[]; unfilled=[]; gaps=[]; events=[]; cash_payment_bases=set();policy_targets=[]
    def finish(status):
        return dict(summary=dict(status=status,policy_mode='dynamic' if policy is not None else 'static',corporate_action_coverage='complete' if corporate_actions.attrs.get('coverage_complete') is True else 'incomplete',initial_cash=initial_cash,ending_cash=cash,cash_payment_bases=sorted(cash_payment_bases),cash_payment_policy="仅已核验实际到账或最终公告明确派付日；公告派付日属于模拟到账",fees=sum(t['fee'] for t in trades),gap_count=len(gaps),execution='next_session_raw_vwap',participation=profile.max_participation,fee_per_side=profile.fee_per_side),daily=pd.DataFrame(daily),trades=pd.DataFrame(trades),unfilled=pd.DataFrame(unfilled),gaps=pd.DataFrame(gaps),events=pd.DataFrame(events),targets=pd.DataFrame(policy_targets),holdings=pd.DataFrame([dict(security_id=s,quantity=q) for s,q in shares.items() if q]))
    cash=float(initial_cash);shares={}
    if not np.isfinite(cash) or cash<0:
        gaps.append(dict(reason='初始现金无效'));return finish('data_error')
    dates=pd.DatetimeIndex(pd.to_datetime(calendar['date'] if isinstance(calendar,pd.DataFrame) else calendar)).sort_values().unique()
    b=bars.copy();b['date']=pd.to_datetime(b.date)
    if b.duplicated(['date','security_id']).any():
        gaps.append(dict(reason='行情重复'));return finish('data_error')
    if policy is not None and (not callable(policy) or (targets is not None and not targets.empty)):
        gaps.append(dict(reason='动态policy必须可调用且不能与非空静态targets混用'));return finish('data_error')
    t=pd.DataFrame(columns=['signal_date','security_id','target_weight']) if targets is None else targets.copy()
    t['signal_date']=pd.to_datetime(t.signal_date)
    t['target_weight']=pd.to_numeric(t.target_weight,errors='coerce')
    if t.duplicated(['signal_date','security_id']).any() or not np.isfinite(t.target_weight).all() or (t.target_weight<0).any() or (t.target_weight>profile.max_position_weight+1e-9).any() or (t.groupby('signal_date').target_weight.sum()>profile.max_equity_weight+1e-9).any():
        gaps.append(dict(reason='目标权重无效'));return finish('data_error')
    if {'isin','identity_status','lot_valid_from'}.issubset(securities.columns):
        for signal_date,snapshot in t.groupby('signal_date'):
            effective=securities.loc[(securities.identity_status=='verified') & (pd.to_datetime(securities.lot_valid_from,errors='coerce')<=signal_date)].copy()
            if 'lot_valid_to' in effective:
                expiry=pd.to_datetime(effective.lot_valid_to,errors='coerce')
                effective=effective.loc[expiry.isna() | (expiry>=signal_date)]
            effective=effective.loc[effective['isin'].notna() & effective['isin'].astype(str).str.strip().ne('')]
            weights={}
            for row in snapshot.itertuples():
                matches=effective.loc[effective.security_id==row.security_id,'isin'].unique()
                if len(matches)==1:
                    isin=str(matches[0]);weights[isin]=weights.get(isin,0.0)+float(row.target_weight)
            if any(weight>profile.max_position_weight+1e-9 for weight in weights.values()):
                gaps.append(dict(date=signal_date,reason='同一已验证ISIN柜台合计目标仓位超限'));return finish('data_error')
    orders={}
    # Sizing uses signal-day prices and account, never tomorrow's return or volume.
    signals={d:g for d,g in t.groupby('signal_date')}
    for d in signals:
        if d not in dates:
            gaps.append(dict(date=d,reason='信号日不在交易日历'))
    actions=corporate_actions.copy().reset_index(drop=True)
    for c in ['effective_date','payment_date']:
        if c in actions: actions[c]=pd.to_datetime(actions[c],errors='coerce')
        else: actions[c]=pd.NaT
    amount_known_on={}
    if 'amount_known_at' in actions:
        for idx,a in actions.iterrows():
            if a.action_type not in ('cash_dividend','privatisation_cash','acquisition','delisting'):continue
            stamp=pd.Timestamp(a.amount_known_at)
            if pd.isna(stamp) or stamp.tzinfo is None:
                gaps.append(dict(reason='现金金额公开时间必须已知且包含时区'));return finish('data_error')
            local=stamp.tz_convert('Asia/Hong_Kong')
            day=local.tz_localize(None).normalize()
            if local>local.normalize()+pd.Timedelta(hours=19):day+=pd.Timedelta(days=1)
            amount_known_on[idx]=day
    entitlements={}; unknown_amounts={}; no_trade_sessions={}; blocked=set()
    def gap(d,s,reason): gaps.append(dict(date=d,security_id=s,reason=reason))
    def trade_evidence(quote):
        present=quote.get('quote_present');volume=quote.get('volume');amount=quote.get('amount');valid=quote.get('data_valid')
        if pd.isna(valid) or valid!=True:return None,'当日行情缺少有效数据证据'
        if pd.isna(present) or present not in (True,False) or pd.isna(volume) or not np.isfinite(volume):
            return None,'缺少明确成交状态或成交量'
        if volume==0 and pd.notna(amount) and amount!=0:return None,'零成交量与成交额不一致'
        # 明确零量足以证明未成交；缺失成交额保持缺失，不要求执行手数。
        if present==False and volume==0:return False,None
        if present==False or volume<=0:return None,'成交状态与成交量不一致'
        return True,None
    def historical_lot(s,d):
        if 'lot_valid_from' not in securities: return None
        records=securities.loc[securities.security_id==s].copy()
        valid=pd.to_datetime(records.lot_valid_from,errors='coerce')<=d
        if 'lot_valid_to' in records: valid &= pd.to_datetime(records.lot_valid_to,errors='coerce').isna() | (pd.to_datetime(records.lot_valid_to,errors='coerce')>=d)
        records=records.loc[valid]
        if len(records)!=1 or records.iloc[0].get('identity_status')!='verified': return None
        value=records.iloc[0].lot_size
        return float(value) if pd.notna(value) and np.isfinite(value) and value>0 else None
    def security_currency(s):
        if 'currency' not in securities: return None
        currencies=securities.loc[securities.security_id==s,'currency'].dropna().unique()
        return str(currencies[0]) if len(currencies)==1 and str(currencies[0]).strip() else None
    def fx_rate(s,quotes):
        currency=security_currency(s)
        if currency is None or s not in quotes.index or 'fx_to_hkd' not in quotes: return None
        fx=quotes.loc[s,'fx_to_hkd']
        if pd.isna(fx) or not np.isfinite(fx) or fx<=0 or (currency=='HKD' and fx!=1): return None
        return float(fx)
    def cash_fx(s,currency,quotes):
        if currency=='HKD': return 1.0
        if currency!=security_currency(s): return None
        return fx_rate(s,quotes)
    event_dates=pd.DatetimeIndex(pd.concat([actions.effective_date,actions.payment_date,pd.Series(amount_known_on,dtype='datetime64[ns]')]).dropna())
    timeline=dates.union(event_dates[(event_dates>=dates.min()) & (event_dates<=dates.max())].unique()).sort_values()
    for d in timeline:
        quotes=b.loc[b.date==d].set_index('security_id')
        # Ex-date rights are based on shares owned before today's trading.
        for idx,a in actions.iterrows():
            s=a.security_id; kind=a.action_type
            if a.effective_date==d:
                if a.get('verified',False)!=True:
                    if shares.get(s,0):gap(d,s,'公司行动未经证据核实');blocked.add(s)
                    continue
                if kind in ('cash_dividend','privatisation_cash','acquisition','delisting'):
                    q=shares.get(s,0)
                    if q:
                        source=a.get('source_url');basis=a.get('payment_basis')
                        if not isinstance(source,str) or not source.strip() or basis not in ('actual_account_receipt','issuer_final_schedule_simulated'):
                            gap(d,s,'现金公司行动缺少来源或明确到账依据');blocked.add(s);continue
                        amount=a.get('cash_per_share_hkd',np.nan)
                        currency='HKD'
                        if pd.isna(amount):
                            amount=a.get('cash_per_share',np.nan);currency=a.get('cash_currency')
                        if pd.isna(currency) or currency not in ('HKD',security_currency(s)):
                            gap(d,s,'现金公司行动缺少明确或匹配的币种');blocked.add(s);continue
                        if not np.isfinite(amount) or amount<0 or (pd.notna(a.payment_date) and a.payment_date<d):
                            gap(d,s,'现金公司行动付款日或金额无效')
                            if kind!='cash_dividend':blocked.add(s)
                        else:
                            if pd.isna(a.payment_date):gap(d,s,'现金公司行动缺少已核实付款日，保留应收不入现金')
                            receivable=q*float(amount)
                            if idx in amount_known_on and amount_known_on[idx]>d:
                                unknown_amounts[idx]=receivable
                                receivable=np.nan
                            entitlements[idx]=(s,receivable,kind,q,currency)
                            cash_payment_bases.add(basis)
                            if kind!='cash_dividend':
                                blocked.add(s);shares[s]=0
                            events.append(dict(date=d,security_id=s,action_type=kind,event_type='entitlement',cash_flow=0,cash_after=cash,receivable_amount=receivable,entitled_quantity=q,cash_currency=currency,shares_cancelled=q if kind!='cash_dividend' else 0,payment_basis=basis,source_url=source))
                elif kind in ('split','consolidation'):
                    multiplier=a.get('share_multiplier',np.nan)
                    if not np.isfinite(multiplier) or multiplier<=0:
                        if shares.get(s,0):gap(d,s,'拆并股比例缺失');blocked.add(s)
                    else:
                        shares[s]=shares.get(s,0)*float(multiplier)
                        for execution_date,pending_orders in orders.items():
                            if execution_date>=d:
                                for order in pending_orders:
                                    if order['security_id']==s: order['quantity']*=float(multiplier)
                        events.append(dict(date=d,security_id=s,action_type=kind,share_multiplier=float(multiplier),cash_flow=0))
                elif shares.get(s,0):
                    gap(d,s,'未支持的公司行动类型: '+str(kind));blocked.add(s)
            if idx in unknown_amounts and amount_known_on[idx]<=d:
                sid,_,event_kind,quantity,currency=entitlements[idx]
                recognized=unknown_amounts.pop(idx)
                entitlements[idx]=(sid,recognized,event_kind,quantity,currency)
                events.append(dict(date=d,security_id=sid,action_type=event_kind,event_type='amount_recognition',cash_flow=0,cash_after=cash,receivable_amount=recognized,entitled_quantity=quantity,cash_currency=currency,source_url=a.source_url))
            if a.payment_date==d and idx in entitlements:
                if idx in unknown_amounts:
                    gap(d,s,'派付日金额尚未公开，不能生成现金');continue
                s,amount,kind,q,currency=entitlements[idx]
                rate=cash_fx(s,currency,quotes)
                if rate is None:
                    gap(d,s,'付款日缺少明确汇率，未生成现金');continue
                entitlements.pop(idx)
                amount*=rate
                cash+=amount
                events.append(dict(date=d,security_id=s,action_type=kind,event_type='payment',cash_flow=amount,cash_after=cash,payment_basis=a.payment_basis,source_url=a.source_url))
            if pd.isna(a.effective_date) and a.payment_date==d and shares.get(s,0):gap(d,s,'缺少权益生效日，无法确定应收金额')
        if d not in dates:
            continue
        pending=orders.get(d,[])
        pending.sort(key=lambda x:x['quantity']) # Sell first to release actual cash.
        for order in pending:
            s=order['security_id']; wanted=order['quantity']
            reason=None;lot=None
            if s in blocked:reason='公司行动状态未完成，禁止交易'
            elif s not in quotes.index:reason='缺少当日成交数据';gap(d,s,reason)
            else:
                q=quotes.loc[s]
                present,reason=trade_evidence(q)
                if reason is not None:gap(d,s,reason)
                elif present is False:reason='当日没有成交，订单未执行'
                elif not np.isfinite(q.get('vwap',np.nan)) or q.vwap<=0:
                    reason='存在成交但缺少真实VWAP';gap(d,s,reason)
            if reason is None:
                lot=historical_lot(s,d)
                if lot is None:reason='缺少已核实证券身份或当日有效交易单位';gap(d,s,reason)
            if reason is None and fx_rate(s,quotes) is None:
                reason='缺少成交日币种或有效汇率';gap(d,s,reason)
            if reason:
                unfilled.append(dict(date=d,security_id=s,signal_date=order['signal_date'],quantity=wanted,reason=reason));continue
            native_price=float(quotes.loc[s,'vwap']);fx=fx_rate(s,quotes);price=native_price*fx;volume=float(quotes.loc[s,'volume'])
            capacity=np.floor(volume*profile.max_participation/lot)*lot
            amount=min(abs(wanted),capacity)
            if wanted>0:amount=min(amount,np.floor(cash/(price*(1+profile.fee_per_side))/lot)*lot)
            else:amount=min(amount,shares.get(s,0))
            amount=np.floor(amount/lot)*lot
            delta=float(np.sign(wanted)*amount)
            if delta:
                fee=abs(delta)*price*profile.fee_per_side
                cash-=delta*price+fee;shares[s]=shares.get(s,0)+delta
                trades.append(dict(date=d,signal_date=order['signal_date'],security_id=s,quantity=abs(delta),signed_quantity=delta,side='buy' if delta>0 else 'sell',price=native_price,currency=security_currency(s),fx_to_hkd=fx,price_hkd=price,value_hkd=abs(delta)*price,fee=fee,cash_after=cash,lot_size=lot,day_volume=volume))
            if abs(delta-wanted)>1e-8:
                unfilled.append(dict(date=d,signal_date=order['signal_date'],security_id=s,quantity=wanted-delta,reason='成交量、现金或整手限制'))
        value=0.;complete=True;quote_details=[]
        for s,q in shares.items():
            if not q:continue
            if s not in quotes.index or not np.isfinite(quotes.loc[s,'raw_close']) or quotes.loc[s,'raw_close']<=0:
                complete=False;gap(d,s,'缺少当日明确未复权估值');continue
            fx=fx_rate(s,quotes)
            if fx is None:
                complete=False;gap(d,s,'持仓缺少当日币种或有效汇率');continue
            present,reason=trade_evidence(quotes.loc[s])
            if reason is not None:
                no_trade_sessions[s]=None;sessions_without_trade=None
                gap(d,s,reason+'，无法统计无成交天数')
            else:
                if present:no_trade_sessions[s]=0
                elif no_trade_sessions.get(s) is not None:no_trade_sessions[s]+=1
                sessions_without_trade=no_trade_sessions.get(s)
            value+=q*float(quotes.loc[s,'raw_close'])*fx
            quote_details.append(dict(security_id=s,quantity=q,raw_close=float(quotes.loc[s,'raw_close']),currency=security_currency(s),fx_to_hkd=fx,value_hkd=q*float(quotes.loc[s,'raw_close'])*fx,quote_present=present,sessions_without_trade=sessions_without_trade))
        receivables=0.0
        for sid,amount,kind,q,currency in entitlements.values():
            if not np.isfinite(amount):
                complete=False;gap(d,sid,'分派权益已保留，但金额尚未公开，账户估值不完整');receivables=np.nan;continue
            rate=cash_fx(sid,currency,quotes)
            if rate is None:
                complete=False;gap(d,sid,'应收现金缺少当日汇率');receivables=np.nan
            else: receivables+=amount*rate
        equity=cash+value+receivables if complete else np.nan
        daily.append(dict(date=d,cash=cash,receivables=receivables,holdings_value=value if complete else np.nan,equity=equity,valuation_complete=complete,positions=quote_details))
        if policy is not None:
            if not complete:
                gap(d,None,'信号日账户估值不完整，无法调用策略');continue
            account=dict(cash=cash,equity=equity,receivables=receivables,
                         holdings=pd.DataFrame([dict(security_id=s,quantity=q) for s,q in shares.items() if q],columns=['security_id','quantity']),
                         bars=quotes.reset_index().copy(deep=True))
            try:
                decision=policy(d,account)
                if not isinstance(decision,pd.DataFrame) or not {'security_id','target_weight'}.issubset(decision.columns):
                    raise ValueError('policy必须返回security_id和target_weight两列的目标表')
                decision=decision[['security_id','target_weight']].copy()
                weights=pd.to_numeric(decision.target_weight,errors='coerce')
                if decision.security_id.isna().any() or decision.security_id.duplicated().any() or not np.isfinite(weights).all() or weights.lt(0).any() or weights.gt(profile.max_position_weight+1e-9).any() or weights.sum()>profile.max_equity_weight+1e-9:
                    raise ValueError('动态策略目标权重无效或超限')
                decision['target_weight']=weights;decision['signal_date']=d
                if {'isin','identity_status','lot_valid_from'}.issubset(securities.columns):
                    effective=securities.loc[securities.identity_status.eq('verified') & pd.to_datetime(securities.lot_valid_from,errors='coerce').le(d)]
                    if 'lot_valid_to' in effective:
                        expiry=pd.to_datetime(effective.lot_valid_to,errors='coerce')
                        effective=effective.loc[expiry.isna() | expiry.ge(d)]
                    issuer_weights={}
                    for item in decision.itertuples():
                        identities=effective.loc[effective.security_id.eq(item.security_id),'isin'].dropna().unique()
                        if len(identities)==1 and str(identities[0]).strip():
                            issuer=str(identities[0]);issuer_weights[issuer]=issuer_weights.get(issuer,0.)+item.target_weight
                    if any(weight>profile.max_position_weight+1e-9 for weight in issuer_weights.values()):
                        raise ValueError('动态策略同一ISIN合计目标仓位超限')
                signals[d]=decision
                policy_targets.extend(decision.to_dict('records'))
            except Exception as exc:
                gap(d,None,'动态策略失败，保留实际账户: '+str(exc));continue
        if d in signals:
            future=dates[dates>d]
            if not len(future):
                for s in signals[d].security_id:unfilled.append(dict(date=d,security_id=s,reason='回放区间没有下一交易日'))
                continue
            if not complete:
                gap(d,None,'信号日账户估值不完整，无法计算订单');continue
            snapshot=signals[d].set_index('security_id').target_weight.to_dict()
            for s,q in shares.items():
                if q and s not in snapshot: snapshot[s]=0.0
            for s,weight in snapshot.items():
                if s in blocked:
                    if weight:gap(d,s,'公司行动已冻结或注销股份，不可下单')
                    continue
                if s not in quotes.index or not np.isfinite(quotes.loc[s,'raw_close']) or quotes.loc[s,'raw_close']<=0:
                    gap(d,s,'信号日缺少真实参考价');continue
                fx=fx_rate(s,quotes)
                if fx is None:
                    gap(d,s,'信号日缺少币种或有效汇率，不可下单');continue
                desired=float(weight)*equity/(float(quotes.loc[s,'raw_close'])*fx)
                delta=desired-shares.get(s,0)
                if abs(delta)>1e-8:orders.setdefault(future[0],[]).append(dict(signal_date=d,security_id=s,quantity=delta))
    return finish('ok' if not gaps and corporate_actions.attrs.get('coverage_complete') is True else 'incomplete')
