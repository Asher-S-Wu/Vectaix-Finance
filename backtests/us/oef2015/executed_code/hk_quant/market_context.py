"""使用当日可见收益构建市场与风险输入，不拼接不同复权时点的价格水平。"""
import numpy as np
import pandas as pd


def market_context_from_inputs(inputs,calendar):
    returns=inputs.pivot(index='date',columns='security_id',values='return_1').reindex(calendar)
    quoted=inputs.pivot(index='date',columns='security_id',values='quote_present').reindex(calendar).eq(True)
    bias=inputs.pivot(index='date',columns='security_id',values='bias_60').reindex(calendar)
    median=returns.where(quoted).median(axis=1)
    level=(1+median).cumprod()
    result=pd.DataFrame({'date':calendar})
    for length in (20,60,252):
        result[f'market_momentum_{length}']=(level/level.shift(length)-1).to_numpy()
    result['market_volatility_60']=(median.rolling(60,min_periods=30).std()*np.sqrt(252)).to_numpy()
    valid=quoted&bias.notna()
    result['market_breadth_60']=((bias.gt(0)&valid).sum(axis=1)/valid.sum(axis=1)).to_numpy()
    return result


def risk_returns_from_inputs(inputs,securities):
    values=inputs.pivot(index='date',columns='security_id',values='return_1').sort_index()
    if np.isinf(values.to_numpy(dtype=float)).any():raise ValueError('风险收益含无穷值')
    master=securities.set_index('security_id')
    allowed=pd.DataFrame(False,index=values.index,columns=values.columns)
    for security in values:
        record=master.loc[security]
        if record.asset_type not in ('equity','reit'):continue
        if record.identity_status=='partially_verified':
            # These are already daily-identity-masked return_1 observations.
            # Non-observation dates stay missing, including isolated proofs.
            allowed[security]=values[security].notna()
            continue
        if record.identity_status!='verified' or pd.isna(record.identity_valid_from):continue
        valid=values.index>=pd.Timestamp(record.identity_valid_from)
        if pd.notna(record.identity_valid_to):valid &= values.index<pd.Timestamp(record.identity_valid_to)
        allowed[security]=valid
    # Both endpoints of a one-session return must belong to this identity.
    partial=master.identity_status.reindex(values.columns).eq('partially_verified')
    endpoints=allowed&allowed.shift(1,fill_value=False)
    endpoints.loc[:,partial]=allowed.loc[:,partial]
    return values.where(endpoints)
