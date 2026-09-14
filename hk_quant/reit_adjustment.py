"""按公开时点构建现金分派价格复权；不是现金财富回放或训练许可。"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import pandas as pd

from hk_quant.distribution_versions import distributions_asof


@dataclass
class AdjustmentResult:
    history: pd.DataFrame
    selected_events: pd.DataFrame
    factors: pd.DataFrame
    gaps: pd.DataFrame
    status: str
    coverage_complete: bool = False
    return_basis: str = 'corporate_action_adjusted_price_return_not_cash_wealth'


def adjusted_history_asof(raw_bars, calendar, event_versions, cutoff, quote_currency):
    """单证券、无填价的现金分派复权，返回独立快照和未解决原因。

    calendar 是明确提供的香港交易日序列；raw_bars 必须包含 security_id、
    date、raw_close，且为已收盘日线。当日日线仅在19:00香港时间以后纳入，
    此边界是本接口的信息集约定，不是对交易所特殊交易日的认证。
    供股须有公开条款与当日除权参考价佐证；不代表认购或资金到账。
    覆盖度始终未认证，不产生全局训练批准。
    """
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError('截止时间必须包含时区')
    if not isinstance(quote_currency, str) or not quote_currency.strip():
        raise ValueError('必须明确报价币种')
    if not {'security_id', 'date', 'raw_close'}.issubset(raw_bars.columns):
        raise ValueError('原始日线必须包含证券、日期和原始收盘价')
    if raw_bars.security_id.isna().any() or raw_bars.security_id.nunique() != 1:
        raise ValueError('每次只能处理一只明确的证券')
    history = raw_bars.copy(deep=True)
    dates = pd.DatetimeIndex(pd.to_datetime(history.date))
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar))
    for values in (dates, sessions):
        if values.tz is not None or values.hasnans or values.has_duplicates or not values.equals(values.normalize()):
            raise ValueError('交易日须为不重复、无时区的明确日期')
    if sessions.empty or not sessions.is_monotonic_increasing or not dates.isin(sessions).all():
        raise ValueError('必须提供有序交易日历且包含全部原始日线日期')
    history['date'] = dates
    local = cutoff.tz_convert('Asia/Hong_Kong')
    today = local.tz_localize(None).normalize()
    completed = today if local.hour >= 19 else today - pd.Timedelta(days=1)
    history = history.loc[history.date.le(completed)].sort_values('date').reset_index(drop=True)
    security = raw_bars.security_id.iloc[0]
    selected = distributions_asof(event_versions, cutoff)
    selected = selected.loc[selected.security_id.eq(security)].copy().reset_index(drop=True)
    multipliers = [Decimal(1) for _ in range(len(history))]
    unresolved = pd.Series(False, index=history.index)
    gaps, factors, grouped, rights = [], [], {}, {}

    def gap(row, reason, ex):
        gaps.append(dict(security_id=security, event_id=row['event_id'], ex_date=ex,
                         reason=reason, cash_currency=row.get('cash_currency'),
                         quote_currency=quote_currency, published_at=row['published_at']))
        unresolved.loc[:] |= True if pd.isna(ex) else history.date.lt(ex)

    for row in selected.to_dict('records'):
        ex = pd.to_datetime(row.get('ex_date'), errors='coerce')
        if pd.isna(ex) or ex.tzinfo is not None or ex != ex.normalize():
            gap(row, 'unknown_ex_date', pd.NaT)
            continue
        if ex > today:
            continue
        if row.get('event_type') == 'rights_issue':
            try:
                numbers=[Decimal(row[key]) for key in ('rights_new_units','rights_old_units','rights_subscription_price')]
                legal=all(value.is_finite() and value>0 for value in numbers)
            except (InvalidOperation,KeyError,TypeError):
                legal=False
            if row.get('rights_terms_public') is not True or not legal:
                gap(row,'missing_public_rights_terms',ex);continue
            if row.get('rights_currency')!=quote_currency or 'cancel' in str(row.get('announcement_status','')).lower():
                gap(row,'unresolved_rights_currency_or_cancellation',ex);continue
            rights.setdefault(ex,[]).append((row,numbers))
            continue
        if row.get('event_type') not in ('cash_distribution', 'cash_distribution_with_scrip_option'):
            gap(row, 'unsupported_event_type', ex)
            continue
        if (row.get('conditional_cash_amount') != False or
                row.get('cash_amount_status') != 'declared_payment_currency_amount' or
                'cancel' in str(row.get('announcement_status', '')).lower()):
            gap(row, 'cash_amount_not_final', ex)
            continue
        if row.get('cash_currency') != quote_currency:
            gap(row, 'cash_currency_mismatch', ex)
            continue
        try:
            value = row.get('cash_per_unit_decimal')
            amount = Decimal(value) if isinstance(value, str) else Decimal('NaN')
            valid_amount = amount.is_finite() and amount >= 0
        except InvalidOperation:
            valid_amount = False
        if not valid_amount:
            gap(row, 'invalid_cash_amount', ex)
            continue
        grouped.setdefault(ex, []).append((row, amount))

    for ex in sorted(set(grouped)|set(rights)):
        entries=grouped.get(ex,[])
        right_entries=rights.get(ex,[])
        event_rows=[row for row,_ in entries]+[row for row,_ in right_entries]
        if ex not in sessions or sessions.get_loc(ex) == 0:
            for row in event_rows:
                gap(row, 'missing_prior_calendar_session', ex)
            continue
        prior = sessions[sessions.get_loc(ex) - 1]
        reference = history.loc[history.date.eq(prior), 'raw_close']
        price = (Decimal(str(reference.iloc[0]))
                 if len(reference) and pd.notna(reference.iloc[0]) else Decimal('NaN'))
        if not price.is_finite() or price <= 0:
            for row in event_rows:
                gap(row, 'missing_prior_session_close', ex)
            continue
        cash = sum((amount for _, amount in entries), Decimal(0))
        theoretical=price-cash
        if right_entries:
            if len(right_entries)!=1:
                for row in event_rows:gap(row,'multiple_rights_same_date',ex)
                continue
            new,old,subscription=right_entries[0][1]
            ratio=new/old
            theoretical=(theoretical+ratio*subscription)/(1+ratio)
            reference_quote=history.loc[history.date.eq(ex),'prevClose'] if 'prevClose' in history else pd.Series(dtype=float)
            observed=Decimal(str(reference_quote.iloc[0])) if len(reference_quote) and pd.notna(reference_quote.iloc[0]) else Decimal('NaN')
            # Half-cent tolerance is a source-price rounding check, not an
            # assumption that the underwriting or investor subscription completed.
            if not observed.is_finite() or observed<=0 or abs(observed-theoretical)>Decimal('0.005'):
                for row in event_rows:gap(row,'missing_or_inconsistent_ex_rights_reference',ex)
                continue
        factor = theoretical / price
        if factor <= 0:
            for row in event_rows:
                gap(row, 'nonpositive_theoretical_factor', ex)
            continue
        # A partly unresolved same-day bundle must never advertise a valid factor.
        if any(item['ex_date'] == ex for item in gaps):
            continue
        factors.append(dict(ex_date=ex, prior_session=prior, reference_raw_close=str(price),
                            cash_total_decimal=str(cash), cash_currency=quote_currency,
                            factor_decimal=str(factor), event_ids=tuple(row['event_id'] for row in event_rows),
                            adjustment_kind='rights_issue' if right_entries else 'cash_distribution',
                            cash_settlement_or_subscription_assumed=False))
        for index in history.index[history.date.lt(ex)]:
            multipliers[index] *= factor

    history['adjustment_factor'] = [float('nan') if unresolved.iloc[i] else float(f)
                                    for i, f in enumerate(multipliers)]
    history['adjusted_close'] = pd.to_numeric(history.raw_close, errors='raise') * history.adjustment_factor
    history['adjustment_status'] = ['unresolved' if x else 'resolved' for x in unresolved]
    history.attrs.update(as_of=cutoff.isoformat(), quote_currency=quote_currency,
                         coverage_complete=False, approved_for_training=False,
                         return_basis='corporate_action_adjusted_price_return_not_cash_wealth')
    return AdjustmentResult(history, selected, pd.DataFrame(factors),
                            pd.DataFrame(gaps), 'unresolved' if gaps else 'resolved')
