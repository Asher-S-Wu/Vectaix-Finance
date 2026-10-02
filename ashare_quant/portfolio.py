"""CNY research advice with A-share execution gates and audited rounded quantities.

This produces indicative allocations, never orders. Market risk mathematics are
market independent; fees, declaration quantities, limits and T+1 are CN-specific.
"""
from dataclasses import dataclass
import re

import cvxpy as cp
import numpy as np
import pandas as pd

from .rules import buy_quantity, execution_status, lot_rule, sell_quantity, transaction_fee

SECURITY_PATTERN = re.compile(r'^\d{6}\.(SH|SZ|BJ)$')
HOLDING_COLUMNS = {'security_id', 'quantity', 'average_cost', 'acquired_date', 'sellable_quantity'}


@dataclass(frozen=True)
class RiskProfile:
    max_position_weight: float = .10
    max_equity_weight: float = .95
    target_annual_volatility: float = .15
    max_participation: float = .01
    horizon: int = 20
    commission_rate: float = .0003
    min_commission: float = 5.
    fee_per_side: float = 0.  # Optional additional friction/stress, not the CN tax schedule.

    def __post_init__(self):
        values = [v for k, v in vars(self).items() if k != 'horizon']
        if not all(np.isfinite(v) for v in values):
            raise ValueError('风险参数必须为有限数')
        if not 0 < self.max_position_weight <= self.max_equity_weight <= 1:
            raise ValueError('仓位上限必须满足 0 < 单股上限 <= 总仓位上限 <= 1')
        if self.target_annual_volatility <= 0 or not 0 < self.max_participation <= 1:
            raise ValueError('波动目标或成交参与率无效')
        if self.horizon not in (1, 5, 20, 60) or isinstance(self.horizon, bool):
            raise ValueError('预测期限仅支持 1、5、20、60')
        if not 0 <= self.commission_rate < 1 or self.min_commission < 0 or not 0 <= self.fee_per_side < 1:
            raise ValueError('手续费参数无效')


def validate_holdings(holdings):
    """Validate both API formats without inventing settlement-age information."""
    if not isinstance(holdings, pd.DataFrame) or not {'security_id', 'quantity'} <= set(holdings):
        raise ValueError('持仓必须含 security_id 和 quantity')
    if set(holdings) - HOLDING_COLUMNS:
        raise ValueError('持仓含未知字段')
    frame = holdings.copy()
    if frame.security_id.isna().any() or not frame.security_id.map(lambda s: isinstance(s, str) and bool(SECURITY_PATTERN.fullmatch(s))).all():
        raise ValueError('证券代码必须为六位数字加 .SH、.SZ 或 .BJ')
    if frame.security_id.duplicated().any():
        raise ValueError('持仓含重复证券')
    for field in ('quantity', 'average_cost', 'sellable_quantity'):
        if field not in frame:
            continue
        frame[field] = pd.to_numeric(frame[field], errors='raise')
        values = frame[field] if field == 'quantity' else frame[field].dropna()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'{field} 必须为非负有限数')
        if field != 'average_cost' and (values % 1 != 0).any():
            raise ValueError('持仓及可卖数量必须为整数股')
    if 'sellable_quantity' in frame and frame.sellable_quantity.gt(frame.quantity).any():
        raise ValueError('可卖数量不得超过持仓数量')
    if 'acquired_date' in frame:
        supplied = frame.acquired_date.notna() & frame.acquired_date.astype(str).ne('')
        parsed = pd.to_datetime(frame.loc[supplied, 'acquired_date'], errors='raise')
        if parsed.isna().any() or getattr(parsed.dt, 'tz', None) is not None:
            raise ValueError('acquired_date 必须为不带时区的有效日期')
        frame.loc[supplied, 'acquired_date'] = parsed.dt.strftime('%Y-%m-%d')
        frame.loc[~supplied, 'acquired_date'] = None
    return frame


def _sale_quantity(wanted, available, holding, sid):
    """Round desired sales upward, respecting minimum declaration and availability."""
    minimum, step = lot_rule(sid)
    wanted = max(0., float(wanted))
    rounded = max(minimum, int(np.ceil(wanted / step - 1e-7)) * step) if wanted else 0
    return sell_quantity(min(rounded, available), int(holding), sid)


def _risk_estimates(returns, asof, ids):
    dates = pd.DatetimeIndex(pd.to_datetime(returns.index, errors='coerce'))
    if dates.isna().any() or dates.duplicated().any() or returns.columns.duplicated().any():
        raise ValueError('风险历史日期或证券列无效/重复')
    history = returns.copy()
    history.index = dates
    history = history.loc[dates < asof].sort_index().tail(252)
    history = history.apply(pd.to_numeric, errors='coerce').where(lambda x: np.isfinite(x))
    market = history.median(axis=1)
    variance = float(market.var(ddof=1) * 252)
    if market.notna().sum() < 60 or not np.isfinite(variance) or variance <= 0:
        raise ValueError('市场风险因子缺少至少60日真实历史或方差不可估计')
    estimates, failures = {}, {}
    for sid in ids:
        paired = history[sid].notna() & market.notna() if sid in history else pd.Series(False, index=history.index)
        count = int(paired.sum())
        if count < 60:
            failures[sid] = '个股风险历史不足60对真实观察'
            continue
        x = np.column_stack([np.ones(count), market.loc[paired].to_numpy(float)])
        y = history.loc[paired, sid].to_numpy(float)
        coeff, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ coeff
        residual_variance = float(residual @ residual / (count - 2) * 252)
        if rank != 2 or not np.isfinite(coeff).all() or not np.isfinite(residual_variance):
            failures[sid] = '风险因子无法识别或估计无效'
            continue
        estimates[sid] = dict(beta=float(coeff[1]), residual_variance_annual=residual_variance,
                              intercept_daily=float(coeff[0]), observations=count)
    return estimates, failures, dict(
        risk_model='market_single_factor_ols_independent_residuals', risk_window_sessions=252,
        risk_window_start=history.index.min().isoformat(), risk_window_end=history.index.max().isoformat(),
        market_observations=int(market.notna().sum()), market_variance_annual=variance,
        risk_estimates=estimates, risk_data_policy='仅信号日前最近252交易日；至少60对真实观察；不填补缺失收益',
        risk_limitations='市场单因子和独立残差，未覆盖全部行业、共同风险及停牌流动性风险，不保证实际波动率')


def advise_portfolio(forecasts, holdings, cash, latest_bars, securities, returns,
                     profile=RiskProfile(), allocation='utility'):
    excluded, risk_summary = [], {}

    def error(reason, status='data_error'):
        return dict(status=status, reason=reason, currency='CNY', recommendations=pd.DataFrame(),
                    excluded=pd.DataFrame(excluded, columns=['security_id', 'reason']), summary=risk_summary)

    try:
        holdings = validate_holdings(holdings)
        cash = float(cash)
        if not np.isfinite(cash) or cash < 0:
            raise ValueError('现金必须为非负有限数')
        if allocation not in ('utility', 'equal_weight'):
            raise ValueError('未知配置方式')
        if latest_bars.empty or latest_bars.security_id.duplicated().any() or securities.security_id.duplicated().any():
            raise ValueError('市价或证券身份缺失/重复')
        asof = pd.to_datetime(latest_bars.date, errors='raise').max().normalize()
        if pd.isna(asof):
            raise ValueError('行情日期缺失')
    except (ValueError, TypeError, AttributeError) as exc:
        return error(str(exc))
    bars, sec = latest_bars.set_index('security_id'), securities.set_index('security_id')
    owned = holdings.set_index('security_id')
    quantities_by_id = owned.quantity.to_dict()
    for sid, quantity in quantities_by_id.items():
        if not quantity:
            continue
        if sid not in bars.index or not np.isfinite(bars.loc[sid, 'raw_close']) or bars.loc[sid, 'raw_close'] <= 0:
            return error('持仓缺少明确市价: ' + sid)
        if sid not in sec.index or sec.loc[sid, 'currency'] != 'CNY':
            return error('持仓缺少CNY计价证券身份: ' + sid)
        acquired = owned.loc[sid].get('acquired_date')
        if pd.notna(acquired) and pd.Timestamp(acquired) > asof:
            return error('持仓取得日期晚于快照日期: ' + sid)
        if pd.notna(acquired) and pd.Timestamp(acquired) == asof and float(owned.loc[sid].get('sellable_quantity', 0) or 0) > 0:
            return error('当日买入持仓不可声明为可卖数量: ' + sid)
    equity = cash + sum(q * float(bars.loc[sid, 'raw_close']) for sid, q in quantities_by_id.items() if q)
    if equity <= 0:
        return error('账户资产必须大于零')
    selected = forecasts.loc[(forecasts.horizon == profile.horizon) &
                             (pd.to_datetime(forecasts.date).dt.normalize() == asof)].copy()
    if selected.security_id.duplicated().any():
        return error('同一证券有重复预测')
    selected = selected.set_index('security_id')
    valid = selected.loc[selected.status.eq('ok') & np.isfinite(selected.expected_return)]
    permissions = {}

    def permission(sid):
        if sid in permissions:
            return permissions[sid]
        base_reason = None
        if sid not in sec.index or sec.loc[sid].get('identity_status') != 'verified':
            base_reason = 'identity_unverified'
        elif not SECURITY_PATTERN.fullmatch(str(sid)) or sec.loc[sid].get('currency') != 'CNY' or sec.loc[sid].get('asset_type', 'equity') != 'equity':
            base_reason = 'not_cny_ashare_equity'
        elif not np.isfinite(sec.loc[sid].get('lot_size', np.nan)) or sec.loc[sid, 'lot_size'] <= 0:
            base_reason = 'missing_lot_size'
        elif sid not in bars.index or pd.Timestamp(bars.loc[sid, 'date']).normalize() != asof:
            base_reason = 'missing_current_quote'
        elif not np.isfinite(bars.loc[sid].get('adv20_amount', np.nan)) or bars.loc[sid, 'adv20_amount'] <= 0:
            base_reason = 'missing_liquidity_history'
        elif sid not in valid.index:
            base_reason = 'forecast_unavailable'
        if base_reason:
            result = (False, base_reason, False, base_reason, 0.)
        else:
            bar = bars.loc[sid].to_dict()
            # Advice is a close-price estimate; execution must recheck the actual order session.
            bar['execution_price'] = bar['raw_close']
            can_buy, buy_reason = execution_status(bar, 'buy')
            acquired = owned.loc[sid].get('acquired_date') if sid in owned.index else None
            if pd.isna(acquired):
                acquired = None
            supplied = owned.loc[sid].get('sellable_quantity') if sid in owned.index else None
            if pd.notna(supplied):
                sellable = float(supplied)
            elif acquired is not None and pd.Timestamp(acquired) < asof:
                sellable = float(quantities_by_id.get(sid, 0))
            else:
                sellable = 0.
            can_sell, sell_reason = execution_status(bar, 'sell', acquired_date=acquired)
            if can_sell and quantities_by_id.get(sid, 0) and sellable == 0:
                can_sell, sell_reason = False, 'unknown_acquisition_date' if acquired is None else 't_plus_one'
            result = (can_buy, buy_reason, can_sell, sell_reason, sellable)
        permissions[sid] = result
        return result

    candidates = []
    for sid in valid.sort_values('expected_return', ascending=False).index:
        if permission(sid)[0]:
            candidates.append(sid)
        else:
            excluded.append(dict(security_id=sid, reason=permission(sid)[1]))
    ids = list(dict.fromkeys(candidates + [sid for sid, q in quantities_by_id.items() if q]))
    if not ids:
        return error('没有具备A股交易证据、有效预测和CNY市价的候选证券')
    try:
        estimates, failures, risk_summary = _risk_estimates(returns, asof, ids)
    except (ValueError, TypeError) as exc:
        return error(str(exc))
    for sid, reason in failures.items():
        excluded.append(dict(security_id=sid, reason=reason))
        if quantities_by_id.get(sid, 0):
            return error('实际持仓风险无法估计: ' + sid + '；' + reason)
    ids = [sid for sid in ids if sid in estimates]
    if not ids:
        return error('所有候选均缺少可估计的真实风险历史')
    prices = np.array([float(bars.loc[sid, 'raw_close']) for sid in ids])
    quantities = np.array([quantities_by_id.get(sid, 0) for sid in ids], dtype=float)
    beta = np.array([estimates[sid]['beta'] for sid in ids])
    residual = np.array([estimates[sid]['residual_variance_annual'] for sid in ids])
    market_variance = risk_summary['market_variance_annual']
    mu = np.array([float(valid.loc[sid, 'expected_return']) if sid in valid.index else 0 for sid in ids])
    minimum, maximum = quantities.copy(), quantities.copy()
    capacities = []
    for i, sid in enumerate(ids):
        can_buy, _, can_sell, _, sellable = permission(sid)
        amount = float(bars.loc[sid].get('amount', bars.loc[sid].get('adv20_amount', 0)))
        adv = float(bars.loc[sid].get('adv20_amount', 0))
        capacity = profile.max_participation * min(adv, amount) if np.isfinite([adv, amount]).all() else 0.
        capacities.append(max(0., capacity))
        if can_buy:
            maximum[i] += buy_quantity(capacity, prices[i], sid)
        if can_sell:
            limit = min(sellable, np.floor(capacity / prices[i] + 1e-9))
            limit = sell_quantity(limit, int(quantities[i]), sid)
            minimum[i] -= max(0, limit)
    current = quantities * prices / equity
    lower, upper = minimum * prices / equity, maximum * prices / equity
    def bounded_rate(sid, price, side, quantity):
        # Every real declaration is at least the venue minimum, except an entire
        # smaller inherited balance. Charging its fee/notional ratio is a safe
        # continuous upper bound without reserving fees for unplaced orders.
        minimum_order, _ = lot_rule(sid)
        if side == 'sell' and quantity > 0:
            minimum_order = min(minimum_order, int(quantity))
        notional = minimum_order * price
        return (transaction_fee(notional, side, asof, sid, commission_rate=profile.commission_rate,
                                min_commission=profile.min_commission) + .01) / notional + profile.fee_per_side

    rate_buy = np.array([bounded_rate(sid, prices[i], 'buy', quantities[i]) for i,sid in enumerate(ids)])
    rate_sell = np.array([bounded_rate(sid, prices[i], 'sell', quantities[i]) for i,sid in enumerate(ids)])
    w = cp.Variable(len(ids))
    buys, sales = cp.Variable(len(ids), nonneg=True), cp.Variable(len(ids), nonneg=True)
    fee_bound = rate_buy @ buys + rate_sell @ sales
    # A scalar NAV avoids repeating the full fee vector in every position cap.
    nav = cp.Variable()
    risk_norm = cp.norm(cp.hstack([cp.multiply(np.sqrt(residual), w), np.sqrt(market_variance) * (beta @ w)]), 2)
    # Avoid CVXPY Sum shape inference over uninitialized NumPy memory.
    risk_variance = cp.sum_squares(cp.multiply(np.sqrt(residual), w)) + market_variance * cp.square(beta @ w)
    total_weight = np.ones(len(ids)) @ w
    constraints = [nav + fee_bound == 1, w == current + buys - sales, buys <= upper-current, sales <= current-lower,
                   w >= lower, w <= upper, total_weight <= nav, nav >= .01]
    # Unavoidable frozen exposure is retained, while adjustable exposure follows the original caps.
    worst_fee = float(rate_sell.max()*current.sum() + rate_buy.max())
    conservative_nav = max(.01, 1 - worst_fee)
    frozen_overweight = lower > profile.max_position_weight * conservative_nav
    if frozen_overweight.any():
        constraints.append(w[frozen_overweight] == lower[frozen_overweight])
    if (~frozen_overweight).any():
        constraints.append(w[~frozen_overweight] <= profile.max_position_weight * nav)
    if lower.sum() > profile.max_equity_weight * conservative_nav:
        constraints.append(total_weight == lower.sum())
    else:
        constraints.append(total_weight <= profile.max_equity_weight * nav)
    try:
        feasibility = cp.Problem(cp.Minimize(risk_norm-profile.target_annual_volatility*nav), constraints)
        feasibility.solve(solver='CLARABEL')
        if feasibility.status != 'optimal':
            return error('现金、成交容量和冻结持仓无法满足连续约束: ' + str(feasibility.status), 'infeasible')
        shortfall = max(0., float(feasibility.value))
        reachable = shortfall <= 1e-7
        optimum = w.value.copy()
        if reachable:
            if allocation == 'utility':
                objective = cp.Maximize(mu@w - fee_bound - risk_variance*profile.horizon/252)
            else:
                equal = np.full(len(ids), min(profile.max_position_weight, profile.max_equity_weight/len(ids)))
                objective = cp.Minimize(cp.sum_squares(w-equal) + fee_bound * 1e-5)
            problem = cp.Problem(objective, [*constraints, risk_norm <= profile.target_annual_volatility*nav])
            problem.solve(solver='CLARABEL')
            if problem.status != 'optimal':
                return error('账户优化未完成: ' + str(problem.status), 'solver_error')
            optimum = w.value.copy()
    except cp.error.SolverError as exc:
        return error(str(exc), 'solver_error')
    target = quantities.copy()
    for i, sid in enumerate(ids):
        delta = max(0., float(optimum[i])) * equity / prices[i] - quantities[i]
        if delta > 1e-5:
            target[i] += buy_quantity(min(delta * prices[i], capacities[i]), prices[i], sid)
        elif delta < -1e-5:
            target[i] -= _sale_quantity(-delta, quantities[i]-minimum[i], quantities[i], sid)

    def actual_fees():
        return np.array([transaction_fee(abs(target[i]-quantities[i])*prices[i], 'buy' if target[i]>quantities[i] else 'sell',
                        asof, sid, commission_rate=profile.commission_rate, min_commission=profile.min_commission)
                        + abs(target[i]-quantities[i])*prices[i]*profile.fee_per_side if target[i]!=quantities[i] else 0.
                        for i, sid in enumerate(ids)])

    fees = actual_fees()
    post_fee_equity = equity - fees.sum()
    target_cash = equity - target@prices - fees.sum()
    if post_fee_equity <= 0:
        return error('费用超过账户资产', 'rounding_infeasible')
    weights = target*prices/post_fee_equity
    volatility = float(np.sqrt(residual@(weights**2) + market_variance*(beta@weights)**2))
    violations = []
    for i, sid in enumerate(ids):
        if weights[i] > profile.max_position_weight + 1e-7:
            violations.append(dict(constraint='single_position', security_id=sid, actual=float(weights[i]),
                limit=profile.max_position_weight, unavoidable=bool(target[i] <= minimum[i]+1e-7)))
    if weights.sum() > profile.max_equity_weight + 1e-7:
        violations.append(dict(constraint='total_equity', actual=float(weights.sum()), limit=profile.max_equity_weight,
                               unavoidable=bool(np.all(target <= minimum+1e-7))))
    if volatility > profile.target_annual_volatility + 1e-7:
        violations.append(dict(constraint='annual_volatility', actual=volatility,
                               limit=profile.target_annual_volatility, unavoidable=not reachable))
    if target_cash < -1e-6:
        return error('整股后的实际手续费造成现金不足', 'rounding_infeasible')
    rows = []
    for i, sid in enumerate(ids):
        delta = target[i]-quantities[i]
        can_buy, buy_reason, can_sell, sell_reason, sellable = permission(sid)
        if (delta > 0 and not can_buy) or (delta < 0 and (not can_sell or -delta > sellable+1e-7)) or abs(delta)*prices[i] > capacities[i]+1e-6:
            return error('整股订单不符合交易许可、T+1或参与率', 'rounding_infeasible')
        action = '买入' if delta > 0 and quantities[i] == 0 else '加仓' if delta > 0 else '卖出' if delta < 0 and target[i] == 0 else '减仓' if delta < 0 else '持有'
        reason = '预期收益、A股成本和账户风险联合优化'
        if not can_sell and quantities[i] > 0:
            reason += '；卖出受限: ' + str(sell_reason)
        if not can_buy:
            reason += '；买入受限: ' + str(buy_reason)
        rows.append(dict(security_id=sid, action=action, current_quantity=int(quantities[i]),
            target_quantity=int(target[i]), order_quantity=int(abs(delta)), target_weight=float(weights[i]),
            reference_price=float(prices[i]), reference_price_cny=float(prices[i]), currency='CNY',
            target_value_cny=float(target[i]*prices[i]), estimated_fee=float(fees[i]),
            buy_allowed=bool(can_buy), sell_allowed=bool(can_sell), buy_block_reason=None if can_buy else buy_reason,
            sell_block_reason=None if can_sell else sell_reason, reason=reason))
    summary = dict(allocation=allocation, currency='CNY', equity=float(equity), post_fee_equity=float(post_fee_equity),
        current_cash=cash, target_cash=float(target_cash), estimated_fees=float(fees.sum()), annual_volatility=volatility,
        target_equity_weight=float(weights.sum()), coverage_count=len(selected), candidate_count=len(ids),
        constraints_satisfied=not violations, constraint_violations=violations,
        controllable_constraints_satisfied=all(v['unavoidable'] for v in violations),
        volatility_target_reachable=bool(reachable), minimum_volatility_constraint_shortfall=shortfall,
        minimum_retained_quantities={sid:float(minimum[i]) for i,sid in enumerate(ids)},
        as_of=asof.isoformat(), solver='CLARABEL', **risk_summary,
        execution_basis='收盘参考价格的研究建议；执行前须重新核对账户可卖数量、实际盘口和当日涨跌停价',
        settlement_policy='A股T+1；未提供取得日期或明确可卖数量的持仓不建议卖出',
        fee_policy='CNY佣金及最低佣金、按交易日卖方印花税、过户费；另加显式费用压力参数',
        optimization_fee_policy='连续优化按最小有效申报金额对应的费率上界约束；最终费用按实际整数股订单逐笔核算',
        sector_constraints='未取得可靠行业分类，未设行业约束')
    return dict(status='constraints_unmet' if violations else 'ok', reason='原风险目标存在未满足项' if violations else None,
                currency='CNY', recommendations=pd.DataFrame(rows), excluded=pd.DataFrame(excluded,columns=['security_id','reason']),
                summary=summary)
