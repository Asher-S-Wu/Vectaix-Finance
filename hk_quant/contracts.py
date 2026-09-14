"""预测和账户模块共同使用的数据定义。"""
from dataclasses import dataclass


FORECAST_RETURN_BASIS = "HKD source-adjusted price return"
CASH_DIVIDEND_ACCOUNTING = (
    "Source-adjusted price return is not a one-share-plus-cash total return. "
    "Cash dividend receipts are accounted for separately by replay under verified payment evidence."
)


@dataclass(frozen=True)
class RiskProfile:
    max_position_weight: float = .10
    max_equity_weight: float = .95
    target_annual_volatility: float = .15
    fee_per_side: float = .0025
    max_participation: float = .01
    horizon: int = 20

    def __post_init__(self):
        if not 0 < self.max_position_weight <= self.max_equity_weight <= 1:
            raise ValueError('仓位上限必须满足 0 < 单股上限 <= 总仓位上限 <= 1')
        if not 0 < self.target_annual_volatility or not 0 <= self.fee_per_side < 1:
            raise ValueError('波动目标或费用无效')
        if not 0 < self.max_participation <= 1 or self.horizon not in (1, 5, 20, 60):
            raise ValueError('成交参与率或预测期限无效')


# bars: date, security_id, raw_close, adj_close, vwap, volume, amount,
#       cum_adjfactor, open_adj, high_adj, low_adj, total_mv, free_mv,
#       total_share, free_share, turnover_ratio, quote_present.
# securities: security_id, exchange_code, name, isin, list_date, delist_date,
#             currency, asset_type, lot_size, identity_status.
# forecasts: date, security_id, horizon, score, probability_up, expected_return,
#            q10, q50, q90, status, reasons, model_version, data_as_of.
# holdings: security_id, quantity, average_cost (optional); cash is HKD.
# corporate_actions: security_id, announced_at, effective_date, payment_date,
#                    action_type, cash_per_share, share_multiplier, verified, payment_basis, source_url.
# Cash requires verified evidence, source_url and explicit payment_basis:
# actual_account_receipt or issuer_final_schedule_simulated. The simulated basis
# uses the explicit payment date in a final issuer announcement, never its
# publication date or a cheque despatch deadline. Live advice uses actual cash.
# Cash exits cancel shares on effective_date and recognize receivables until paid.
