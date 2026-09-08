from __future__ import annotations

import numpy as np
import pandas as pd


HORIZON = 21
PREDICTION_STRIDE = 5
SIGNAL_TIME = "17:10:00+08:00"
FEATURES = {
    "return_1": "P/P.shift(1)-1",
    "return_5": "P/P.shift(5)-1",
    "return_21": "P/P.shift(21)-1",
    "return_63": "P/P.shift(63)-1",
    "return_126": "P/P.shift(126)-1",
    "return_252": "P/P.shift(252)-1",
    "momentum_12_1": "P.shift(21)/P.shift(252)-1",
    "distance_ma_21": "P/mean(P,21)-1",
    "distance_ma_63": "P/mean(P,63)-1",
    "distance_high_252": "P/max(P,252)-1",
    "distance_low_252": "P/min(P,252)-1",
    "volatility_21": "std(return_1,21,ddof=1)*sqrt(252)",
    "volatility_63": "std(return_1,63,ddof=1)*sqrt(252)",
    "volatility_ratio_21_126": "std(return_1,21)/std(return_1,126)",
    "downside_volatility_63": "sqrt(mean(min(return_1,0)^2,63)*252)",
    "distance_high_63": "P/max(P,63)-1",
    "intraday_return": "P/O-1",
    "overnight_gap": "O/P.shift(1)-1",
    "overnight_gap_mean_5": "mean(overnight_gap,5)",
    "range_mean_21": "mean((H-L)/P.shift(1),21)",
    "wick_asymmetry_mean_21": "mean(((H-max(O,P))-(min(O,P)-L))/P.shift(1),21)",
    "log_turnover_mean_21": "log1p(mean(amount*1000,21))",
    "turnover_ratio_5_63": "mean(amount*1000,5)/mean(amount*1000,63)",
    "money_weighted_return_21": "sum(return_1*amount*1000,21)/sum(amount*1000,21)",
    "illiquidity_21": "mean(abs(return_1)/(amount*1000),21)*1e6",
    "excess_return_21": "return_21-market_return_21",
    "excess_return_63": "return_63-market_return_63",
    "beta_126": "cov(return_1,market_return_1,126,ddof=1)/var(market_return_1,126,ddof=1)",
    "market_return_21": "HS300.close/HS300.close.shift(21)-1",
    "market_volatility_21": "std(market_return_1,21,ddof=1)*sqrt(252)",
}
FEATURE_KEYS = tuple(FEATURES)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = denominator.gt(0) & np.isfinite(denominator) & np.isfinite(numerator)
    return numerator.div(denominator.where(valid)).where(valid)


def _bar_flags(frame: pd.DataFrame) -> dict[str, pd.Series]:
    ohlc = frame[["open", "high", "low", "close"]]
    return {
        "missing_price": ~np.isfinite(ohlc).all(axis=1),
        "nonpositive_price": ohlc.le(0).any(axis=1),
        "invalid_ohlc": (frame.high + 1e-8 < ohlc.max(axis=1)) | (frame.low - 1e-8 > ohlc.min(axis=1)),
        "invalid_volume": ~np.isfinite(frame.volume) | frame.volume.le(0),
        "invalid_turnover": ~np.isfinite(frame.turnover) | frame.turnover.le(0),
    }


def _forward_count(flag: pd.Series) -> pd.Series:
    # A value at t covers t+1 through t+22, including both label endpoints.
    return flag.astype(float).rolling(HORIZON + 1, min_periods=HORIZON + 1).sum().shift(-(HORIZON + 1))


def build_general_factors(dataset: dict) -> dict:
    """Build fixed factors and separately retain every planned observation outcome."""
    calendar = pd.DatetimeIndex(dataset["calendar"]).as_unit("ns")
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("官方交易日历必须非空、唯一并递增。")
    stocks = dataset["stocks"].sort_values("symbol").reset_index(drop=True)
    if (len(stocks) != 537 or len(dataset["holdout_symbols"]) != 105
            or set(dataset["training_symbols"]) & set(dataset["holdout_symbols"])
            or set(stocks.symbol) != set(dataset["training_symbols"]) | set(dataset["holdout_symbols"])):
        raise ValueError("本次冻结范围必须是 537 股、105 股永久留出，训练与留出互斥且覆盖全部股票。")
    prices = dataset["prices"]
    if prices.duplicated(["symbol", "date"]).any() or not prices.date.isin(calendar).all():
        raise ValueError("原始股价含重复证券日期或非官方交易日。")
    benchmark = dataset["benchmark"].set_index("date").reindex(calendar)
    benchmark_flags = _bar_flags(benchmark)
    benchmark_valid = ~pd.DataFrame(benchmark_flags).any(axis=1)
    market_price = benchmark.close.where(benchmark_valid)
    market_r1 = _ratio(market_price, market_price.shift(1)) - 1
    market_r21 = _ratio(market_price, market_price.shift(21)) - 1
    market_r63 = _ratio(market_price, market_price.shift(63)) - 1
    market_variance = market_r1.rolling(126, min_periods=126).var(ddof=1)
    market_volatility = market_r1.rolling(21, min_periods=21).std(ddof=1) * np.sqrt(252)
    grid = calendar[::PREDICTION_STRIDE]
    dates_to_keep = grid.union(calendar[-1:])
    date_series = pd.Series(calendar, index=calendar)
    label_start = date_series.shift(-1)
    label_end = date_series.shift(-(HORIZON + 1))
    factors, coverage, latest = [], [], []
    price_groups = {symbol: frame for symbol, frame in prices.groupby("symbol", sort=False)}
    suspensions = dataset["suspensions"]
    source_columns = ["date", "open", "high", "low", "close", "volume", "turnover", "adj_factor"]
    for stock in stocks.to_dict("records"):
        symbol = stock["symbol"]
        # Empty successful source tables remain empty observations on this calendar.
        observed = prices.loc[prices.symbol.eq(symbol), source_columns] if symbol not in price_groups else price_groups[symbol][source_columns]
        frame = observed.set_index("date").reindex(calendar)
        flags = _bar_flags(frame)
        flags["missing_observation"] = pd.Series(~calendar.isin(observed.date), index=calendar)
        flags["invalid_adjustment"] = ~np.isfinite(frame.adj_factor) | frame.adj_factor.le(0)
        suspension = suspensions.loc[suspensions.symbol.eq(symbol) & suspensions.suspend_type.eq("S")]
        full_days = suspension.loc[suspension.suspend_timing.isna() | suspension.suspend_timing.eq(""), "date"]
        flags["full_day_suspension"] = pd.Series(calendar.isin(full_days), index=calendar)
        valid = ~pd.DataFrame(flags).any(axis=1)
        adjusted = frame[["open", "high", "low", "close"]].mul(frame.adj_factor, axis=0).where(valid, axis=0)
        p, o, h, lo = [adjusted[key] for key in ("close", "open", "high", "low")]
        amount = frame.turnover.where(valid)
        returns = {k: _ratio(p, p.shift(k)) - 1 for k in (1, 5, 21, 63, 126, 252)}
        r1 = returns[1]
        std21 = r1.rolling(21, min_periods=21).std(ddof=1)
        gap = _ratio(o, p.shift(1)) - 1
        f = {f"return_{k}": values for k, values in returns.items()}
        f.update({
            "momentum_12_1": _ratio(p.shift(21), p.shift(252)) - 1,
            "distance_ma_21": _ratio(p, p.rolling(21, min_periods=21).mean()) - 1,
            "distance_ma_63": _ratio(p, p.rolling(63, min_periods=63).mean()) - 1,
            "distance_high_252": _ratio(p, p.rolling(252, min_periods=252).max()) - 1,
            "distance_low_252": _ratio(p, p.rolling(252, min_periods=252).min()) - 1,
            "volatility_21": std21 * np.sqrt(252),
            "volatility_63": r1.rolling(63, min_periods=63).std(ddof=1) * np.sqrt(252),
            "volatility_ratio_21_126": _ratio(std21, r1.rolling(126, min_periods=126).std(ddof=1)),
            "downside_volatility_63": np.sqrt(r1.clip(upper=0).pow(2).rolling(63, min_periods=63).mean() * 252),
            "distance_high_63": _ratio(p, p.rolling(63, min_periods=63).max()) - 1,
            "intraday_return": _ratio(p, o) - 1,
            "overnight_gap": gap,
            "overnight_gap_mean_5": gap.rolling(5, min_periods=5).mean(),
            "range_mean_21": _ratio(h - lo, p.shift(1)).rolling(21, min_periods=21).mean(),
            "wick_asymmetry_mean_21": _ratio((h - np.maximum(o, p)) - (np.minimum(o, p) - lo), p.shift(1)).rolling(21, min_periods=21).mean(),
            "log_turnover_mean_21": np.log1p(amount.rolling(21, min_periods=21).mean()),
            "turnover_ratio_5_63": _ratio(amount.rolling(5, min_periods=5).mean(), amount.rolling(63, min_periods=63).mean()),
            "money_weighted_return_21": _ratio((r1 * amount).rolling(21, min_periods=21).sum(), amount.rolling(21, min_periods=21).sum()),
            "illiquidity_21": _ratio(r1.abs(), amount).rolling(21, min_periods=21).mean() * 1e6,
            "excess_return_21": returns[21] - market_r21,
            "excess_return_63": returns[63] - market_r63,
            "beta_126": _ratio(r1.rolling(126, min_periods=126).cov(market_r1, ddof=1), market_variance),
            "market_return_21": market_r21,
            "market_volatility_21": market_volatility,
        })
        features = pd.DataFrame(f, index=calendar)[list(FEATURE_KEYS)]
        finite = pd.Series(np.isfinite(features.to_numpy()).all(axis=1), index=calendar)
        feature_reason = pd.Series("available", index=calendar, dtype="string")
        feature_reason.loc[~finite] = "incomplete_window_or_nonpositive_denominator"
        feature_reason.loc[np.arange(len(calendar)) < 252] = "initial_252_session_warmup"
        for name in reversed(tuple(flags)):
            feature_reason.loc[flags[name]] = name
        label_flags = {key: _forward_count(value) for key, value in flags.items()}
        label_flags["any_suspension"] = _forward_count(pd.Series(calendar.isin(suspension.date), index=calendar))
        after_delisting = pd.Series(False, index=calendar)
        if pd.notna(stock["delist_date"]):
            after_delisting = pd.Series(calendar >= stock["delist_date"], index=calendar)
        label_flags["at_or_after_delisting"] = _forward_count(after_delisting)
        label_valid = label_end.notna() & ~pd.DataFrame(label_flags).gt(0).any(axis=1)
        future_return = (_ratio(o.shift(-(HORIZON + 1)), o.shift(-1)) - 1).where(label_valid)
        label_valid &= np.isfinite(future_return)
        outcome = pd.Series("scorable", index=calendar, dtype="string")
        for name in reversed(tuple(label_flags)):
            outcome.loc[label_flags[name].gt(0)] = name
        outcome.loc[label_end.isna()] = "pending_horizon"
        if (outcome.isna().any() or label_valid.isna().any()
                or not np.array_equal(outcome.eq("scorable").to_numpy(dtype=bool), label_valid.to_numpy(dtype=bool))):
            raise ValueError(f"{symbol} 标签有效性与缺失原因不一致。")
        meta = pd.DataFrame({
            "date": calendar, "symbol": symbol, "split": stock["split"],
            "current_list_status": stock["list_status"],
            "signal_at": calendar.strftime("%Y-%m-%d") + "T" + SIGNAL_TIME,
            "feature_available": finite.to_numpy(), "feature_status": feature_reason.to_numpy(),
            "missing_feature_count": (~np.isfinite(features.to_numpy())).sum(axis=1),
            "label_start": label_start.to_numpy(), "label_end": label_end.to_numpy(),
            "label_status": outcome.to_numpy(), "forward_return": future_return.to_numpy(),
            "target": (future_return > 0).astype(float).where(label_valid).to_numpy(),
            "within_listing_lifecycle": (calendar >= stock["list_date"]) & ~after_delisting.to_numpy(),
        }, index=calendar)
        meta["lifecycle_status"] = "listed_window"
        meta.loc[calendar < stock["list_date"], "lifecycle_status"] = "not_yet_listed"
        meta.loc[after_delisting, "lifecycle_status"] = "already_delisted"
        for name, values in label_flags.items():
            meta[f"label_{name}_days"] = values.to_numpy()
        for name, values in flags.items():
            meta[f"signal_{name}"] = values.to_numpy()
        coverage.append(meta.loc[grid].reset_index(drop=True))
        latest.append(meta.loc[calendar[-1:]].reset_index(drop=True))
        included = finite & calendar.isin(dates_to_keep)
        factors.append(pd.concat([meta.loc[included, ["date", "symbol"]], features.loc[included]], axis=1).reset_index(drop=True))
    factor_frame = pd.concat(factors, ignore_index=True).sort_values(["date", "symbol"], ignore_index=True)
    coverage_frame = pd.concat(coverage, ignore_index=True).sort_values(["date", "symbol"], ignore_index=True)
    latest_frame = pd.concat(latest, ignore_index=True).sort_values("symbol", ignore_index=True)
    if factor_frame.empty or not np.isfinite(factor_frame[list(FEATURE_KEYS)].to_numpy()).all():
        raise ValueError("没有可用的有限因子行。")
    return {
        "factors": factor_frame, "coverage": coverage_frame, "latest_coverage": latest_frame,
        "calendar": calendar, "prediction_dates": grid, "stocks": stocks,
        "training_symbols": dataset["training_symbols"], "holdout_symbols": dataset["holdout_symbols"],
        "provenance": {
            **dataset["provenance"], "featureNames": FEATURES,
            "factorRows": len(factor_frame), "plannedRows": len(coverage_frame),
            "plannedDates": len(grid), "featureAvailableRows": int(coverage_frame.feature_available.sum()),
            "scorableLabelRows": int(coverage_frame.label_status.eq("scorable").sum()),
            "signalTime": "17:10 Asia/Shanghai, historical information-cutoff convention, not historical issuance proof",
            "featurePriceConvention": "Same-day raw OHLC times same-day adj_factor; no future endpoint normalization; pre_close unused.",
            "missingDataPolicy": "No observation, rate, factor, probability, label or return is imputed. Every planned date and frozen symbol remains in coverage.",
            "labelPolicy": "Next official session open to 22nd future session open, adjusted; every one of the 22 price dates must be valid and contain no S suspension record or delisting boundary.",
            "statisticalWindows": "Official SSE/SZSE calendar; complete windows; std/cov ddof=1; strictly positive finite denominators.",
            "lifecyclePolicy": "Current list status and listing/delisting dates are coverage metadata only, never model inputs or prediction filters.",
        },
    }
