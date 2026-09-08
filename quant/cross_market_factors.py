from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


HORIZON = 21
PREDICTION_STRIDE = 5
SIGNAL_TIME = "19:10:00+08:00"
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
    "log_turnover_ratio_21_252": "log(mean(A,21)/mean(A,252))",
    "turnover_ratio_5_63": "mean(A,5)/mean(A,63)",
    "money_weighted_return_21": "sum(return_1*A,21)/sum(A,21)",
    "illiquidity_ratio_21_252": "mean(abs(return_1)/A,21)/mean(abs(return_1)/A,252)",
    "excess_return_21": "return_21-local_market_return_21",
    "excess_return_63": "return_63-local_market_return_63",
    "beta_126": "cov(return_1,local_market_return_1,126,ddof=1)/var(local_market_return_1,126,ddof=1)",
    "market_return_21": "local_benchmark.close/local_benchmark.close.shift(21)-1",
    "market_volatility_21": "std(local_market_return_1,21,ddof=1)*sqrt(252)",
}
FEATURE_KEYS = tuple(FEATURES)
IDENTITY_KEYS = ["date", "market", "symbol", "issuer_id"]


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = denominator.gt(0) & np.isfinite(denominator) & np.isfinite(numerator)
    return numerator.div(denominator.where(valid)).where(valid)


def _calendar(values, market: str) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(values).as_unit("ns")
    if (calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing
            or calendar.tz is not None or calendar.hasnans or not calendar.equals(calendar.normalize())):
        raise ValueError(f"{market} 本地交易日历须为非空、无时区、唯一递增的自然日期。")
    return calendar


def _bar_flags(frame: pd.DataFrame, market: str) -> dict[str, pd.Series]:
    columns = ["open", "high", "low", "close"] if market == "A" else ["open", "close"]
    price = frame[columns]
    invalid_ohlc = pd.Series(False, index=frame.index)
    if market == "A":
        invalid_ohlc = ((frame.high + 1e-8 < price.max(axis=1))
                        | (frame.low - 1e-8 > price.min(axis=1)))
    return {
        "missing_price": ~np.isfinite(price).all(axis=1),
        "nonpositive_price": price.le(0).any(axis=1),
        "invalid_ohlc": invalid_ohlc,
        "invalid_volume": ~np.isfinite(frame.volume) | frame.volume.le(0),
        "invalid_turnover": ~np.isfinite(frame.turnover) | frame.turnover.le(0),
    }


def _forward_count(flag: pd.Series) -> pd.Series:
    # At t, the next 22 local price dates are t+1 through t+22.
    return flag.astype(float).rolling(HORIZON + 1, min_periods=HORIZON + 1).sum().shift(-(HORIZON + 1))


def _features(frame: pd.DataFrame, valid: pd.Series, benchmark: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    adjusted = frame[["open", "close"]].mul(frame.adj_factor, axis=0).where(valid, axis=0)
    p, o = adjusted.close, adjusted.open
    amount = frame.turnover.where(valid)
    returns = {k: _ratio(p, p.shift(k)) - 1 for k in (1, 5, 21, 63, 126, 252)}
    r1 = returns[1]
    std21 = r1.rolling(21, min_periods=21).std(ddof=1)
    market_r1 = _ratio(benchmark, benchmark.shift(1)) - 1
    market_r21 = _ratio(benchmark, benchmark.shift(21)) - 1
    market_r63 = _ratio(benchmark, benchmark.shift(63)) - 1
    market_var = market_r1.rolling(126, min_periods=126).var(ddof=1)
    gap = _ratio(o, p.shift(1)) - 1
    daily_illiquidity = _ratio(r1.abs(), amount)
    result = {f"return_{k}": values for k, values in returns.items()}
    result.update({
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
        "log_turnover_ratio_21_252": np.log(_ratio(amount.rolling(21, min_periods=21).mean(), amount.rolling(252, min_periods=252).mean())),
        "turnover_ratio_5_63": _ratio(amount.rolling(5, min_periods=5).mean(), amount.rolling(63, min_periods=63).mean()),
        "money_weighted_return_21": _ratio((r1 * amount).rolling(21, min_periods=21).sum(), amount.rolling(21, min_periods=21).sum()),
        "illiquidity_ratio_21_252": _ratio(daily_illiquidity.rolling(21, min_periods=21).mean(), daily_illiquidity.rolling(252, min_periods=252).mean()),
        "excess_return_21": returns[21] - market_r21,
        "excess_return_63": returns[63] - market_r63,
        "beta_126": _ratio(r1.rolling(126, min_periods=126).cov(market_r1, ddof=1), market_var),
        "market_return_21": market_r21,
        "market_volatility_21": market_r1.rolling(21, min_periods=21).std(ddof=1) * np.sqrt(252),
    })
    return pd.DataFrame(result, index=frame.index)[list(FEATURE_KEYS)], o


def _stock_meta(stock: dict, market: str, local: pd.DatetimeIndex, union: pd.DatetimeIndex) -> pd.DataFrame:
    after_delisting = np.zeros(len(union), dtype=bool)
    if pd.notna(stock["delist_date"]):
        after_delisting = union >= stock["delist_date"]
    before_listing = union < stock["list_date"]
    market_open = union.isin(local)
    # Count only calendar sessions actually supplied; pre-snapshot days are never invented.
    sessions = np.maximum(local.searchsorted(union, side="right") - local.searchsorted(stock["list_date"], side="left"), 0)
    lifecycle = np.full(len(union), "listed_window", dtype=object)
    lifecycle[before_listing] = "not_yet_listed"
    lifecycle[after_delisting] = "already_delisted"
    return pd.DataFrame({
        "date": union, "market": market, "symbol": stock["symbol"], "issuer_id": stock["issuer_id"],
        "split": stock["split"], "identity_status": stock["identity_status"],
        "current_list_status": stock["list_status"],
        "signal_at": union.strftime("%Y-%m-%d") + "T" + SIGNAL_TIME,
        "market_open": market_open,
        "within_listing_lifecycle": ~before_listing & ~after_delisting,
        "lifecycle_status": lifecycle,
        "local_snapshot_sessions_since_listing": sessions,
        "eligible_after_warmup": market_open & ~before_listing & ~after_delisting & (sessions >= 253),
    }, index=union)


def build_cross_market_factors(dataset: dict) -> dict:
    """Build multiplicatively adjusted research-price factors, retaining all planned coverage."""
    inputs = dataset["markets"]
    if set(inputs) != {"A", "HK"}:
        raise ValueError("跨市场输入必须明确包含 A 与 HK 两个市场。")
    calendars = {market: _calendar(inputs[market]["calendar"], market) for market in ("A", "HK")}
    union = calendars["A"].union(calendars["HK"]).sort_values()
    grid = union[::PREDICTION_STRIDE]
    keep = grid.union(union[-1:])
    stock_frames = []
    required_stock = ["symbol", "issuer_id", "split", "list_date", "delist_date", "list_status", "identity_status"]
    for market in ("A", "HK"):
        stocks = inputs[market]["stocks"].copy()
        if not set(required_stock).issubset(stocks.columns) or stocks.empty or stocks.symbol.duplicated().any():
            raise ValueError(f"{market} 股票池缺少身份字段、为空或证券重复。")
        if (not stocks.symbol.map(lambda value: isinstance(value, str) and bool(value)).all()
                or not stocks.identity_status.map(lambda value: isinstance(value, str) and bool(value)).all()):
            raise ValueError(f"{market} 股票代码及身份状态须为非空字符串。")
        stocks["market"] = market
        stocks["list_date"] = pd.to_datetime(stocks.list_date, errors="raise").dt.as_unit("ns")
        stocks["delist_date"] = pd.to_datetime(stocks.delist_date, errors="raise").dt.as_unit("ns")
        if (stocks.list_date.isna().any() or (stocks.delist_date.notna() & stocks.delist_date.le(stocks.list_date)).any()):
            raise ValueError(f"{market} 上市/退市日期缺失或顺序错误。")
        admitted = stocks.identity_status.eq("admitted")
        if (not stocks.loc[admitted, "issuer_id"].map(lambda value: isinstance(value, str) and bool(value)).all()
                or not stocks.loc[admitted, "split"].isin(["train", "heldout"]).all()):
            raise ValueError(f"{market} 已准入股票须有非空公司身份及固定训练/留出分组。")
        if stocks.loc[admitted].duplicated("issuer_id").any():
            raise ValueError(f"{market} 同一已准入公司有多个柜台，须先完成明确的主柜台选择。")
        stock_frames.append(stocks)
    stocks = pd.concat(stock_frames, ignore_index=True).sort_values(["market", "symbol"], ignore_index=True)
    admitted_stocks = stocks.loc[stocks.identity_status.eq("admitted")]
    if admitted_stocks.groupby("issuer_id").split.nunique().gt(1).any():
        raise ValueError("同一公司在 A/H 两地跨越训练和留出组。")

    factors, coverage, latest, market_meta = [], [], [], {}
    source_columns = ["date", "open", "high", "low", "close", "volume", "turnover", "adj_factor"]
    for market in ("A", "HK"):
        source = inputs[market]
        local = calendars[market]
        market_stocks = stocks.loc[stocks.market.eq(market)]
        prices, suspensions = source["prices"], source["suspensions"]
        if (prices.duplicated(["symbol", "date"]).any() or not prices.date.isin(local).all()
                or not prices.symbol.isin(market_stocks.symbol).all()):
            raise ValueError(f"{market} 行情含重复日期、非交易日或池外证券。")
        if not suspensions.empty:
            if (not {"symbol", "date", "suspend_type", "suspend_timing"}.issubset(suspensions.columns)
                    or not suspensions.symbol.isin(market_stocks.symbol).all()
                    or not suspensions.suspend_type.isin(["S", "R"]).all()):
                raise ValueError(f"{market} 停复牌表的证券或事件类型无效。")
        benchmark = source["benchmark"]
        if benchmark.date.duplicated().any() or not pd.DatetimeIndex(benchmark.date).as_unit("ns").sort_values().equals(local):
            raise ValueError(f"{market} 基准收盘须逐日本地交易日完整且唯一。")
        benchmark_close = benchmark.set_index("date").reindex(local).close
        if not np.isfinite(benchmark_close).all() or not benchmark_close.gt(0).all():
            raise ValueError(f"{market} 基准收盘包含非有限或非正值。")
        local_dates = pd.Series(local, index=local)
        label_start = local_dates.shift(-1)
        label_end = local_dates.shift(-(HORIZON + 1))
        grouped = {symbol: frame[source_columns].set_index("date") for symbol, frame in prices.groupby("symbol", sort=False)}
        for stock in market_stocks.to_dict("records"):
            symbol = stock["symbol"]
            if symbol in grouped:
                observed = grouped[symbol]
            else:
                observed = prices.loc[prices.symbol.eq(symbol), source_columns].set_index("date")
            frame = observed.reindex(local)
            flags = _bar_flags(frame, market)
            flags["missing_observation"] = pd.Series(~local.isin(observed.index), index=local)
            flags["invalid_adjustment"] = ~np.isfinite(frame.adj_factor) | frame.adj_factor.le(0)
            if suspensions.empty:
                suspension_dates = pd.DatetimeIndex([])
                full_days = pd.DatetimeIndex([])
            else:
                suspended = suspensions.loc[suspensions.symbol.eq(symbol) & suspensions.suspend_type.eq("S")]
                suspension_dates = pd.DatetimeIndex(suspended.date)
                full_days = pd.DatetimeIndex(suspended.loc[suspended.suspend_timing.isna() | suspended.suspend_timing.eq(""), "date"])
            flags["full_day_suspension"] = pd.Series(local.isin(full_days), index=local)
            valid = ~pd.DataFrame(flags).any(axis=1)
            feature_values, adjusted_open = _features(frame, valid, benchmark_close)
            finite = pd.Series(np.isfinite(feature_values.to_numpy()).all(axis=1), index=local)
            admitted = stock["identity_status"] == "admitted"
            feature_available = finite & admitted
            feature_status = pd.Series("available", index=local, dtype="string")
            feature_status.loc[~finite] = "incomplete_window_or_nonpositive_denominator"
            feature_status.loc[np.arange(len(local)) < 252] = "initial_252_local_session_warmup"
            for name in reversed(tuple(flags)):
                feature_status.loc[flags[name]] = name
            if not admitted:
                feature_status[:] = "identity_unverified"
            label_flags = {name: _forward_count(value) for name, value in flags.items()}
            label_flags["any_known_suspension"] = _forward_count(pd.Series(local.isin(suspension_dates), index=local))
            after_delisting = pd.Series(False, index=local)
            if pd.notna(stock["delist_date"]):
                after_delisting = pd.Series(local >= stock["delist_date"], index=local)
            label_flags["at_or_after_delisting"] = _forward_count(after_delisting)
            label_valid = label_end.notna() & ~pd.DataFrame(label_flags).gt(0).any(axis=1)
            forward_return = (_ratio(adjusted_open.shift(-(HORIZON + 1)), adjusted_open.shift(-1)) - 1).where(label_valid)
            label_valid &= np.isfinite(forward_return)
            label_status = pd.Series("scorable", index=local, dtype="string")
            for name in reversed(tuple(label_flags)):
                label_status.loc[label_flags[name].gt(0)] = name
            label_status.loc[label_end.isna()] = "pending_horizon"
            if (label_valid.isna().any() or label_status.isna().any()
                    or not np.array_equal(label_status.eq("scorable").to_numpy(dtype=bool), label_valid.to_numpy(dtype=bool))):
                raise ValueError(f"{market}/{symbol} 标签有效性与不可评分原因不一致。")
            meta = _stock_meta(stock, market, local, union)
            meta["feature_available"] = False
            meta["feature_status"] = "market_closed"
            meta["feature_values_finite"] = False
            meta["missing_feature_count"] = pd.Series(pd.NA, index=union, dtype="Int64")
            meta["label_start"] = pd.Series(pd.NaT, index=union, dtype="datetime64[ns]")
            meta["label_end"] = pd.Series(pd.NaT, index=union, dtype="datetime64[ns]")
            meta["label_status"] = "market_closed"
            meta["forward_return"] = np.nan
            meta["target"] = np.nan
            meta.loc[local, "feature_available"] = feature_available.to_numpy(dtype=bool)
            meta.loc[local, "feature_values_finite"] = finite.to_numpy(dtype=bool)
            meta.loc[local, "feature_status"] = feature_status.to_numpy()
            meta.loc[local, "missing_feature_count"] = (~np.isfinite(feature_values.to_numpy())).sum(axis=1)
            meta.loc[local, "label_start"] = label_start.to_numpy()
            meta.loc[local, "label_end"] = label_end.to_numpy()
            meta.loc[local, "label_status"] = label_status.to_numpy()
            meta.loc[local, "forward_return"] = forward_return.to_numpy()
            meta.loc[local, "target"] = (forward_return > 0).astype(float).where(label_valid).to_numpy()
            for name, values in label_flags.items():
                meta[f"label_{name}_days"] = values.reindex(union)
            for name, values in flags.items():
                meta[f"signal_{name}"] = values.astype("boolean").reindex(union)
            coverage.append(meta.loc[grid].reset_index(drop=True))
            latest.append(meta.loc[union[-1:]].reset_index(drop=True))
            included = local[feature_available.to_numpy(dtype=bool) & local.isin(keep)]
            factors.append(pd.concat([meta.loc[included, IDENTITY_KEYS], feature_values.loc[included]], axis=1).reset_index(drop=True))
        market_meta[market] = {
            "calendar": local, "stocks": market_stocks.reset_index(drop=True),
            "training_symbols": market_stocks.loc[market_stocks.split.eq("train") & market_stocks.identity_status.eq("admitted"), "symbol"].tolist(),
            "holdout_symbols": market_stocks.loc[market_stocks.split.eq("heldout") & market_stocks.identity_status.eq("admitted"), "symbol"].tolist(),
            "provenance": source["provenance"],
        }
    factor_frame = pd.concat(factors, ignore_index=True).sort_values(["date", "market", "symbol"], ignore_index=True)
    coverage_frame = pd.concat(coverage, ignore_index=True).sort_values(["date", "market", "symbol"], ignore_index=True)
    latest_frame = pd.concat(latest, ignore_index=True).sort_values(["market", "symbol"], ignore_index=True)
    if (factor_frame.empty or factor_frame.duplicated(["date", "market", "symbol"]).any()
            or not np.isfinite(factor_frame[list(FEATURE_KEYS)].to_numpy()).all()
            or len(coverage_frame) != len(grid) * len(stocks)
            or coverage_frame.duplicated(["date", "market", "symbol"]).any()):
        raise ValueError("跨市场因子或全股票计划网格存在空集、非有限值、重复或缺行。")
    summaries = {}
    for market, local in calendars.items():
        rows = coverage_frame.loc[coverage_frame.market.eq(market)]
        summaries[market] = {
            "localSessions": len(local), "stocks": int(rows.symbol.nunique()),
            "issuerAdmittedStocks": int((stocks.market.eq(market) & stocks.identity_status.eq("admitted")).sum()),
            "plannedRows": len(rows), "marketClosedRows": int((~rows.market_open).sum()),
            "eligibleAfterWarmupRows": int(rows.eligible_after_warmup.sum()),
            "featureAvailableRows": int(rows.feature_available.sum()),
            "scorableLabelRows": int(rows.label_status.eq("scorable").sum()),
            "identityUnverifiedRows": int((rows.market_open & rows.identity_status.ne("admitted")).sum()),
            "calendarSha256": hashlib.sha256("\n".join(local.strftime("%Y-%m-%d")).encode()).hexdigest(),
            "suspensionInterpretation": "Only explicit S records exclude otherwise observed bars/labels; an empty table does not establish absence of suspension. Missing or zero-volume price dates remain invalid.",
            "ohlcEnvelopeChecked": market == "A",
        }
    return {
        "factors": factor_frame, "coverage": coverage_frame, "latest_coverage": latest_frame,
        "calendar": union, "prediction_dates": grid, "stocks": stocks, "markets": market_meta,
        "training_symbols": admitted_stocks.loc[admitted_stocks.split.eq("train"), "symbol"].tolist(),
        "holdout_symbols": admitted_stocks.loc[admitted_stocks.split.eq("heldout"), "symbol"].tolist(),
        "provenance": {
            **dataset["provenance"], "featureNames": FEATURES, "factorRows": len(factor_frame),
            "plannedRows": len(coverage_frame), "plannedDates": len(grid), "marketFactorCoverage": summaries,
            "calendarSha256": hashlib.sha256("\n".join(union.strftime("%Y-%m-%d")).encode()).hexdigest(),
            "signalTime": "19:10 Asia/Shanghai historical information-cutoff convention; no proof of historical publication or real issuance",
            "featurePriceConvention": "P and O are raw close/open times same-day positive multiplicative adj_factor. HK uses separately verified supplier qfq factors. HFQ additive cash and reconstructed reinvestment indices are not used.",
            "labelPolicy": "Positive multiplicatively adjusted research-price change from next local-session open to 22nd future local-session open; all 22 observed price dates must be valid, with no known S event or delisting boundary. Not investor total return, received cash, reinvestment performance or executable profit.",
            "calendarPolicy": "All rolling windows and labels use the security's own market calendar. The union calendar supplies the fixed five-session prediction grid; closed-market rows remain explicitly closed and never generate factors.",
            "warmupPolicy": "At least 253 local sessions present in the supplied calendar on/after listing, while inside listing lifecycle and market open; no pre-snapshot calendar sessions inferred. Eligibility is coverage metadata, not a future-availability feature filter.",
            "identityPolicy": "Only identity_status=admitted can generate factors; a verified issuer cannot cross train/heldout groups or have multiple admitted counters in one market. Unverified identities retain all coverage rows.",
            "liquidityConvention": "A is original positive turnover in a consistently recorded local currency/unit. Both liquidity-level features use own-stock 21/252 ratios, without currency conversion or cross-sectional statistics.",
            "missingDataPolicy": "No prices, adjustment factors, suspended sessions, features, probabilities or outcomes are filled. Feature admission never depends on future label availability.",
            "statisticalWindows": "Complete native-calendar windows; std/cov ddof=1; finite strictly positive denominators; 28 fixed factors.",
            "lifecyclePolicy": "Listing/delisting/current status are metadata; future delisting does not remove current features, and crossing a delisting boundary makes the outcome unavailable.",
        },
    }
