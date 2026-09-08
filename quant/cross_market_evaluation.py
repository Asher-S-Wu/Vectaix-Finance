from __future__ import annotations

import numpy as np
import pandas as pd

from .general_training import coverage_metrics, probability_metrics


BLOCK = 12
REPEATS = 2000


def intervals(frame: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    """Resample original union-calendar blocks, retaining every stock on each date."""
    if frame.date.nunique() < BLOCK or len(dates) < BLOCK:
        return {"status": "unavailable", "reason": "fewer_than_12_scorable_dates"}
    y, p, b = [frame[k].to_numpy(float) for k in ("target", "probability", "base_probability")]
    daily = pd.DataFrame({
        "date": frame.date.to_numpy(), "accuracy": ((p >= .5) == y).astype(float),
        "brier": (p-y)**2, "brierMinusFrequency": (p-y)**2-(b-y)**2,
        "positive": y, "negative": 1-y,
        "tp": ((p >= .5) & (y == 1)).astype(float),
        "tn": ((p < .5) & (y == 0)).astype(float),
    }).groupby("date").mean().reindex(dates)
    if "local_probability" in frame:
        local = frame.local_probability.to_numpy(float)
        pair = pd.DataFrame({"date": frame.date.to_numpy(),
                             "brierMinusLocal": (p-y)**2-(local-y)**2,
                             "accuracyMinusLocal": ((p >= .5) == y).astype(float)-((local >= .5) == y)})
        daily = daily.join(pair.groupby("date").mean())
    rng = np.random.default_rng(42)
    starts = rng.integers(0, len(dates)-BLOCK+1, size=(REPEATS, int(np.ceil(len(dates)/BLOCK))))
    draws = (starts[:, :, None]+np.arange(BLOCK)).reshape(REPEATS, -1)[:, :len(dates)]
    observed = daily.accuracy.notna().to_numpy(float)
    # Zero is an absent contribution to the sum; no missing forecast or label is invented.
    sums = np.where(np.isfinite(daily.to_numpy()), daily.to_numpy(), 0.)[draws].sum(axis=1)
    count = observed[draws].sum(axis=1)
    values = sums/np.where(count > 0, count, np.nan)[:, None]
    result = {"status": "available", "blockPredictionDates": BLOCK, "draws": REPEATS, "seed": 42,
              "interpretation": "Conditional on frozen securities; overlapping market outcomes are not independent observations"}
    for i, key in enumerate(daily.columns):
        if key not in ("positive", "negative", "tp", "tn"):
            result[key] = np.quantile(values[:, i], [.025, .975]).tolist() if np.isfinite(values[:, i]).all() else None
    balanced = .5*(sums[:, 5]/np.where(sums[:, 3] > 0, sums[:, 3], np.nan)
                   +sums[:, 6]/np.where(sums[:, 4] > 0, sums[:, 4], np.nan))
    result["balancedAccuracy"] = np.quantile(balanced, [.025, .975]).tolist() if np.isfinite(balanced).all() else None
    return result


def score(frame: pd.DataFrame, dates: pd.DatetimeIndex, *, with_intervals=True) -> dict:
    result = probability_metrics(frame, dates, intervals=False)
    if with_intervals and len(frame):
        result["bootstrap95"] = intervals(frame, dates)
    return result


def compare(joint: pd.DataFrame, local: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    paired = joint.merge(local[["date", "symbol", "target", "probability"]], on=["date", "symbol"],
                         suffixes=("", "_local"), validate="one_to_one")
    if paired.empty:
        return {"status": "unavailable", "reason": "no_common_scorable_forecasts"}
    if not np.array_equal(paired.target.to_numpy(), paired.target_local.to_numpy()):
        raise ValueError("跨模型配对标签不同。")
    paired = paired.rename(columns={"probability_local": "local_probability"})
    y, p, b = [paired[k].to_numpy(float) for k in ("target", "probability", "local_probability")]
    means = pd.DataFrame({"date": paired.date.to_numpy(), "jointBrier": (p-y)**2, "localBrier": (b-y)**2,
                          "accuracyDifference": ((p >= .5) == y).astype(float)-((b >= .5) == y)}).groupby("date").mean().mean()
    return {"status": "available", "rows": len(paired), "dates": int(paired.date.nunique()),
            "jointOutsidePair": len(joint)-len(paired), "localOutsidePair": len(local)-len(paired),
            **{k: float(v) for k, v in means.items()},
            "brierSkillVersusLocal": float(1-means.jointBrier/means.localBrier) if means.localBrier > 0 else None,
            "bootstrap95": intervals(paired, dates)}


def evaluate(dataset: dict, predictions: pd.DataFrame) -> dict:
    reports = {}
    for market in ("A", "HK"):
        reports[market] = {}
        for cohort, split, start, end in (
            ("exploratory_training_2019_2023", "train", "2019-01-01", "2023-12-31"),
            ("exploratory_training_2024_plus", "train", "2024-01-01", "2026-08-31"),
            ("exploratory_heldout_2024_plus", "heldout", "2024-01-01", "2026-08-31"),
        ):
            planned = dataset["coverage"].loc[lambda x: x.market.eq(market) & x.split.eq(split) & x.date.between(start, end)]
            dates = dataset["prediction_dates"]
            dates = dates[(dates >= start) & (dates <= end)]
            selected = predictions.loc[lambda x: x.market.eq(market) & x.split.eq(split) & x.date.between(start, end)]
            result = {"market": market, "split": split, "historicalExplorationOnly": True, "models": {}}
            scored = {}
            for stream in ("JOINT", "A_ONLY", "HK_ONLY"):
                forecasts = selected.loc[selected.model.eq(stream)]
                valid = forecasts.loc[forecasts.probability.notna() & forecasts.label_status.eq("scorable")]
                scored[stream] = valid
                cov = coverage_metrics(planned, forecasts)
                eligible = planned.loc[planned.eligible_after_warmup & planned.market_open]
                eligible_issued = eligible[["date", "symbol"]].merge(forecasts.loc[forecasts.probability.notna(), ["date", "symbol"]],
                                                                     on=["date", "symbol"], validate="one_to_one")
                mature = forecasts.loc[forecasts.probability.notna() & forecasts.label_status.ne("pending_horizon")]
                cov.update(eligibleAfterWarmupRows=len(eligible), eligibleIssuedRows=len(eligible_issued),
                           eligibleIssuanceRate=len(eligible_issued)/len(eligible) if len(eligible) else None,
                           maturedScorableRate=len(valid)/len(mature) if len(mature) else None)
                per_stock = [{"symbol": symbol, "metrics": score(valid.loc[valid.symbol.eq(symbol)], dates, with_intervals=False)}
                             for symbol in sorted(planned.symbol.unique())]
                result["models"][stream] = {"metrics": score(valid, dates), "coverage": cov, "perStock": per_stock,
                                             "perYear": {str(year): score(valid.loc[valid.date.dt.year.eq(year)], dates[dates.year == year])
                                                         for year in sorted(planned.date.dt.year.unique())}}
            result["jointVersusLocal"] = compare(scored["JOINT"], scored[f"{market}_ONLY"], dates)
            reports[market][cohort] = result
    return reports
