from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .factors import FACTOR_KEYS, LABEL_HORIZON
from .trend_factors import TREND_FACTOR_KEYS


CANDIDATES = ('core_logistic', 'expanded_logistic', 'expanded_trees', 'expanded_ensemble')
CANDIDATE_NAMES = {
    'core_logistic': '原有八因子概率模型',
    'expanded_logistic': '扩展因子线性模型',
    'expanded_trees': '扩展因子浅层树模型',
    'expanded_ensemble': '扩展因子双模型平均',
}
MIN_TRAINING = 504
TRAINING_YEARS = 5
MIN_CALIBRATION = 18
MAX_CALIBRATION = 36
INTERVAL_COVERAGE = 0.8


def training_rows(labels: pd.DataFrame, signal: pd.Timestamp) -> pd.DataFrame:
    return labels.loc[(labels.date >= signal - pd.DateOffset(years=TRAINING_YEARS)) & (labels.label_end < signal)].copy()


def _logits(probabilities) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all() or ((values <= 0) | (values >= 1)).any():
        raise ValueError('模型原始概率必须严格位于零和一之间。')
    return np.log(values / (1 - values)).reshape(-1, 1)


def _calibrate(history: pd.DataFrame, current: float):
    labels = (history.target > 0).astype(int)
    if labels.value_counts().reindex([0, 1], fill_value=0).min() < 3:
        raise ValueError('概率校准至少需要三次上涨和三次未上涨的成熟记录。')
    calibrator = LogisticRegression(C=0.5, solver='lbfgs', max_iter=1000)
    calibrator.fit(_logits(history.raw_probability), labels)
    probability = float(calibrator.predict_proba(_logits([current]))[0, 1])
    return probability, calibrator


def _fit_raw(train: pd.DataFrame, current: pd.DataFrame, expanded_keys: tuple[str, ...]):
    target = (train.target > 0).astype(int)
    if target.nunique() != 2:
        raise ValueError('训练窗口未同时包含上涨和未上涨，停止训练。')
    # Twenty-one overlapping daily labels must not carry twenty-one times
    # the regularization weight of one return interval.
    weights = np.full(len(train), 1 / LABEL_HORIZON)
    core = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    linear = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    trees = HistGradientBoostingClassifier(
        max_iter=80, learning_rate=0.05, max_leaf_nodes=7, max_depth=2,
        min_samples_leaf=63, l2_regularization=1.0, early_stopping=False, random_state=42,
    )
    core.fit(train[list(FACTOR_KEYS)], target, logisticregression__sample_weight=weights)
    linear.fit(train[list(expanded_keys)], target, logisticregression__sample_weight=weights)
    trees.fit(train[list(expanded_keys)], target, sample_weight=weights)
    raw = {
        'core_logistic': float(core.predict_proba(current[list(FACTOR_KEYS)])[0, 1]),
        'expanded_logistic': float(linear.predict_proba(current[list(expanded_keys)])[0, 1]),
        'expanded_trees': float(trees.predict_proba(current[list(expanded_keys)])[0, 1]),
    }
    raw['expanded_ensemble'] = (raw['expanded_logistic'] + raw['expanded_trees']) / 2
    returns = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    returns.fit(train[list(expanded_keys)], train.target, ridge__sample_weight=weights)
    expected = float(returns.predict(current[list(expanded_keys)])[0])
    models = {'core_logistic': core, 'expanded_logistic': linear, 'expanded_trees': trees, 'return_model': returns}
    return raw, expected, models


def walk_forward(
    factors: pd.DataFrame, labels: pd.DataFrame, calendar: pd.DatetimeIndex,
    old_alpha: float, expanded_keys: tuple[str, ...] = TREND_FACTOR_KEYS,
    progress=None,
):
    by_date = factors.set_index('date', drop=False)
    label_dates = labels.set_index('date')
    grid = [date for date in calendar[::LABEL_HORIZON]
            if date in by_date.index and len(training_rows(labels, date)) >= MIN_TRAINING]
    latest_date = calendar[-1]
    if latest_date not in by_date.index:
        raise ValueError('最新正式交易日没有完整因子，不能输出过期预测。')
    dates = sorted(set(grid + [latest_date]))
    records = {candidate: [] for candidate in CANDIDATES}
    return_history = []
    latest_models = None
    for index, signal in enumerate(dates):
        current = by_date.loc[[signal]]
        train = training_rows(labels, signal)
        if len(train) < MIN_TRAINING:
            raise ValueError('最新信号成熟训练数据不足。')
        raw, expected, models = _fit_raw(train, current, expanded_keys)
        old_train = labels.loc[(labels.date >= signal - pd.DateOffset(years=3)) & (labels.label_end < signal)]
        old = make_pipeline(StandardScaler(), Ridge(alpha=old_alpha))
        old.fit(old_train[list(FACTOR_KEYS)], old_train.target)
        old_return = float(old.predict(current[list(FACTOR_KEYS)])[0])
        # An unconditional comparator is estimated only from disjoint mature labels.
        independent_train = train.loc[train.date.isin(calendar[::LABEL_HORIZON])]
        base_probability = float((int((independent_train.target > 0).sum()) + 1) / (len(independent_train) + 2))
        scale = float(current.annualVolatility.iloc[0] * math.sqrt(LABEL_HORIZON / 252))
        if not np.isfinite(expected) or not np.isfinite(scale) or scale <= 0:
            raise ValueError('收益预测或波动尺度无效。')
        position = calendar.get_loc(signal)
        if signal in label_dates.index:
            realized = label_dates.loc[signal]
            label_start, label_end, target = realized.label_start, realized.label_end, float(realized.target)
        else:
            if position + LABEL_HORIZON + 1 < len(calendar):
                raise ValueError('历史预测缺少本应已成熟的真实收益标签。')
            # No future prices are synthesized. Unmatured predictions are explicitly unlabeled.
            label_start = calendar[position + 1] if position + 1 < len(calendar) else pd.NaT
            label_end, target = pd.NaT, np.nan
        matured_returns = [r for r in return_history if r['label_end'] < signal][-MAX_CALIBRATION:]
        lower, upper, radius = None, None, None
        if len(matured_returns) >= MIN_CALIBRATION:
            residuals = np.array([abs(r['target'] - r['expected_return']) / r['scale'] for r in matured_returns])
            rank = math.ceil((len(residuals) + 1) * INTERVAL_COVERAGE)
            radius = float(np.sort(residuals)[rank - 1])
            lower, upper = expected - radius * scale, expected + radius * scale
        calibration_models = {}
        for candidate in CANDIDATES:
            matured = [r for r in records[candidate] if r['label_end'] < signal][-MAX_CALIBRATION:]
            probability, calibrator = None, None
            if len(matured) >= MIN_CALIBRATION:
                probability, calibrator = _calibrate(pd.DataFrame(matured), raw[candidate])
            calibration_models[candidate] = calibrator
            records[candidate].append({
                'date': signal, 'symbol': str(current.symbol.iloc[0]), 'candidate': candidate,
                'scheduled': signal in grid, 'label_start': label_start, 'label_end': label_end, 'target': target,
                'raw_probability': raw[candidate], 'probability': probability,
                'base_probability': base_probability, 'old_direction': old_return > 0,
                'old_expected_return': old_return, 'expected_return': expected,
                'lower_return': lower, 'upper_return': upper, 'scale': scale,
                'training_samples': len(train), 'training_start': train.date.min(),
                'training_last_label_end': train.label_end.max(),
                'calibration_samples': len(matured),
                'calibration_last_label_end': matured[-1]['label_end'] if matured else pd.NaT,
            })
        return_history.append({'date': signal, 'label_end': label_end, 'target': target, 'expected_return': expected, 'scale': scale})
        latest_models = {
            'classifiers': models, 'calibrators': calibration_models, 'return_radius': radius,
            'factor_keys': expanded_keys, 'core_factor_keys': FACTOR_KEYS,
            'signal_date': signal, 'training_last_label_end': train.label_end.max(),
        }
        if progress is not None and (index % 20 == 0 or index == len(dates) - 1):
            progress(index + 1, len(dates))
    combined = pd.concat([pd.DataFrame(rows) for rows in records.values()], ignore_index=True)
    core_probabilities = combined.loc[combined.candidate == 'core_logistic', ['date', 'probability']].rename(columns={'probability': 'core_probability'})
    combined = combined.merge(core_probabilities, on='date', validate='many_to_one')
    return combined, latest_models
