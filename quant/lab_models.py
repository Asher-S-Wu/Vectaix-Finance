from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, ndtr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .lab_binning import install_batch_binning


HORIZON = 21
PREDICTION_STRIDE = 5
REFIT_STRIDE = 21
MIN_TRAINING = 504
TRAINING_YEARS = 5
CALIBRATION_YEARS = 3
MIN_CALIBRATION_DATES = 52
MODEL_WORKERS = 3


@dataclass(frozen=True)
class Expert:
    id: str
    scope: str
    features: str
    family: str
    strength: float


BASE_EXPERTS = (
    Expert('local_core_lr_01', 'local', 'core', 'logistic', 0.1),
    Expert('local_core_lr_1', 'local', 'core', 'logistic', 1.0),
    Expert('local_compact_lr_01', 'local', 'compact', 'logistic', 0.1),
    Expert('local_compact_lr_1', 'local', 'compact', 'logistic', 1.0),
    Expert('pool_compact_lr_01', 'pool', 'compact', 'logistic', 0.1),
    Expert('pool_compact_lr_1', 'pool', 'compact', 'logistic', 1.0),
    Expert('local_drivers_lr_01', 'local', 'drivers', 'logistic', 0.1),
    Expert('pool_drivers_lr_01', 'pool', 'drivers', 'logistic', 0.1),
    Expert('local_drivers_tree_3', 'local', 'drivers', 'tree', 3),
    Expert('local_drivers_tree_7', 'local', 'drivers', 'tree', 7),
    Expert('pool_drivers_tree_7', 'pool', 'drivers', 'tree', 7),
    Expert('pool_drivers_tree_15', 'pool', 'drivers', 'tree', 15),
    Expert('sector_drivers_tree_7', 'sector', 'drivers', 'tree', 7),
    Expert('pool_drivers_ridge_10', 'pool', 'drivers', 'ridge', 10),
    Expert('pool_drivers_ridge_100', 'pool', 'drivers', 'ridge', 100),
)
EVENT_EXPERTS = (
    Expert('local_event_compact_lr_01', 'local', 'event_compact', 'logistic', 0.1),
    Expert('pool_event_compact_lr_01', 'pool', 'event_compact', 'logistic', 0.1),
    Expert('local_event_drivers_tree_7', 'local', 'event_drivers', 'tree', 7),
    Expert('pool_event_drivers_tree_7', 'pool', 'event_drivers', 'tree', 7),
)
EXPERTS = BASE_EXPERTS + EVENT_EXPERTS
BLEND_ID = 'blend_local_pool_tree_7'
BASE_EXPERT_IDS = tuple(expert.id for expert in BASE_EXPERTS) + (BLEND_ID,)
EXPERT_IDS = BASE_EXPERT_IDS + tuple(expert.id for expert in EVENT_EXPERTS)
BASE_CANDIDATE_IDS = tuple(f'{expert}:{mode}' for expert in BASE_EXPERT_IDS for mode in ('raw', 'monotone'))
CANDIDATE_IDS = tuple(f'{expert}:{mode}' for expert in EXPERT_IDS for mode in ('raw', 'monotone'))


def model_specification() -> dict:
    return {
        'experts': [asdict(expert) for expert in EXPERTS],
        'blend': {'id': BLEND_ID, 'members': ['local_drivers_tree_7', 'pool_drivers_tree_7'], 'weights': [0.5, 0.5]},
        'candidateCount': len(CANDIDATE_IDS), 'candidates': CANDIDATE_IDS,
        'priceOnlyCandidates': BASE_CANDIDATE_IDS,
        'eventHypothesis': '已公布年度或中期业绩后的价格反应可能具有不同延续性；只新增四个固定模型，不根据本轮结果改变事件窗口。',
        'trainingYears': TRAINING_YEARS, 'minimumTrainingRowsPerStock': MIN_TRAINING,
        'horizonTradingDays': HORIZON, 'predictionStride': PREDICTION_STRIDE, 'refitStride': REFIT_STRIDE,
        'modelWorkers': MODEL_WORKERS, 'mathThreadsPerFit': 1,
        'weightedBinning': '固定scikit-learn 1.9.0，分箱分位数批量计算并共享排序；保留原权重、分位数、去重与截断规则。',
        'constantFeatures': '每个训练窗口内删除完全不变化的输入，再标准化；不依据未来数据筛选。',
        'tree': {'iterations': 100, 'learningRate': 0.03, 'earlyStopping': False, 'seed': 42},
        'sharedTrainingWeights': '每个日期全部股票合计权重1/21；本股训练每行1/21，不将重叠标签视为独立实验。',
        'ridgeProbability': '以观测21日波动标准化收益训练，正态分布函数将预测值映为初始上涨概率；另比较历史校准。',
        'calibration': {'years': CALIBRATION_YEARS, 'minimumDistinctForecastDates': MIN_CALIBRATION_DATES,
                        'scope': '各候选共享全部股票的历史样本外预测，日期内等权',
                        'slopeMinimum': 0, 'identityPenalty': 1.0,
                        'purpose': '保持原始概率排序；斜率为零时仅表示未能识别原始信号，不反转方向。'},
    }


def _fit(expert: Expert, train: pd.DataFrame, columns: list[str]):
    if (train.target > 0).nunique() != 2:
        raise ValueError(f'{expert.id} 训练标签未同时包含上涨与未上涨。')
    multiplicity = train.groupby('date').symbol.transform('size').to_numpy()
    weights = 1 / (multiplicity * HORIZON)
    if expert.family == 'logistic':
        model = make_pipeline(VarianceThreshold(), StandardScaler(), LogisticRegression(C=expert.strength, max_iter=1000))
        model.fit(train[columns], train.target > 0,
                  standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
    elif expert.family == 'tree':
        members = train.symbol.nunique()
        model = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.03, max_leaf_nodes=int(expert.strength),
            max_depth=2 if expert.strength <= 7 else 3,
            min_samples_leaf=63 * members, l2_regularization=1.0,
            early_stopping=False, random_state=42,
        )
        model.fit(train[columns], train.target > 0, sample_weight=weights)
    elif expert.family == 'ridge':
        model = make_pipeline(VarianceThreshold(), StandardScaler(), Ridge(alpha=expert.strength))
        scale = train.annualVolatility.to_numpy() * np.sqrt(HORIZON / 252)
        if (scale <= 0).any() or not np.isfinite(scale).all():
            raise ValueError('收益模型的历史波动尺度无效。')
        model.fit(train[columns], train.target.to_numpy() / scale,
                  standardscaler__sample_weight=weights, ridge__sample_weight=weights)
    else:
        raise ValueError(f'未知模型家族：{expert.family}')
    return model


def _probability(expert: Expert, model, current: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = ndtr(model.predict(current[columns])) if expert.family == 'ridge' else model.predict_proba(current[columns])[:, 1]
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('候选模型产生无效概率。')
    return np.asarray(values, dtype=float)


def fit_predict_experts(train: pd.DataFrame, current: pd.DataFrame, feature_groups: dict, sectors: dict):
    counts = train.groupby('symbol').size()
    eligible = counts.index[counts >= MIN_TRAINING].tolist()
    current = current.loc[current.symbol.isin(eligible)].copy()
    train = train.loc[train.symbol.isin(eligible)].copy()
    if current.empty:
        raise ValueError('该训练时点没有达到最小历史要求的证券。')
    bundles, outputs = {}, []
    def fit_expert(expert):
        columns = list(feature_groups[expert.features])
        bundle = {'specification': asdict(expert), 'columns': columns, 'models': {}}
        values = pd.Series(index=current.index, dtype=float)
        if expert.scope == 'local':
            groups = [(symbol, [symbol]) for symbol in current.symbol.unique()]
        elif expert.scope == 'pool':
            groups = [('all', eligible)]
        elif expert.scope == 'sector':
            groups = [(sector, [symbol for symbol in eligible if sectors[symbol] == sector]) for sector in sorted(set(sectors.values()))]
        else:
            raise ValueError('未知训练范围。')
        for key, symbols in groups:
            observed = current.loc[current.symbol.isin(symbols)]
            if observed.empty:
                continue
            training = train.loc[train.symbol.isin(symbols)]
            model = _fit(expert, training, columns)
            values.loc[observed.index] = _probability(expert, model, observed, columns)
            bundle['models'][key] = model
        if values.isna().any():
            raise ValueError('候选没有覆盖本轮全部有效股票。')
        output = current[['date', 'symbol']].assign(expert=expert.id, raw_probability=values.to_numpy())
        return expert.id, bundle, output
    # OpenMP's thread limit is local to the calling thread, so set it in each worker too.
    with ThreadPoolExecutor(max_workers=MODEL_WORKERS, initializer=threadpool_limits, initargs=(1,)) as executor:
        for expert_id, bundle, output in executor.map(fit_expert, EXPERTS):
            outputs.append(output)
            bundles[expert_id] = bundle
    result = pd.concat(outputs, ignore_index=True)
    members = result.loc[result.expert.isin(['local_drivers_tree_7', 'pool_drivers_tree_7'])]
    blend = members.groupby(['date', 'symbol'], as_index=False).raw_probability.mean().assign(expert=BLEND_ID)
    return pd.concat([result, blend], ignore_index=True), bundles


def raw_walk_forward(dataset: dict, output_dir, progress):
    install_batch_binning()
    factors, labels, calendar = dataset['factors'], dataset['labels'], dataset['calendar']
    securities = dataset['universe']['stocks'] + dataset['universe']['peerStocks']
    sectors = {stock['symbol']: stock['sector'] for stock in securities}
    forecast_dates = calendar[::PREDICTION_STRIDE]
    refits = [date for date in calendar[::REFIT_STRIDE]
              if date >= pd.Timestamp('2014-01-01')]
    records, fit_audits = [], []
    for index, signal in enumerate(refits):
        stop = refits[index + 1] if index + 1 < len(refits) else calendar[-1] + pd.Timedelta(days=1)
        current = factors.loc[(factors.date >= signal) & (factors.date < stop) & factors.date.isin(forecast_dates)]
        if current.empty:
            continue
        training = labels.loc[(labels.date >= signal - pd.DateOffset(years=TRAINING_YEARS)) & (labels.label_end < signal)]
        raw, _ = fit_predict_experts(training, current, dataset['feature_groups'], sectors)
        historical = training.loc[training.date.isin(calendar[::HORIZON])]
        rates = historical.groupby('symbol').target.agg(['size', lambda values: int((values > 0).sum())])
        rates.columns = ['observations', 'up']
        rates['base_probability'] = (rates.up + 1) / (rates.observations + 2)
        raw = raw.merge(rates[['base_probability']], left_on='symbol', right_index=True, validate='many_to_one')
        raw['fit_date'] = signal
        raw['training_last_label_end'] = training.label_end.max()
        raw = raw.merge(labels[['date', 'symbol', 'label_start', 'label_end', 'target']], on=['date', 'symbol'], how='left', validate='many_to_one')
        records.append(raw)
        fit_audits.append({'date': signal.date().isoformat(), 'lastMaturedLabelEnd': training.label_end.max().date().isoformat(),
                           'trainingRows': len(training), 'stocks': int(raw.symbol.nunique()), 'forecastRows': len(raw)})
        if index % 6 == 0 or index == len(refits) - 1:
            pd.concat(records, ignore_index=True).to_csv(output_dir / 'raw-predictions.csv', index=False, date_format='%Y-%m-%d')
            progress(index + 1, len(refits))
    if not records:
        raise ValueError('没有完成任何原始样本外预测。')
    history = pd.concat(records, ignore_index=True).sort_values(['expert', 'date', 'symbol']).reset_index(drop=True)
    latest = calendar[-1]
    train = labels.loc[(labels.date >= latest - pd.DateOffset(years=TRAINING_YEARS)) & (labels.label_end < latest)]
    current = factors.loc[factors.date == latest]
    if not set(dataset['target_symbols']).issubset(current.symbol):
        raise ValueError('原四只股票最新日期缺少有效因子。')
    latest_raw, models = fit_predict_experts(train, current, dataset['feature_groups'], sectors)
    return history, latest_raw, models, fit_audits


def _logit(probabilities):
    # Protect only floating-point endpoints in the logarithm; these are not data substitutes.
    epsilon = np.finfo(float).eps
    values = np.clip(np.asarray(probabilities, dtype=float), epsilon, 1 - epsilon)
    return np.log(values) - np.log1p(-values)


def fit_monotone_calibrator(history: pd.DataFrame) -> dict:
    score = _logit(history.raw_probability)
    target = (history.target.to_numpy() > 0).astype(float)
    counts = history.groupby('date').symbol.transform('size').to_numpy()
    weights = PREDICTION_STRIDE / (HORIZON * counts)
    if len(np.unique(target)) != 2:
        raise ValueError('共享校准记录必须同时有上涨和未上涨。')
    def objective(parameters):
        slope, intercept = parameters
        logits = slope * score + intercept
        error = weights * (expit(logits) - target)
        loss = np.sum(weights * (np.logaddexp(0, logits) - target * logits)) + 0.5 * ((slope - 1) ** 2 + intercept ** 2)
        gradient = np.array([np.sum(error * score) + slope - 1, np.sum(error) + intercept])
        return float(loss), gradient
    solution = minimize(objective, np.array([1.0, 0.0]), method='L-BFGS-B', jac=True,
                        bounds=[(0, None), (None, None)], options={'maxiter': 200, 'ftol': 1e-10})
    if not solution.success or not np.isfinite(solution.x).all():
        raise ValueError(f'单调概率校准未收敛：{solution.message}')
    return {'slope': float(solution.x[0]), 'intercept': float(solution.x[1]),
            'distinctDates': int(history.date.nunique()), 'rows': len(history),
            'firstDate': history.date.min().date().isoformat(),
            'lastLabelEnd': history.label_end.max().date().isoformat()}


def apply_calibrator(calibrator: dict, probabilities) -> np.ndarray:
    return expit(calibrator['slope'] * _logit(probabilities) + calibrator['intercept'])


def calibrate_walk_forward(history: pd.DataFrame, latest_raw: pd.DataFrame, progress):
    rows, latest_rows, calibrators, audits = [], [], {}, []
    for index, expert in enumerate(EXPERT_IDS):
        past = history.loc[history.expert == expert].sort_values(['date', 'symbol'])
        for signal, current in past.groupby('date', sort=True):
            matured = past.loc[(past.date >= signal - pd.DateOffset(years=CALIBRATION_YEARS)) & (past.label_end < signal)]
            if matured.date.nunique() < MIN_CALIBRATION_DATES:
                continue
            calibrated = fit_monotone_calibrator(matured)
            raw_rows = current.assign(candidate=f'{expert}:raw', probability=current.raw_probability)
            calibrated_rows = current.assign(candidate=f'{expert}:monotone', probability=apply_calibrator(calibrated, current.raw_probability))
            rows.extend([raw_rows, calibrated_rows])
            audits.append({'expert': expert, 'date': signal.date().isoformat(), **calibrated})
        current = latest_raw.loc[latest_raw.expert == expert]
        signal = current.date.iloc[0]
        matured = past.loc[(past.date >= signal - pd.DateOffset(years=CALIBRATION_YEARS)) & (past.label_end < signal)]
        if matured.date.nunique() < MIN_CALIBRATION_DATES:
            raise ValueError('最新预测缺少足够的共享校准历史。')
        calibrated = fit_monotone_calibrator(matured)
        calibrators[expert] = calibrated
        latest_rows.extend([
            current.assign(candidate=f'{expert}:raw', probability=current.raw_probability),
            current.assign(candidate=f'{expert}:monotone', probability=apply_calibrator(calibrated, current.raw_probability)),
        ])
        progress(index + 1, len(EXPERT_IDS))
    return pd.concat(rows, ignore_index=True), pd.concat(latest_rows, ignore_index=True), calibrators, audits
