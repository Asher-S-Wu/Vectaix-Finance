"""各预测任务独立报告可用性，整体失败不抹去其他任务的真实输出。"""
import numpy as np
import pandas as pd


PREDICTION_SCHEMA = 'independent-task-availability-v1'
MODEL_REJECTION_STATUSES = ('unrepresentable_prediction', 'invalid_return_range', 'invalid_quantile_order')
TASK_COLUMNS = {
    'score': ['score'],
    'probability_up': ['probability_up', 'baseline_probability'],
    'intervals': ['q10', 'q50', 'q90', 'baseline_q10', 'baseline_q50', 'baseline_q90'],
    'expected_return': ['expected_return'],
}
TASK_STATUS_COLUMNS = {'score': 'score_status', 'probability_up': 'probability_status',
                       'intervals': 'interval_status', 'expected_return': 'expected_return_status'}
STATUS_REASONS = {'insufficient_model_inputs': '必要输入不足', 'unrepresentable_prediction': '预测数值无法表示',
                  'invalid_return_range': '预测收益低于-100%，未裁剪',
                  'invalid_quantile_order': '预测分位数交叉，未排序修正'}


def classify_task(frame, task):
    values = frame[TASK_COLUMNS[task]].to_numpy(float)
    status = pd.Series('ok', index=frame.index, dtype=object)
    status.loc[~np.isfinite(values).all(axis=1)] = 'unrepresentable_prediction'
    if task == 'probability_up':
        status.loc[((values < 0) | (values > 1)).any(axis=1)] = 'unrepresentable_prediction'
    if task in ('expected_return', 'intervals'):
        status.loc[(values < -1).any(axis=1)] = 'invalid_return_range'
    if task == 'intervals':
        crossed = ((values[:, 0] > values[:, 1]) | (values[:, 1] > values[:, 2])
                   | (values[:, 3] > values[:, 4]) | (values[:, 4] > values[:, 5]))
        status.loc[crossed] = 'invalid_quantile_order'
    return status


def aggregate_task_status(frame):
    fields = list(TASK_STATUS_COLUMNS.values())
    statuses = frame[fields]
    if not statuses.isin(('ok', 'insufficient_model_inputs', *MODEL_REJECTION_STATUSES)).all().all():
        raise ValueError('任务可用状态缺失或未知，不能推断为输入不足')
    aggregate = pd.Series('ok', index=frame.index, dtype=object)
    aggregate.loc[statuses.eq('insufficient_model_inputs').any(axis=1)] = 'insufficient_model_inputs'
    for status in MODEL_REJECTION_STATUSES:
        aggregate.loc[statuses.eq(status).any(axis=1)] = status
    return aggregate


def store_task_outputs(out, values):
    """只为各任务合法的行写入真实数值，其他任务互不影响。"""
    for task, columns in TASK_COLUMNS.items():
        status = classify_task(values, task)
        out.loc[values.index, TASK_STATUS_COLUMNS[task]] = status
        ready = status.index[status.eq('ok')]
        out.loc[ready, columns] = values.loc[ready, columns]
        failed = status.index[status.ne('ok')]
        out.loc[failed, 'reasons'] += task + ': ' + status.loc[failed].map(STATUS_REASONS) + '；'
    out.loc[values.index, 'status'] = aggregate_task_status(out.loc[values.index])


def validate_task_outputs(frame):
    """读取已存预测时检查每任务状态与数值一致，不接受旧的整行推断。"""
    aggregate = aggregate_task_status(frame)
    if not aggregate.eq(frame.status).all():
        raise ValueError('整体状态与独立任务状态不一致')
    for task, columns in TASK_COLUMNS.items():
        states = frame[TASK_STATUS_COLUMNS[task]]
        valid = states.eq('ok')
        if not classify_task(frame.loc[valid], task).eq('ok').all():
            raise ValueError(f'{task}声明可用但预测数值无效')
        if frame.loc[~valid, columns].notna().any().any():
            raise ValueError(f'{task}声明不可用但仍携带预测数值')
