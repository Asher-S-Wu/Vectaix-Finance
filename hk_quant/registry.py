"""只有合格模型与完整快照才能成为正式发布。"""
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import HORIZONS
from .paths import MODELS
from .prediction_tasks import PREDICTION_SCHEMA,TASK_STATUS_COLUMNS,TASK_COLUMNS,validate_task_outputs


def publication_artifact(models_root,relative_path):
    """发布记录只使用发布目录内的可迁移相对路径。"""
    relative=Path(relative_path)
    if relative.is_absolute() or '\\' in relative_path or '..' in relative.parts:
        raise ValueError('发布文件必须使用目录内的相对路径')
    root=Path(models_root).resolve();target=(root/relative).resolve()
    if not target.is_relative_to(root):raise ValueError('发布文件超出模型目录')
    return target


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def publish_snapshot(forecast_path, model_version, data_as_of, model_acceptance, snapshot_audit, models_root=MODELS):
    models_root = Path(models_root).resolve()
    if model_acceptance.get('eligible') is not True:
        result = {'status': 'not_published', 'eligible': False, 'model_version': model_version,
                  'reason': '模型尚未通过验收', 'model_acceptance': model_acceptance}
        _atomic_json(models_root / 'publication_status.json', result)
        return result
    if snapshot_audit.get('critical_gap_count') != 0 or snapshot_audit.get('freshness_passed') is not True:
        result = {'status': 'not_published', 'eligible': False, 'model_version': model_version,
                  'reason': '当日快照存在关键缺口或数据尚未完整', 'snapshot_audit': snapshot_audit}
        _atomic_json(models_root / 'publication_status.json', result)
        return result
    forecast_path = Path(forecast_path).resolve()
    frame = pd.read_parquet(forecast_path)
    required = {'date','security_id','horizon','model_version','data_as_of','status',
                'prediction_schema','reasons','explanations','return_basis',*TASK_STATUS_COLUMNS.values(),
                *(column for columns in TASK_COLUMNS.values() for column in columns)}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError('预测快照为空或字段不完整')
    day = pd.Timestamp(data_as_of).normalize()
    if not frame.model_version.eq(model_version).all() or not pd.to_datetime(frame.date).dt.normalize().eq(day).all():
        raise ValueError('预测版本或日期与发布声明不一致')
    if not pd.to_datetime(frame.data_as_of).dt.normalize().eq(day).all():
        raise ValueError('预测数据日期与发布声明不一致')
    if frame.duplicated(['date','security_id','horizon']).any():
        raise ValueError('预测快照有重复股票期限')
    if not frame.groupby('security_id').horizon.apply(lambda values:set(values)==set(HORIZONS)).all():
        raise ValueError('预测快照未覆盖四个期限')
    if not frame.prediction_schema.eq(PREDICTION_SCHEMA).all():
        raise ValueError('预测任务输出格式不符合当前发布规范')
    validate_task_outputs(frame)
    ready = frame[frame.status.eq('ok')]
    if ready.empty:
        raise ValueError('快照没有可用预测')
    numbers = ready[['score','probability_up','expected_return','q10','q50','q90']]
    if not np.isfinite(numbers).all().all() or not ready.probability_up.between(0,1).all():
        raise ValueError('预测数值无效')
    if not (ready.q10.le(ready.q50) & ready.q50.le(ready.q90)).all():
        raise ValueError('预测区间发生交叉')
    stamp = datetime.now().astimezone().isoformat()
    certificate = {'eligible': True, 'model_version': model_version, 'data_as_of': day.date().isoformat(),
                   'prediction_schema':PREDICTION_SCHEMA,
                   'created_at': stamp, 'model_acceptance': model_acceptance, 'snapshot_audit': snapshot_audit}
    publication_directory=publication_artifact(models_root,f'publications/{model_version}')
    publication_directory.mkdir(parents=True,exist_ok=True)
    snapshot_path=publication_directory/f'{day:%Y%m%d}.parquet'
    pending_snapshot=snapshot_path.with_suffix('.parquet.pending')
    shutil.copyfile(forecast_path,pending_snapshot)
    pending_snapshot.replace(snapshot_path)
    certificate_path = publication_directory / f'{day:%Y%m%d}.json'
    _atomic_json(certificate_path, certificate)
    active = {'eligible': True, 'model_version': model_version, 'data_as_of': day.date().isoformat(),
              'prediction_schema':PREDICTION_SCHEMA,
              'forecast_path': snapshot_path.relative_to(models_root).as_posix(),
              'acceptance_path': certificate_path.relative_to(models_root).as_posix(),
              'published_at': stamp}
    _atomic_json(models_root / 'active.json', active)
    _atomic_json(models_root / 'publication_status.json', {'status':'published', **active})
    return active
