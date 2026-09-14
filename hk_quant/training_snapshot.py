"""把有来源的收益标签修复应用到独立训练快照，保留原数据与确认期。"""
import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from . import HORIZONS
from .collect import write_json
from .training import dataset_lineage


def apply_label_patches(frame, patches, cutoff):
    """只修改现有行的缺失收益，拒绝未知证券、改变期限或覆盖已有结果。"""
    result = frame.copy()
    if patches.empty:
        return result
    patches = patches.copy()
    for field in ('date', 'label_end'):
        patches[field] = pd.to_datetime(patches[field])
    keys = ['date', 'security_id', 'horizon']
    if patches.duplicated(keys).any():
        raise ValueError('收益补丁键重复')
    if patches[['date', 'security_id', 'label_end']].isna().any().any():
        raise ValueError('补丁缺少证券、日期或收益结束日')
    if not patches.horizon.isin(HORIZONS).all():
        raise ValueError('收益补丁期限不受支持')
    if patches.label_end.gt(pd.Timestamp(cutoff)).any():
        raise ValueError('收益标签尚未到修复截止日期')
    values = pd.to_numeric(patches.new_fwd_return, errors='coerce')
    if not np.isfinite(values).all() or values.lt(-1).any():
        raise ValueError('修复收益无效或低于-100%')
    if not patches.old_fwd_return.isna().all():
        raise ValueError('收益修复不得覆盖已有标签')
    if patches.source_url.isna().any() or not patches.source_url.astype(str).str.startswith('https://').all():
        raise ValueError('收益修复缺少证据来源')
    frame_keys = pd.MultiIndex.from_arrays([pd.to_datetime(result.date), result.security_id])
    if frame_keys.has_duplicates:
        raise ValueError('训练数据证券日期重复')
    positions = frame_keys.get_indexer(pd.MultiIndex.from_frame(patches[['date', 'security_id']]))
    if (positions < 0).any():
        raise ValueError('收益修复指向不存在的样本')
    for position, row in zip(positions, patches.itertuples(index=False)):
        label = f'fwd_return_{row.horizon}'
        end = f'label_end_{row.horizon}'
        if pd.Timestamp(result.iloc[position][end]) != row.label_end:
            raise ValueError('收益修复不能改变原预测结束日')
        if pd.notna(result.iloc[position][label]):
            raise ValueError('不得覆盖已有收益标签')
        result.iloc[position, result.columns.get_loc(label)] = float(row.new_fwd_return)
    return result


def build_training_snapshot(source, destination, patch_paths, cutoff='2023-12-31'):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('目标训练快照已存在，请检查已有构建状态')
    patches = pd.concat([pd.read_parquet(path) for path in patch_paths], ignore_index=True)
    if patches.duplicated(['date', 'security_id', 'horizon']).any():
        raise ValueError('多个来源补丁有重复预测键')
    # Verify every referenced original before any destination is created.
    for security, group in patches.groupby('security_id'):
        frame = pd.read_parquet(source / 'features' / f'{quote(security, safe=".!-")}.parquet')
        apply_label_patches(frame, group, cutoff)
    destination.mkdir(parents=True)
    state = dict(status='copying', source_root=str(source), label_cutoff=cutoff,
        label_patch_sources=[str(Path(path).resolve()) for path in patch_paths],
        raw_evidence_policy='Raw source files remain at the source root; derived training inputs are separate physical copies.')
    write_json(destination / 'build_status.json', state)
    for directory in ('bars', 'features', 'normalized', 'training_samples'):
        shutil.copytree(source / directory, destination / directory)
    for path in source.iterdir():
        if path.is_file() and path.name not in ('build_status.json', 'normalization_manifest.json'):
            shutil.copy2(path, destination / path.name)
    reference = destination / 'references'
    reference.mkdir()
    shutil.copy2(source / 'references/calendar.parquet', reference / 'calendar.parquet')
    for name in ('endpoint_identity_review', 'development_label_gaps'):
        shutil.copytree(source / 'references' / name, reference / name)
    write_json(destination / 'build_status.json', {**state, 'status': 'patching_labels'})
    counts = {'features': 0, 'normalized': 0, 'training_samples': 0}
    for security, group in patches.groupby('security_id'):
        path = destination / 'features' / f'{quote(security, safe=".!-")}.parquet'
        frame = pd.read_parquet(path)
        apply_label_patches(frame, group, cutoff).to_parquet(path, index=False, row_group_size=252)
        counts['features'] += len(group)
    for directory in ('normalized', 'training_samples'):
        for year, group in patches.groupby(pd.to_datetime(patches.date).dt.year):
            path = destination / directory / f'{year}.parquet'
            frame = pd.read_parquet(path)
            keys = pd.MultiIndex.from_frame(frame[['date', 'security_id']])
            selected = group.loc[pd.MultiIndex.from_frame(group[['date', 'security_id']]).isin(keys)]
            apply_label_patches(frame, selected, cutoff).to_parquet(path, index=False, row_group_size=200_000)
            counts[directory] += len(selected)
    if counts['features'] != len(patches) or counts['normalized'] != len(patches):
        raise ValueError('标签修复未完整进入因子与标准化数据')
    patches.to_parquet(destination / 'label_patches.parquet', index=False)
    manifest = json.loads((destination / 'feature_manifest.json').read_text(encoding='utf-8'))
    manifest['label_revision'] = dict(protocol='reviewed-missing-label-patches-v1', patches=len(patches),
        source_root=str(source), cutoff=cutoff, inputs_changed=False, full_label_coverage_complete=False)
    write_json(destination / 'feature_manifest.json', manifest)
    write_json(destination / 'normalization_manifest.json', dataset_lineage(destination))
    audit = dict(patches=len(patches), patched_sample_counts=counts, features_changed=False,
        rows_added=0, existing_values_overwritten=0, confirmation_rows_changed=0,
        full_label_coverage_complete=False, source_root=str(source))
    write_json(destination / 'label_patch_audit.json', audit)
    write_json(destination / 'build_status.json', {**state, 'status': 'complete'})
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--patch', type=Path, action='append', required=True)
    args = parser.parse_args()
    print(json.dumps(build_training_snapshot(args.source, args.destination, args.patch), ensure_ascii=False), flush=True)
