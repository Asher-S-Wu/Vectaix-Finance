"""在独立目录构建完整数据版本，不覆盖正在训练的快照。"""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from .collect import write_json
from .data import prepare
from .features import build_features
from .training import normalize_dataset


def rebuild_snapshot(source_root,destination,start,end):
    source_root,destination=Path(source_root).resolve(),Path(destination).resolve()
    collection=json.loads((source_root/'collection_status.json').read_text(encoding='utf-8'))
    if (collection['status']!='complete' or collection['requested_start']>start
        or collection['requested_end']<end):
        raise ValueError('源采集尚未完整覆盖请求日期')
    if destination.exists():raise ValueError('目标版本已存在；应先核对已有构建进程及结果')
    destination.mkdir(parents=True)
    origin={'source_root':str(source_root),'start':start,'end':end,
            'created_at':datetime.now().astimezone().isoformat(),'status':'copying_sources'}
    write_json(destination/'build_status.json',origin)
    for folder in ('source','references','raw'):
        shutil.copytree(source_root/folder,destination/folder)
    for filename in ('financial_versions.parquet','collection_status.json'):
        shutil.copy2(source_root/filename,destination/filename)
    for stage,operation in (
        ('preparing',lambda:prepare(destination,start,end)),
        ('building_features',lambda:build_features(destination)),
        ('normalizing',lambda:normalize_dataset(destination)),
    ):
        origin['status']=stage
        write_json(destination/'build_status.json',origin)
        print(stage,flush=True)
        operation()
    origin['status']='complete'
    origin['completed_at']=datetime.now().astimezone().isoformat()
    write_json(destination/'build_status.json',origin)
    return origin


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--start',default='20100101')
    parser.add_argument('--end',required=True)
    args=parser.parse_args()
    rebuild_snapshot(args.source_root,args.destination,args.start,args.end)
