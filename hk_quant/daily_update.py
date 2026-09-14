"""生成每日港股候选预测和验收报告。"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from .collect import write_json
from .report import build_report
from .training import export_candidate


def run_daily_update(data_root, results_root, model_root):
    data_root = Path(data_root)
    results_root = Path(results_root)
    model_root = Path(model_root)
    status_path = results_root / 'daily_update_status.json'
    build_path = data_root / 'build_status.json'
    if not build_path.exists():
        raise ValueError('缺少数据构建状态')
    build = json.loads(build_path.read_text(encoding='utf-8'))
    if build.get('status') != 'complete':
        raise ValueError('数据版本仍在构建，不能生成每日预测')
    candidate = export_candidate(data_root, results_root, model_root)
    report = build_report(results_root, data_root)
    # 每日流程只生成可审阅候选；正式发布必须由 registry.publish_snapshot 写入发布证书。
    published = False
    result = {
        'status': 'published' if published else 'candidate_only',
        'published': published,
        'model_version': candidate.get('model_version'),
        'data_as_of': candidate.get('data_as_of'),
        'model_trained_as_of': candidate.get('model_trained_as_of'),
        'forecast_path': candidate.get('forecast_path'),
        'candidate_status': candidate.get('status'),
        'report_eligible': report.get('eligible') is True,
        'publication_reason': '每日流程未执行正式发布；需独立验收证书和发布写入',
        'generated_at': datetime.now().astimezone().isoformat(),
    }
    write_json(status_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description='生成每日港股候选预测和训练报告')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    args = parser.parse_args()
    result = run_daily_update(args.data_root, args.results_root, args.model_root)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] == 'published' else 2


if __name__ == '__main__':
    raise SystemExit(main())
