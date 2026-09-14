"""把真实训练指标写成可阅读报告，未完成的验收保持未完成。"""
import argparse
import html
import json
from datetime import datetime
from pathlib import Path

from .collect import write_json
from .contracts import FORECAST_RETURN_BASIS, CASH_DIVIDEND_ACCOUNTING
from .paths import DATA, RESULTS

TASK_LABELS = {'score':'排名','probability_up':'上涨概率','intervals':'收益区间','expected_return':'预期收益'}


def build_report(results_root=RESULTS, data_root=DATA):
    protocol = json.loads((results_root/'training_protocol.json').read_text(encoding='utf-8'))
    audit = json.loads((data_root/'data_audit.json').read_text(encoding='utf-8'))
    feature_manifest = json.loads((data_root/'feature_manifest.json').read_text(encoding='utf-8'))
    identity_audit_path = data_root/'dated_identity_audit.json'
    identity_audit = json.loads(identity_audit_path.read_text(encoding='utf-8')) if identity_audit_path.exists() else {}
    status = json.loads((results_root/'training_status.json').read_text(encoding='utf-8'))
    rows, coverage, pending = [], [], []
    bundle_metrics = None
    component_mature_coverage = {}
    label_fields = ('matured_rows','matured_label_rows','missing_matured_label_rows','matured_label_coverage')
    entries=[(kind,results_root/'development'/f'{kind}_metrics.json') for kind in protocol['kinds']]
    if (results_root/'bundle_development_metrics.json').exists():
        entries.append(('统一分任务组合',results_root/'bundle_development_metrics.json'))
    for kind,path in entries:
        if not path.exists():
            pending.append(kind)
            continue
        metrics = json.loads(path.read_text(encoding='utf-8'))
        if kind == '统一分任务组合':
            bundle_metrics = metrics
        else:
            component_mature_coverage[kind] = {key:metrics.get(key)
                for key in (*label_fields, 'label_coverage_complete')}
        coverage.append({'model':kind,'total_rows':metrics['total_forecast_rows'],
                         'scored_rows':metrics['scored_forecast_rows'],'excluded_rows':metrics['excluded_status_rows'],
                         'status_counts':metrics['status_counts'],
                         'input_unavailable_rows':metrics.get('input_unavailable_rows'),
                         'model_rejected_rows':metrics.get('model_rejected_rows'),
                         'model_output_coverage_complete':metrics.get('model_output_coverage_complete'),
                         'task_coverage':metrics.get('task_coverage'),
                         'task_valid_samples':metrics.get('task_valid_samples'),
                         'horizon_status_coverage':{
                             str(h):{key:values.get(key) for key in ('status_counts','input_unavailable_rows',
                                 'input_eligible_forecast_rows','model_rejected_rows','matured_model_rejected_rows',
                                 'model_rejection_rate','model_output_coverage_complete',
                                 'task_coverage','task_valid_samples')}
                             for h,values in metrics['horizons'].items()}})
        for horizon, values in metrics['horizons'].items():
            rows.append({'model':kind,'horizon':int(horizon),'rank_ic':values['ic']['mean'],
                'rank_ic_ci95_lower':values['ic']['ci_lower'],'positive_ic_year_ratio':values['ic']['positive_year_ratio'],
                'brier_skill':values['brier']['skill'],'calibration_error':values['calibration']['ece'],
                'interval_coverage':values['interval']['coverage'],'interval_loss':values['interval']['pinball_model'],
                'baseline_interval_loss':values['interval']['pinball_baseline'],
                'interval_loss_improved':values['interval']['pinball_improved']})
    bundle_exists = (results_root/'bundle_development_metrics.json').exists()
    decision_path = results_root/'calibrated_release_decision.json'
    decision = json.loads(decision_path.read_text(encoding='utf-8')) if decision_path.exists() else {}
    confirmation_source = decision.get('prediction_confirmation_source')
    confirmation_metrics = (json.loads((results_root/confirmation_source/'metrics.json').read_text(encoding='utf-8'))
                            if confirmation_source else None)
    monthly_passed = decision.get('checks', {}).get('confirmation.predictions.monthly_frozen_validation', {}).get('passed') is True
    formal_monthly = (confirmation_metrics is not None and confirmation_metrics.get('formal_confirmation') is True
                      and confirmation_metrics.get('retrain_frequency') == 'monthly')
    labels_complete = all(metrics is not None and metrics.get('label_coverage_complete') is True
                          and metrics.get('missing_matured_label_rows') == 0
                          for metrics in (bundle_metrics, confirmation_metrics))
    model_outputs_complete = all(metrics is not None and metrics.get('model_output_coverage_complete') is True
                                 and metrics.get('model_rejected_rows') == 0
                                 for metrics in (bundle_metrics, confirmation_metrics))
    prediction_accepted = (decision.get('prediction_acceptance_passed') is True and monthly_passed
                           and formal_monthly and labels_complete and model_outputs_complete
                           and audit.get('terminal_return_coverage_complete') is True)
    mature_coverage = {}
    if bundle_metrics is not None:
        mature_coverage['development'] = {key:bundle_metrics.get(key) for key in label_fields}
    confirmation_evaluation = '未完成：缺少完整月度冻结确认记录'
    if confirmation_metrics is not None:
        static = not formal_monthly
        scope = 'static_confirmation_diagnostic' if static else 'monthly_confirmation'
        mature_coverage[scope] = {key:confirmation_metrics.get(key) for key in label_fields}
        if static:
            confirmation_evaluation = '未完成：现有结果来自静态历史诊断，不能替代完整月度冻结确认'
        elif prediction_accepted:
            confirmation_evaluation = '月度冻结确认已通过预测验收；正式发布仍需组合和数据验收'
    evaluation_scope = ('各任务的预测统计使用该任务可用且实际收益已知的成熟样本，不能证明全市场或退市终止收益覆盖。'
                        '输入不足与模型输出无效分别记录；输入具备但模型输出无效的记录计入完整能力验收。'
                        + ('预测能力已通过正式验收。' if prediction_accepted else '预测能力尚未完成正式验收。'))
    payload = {'generated_at':datetime.now().astimezone().isoformat(),'status':status,'eligible':False,
        'release_status':'未正式发布；本报告仅展示已完成的开发期评价，不构成全套验收通过',
        'development_period':'2016—2023','pending_models':[] if bundle_exists else pending,
        'pending_component_metrics':pending if bundle_exists else [],
        'prediction_metrics':rows,'forecast_coverage':coverage,
        'prediction_acceptance_passed':prediction_accepted,
        'prediction_evaluation_scope':evaluation_scope,'mature_label_coverage':mature_coverage,
        'component_mature_label_coverage':component_mature_coverage,
        'component_comparison_scope':'候选模型按同一任务在所有候选上的共同可评分子样本比较，样本日期、证券、期限一致且收益已成熟。不同任务使用各自的分母，此比较不构成全市场完整能力验收。',
        'release_decision_path':str(decision_path) if decision_path.exists() else None,
        'return_basis':FORECAST_RETURN_BASIS,'cash_dividend_accounting':CASH_DIVIDEND_ACCOUNTING,
        'portfolio_evaluation':'未完成','confirmation_evaluation':confirmation_evaluation,
        'data_as_of':feature_manifest.get('data_as_of'),
        'feature_rows':feature_manifest.get('rows'),
        'feature_eligible_rows':feature_manifest.get('eligible_rows'),
        'feature_securities':feature_manifest.get('securities'),
        'data_audit_snapshot':{key:value for key,value in audit.items() if key!='monthly'},
        'identity_audit':identity_audit,
        'training_sample':protocol['training_sampling'],'cash_policy':protocol['execution']['cash_policy']}
    write_json(results_root/'training_report.json',payload)
    def number(value, percent=False):
        if value is None:return '无可用结果'
        return f'{value:.2%}' if percent else f'{value:.6f}'
    body = []
    for row in rows:
        values=[row['model'],str(row['horizon']),number(row['rank_ic']),number(row['rank_ic_ci95_lower']),
                number(row['brier_skill'],True),number(row['calibration_error'],True),number(row['interval_coverage'],True),
                '通过' if row['interval_loss_improved'] else '未通过']
        body.append('<tr>'+''.join(f'<td>{html.escape(value)}</td>' for value in values)+'</tr>')
    coverage_body = ''.join('<tr>'+''.join(f'<td>{html.escape(str(row[key]) if row[key] is not None else "未核验")}</td>'
                           for key in ('model','total_rows','scored_rows','input_unavailable_rows','model_rejected_rows'))+'</tr>' for row in coverage)
    task_rows = []
    for row in coverage:
        for task,label in TASK_LABELS.items():
            counts = row['task_coverage'].get(task) if row['task_coverage'] is not None else None
            samples = row['task_valid_samples'].get(task) if row['task_valid_samples'] is not None else None
            numbers = [counts.get(key) if counts is not None else None
                       for key in ('total_forecast_rows','input_eligible_forecast_rows','scored_forecast_rows')]
            numbers.append(samples)
            numbers.extend(counts.get(key) if counts is not None else None
                           for key in ('model_rejected_rows','input_unavailable_rows'))
            values = [row['model'],label] + ['未核验' if value is None else f'{value:,}' for value in numbers]
            task_rows.append('<tr>'+''.join(f'<td>{html.escape(value)}</td>' for value in values)+'</tr>')
    task_coverage_table = ('<h2>各预测任务覆盖</h2>'
        '<p>“输出可用”统计能够给出该任务结果的记录；成熟有效样本还要求预测期限已经结束且实际收益已知。四项任务分别计算，缺少核验记录的项目显示“未核验”。</p>'
        '<table aria-label="预测任务覆盖"><thead><tr><th>模型</th><th>任务</th><th>全部记录</th>'
        '<th>输入可用</th><th>输出可用</th><th>成熟已知收益有效样本</th><th>模型拒绝</th><th>输入不足</th></tr></thead>'
        '<tbody>'+''.join(task_rows)+'</tbody></table>' if task_rows else '')
    component_label_rows = []
    for kind, counts in component_mature_coverage.items():
        complete = counts['label_coverage_complete']
        values = [kind] + ['未核验' if counts[key] is None else f'{counts[key]:,}'
            for key in ('matured_rows','matured_label_rows','missing_matured_label_rows')]
        values += ['未核验' if counts['matured_label_coverage'] is None else number(counts['matured_label_coverage'], True),
                   '未核验' if complete is None else ('完整' if complete is True else '不完整')]
        component_label_rows.append('<tr>'+''.join(f'<td>{html.escape(value)}</td>' for value in values)+'</tr>')
    component_label_table = ('<h2>候选模型收益标签覆盖</h2>'
        '<p>各模型的收益标签缺失与模型输出无效分别统计。</p>'
        '<table aria-label="候选模型收益标签覆盖"><thead><tr><th>模型</th><th>收益期已结束记录</th>'
        '<th>已知收益标签</th><th>缺失收益标签</th><th>标签覆盖率</th><th>标签完整性</th></tr></thead>'
        '<tbody>'+''.join(component_label_rows)+'</tbody></table>' if component_label_rows else '')
    coverage_labels = {'development':'开发期','static_confirmation_diagnostic':'静态确认诊断','monthly_confirmation':'月度确认'}
    label_statements = []
    for period,counts in mature_coverage.items():
        missing = counts['missing_matured_label_rows']
        if missing is None:
            statement = f'{coverage_labels[period]}的成熟收益标签覆盖尚未核验。'
        else:
            ratio = counts['matured_label_coverage']
            statement = f'{coverage_labels[period]}成熟收益标签缺失 {missing:,} 条，覆盖率为 {number(ratio,True)}。'
        label_statements.append(statement)
    issues = []
    for row in coverage:
        rejected = row['model_rejected_rows']
        if rejected is None:
            issues.append(f'{row["model"]}的模型输出有效性尚未按完整清单核验。')
        elif rejected:
            issues.append(f'{row["model"]}有 {rejected:,} 条输入具备但模型输出无效的记录，完整预测验收未通过。')
    unresolved = identity_audit.get('remaining_fully_unresolved_securities', audit.get('unresolved_identities', 0))
    if unresolved:issues.append(f'仍有 {unresolved} 个证券身份待核实。')
    if not audit['historical_lot_coverage_complete']:issues.append('历史交易单位证据尚未覆盖完整。')
    if audit.get('lot_source_date_review',{}).get('status')=='existing_facts_require_row_observation_date_revalidation':
        issues.append('旧版手数证据没有区分页面查询日期和股票记录日期，须按原始记录重建；完成前不能用于正式回测。')
    if not audit['corporate_action_cash_coverage_complete']:issues.append('历史公司行动及派付记录尚未覆盖完整。')
    if audit.get('terminal_return_coverage_complete') is not True:issues.append('退市、私有化等终止收益标签覆盖尚未核验完整。')
    issues.append('REIT历史数据缺口单独保留，未用普通股结果冒充REIT验证。')
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>港股模型训练报告</title>
<style>body{{font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#17212b}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #d9e1e8;text-align:left}}th{{background:#edf2f6}}.state{{padding:16px;background:#fff4d6;border-left:4px solid #d89b24}}h2{{margin-top:36px}}</style>
<h1>全港股模型训练报告</h1><p class="state">{html.escape(payload['release_status'])}</p>
<p>开发验证：2016—2023年。未完成候选：{html.escape('、'.join(pending) or '无')}。</p>
<p class="state">{html.escape(evaluation_scope)} {html.escape(confirmation_evaluation)}</p>
<h2>已有样本的预测统计</h2><table><thead><tr><th>模型</th><th>期限（交易日）</th><th>排名相关性</th><th>95%区间下限</th><th>概率相对基准改善</th><th>概率校准误差</th><th>80%区间实际覆盖</th><th>区间损失优于基准</th></tr></thead><tbody>{''.join(body)}</tbody></table>
<p>排名以20日为主；相关性95%区间下限须大于0。概率改善须大于0，校准误差不超过5%；80%预测区间实际覆盖须为75%—85%，且损失优于历史波动基准。</p>
<p>{html.escape(payload['component_comparison_scope'])}</p>
<h2>覆盖情况</h2><table><tr><th>模型</th><th>全部预测记录</th><th>四任务均可用</th><th>输入不足</th><th>模型输出无效</th></tr>{coverage_body}</table>
{task_coverage_table}
{component_label_table}
<p>{html.escape(' '.join(label_statements))}</p>
<p>每只股票、每个预测期限各算一条记录。训练每天固定随机抽取最多256只符合数据条件的证券；全市场推理和评价不做该抽样。缺失预测不补造。</p>
<h2>尚待完成的验收</h2><ul>{''.join('<li>'+html.escape(issue)+'</li>' for issue in issues)}<li>持仓回测与2024年起的冻结历史确认尚未完成。</li></ul>
<p>近期部分结果在旧实验中曾被查看。2024年起的确认属于历史检验，不能称为从未接触过的新留出样本；真正的前瞻记录需上线后另行积累。</p>
<p>回测现金按最终公告明确的派付日模拟，并非投资者真实到账证明；实际持仓建议使用账户提供的真实现金余额。</p>
<p>本轮预测衡量港元计价的来源复权价格收益，不等同于持有一股及收到分红现金的总收益。分红现金由账户回测单独记账；较早实验缓存中的“total return”文字不能作为总回报口径的证明。</p>
<p>生成时间：{html.escape(payload['generated_at'])}</p></html>'''
    (results_root/'training_report.html').write_text(document,encoding='utf-8')
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results-root',type=Path,default=RESULTS)
    parser.add_argument('--data-root',type=Path,default=DATA)
    args=parser.parse_args()
    build_report(args.results_root,args.data_root)
