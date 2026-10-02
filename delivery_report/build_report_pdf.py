#!/usr/bin/env python3
"""Rebuild the A-share research PDF from completed experiment summaries.
Read-only toward the model repository. Requires reportlab; no network calls.
"""
from pathlib import Path
import argparse, hashlib, json
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Flowable

HERE = Path(__file__).resolve().parent
A = argparse.ArgumentParser()
A.add_argument('--repo', type=Path, default=HERE.parents[1] / 'Vectaix-Finance')
A.add_argument('--input', type=Path, help='Use a previously exported report_data.json')
A.add_argument('--output', type=Path, default=HERE)
a = A.parse_args()
a.output.mkdir(parents=True, exist_ok=True)

if a.input:
    data = json.loads(a.input.read_text(encoding='utf-8'))
else:
    r=a.repo; b=r/'backtests/cn/universal'; m=r/'models/cn/universal'; d=r/'data/cn/universal'
    required=['execution_summary.json','reference_summary.json','benchmark_summary.json','pipeline_status.json']
    absent=[p for p in required if not (b/p).is_file()]
    if absent: raise SystemExit('Waiting for completed experiment artifacts: '+', '.join(absent))
    def read(p): return json.loads(p.read_text(encoding='utf-8'))
    pipeline=read(b/'pipeline_status.json')
    if pipeline.get('status') != 'complete': raise SystemExit('Pipeline is not complete')
    active=read(m/'active.json')
    paths={'manifest':d/'feature_manifest.json','audit':d/'data_audit.json','protocol':b/'protocol.json',
           'selection':b/'frozen_architecture.json','common':b/'development_common_universe.json',
           'confirmation':b/'confirmation_summary.json','execution':b/'execution_summary.json',
           'reference':b/'reference_summary.json','benchmarks':b/'benchmark_summary.json',
           'actions':b/'corporate_actions_status.json','active':m/'active.json',
           'latest':m/active['metadata_path'],'frozen':m/'frozen/linear.json',
           'readiness':b/'source_readiness.json','terminal':b/'terminal_claim_evidence.json'}
    data={k:read(p) for k,p in paths.items()}
    data['audit']={k:v for k,v in data['audit'].items() if k!='years'}
    data['actions']={k:data['actions'].get(k) for k in ['raw_rows','accepted_events','unhandled_rows','equivalent_duplicate_rows','rejection_counts','evidence_basis','issuer_verified','actual_accounting_verified','point_in_time_complete','asof_date','collection_status','selected_canonical_count','source_endpoint','source_url','limitations']}
    # Retain metric provenance and stable content hashes, without credentials or logs.
    data['sources']={k:{'path':str(p.relative_to(r)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()}
    data['pipeline_complete']=True
    (a.output/'report_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

manifest=data['manifest']; conf=data['confirmation']['horizons']; common=data['common']
act=data['execution']; refs=data['reference']; bench=data['benchmarks']; actions=data['actions']; latest=data['latest']; frozen=data['frozen']
frozen_position_count = (act['frozen_corporate_action_positions_count']
    if 'frozen_corporate_action_positions_count' in act
    else len(act['frozen_corporate_action_positions']))
assert data.get('pipeline_complete') is True
assert all(v['brier_skill'] < 0 for v in conf.values()), 'Review probability narrative against new results'
assert all(v['pinball'] > v['baseline_pinball'] for v in conf.values()), 'Review interval narrative against new results'
pdfmetrics.registerFont(TTFont('AshareReportSans', str(HERE/'AshareReportSans-Regular.ttf')))
FONT='AshareReportSans'; INK=colors.HexColor('#172027'); GRAY=colors.HexColor('#56616B'); BLUE=colors.HexColor('#215C91'); LIGHT=colors.HexColor('#E7EDF1')
styles={
 'title':ParagraphStyle('title',fontName=FONT,fontSize=23,leading=30,textColor=colors.black,spaceAfter=12,wordWrap='CJK'),
 'h1':ParagraphStyle('h1',fontName=FONT,fontSize=16,leading=22,textColor=colors.black,spaceAfter=10,wordWrap='CJK'),
 'h2':ParagraphStyle('h2',fontName=FONT,fontSize=12,leading=17,textColor=colors.black,spaceBefore=10,spaceAfter=6,wordWrap='CJK'),
 'body':ParagraphStyle('body',fontName=FONT,fontSize=10.2,leading=15.8,textColor=INK,spaceAfter=7,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName=FONT,fontSize=8.6,leading=12.5,textColor=GRAY,spaceAfter=6,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName=FONT,fontSize=9.1,leading=13,textColor=INK,wordWrap='CJK'),
 'head':ParagraphStyle('head',fontName=FONT,fontSize=9.1,leading=13,textColor=colors.white,wordWrap='CJK'),
 'code':ParagraphStyle('code',fontName='Courier',fontSize=8.1,leading=12,textColor=INK,spaceAfter=4,wordWrap='CJK'),
}
story=[]; markdown=[]
def p(s,style='body'):
    story.append(Paragraph(escape(str(s)).replace('\n','<br/>'),styles[style]));markdown.append(str(s)+'\n')
def title(s): p(s,'title')
def h1(s): p(s,'h1')
def h2(s): p(s,'h2')
def gap(n=5): story.append(Spacer(1,n))
def page(): story.append(PageBreak());markdown.append('\n---\n')
def pct(x,n=2): return '不可用' if x is None else f'{x:.{n}%}'
def num(x,n=4): return '不可用' if x is None else f'{x:.{n}f}'
def date(x):return str(x)[:10]
def table(head,rows,widths):
    tab=Table([[Paragraph(escape(str(t)),styles['head']) for t in head]]+[[Paragraph(escape(str(t)),styles['cell']) for t in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
    tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BLUE),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F4F7F9')]),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.append(tab);gap(8)
    markdown.extend([' | '.join(map(str,head)), ' | '.join(['---']*len(head))]+[' | '.join(map(str,row)) for row in rows]+[''])

class ICChart(Flowable):
    def __init__(self):Flowable.__init__(self);self.width=528;self.height=157
    def draw(self):
        c=self.canv;x0=57;x1=388;w=x1-x0;y0=27;scale=lambda v:x0+w*v/.27
        c.setFont(FONT,8.5);c.setStrokeColor(colors.HexColor('#D6DDE1'))
        for tick in [0,.05,.10,.15,.20,.25]:
            x=scale(tick);c.setLineWidth(.4);c.line(x,y0,x,148);c.setFillColor(GRAY);c.drawCentredString(x,12,f'{tick:.2f}')
        c.setFillColor(GRAY);c.drawString(416,141,'均值 [95%区间]')
        for i,key in enumerate(['1','5','20','60']):
            v=conf[key];y=130-i*29;lo,hi,mean=[v[k] for k in ['ic_ci_lower','ic_ci_upper','rank_ic_mean']]
            c.setFillColor(INK);c.drawString(0,y-3,key+' 日');c.setStrokeColor(BLUE);c.setLineWidth(2)
            c.line(scale(lo),y,scale(hi),y);c.line(scale(lo),y-4,scale(lo),y+4);c.line(scale(hi),y-4,scale(hi),y+4)
            c.setFillColor(BLUE);c.circle(scale(mean),y,3.4,fill=1,stroke=0);c.setFillColor(INK);c.setFont(FONT,8.3)
            c.drawString(401,y-3,f'{mean:.3f} [{lo:.3f}, {hi:.3f}]')

# PAGE 1
p('VECTAIX FINANCE  /  A SHARE RESEARCH','small')
title('A股模型训练与留出验证报告')
p(f"数据截止 {manifest['data_as_of']}；人民币口径；回溯研究",'small')
p('四类候选已完成真实数据比较，2024 年同池选择结果为 Ridge。冻结模型在 2025 年起的独立确认区间显示正向排序信息，但四个期限的上涨概率均未优于历史频率基线，当前快照仍未通过收益与执行准入。')
h2('数据规模与实验边界')
p(f"2016-01-04 至 {manifest['data_as_of']}，共 {data['readiness']['endpoints']['daily']['completed_dates']:,} 个交易日；{manifest['securities']:,} 个证券身份、{manifest['rows']:,} 行带日期因子记录，其中 {manifest['eligible_rows']:,} 行满足基础可用条件。保留已退市身份与不可用状态，缺价不补为零收益。[1]")
p('完整特征库含 37 项价格、成交、规模和市场状态变量；每天最多确定性抽样 256 只用于拟合，全可用历史证券用于推断。四个期限为 1、5、20、60 个交易日。')
h2('固定时间协议')
table(['阶段','数据区间与用途'],[
 ['拟合与校准','2016-2022 拟合；2023 单独校准。训练标签结束日严格早于校准起点，校准标签截至 2023-12-29'],
 ['开发选型','2024 信号及已在 2024-12-31 前成熟的标签；固定以 20 日平均日 RankIC 选型'],
 ['独立确认','2025 起至数据截止日，只评估冻结模型；确认集不参与架构选择或调参'],
 ['最新重训','另建截至 2026-09-30 的研究快照；其历史拟合与校准区间不能当作样本外验证'],
],[78,450])
h2('2024 年四类候选的同池选择')
table(['候选','输入与结构','20 日 RankIC'],[
 ['固定因子','4 项预设权重；概率及区间另行校准',num(common['factor']['rank_ic_mean'])],
 ['Ridge  已选','10 项基础特征 + 期限；alpha = 10',num(common['linear']['rank_ic_mean'])],
 ['Small LightGBM','37 项 + 期限；15 叶、160 轮',num(common['lightgbm_small']['rank_ic_mean'])],
 ['Large LightGBM','37 项 + 期限；31 叶、240 轮',num(common['lightgbm_large']['rank_ic_mean'])],
],[108,326,94])
p('比较限定在四候选当日均可评分的证券交集：222 个 IC 日期、1,175,479 条可评分且已观察到成熟结果的记录。证券池形成不依赖未来收益是否可见。这里的样本量不是独立观测数。[2]','small')

# PAGE 2
page();h1('冻结模型的留出预测质量')
p('确认区间为 2025 年起至 2026-09-30。每日按横截面计算 Spearman 相关，再对日期取均值；每个有效横截面至少 20 只证券。评价仅使用截止日已经成熟、已观察到结果且对应任务可用的记录。[3]')
h2('各期限平均日 RankIC 及 95% 区间')
story.append(ICChart());gap(7)
p('区间为 60 交易日循环块自助法，重复 2,000 次。它保留部分日期依赖结构；重叠期限、同日证券相关和来源修订仍限制解释，不能按数百万证券行计算独立样本置信度。','small')
p('1 / 5 / 20 / 60 日可评分且观察到成熟结果的记录分别为 '+ ' / '.join(f"{v['score_available']:,}" for v in conf.values())+' 条；有效 IC 日期分别为 '+ ' / '.join(str(v['ic_dates']) for v in conf.values())+' 日。','small')
p(f"20 日排序的点估计为 {conf['20']['rank_ic_mean']:.4f}，95% 区间 [{conf['20']['ic_ci_lower']:.4f}, {conf['20']['ic_ci_upper']:.4f}]。这是当前历史确认区间中的横截面排序证据，不等于可交易收益、未来稳定性或因果作用。")
h2('绝对概率与收益区间仍有明显限制')
table(['期限','方向准确率','Brier skill','q10-q90 覆盖率','ECE'],[[h+' 日',pct(v['direction_accuracy']),pct(v['brier_skill']),pct(v['interval_coverage']),pct(v['ece'])] for h,v in conf.items()],[53,119,118,138,100])
p('Brier skill = 1 - 模型 Brier / 基线 Brier；负值表示劣于仅用合法历史训练标签频率的基线。四个期限均为负，因此上涨概率不能解释为已验证的成功率。')
p(f"20 日平均预测上涨概率为 {conf['20']['probability_bins'][4]['mean_probability']:.2%}，实际上涨占比为 {conf['20']['actual_up_ratio']:.2%}；该期限全部可评估概率落在 40%-50% 分箱。q10-q90 的名义覆盖为 80%，20 日实际覆盖为 {conf['20']['interval_coverage']:.2%}。")
p('ECE 使用 10 个等宽概率分箱。四期限区间 pinball 损失也均略高于基线。预测、概率与区间可用于继续诊断，不能因为排序相关为正就宣称这些任务全部通过。最新重训模型需等待新的未来数据单独验证。','small')

# PAGE 3
page();h1('组合回放与成本口径')
p('下表均来自冻结模型确认期回放；模拟初始资金为人民币 1,000,000 元，仅为研究参数，不是用户真实资产。每月末按 20 日评分选前 30 只，下一交易日执行，受可交易性与流动性约束。这些收益只对应固定前 30 规则，不能用于验证 API 持仓建议中的效用和风险优化器。[4]')
h2('复权研究单位回放')
rr=refs.get('fee_0.0025',next(iter(refs.values())))
p(f"日期 {date(rr['start_date'])} 至 {date(rr['end_date'])}。目标股票权重 {pct(rr['equity_weight'],0)}，参与率上限 {pct(rr['participation_rate'],0)}；使用下一交易日复权收盘价及原始收盘涨跌停检查。厂商复权研究单位允许分数份额，费用为每边统一费率及滑点压力。")
table(['每边成本','累计收益','年化收益','日度最大回撤'],[[pct(v['fee_per_side']),pct(v.get('total_return')),pct(v.get('cagr')),pct(v.get('max_drawdown_daily'))] for v in refs.values()],[110,139,139,140])
if any(v.get('total_return') is None for v in refs.values()):
    p('终值缺口涉及已退市的 600636.SH。承接证券 400297（国化5）仍代表持有人权益，但本实验未核实其 9 月 30 日估值或可执行退出；不按零回收处理。三档成本下均不能确认组合收益或是否跑赢基准。[6]','small')
p('该口径不等于原始股价加真实股数和现金分红账户，不核算个人精确税款、券商最低佣金或整手差异。年化值按实际自然日换算；历史回放不是实盘表现。','small')
h2('同日期网格的价格指数基准')
names={'000001.SH':'上证综指','000300.SH':'沪深300','000905.SH':'中证500','000852.SH':'中证1000'}
table(['基准','累计收益','年化收益','最大回撤'],[[names.get(k,k)+' '+k,pct(v.get('total_return')),pct(v.get('cagr')),pct(v.get('max_drawdown'))] for k,v in bench.items()],[180,116,116,116])
p('基准为价格指数，不含股息及交易费用；与研究单位组合口径不完全相同，差值不能直接解释为税后超额收益。','small')
h2('真实股数账户的可核实程度')
p(f"日期 {date(act['start_date'])} 至 {date(act['end_date'])}。回放状态：{act['status']}；估值有效：{str(act['valuation_valid']).lower()}；执行有效：{str(act['execution_valid']).lower()}。累计收益 {pct(act.get('total_return'))}，最大回撤 {pct(act.get('max_drawdown'))}。")
p(f"已记录 {act['trade_count']:,} 笔模拟成交、费用 {act['fees']:,.2f} 元；现金股息模拟入账 {act['cash_dividends_paid']:,.2f} 元，另预留税款 {act['dividend_tax_reserve']:,.2f} 元。期末仍有 {frozen_position_count} 个企业行动持仓冻结。原始价格盯市或参考净值不能替代已核实账户结果。")
p('真实股数路径使用原始价格、整股与最低起点、T+1、日涨跌停、停牌、日期相关费用及成交额约束。股息证据为厂商最终实施计划，经日期和数值校验，未逐条核对发行人原文；现金股息先预留 20% 税额，个人精确税后到账未验证。','small')

# PAGE 4
page();h1('来源限制与可复现性')
h2('应保留的限制')
p(f"来源为 Tushare 历史接口缓存，记录端点、日期参数、抓取时间、行数与 SHA256。数据审计登记 {data['audit']['securities']:,} 条主表证券身份，最终日期化因子面板涵盖 {manifest['securities']:,} 个身份；两者分母不同。[1]")
p('发现 11 处未核实的价格与复权参考不连续点，均在 2025 年以前。原始记录保留，滚动窗口从断点后重新积累，标签不跨断点；隔离不表示来源已被修复。')
p('三个 2022 年北交所转板旧身份的历史行情不可用：翰博高新、泰祥股份、观典防务。转板终止日与新板上市日分别留痕，不虚构中间成交，也不把当前代码的历史直接替代旧身份。[5]')
p('北交所独立日历端点为空，本次使用上交所交易日历并与已观察到的北交所报价交叉核对。当期名称和行业仅展示，不回填作历史特征；历史更名覆盖与厂商历史修订、退市终局结果仍未获得独立完整性认证。')
p('本次训练实现 37 项基础价格量能方法及独立预测、解释、组合建议与审计流程。财报版本完整性认证、SHIBOR/LPR 因子、发行人核实的要约收购因子仍为后续扩展。港股源码、展示与命名空间保留；原港股完整权重及训练数据未提供，无法声明权重级复制。')
h2('模型身份与验收状态')
p(f"冻结确认模型：{frozen['model_version']}\n最新重训模型：{latest['model_version']}\n最新拟合使用 {latest['sample_counts']['train']['used']:,} 条跨期限记录，校准使用 {latest['sample_counts']['calibration']['used']:,} 条；不作为样本外业绩。",'small')
p('快照状态 research_ready=true，eligible=false，execution_validated=false。可运行研究接口及查看预测，不构成投资建议、真实下单或收益承诺。', 'small')
h2('复现入口')
p('在仓库根目录、具备既有完整缓存及依赖的 Python 环境中运行：','small')
p('python -m ashare_quant.pipeline --skip-collection','code')
p('python -m ashare_quant.evidence','code')
p('python -m pytest -q tests/test_ashare_*.py','code')
p('python -m ashare_quant.api --host 127.0.0.1 --port 8001','code')
p('服务要求自行安全设置 ASHARE_QUANT_API_KEY。新采集需自行配置已授权的凭据助手；交付物不包含密钥。接口返回的排序与建议均不自动发送交易。','small')
h2('核查所用证据')
p('[1] data/cn/universal/feature_manifest.json、data_audit.json；backtests/cn/universal/source_readiness.json\n[2] backtests/cn/universal/protocol.json、frozen_architecture.json、development_common_universe.json\n[3] backtests/cn/universal/confirmation_summary.json；models/cn/universal/frozen/linear.json\n[4] backtests/cn/universal/execution_summary.json、reference_summary.json、benchmark_summary.json；execution/ 与 reference/ 内的流水\n[5] docs/ashare/board_transfer_coverage.csv、continuity_evidence.csv；范围说明见 docs/ashare/README.md\n[6] backtests/cn/universal/terminal_claim_evidence.json：交易所、发行人及券商原始链接；转让交易起点另引发行人公告镜像并披露原件限制', 'small')
p('本报告同目录的 report_data.json 保留所用汇总值与源文件哈希；build_report_pdf.py 可离线重建 PDF。','small')

def footer(c,doc):
    c.saveState();c.setFont(FONT,8);c.setFillColor(GRAY);c.drawString(42,24,'A股真实数据研究  |  数据截止 2026-09-30');c.drawRightString(570,24,f'{doc.page} / 4');c.restoreState()
path=a.output/'ashare_training_report.zh-CN.pdf'
doc=SimpleDocTemplate(str(path),pagesize=(612,792),rightMargin=42,leftMargin=42,topMargin=36,bottomMargin=42,title='A股模型训练与留出验证报告',author='Vectaix Finance',subject='A-share chronological modeling experiment and limitations')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(a.output/'ashare_training_report.zh-CN.md').write_text('\n'.join(markdown)+'\n',encoding='utf-8')
print(path)
