<div align="center">

# Vectaix Finance

独立美股、A股与港股选股模型，包含训练权重、历史验证和交易复盘。

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ridge](https://img.shields.io/badge/Model-Ridge-546E7A)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-00866F)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![GitHub stars](https://img.shields.io/github/stars/Asher-S-Wu/Vectaix-Finance)](https://github.com/Asher-S-Wu/Vectaix-Finance/stargazers)

[English](README.md) · **简体中文** · [日本語](README.ja.md) · [한국어](README.ko.md)

</div>

Vectaix Finance 分别提供人民币口径的A股研究管线和港元口径的港股研究管线。A股公开版包含已训练的 Ridge、因子评分和 LightGBM 模型；港股部分记录月度选股策略及逐笔交易复盘。

[独立美股模型](#us) · [A股模型](#ashare) · [港股回测](#hong-kong) · [本地运行](#本地运行)

<a id="us"></a>

## 独立美股模型：实验2

**补齐历史数据并训练排序权重后，复用近期区间的对比有所改善，但仍未证明可靠的样本外优势。** 冻结的 `ridge_rank_5y` 在四个期限的近期 IC 点估计均为正，但95%区间全部跨零；概率技能全部为负，按时间顺序嵌套选型的外层平均 IC 仅 **+0.0009**。模型仍不具备实盘资格。

沿用 **2015-06-30** 的 OEF 固定股票池，名单于 **2015-09-02** 发布，共 **101个原始股票类别**。此前缺失的 **12个** 原始身份均已补入有明确边界的历史，增加 **15,284条** 有效报价。101/101拥有部分历史，不代表完整时点股票池或完整终止财富路径。FOXA 的来源截至 **2018-03-27**，早于2019-03-19实际终止日；DD 从 **2015-07-02** 开始，此前非现金分拆未建模。无法支持的并购、现金结算和 CVR 结果继续标为未知。

### 滚动开发与按时间顺序选型

![美股v2六个预先声明候选在2020–2024年的同股票同日期 IC 与等权年均值](docs/assets/us/v2/rolling_selection.png)

六个预先声明候选包括固定因子、12–1动量，以及3年/5年窗口的独立 Ridge 和 LightGBM 排序模型，学习型排序模型使用固定的24个特征及同日居中的收益排名目标；基线保留预先声明的评分公式。完成 **30次候选折次拟合**；每折训练早于上一校准年，在2020–2024各年相同股票日期观测上评估。5年 Ridge 以 **+0.0271** 的年均 IC 入选，5年中3年为正；这是用于选出模型的开发统计，不是独立检验。

![嵌套外层仅由更早内层年份选型，外层 IC 为-0.0175、+0.0275、-0.0075](docs/assets/us/v2/nested_selection.png)

2022年模型仅用2020–2021年内层结果选择，2023、2024年也仅使用各自之前的年份。三次均选择5年 Ridge；外层 IC 为 **−0.0175 / +0.0275 / −0.0075**，平均 **+0.0009**，仅1/3年份为正。这加强了时间隔离，仍受回溯数据版本和固定股票池限制。

### REUSED DIAGNOSTIC：2025–2026是复用诊断，不是新留出集

![美股v2冻结Ridge在复用近期区间的Rank IC及95%块自助法区间](docs/assets/us/v2/reused_diagnostic_ic.png)

最终四个期限分别训练至2023年、以2024年校准，在 **2025-01-01–2026-09-30** 保持冻结。实验1已查看这个时期，因此不能将其重新视为实验2的全新确认集。

| 期限（交易日） | 平均 Rank IC | 95%区间 | Brier skill | 收益区间覆盖率 |
| --- | ---: | --- | ---: | ---: |
| 1 | +0.0083 | [−0.0042, +0.0211] | −0.012% | 79.40% |
| 5 | +0.0158 | [−0.0131, +0.0438] | −0.026% | 81.71% |
| 20 | +0.0242 | [−0.0255, +0.0696] | −0.286% | 82.37% |
| 60 | +0.0705 | [−0.0040, +0.1415] | −0.677% | 83.63% |

在 **37,113条相同成熟股票日期观测、417个 IC 日期** 上，20日 IC 从v1的 **−0.0772** 变为v2的 **+0.0233**，配对变化 **+0.1005**，95%区间 **[+0.0211, +0.1788]**。上表的+0.0242额外包含已恢复的 WBA 观测，因此口径不同。模型和训练数据同时变化，且区间已经看过，这一对比不能识别改善原因，也不是全新独立确认。

![美股v2的概率Brier skill、q10–q90区间覆盖率及成熟观测分母](docs/assets/us/v2/calibration_checks.png)

IC区间使用2,000次60交易日循环块自助抽样。Brier skill相对历史上涨频率基线，负值代表误差更高；收益区间目标覆盖率为80%。重叠观测不是独立样本。**2026-09-30最新重训模型尚无样本外评估**。

**全区间组合绩效无法确认。** 主策略、费用压力测试与101槽位股票池在2025-08-27之后仍持有估值未解决的 WBA，出现 **274个估值缺口交易日**。总收益、年化收益和最大回撤保持空值；陈旧参考价格不作为已验证收益，因此不展示声称绩效成立的净值曲线。

[方法与边界](docs/us/v2/README.md) · [恢复历史图](docs/assets/us/v2/source_recovery.png) · [滚动比较CSV](docs/showcase/us/v2/rolling_selection.csv) · [嵌套选型CSV](docs/showcase/us/v2/nested_selection.csv) · [复用诊断CSV](docs/showcase/us/v2/reused_diagnostic.csv) · [配对比较](docs/showcase/us/v2/paired_recent.csv) · [组合状态](docs/showcase/us/v2/replay_status.csv) · [哈希与口径](docs/showcase/us/v2/summary.json)

无需下载行情或训练即可重建v2的五张图与汇总CSV：`python -m scripts.build_us_v2_showcase`

初始实验完整保留在下方。

<a id="us-v1"></a>

## 独立美股模型：实验1

**冻结确认结果未能证明排序优势。** 四个期限的平均 Rank IC 均为负，20日和60日的95%区间整体低于零；全部期限的收益区间覆盖率均低于80%。该模型尚未通过实盘使用验证。

本次公开数据初始实验的独立美元模型采用 **2015-06-30** 的 OEF 持仓作为固定历史股票池；该名单于 **2015-09-02** 发布，包含 **101个原始股票类别**。Yahoo Finance 公开历史覆盖2014-01-02至2026-09-30。经有限的供应商名称与历史区间核对，接受 **89条证券历史**；另有 **12个原始证券身份** 被排除或无法取得，包括5条未解决的重组沿革、5条缺失历史和2个被复用的代码。预测总数保留全部原始身份。这是范围有限、使用当前版本历史数据的回溯研究，不能视为无幸存者偏差或全美股市场的估计。AKShare 仅用于辅助核查；复权历史不完整及代码复用等问题使其未被用作训练源。

美股管线独立训练固定因子评分、Ridge、小型和大型 LightGBM，不复用A股或港股的已训练权重。2016–2022年用于训练，2023年校准，2024年选型；随后仅对选中的模型冻结参数，在2025-01-01至2026-09-30确认区间评估。

### 美股冻结模型确认结果

![美股冻结 Factor score 的平均日 Rank IC 与95%的60交易日块自助法区间](docs/assets/us/confirmation_ic.png)

| 期限（交易日） | 平均日 Rank IC | 95%区间 | 上涨概率 Brier skill | 收益区间覆盖率 |
| --- | ---: | --- | ---: | ---: |
| 1 | -0.0095 | [-0.0254, 0.0064] | -0.02% | 72.95% |
| 5 | -0.0271 | [-0.0562, 0.0019] | +0.08% | 73.82% |
| 20 | -0.0772 | [-0.1233, -0.0200] | +0.06% | 74.67% |
| 60 | -0.0989 | [-0.1562, -0.0426] | -2.89% | 69.57% |

Rank IC 是每日横截面中评分与后续复权价格收益的 Spearman 相关系数。95%区间采用60交易日循环块自助法，重复2,000次并保留缺失日期位置。指标仅基于已观测且到期的结果；历史身份缺口和供应商复权口径限制了结论的适用范围。

### 美股选型与校准检查

![四个美股候选模型在相同2024年股票日期观测与 IC 日期上的比较](docs/assets/us/model_selection.png)

所选因子评分使用预设排序权重，概率与收益区间校准单独在美股数据上拟合。

按20交易日平均日 Rank IC 选出 **Factor score**，选型统计量为 **0.0292**。四个候选模型使用相同的 **232个 IC 日期** 和 **20,648条已到期股票日期观测**；确认数据不参与选型或调参。

![美股冻结模型各期限概率 Brier skill，相对历史上涨频率基线](docs/assets/us/probability_skill.png)

零线代表历史上涨频率基线；负值表示模型的 Brier 误差更高。排序 IC 为正本身并不证明概率预测优于基线。

![美股 q10–q90收益区间的实际覆盖率与80%目标](docs/assets/us/interval_coverage.png)

q10–q90收益区间目标覆盖率为80%。分母仅包含已到期且相应任务可用的观测；重叠的股票日期观测不能视为独立样本。另行保存的2026-09-30最新重训模型**尚无样本外评估**。执行未验证、资格状态为 false；这些预测诊断不代表已验证的可投资组合收益，也不保证未来表现。

### 有条件的美股研究模拟

![实验1复权研究单位模拟：选中的因子模型落后于两条参照](docs/assets/us/conditional_replay.png)

冻结因子策略的有条件研究模拟累计收益为 **+16.01%**，低于固定101槽位股票池的 **+27.40%** 和 SPY 复权价格代理的 **+30.84%**。单边成本从15提高至30个基点后，模型结果降至 **+10.91%**。模拟采用月度前10名、下一交易日收盘成交、95%投资目标；缺失股票池槽位保留现金。模型与股票池的平均股票敞口分别为94.70%和83.83%，两者并非等敞口比较。这些美元复权研究单位及收益仅作研究展示，执行与实际持股公司行动账本均未验证。

[模拟曲线 CSV](docs/showcase/us/conditional_replay.csv) · [模拟指标 CSV](docs/showcase/us/conditional_replay_summary.csv) · [模拟假设](backtests/us/oef2015/replay_summary.json)

[美股来源与方法](docs/us/README.md) · [来源审计](backtests/us/oef2015/source_audit.json) · [冻结候选模型](models/us/oef2015/frozen/) · [最新重训](models/us/oef2015/latest/) · [选型 CSV](docs/showcase/us/model_selection.csv) · [确认结果 CSV](docs/showcase/us/confirmation.csv) · [图表口径与哈希](docs/showcase/us/summary.json)

仅使用汇总证据重建这五张图及 CSV，无需下载行情或重新训练：`python -m scripts.build_us_showcase`

<a id="ashare"></a>

## A股模型

A股管线计算37项价格、成交量、市值和市场状态因子，在同一可评分股票池内比较固定因子评分、Ridge、小型和大型 LightGBM。2024年开发集按20个交易日的平均日 Rank IC 选出 Ridge，输入为10项预设基础特征加 `log_horizon`。

模型使用2016至2022年数据拟合，2023年单独校准。选型后冻结参数，在2025年起至2026-09-30的确认区间评估；确认数据不参与选型或调参。冻结模型的结果如下：

![冻结 Ridge 在确认区间的1、5、20、60交易日平均日 Rank IC 及95%块自助法区间](docs/assets/cn/confirmation_ic.png)

| 期限（交易日） | 平均日 Rank IC | 95%区间 | 上涨概率 Brier skill |
| --- | ---: | --- | ---: |
| 1 | 0.0608 | [0.0455, 0.0758] | -0.35% |
| 5 | 0.0822 | [0.0503, 0.1145] | -0.54% |
| 20 | 0.1111 | [0.0608, 0.1626] | -0.22% |
| 60 | 0.1610 | [0.0812, 0.2450] | -1.15% |

Rank IC 是每日横截面中评分与后续收益的 Spearman 相关系数。区间采用60个交易日循环块自助法，重复2,000次，反映这段历史中的排序质量。四个期限的概率 Brier skill 均为负，表现低于历史频率基线。另行发布的2026-09-30最新 Ridge 重训权重用于当前输入研究，尚无独立样本外评估。

部分公司行动和证券终局估值仍未核实，组合收益、CAGR和最大回撤尚无法确认。快照记录为 `eligible=false`、`execution_validated=false`。历史结果不保证未来表现。

仓库包含四个[冻结候选模型](models/cn/universal/frozen/)和[最新 Ridge 权重及元数据](models/cn/universal/snapshots/cn-linear-latest-20260930-v1/)。厂商原始输入和个股预测快照需使用者自行取得授权 Tushare 数据后准备。公开版没有可直接服务的完整快照；克隆后可以检查模型、运行合成测试，真实股票排名需要完整的本地输入。

[公开版说明](ASHARE_RELEASE.zh-CN.md) · [中文研究报告 PDF](delivery_report/ashare_training_report.zh-CN.pdf) · [方法与复现](docs/ashare/README.md) · [确认区间指标](backtests/cn/universal/confirmation_summary.json) · [A股源码](ashare_quant/)

### 开发集模型选择

![2024年共同股票池20日 IC：Ridge 0.0556、大型 LightGBM 0.0264、小型 LightGBM 0.0016、因子评分 -0.0128](docs/assets/cn/model_selection.png)

Ridge 在2024年选型比较中的平均日 Rank IC 为0.0556。四个候选模型使用相同的222个 IC 日期和1,175,479条已到期股票日期观测；图中展示的是当时用于冻结选型的开发集统计量。

[模型比较 CSV](docs/showcase/cn/model_selection.csv) · [冻结选型记录](backtests/cn/universal/frozen_architecture.json)

### 概率与收益区间检查

![冻结 Ridge 的1、5、20、60日概率 Brier skill 分别为 -0.35%、-0.54%、-0.22%、-1.15%](docs/assets/cn/probability_skill.png)

零线代表历史上涨频率基线。负值表示模型概率的 Brier 误差高于该基线；排序 IC 为正，并不表示概率预测已经优于基线。

![q10–q90收益区间实际覆盖率为80.02%、80.32%、78.28%、77.26%，目标为80%](docs/assets/cn/interval_coverage.png)

q10–q90收益预测区间的目标覆盖率为80%。1、5、20、60个交易日的实际覆盖率分别为80.02%、80.32%、78.28%、77.26%。分母仅包含收益已到期且区间可用的观测；存在重叠的股票日期观测不能当作独立样本。这些图对应冻结模型的确认结果，最新重训模型尚未完成样本外评估。

[确认指标 CSV](docs/showcase/cn/confirmation.csv) · [图表来源与口径](docs/showcase/cn/summary.json)

<a id="hong-kong"></a>

## 港股回测

港股管线将37项量价与市场特征合成选股分数，每月选取排名前30的港股。在2024-02-01～2026-09-16的历史回测中，小型 LightGBM 扣费年化收益为 **35.56%**，100万港元增至 **222.09万港元**。

![历史回测：模型净值、恒生指数、恒生科技指数及回撤](docs/assets/performance.png)

### 收益与指数对比

区间为 **2024-02-01～2026-09-16**，共644个交易日。四个模型使用同一可评分股票池，月末按20日预测分数选前30只，下一交易日收盘模拟成交。初始资金100万港元，单边费用0.25%，成交参与率上限1%。

| 模型／基准 | 年化收益 | 累计收益 | 最大回撤 | 盈利持仓占比 |
| --- | ---: | ---: | ---: | ---: |
| **小型 LightGBM** | **35.56%** | **122.09%** | -20.10% | 49.16% |
| 大型 LightGBM | 32.28% | 108.30% | -15.49% | 49.21% |
| 线性模型 | 13.37% | 38.98% | -10.36% | 45.32% |
| 因子评分 | 5.83% | 16.02% | -21.38% | 51.78% |
| 恒生指数 | 19.27% | 58.77% | -19.95% | — |
| 恒生科技指数 | 14.02% | 41.08% | -36.32% | — |

小型模型年化比同期恒指高 **16.28个百分点**，累计收益高 **63.33个百分点**。

盈利持仓占比按**已平仓持仓扣费后是否盈利**统计：小型模型为438／891，即49.16%。在单边费用0.50%的压力情景下，年化收益为31.41%。模型收益已扣模拟交易费，基准采用不含分红、未扣费的价格指数。

[逐日净值](docs/showcase/performance_curve.csv) · [四模型与两档费用结果](docs/showcase/model_comparison.csv) · [日期、参数和数据来源](docs/showcase/summary.json)

### 因子如何参与选股

展示模型使用37项输入，包括价格偏离、动量、波动率、成交额、市值和市场宽度。它把这些信息合成排序分数，再按分数选股。

![小型模型前10项输入的训练期重要性](docs/assets/factor_importance.png)

距离252日低点、60日价格偏离分别占全部输入分裂增益的16.40%和15.81%。这张图展示模型在训练中对各项输入的使用程度，组合收益见上方净值图。

[全部输入的重要性](docs/showcase/factor_importance.csv) · [特征计算](hk_quant/features.py) · [模型实现](hk_quant/models.py)

### 一次完整交易复盘

这次历史模拟跟踪**第一次建仓的全部30只股票**，从月末选股、次日买入到分批退出，保留85笔成交和完整资金账目。

| 日期 | 发生了什么 |
| --- | --- |
| 2024-01-31 | 月末评分，选出前30只股票。 |
| 2024-02-01 | 下一交易日收盘买入，按成交额限制缩减部分仓位；含费用投入716,281.52港元。 |
| 2024-03-01 | 到计划退出日，开始卖出这批持仓。 |
| 2024-03-04～03-14 | 部分股票因成交额限制分批卖出；3月14日完成最后一笔。 |

#### 建仓后的账户

股票市值714,495.29港元，现金283,718.48港元。扣除买入费用后，总资产998,213.76港元，现金占28.42%。图中列出最大的4笔持仓，其余26只合并展示；完整明细可下载。

![2024年2月1日建仓后的持仓与现金比例](docs/assets/holdings.png)

[当期选股排名](docs/showcase/first_cycle_selection.csv) · [30只股票的持仓和权重](docs/showcase/first_cycle_holdings.csv)

#### K线上的买卖点

K线对应当期评分前两名：08619.HK（扣费收益−59.91%）与02171.HK（+61.45%）。买卖点覆盖每次模拟成交，包括08619.HK的两次分批卖出。

![评分前两名股票的复权K线与模拟买卖点](docs/assets/trade_candles.png)

| 日期 | 股票 | 方向 | 复权单位 | 复权价格 | 成交额（港元） | 费用（港元） |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2024-02-01 | 08619.HK | 买入 | 4,885.528685 | 2.307179 | 11,271.79 | 28.18 |
| 2024-02-01 | 02171.HK | 买入 | 7,742.082702 | 4.080000 | 31,587.70 | 78.97 |
| 2024-03-01 | 08619.HK | 卖出 | 1,951.329001 | 1.118632 | 2,182.82 | 5.46 |
| 2024-03-01 | 02171.HK | 卖出 | 7,742.082702 | 6.620000 | 51,252.59 | 128.13 |
| 2024-03-04 | 08619.HK | 卖出 | 2,934.199683 | 0.804017 | 2,359.15 | 5.90 |

成交表使用可带小数的复权研究单位。[全部85笔模拟成交](docs/showcase/first_cycle_trades.csv)记录了日期、股票、方向、数量、价格、金额、费用及本期现金变化。

#### 这一期赚了多少

30个持仓中18个盈利，30笔买入、55笔卖出，共85笔成交。买入总成本716,281.52港元，卖出净收入781,194.89港元，**净利润64,913.36港元**，已扣除双边费用3,744.12港元。投入成本收益率为9.06%，对初始100万港元资金贡献6.49%。

![第一期全部30个持仓的扣费盈亏](docs/assets/first_cycle_pnl.png)

[全部30个持仓的成本、收入与收益率](docs/showcase/first_cycle_positions.csv)

`cohort_cash_after` 是这批30个持仓的独立现金账：从100万港元到1,064,913.36港元，不含后续调仓的新持仓。完整策略的账户资产见逐日净值表。

### 回测口径

- 每期使用卖出后的可用现金的95%，分成30个固定名额。买入同时受信号日20日平均成交额和成交日成交额的1%限制；未买足的部分留作现金，未卖完的持仓继续尝试退出。
- 股票行情来自 Tushare `hk_daily_adj`、`hk_adjfactor`，指数来自 `index_global`。回测使用复权研究单位，没有逐笔还原历史整手、分红到账日和真实撮合。
- 年化按实际日历天数计算复合收益；最大回撤来自每日账户净值。期末资产包含未卖出持仓的参考市值，小型模型有1只未退出。参考估值与可成交价格有别，`execution_validated` 为 `false`。
- 展示模型训练截至2023-12-29，使用冻结参数回放。该区间已用于候选模型比较，属于历史对比回测，不作为独立留出验证。

## 本地运行

建议使用 Python 3.12 和仓库锁定的依赖版本。仓库包含源码、测试、A股训练权重与汇总报告，以及港股展示图和案例CSV。厂商行情和个股预测需自行准备授权输入；港股模型权重及完整回测明细另行保存。

```bash
python -m venv .venv
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`；macOS／Linux 使用 `source .venv/bin/activate`。随后运行：

```bash
pip install -r requirements.txt
```

在 macOS／Linux 终端安装测试依赖后，可以运行自包含的A股测试：

```bash
pip install pytest==9.1.1 httpx==0.28.1
python -m pytest -q tests/test_ashare_*.py
```

完整合成集成测试需手动启用，命令见[公开版说明](ASHARE_RELEASE.zh-CN.md#安装与验证)。它在临时目录生成数据，检查管线和API。全量测试范围及港股历史样例缺失情况见[验证记录](docs/ashare/verification.md)。

### 重建A股图表

四张A股图表及其 CSV 表格读取仓库内的聚合 JSON 指标。安装依赖后，在仓库根目录运行：

```bash
python -m scripts.build_ashare_showcase
```

[生成脚本](scripts/build_ashare_showcase.py)检查冻结选型记录和指标算术，输出到 `docs/assets/cn/` 与 `docs/showcase/cn/`。生成图表不需要厂商原始输入，也不会重新训练模型。

### 重建港股结果

已具备本地冻结模型、预测、行情和本次回测结果时，在仓库根目录重建图表：

```bash
python -m scripts.build_showcase
```

以下命令读取已保存的预测和公司行动记录，重跑同口径回测，不训练模型：

```bash
python -m hk_quant.fixed_backtest --forecast-root backtests/hk/fixed_models_20260916 --data-root data/hk/research_20260916 --source-root data/hk/universal --results-root backtests/hk/fixed_execution_20260916 --terminal-actions data/hk/universal/references/terminal_actions/cash_settlements.csv --transfers data/hk/universal/references/terminal_actions/board_transfers.csv
```

本地输入的路径与冻结模型记录格式见[回测入口](hk_quant/fixed_backtest.py)，图表依赖见[生成脚本](scripts/build_showcase.py)。

## 目录

| 路径 | 用途 | 随仓库分发 |
| --- | --- | --- |
| `ashare_quant/` | A股采集、因子、模型、回放、API与组合建议 | 是 |
| `models/cn/universal/` | 四个冻结候选及最新 Ridge 权重、元数据 | 是 |
| `backtests/cn/universal/` | 实验协议、选型记录与汇总评价 | 仅汇总 |
| `docs/ashare/`、`delivery_report/` | A股方法、验证记录与中文报告 | 是 |
| `hk_quant/` | 港股数据处理、因子、模型、冻结回测、API与组合建议 | 是 |
| `scripts/build_ashare_showcase.py` | 读取已提交的聚合指标，生成A股评估图和表格 | 是 |
| `scripts/build_showcase.py` | 读取港股既有结果，生成展示图和案例表格 | 是 |
| `docs/assets/`、`docs/showcase/` | A股与港股README图片及可核对的轻量结果 | 是 |
| `tests/` | A股与港股研究、服务测试 | 是 |
| `legacy/` | 旧版A股及早期港股脚本、评价规则 | 是 |
| `data/` | 厂商输入与生成的数据 | 否 |
| `models/hk/`、`backtests/hk/` | 港股权重、预测与完整回测明细 | 否 |
| `reports/`、`tmp/`、`output/` | 本地报告、临时文件和导出 | 否 |

## API与部署

两套 FastAPI 服务均提供下表接口，预测期限为1、5、20、60个交易日。每套服务使用各自市场的数据和完整预测快照。

### A股API

准备授权输入，并完成[快照与概率披露步骤](docs/ashare/README.md#attach-observed-probability-evidence-before-serving)。自行在环境中安全设置 `ASHARE_QUANT_API_KEY` 后，启动人民币服务：

```bash
python -m ashare_quant.api --host 127.0.0.1 --port 8001
```

真实股票排名和持仓建议需要匹配的数据、预测及活动快照指针，仅有公开权重无法运行完整服务。API返回研究结果和订单建议，不自动下单。

### 港股API

服务需要独立发布的正式快照，由 `models/hk/universal/active.json` 指定。

```powershell
$env:HK_QUANT_API_KEY = "replace-with-a-long-random-key"
python -m hk_quant.api --host 0.0.0.0 --port 8000
```

### 通用接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/health` | 健康检查，无需密钥 |
| GET | `/v1/model/status` | 正式模型与数据日期 |
| GET | `/v1/rankings?horizon=20&limit=50` | 市场排名 |
| GET | `/v1/stocks/{code}/forecast` | 个股预测 |
| POST | `/v1/portfolio/advice` | JSON持仓建议 |
| POST | `/v1/portfolio/advice/csv` | CSV持仓建议 |

除健康检查外，两套服务的接口均需 `X-API-Key` 请求头。港股持仓建议以港元记账，输出整手订单建议，供使用者决策。

### 港股部署

部署使用 Zeabur Dev 的 Python运行环境，安装命令为 `pip install -r requirements.txt`，启动命令为 `python -m hk_quant.api --host 0.0.0.0 --port $PORT`。设置 `HK_QUANT_API_KEY`，并单独挂载匹配的数据和正式快照，无需 Docker。

<a id="credentials"></a>

## 关于作者

作者 **Shuai Wu** 获得 WorldQuant Challenge **Gold Level（金级）**。

<p align="center">
  <img src="docs/assets/credentials/worldquant-challenge-gold.png" width="420" alt="Shuai Wu — WorldQuant Challenge Gold Level">
</p>

## Star History

<p align="center">
  <a href="https://www.star-history.com/#Asher-S-Wu/Vectaix-Finance&amp;Date">
    <img src="https://api.star-history.com/svg?repos=Asher-S-Wu/Vectaix-Finance&amp;type=Date" width="800" alt="GitHub Stars 增长折线图">
  </a>
</p>
