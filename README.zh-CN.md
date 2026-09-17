<div align="center">

# Vectaix Finance

港股多因子选股，从评分到交易复盘。

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-00866F)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![GitHub stars](https://img.shields.io/github/stars/Asher-S-Wu/Vectaix-Finance)](https://github.com/Asher-S-Wu/Vectaix-Finance/stargazers)

[English](README.md) · **简体中文** · [日本語](README.ja.md) · [한국어](README.ko.md)

</div>

Vectaix Finance 将37项量价与市场特征合成选股分数，每月选取排名前30的港股。在2024-02-01～2026-09-16的历史回测中，小型 LightGBM 扣费年化收益为 **35.56%**，100万港元增至 **222.09万港元**。

![历史回测：模型净值、恒生指数、恒生科技指数及回撤](docs/assets/performance.png)

## 收益与指数对比

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

## 因子如何参与选股

展示模型使用37项输入，包括价格偏离、动量、波动率、成交额、市值和市场宽度。它把这些信息合成排序分数，再按分数选股。

![小型模型前10项输入的训练期重要性](docs/assets/factor_importance.png)

距离252日低点、60日价格偏离分别占全部输入分裂增益的16.40%和15.81%。这张图展示模型在训练中对各项输入的使用程度，组合收益见上方净值图。

[全部输入的重要性](docs/showcase/factor_importance.csv) · [特征计算](hk_quant/features.py) · [模型实现](hk_quant/models.py)

## 一次完整交易复盘

这次历史模拟跟踪**第一次建仓的全部30只股票**，从月末选股、次日买入到分批退出，保留85笔成交和完整资金账目。

| 日期 | 发生了什么 |
| --- | --- |
| 2024-01-31 | 月末评分，选出前30只股票。 |
| 2024-02-01 | 下一交易日收盘买入，按成交额限制缩减部分仓位；含费用投入716,281.52港元。 |
| 2024-03-01 | 到计划退出日，开始卖出这批持仓。 |
| 2024-03-04～03-14 | 部分股票因成交额限制分批卖出；3月14日完成最后一笔。 |

### 建仓后的账户

股票市值714,495.29港元，现金283,718.48港元。扣除买入费用后，总资产998,213.76港元，现金占28.42%。图中列出最大的4笔持仓，其余26只合并展示；完整明细可下载。

![2024年2月1日建仓后的持仓与现金比例](docs/assets/holdings.png)

[当期选股排名](docs/showcase/first_cycle_selection.csv) · [30只股票的持仓和权重](docs/showcase/first_cycle_holdings.csv)

### K线上的买卖点

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

### 这一期赚了多少

30个持仓中18个盈利，30笔买入、55笔卖出，共85笔成交。买入总成本716,281.52港元，卖出净收入781,194.89港元，**净利润64,913.36港元**，已扣除双边费用3,744.12港元。投入成本收益率为9.06%，对初始100万港元资金贡献6.49%。

![第一期全部30个持仓的扣费盈亏](docs/assets/first_cycle_pnl.png)

[全部30个持仓的成本、收入与收益率](docs/showcase/first_cycle_positions.csv)

`cohort_cash_after` 是这批30个持仓的独立现金账：从100万港元到1,064,913.36港元，不含后续调仓的新持仓。完整策略的账户资产见逐日净值表。

## 回测口径

- 每期使用卖出后的可用现金的95%，分成30个固定名额。买入同时受信号日20日平均成交额和成交日成交额的1%限制；未买足的部分留作现金，未卖完的持仓继续尝试退出。
- 股票行情来自 Tushare `hk_daily_adj`、`hk_adjfactor`，指数来自 `index_global`。回测使用复权研究单位，没有逐笔还原历史整手、分红到账日和真实撮合。
- 年化按实际日历天数计算复合收益；最大回撤来自每日账户净值。期末资产包含未卖出持仓的参考市值，小型模型有1只未退出。参考估值与可成交价格有别，`execution_validated` 为 `false`。
- 展示模型训练截至2023-12-29，使用冻结参数回放。该区间已用于候选模型比较，属于历史对比回测，不作为独立留出验证。

## 本地运行

仓库提供源码、测试、5张回测图、作者资质证书和可直接查看的汇总／案例CSV。**完整行情、模型权重、预测及批量回测明细保留本地，不随当前源码分发。**

```bash
python -m venv .venv
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`；macOS／Linux 使用 `source .venv/bin/activate`。随后运行：

```bash
pip install -r requirements.txt
```

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

| 路径 | 用途 | 随当前源码分发 |
| --- | --- | --- |
| `hk_quant/` | 港股数据处理、因子、模型、冻结回测、API与组合建议 | 是 |
| `scripts/build_showcase.py` | 读取现有结果，生成展示图和案例表格 | 是 |
| `docs/assets/`、`docs/showcase/` | README图片及可核对的轻量结果 | 是 |
| `tests/` | 港股研究与服务测试 | 是 |
| `legacy/` | 旧版A股及早期港股脚本、旧评价规则 | 是 |
| `data/`、`models/` | 原始行情、特征、模型权重与正式快照 | 否 |
| `backtests/`、`reports/` | 完整回测明细与自动报告 | 否 |
| `tmp/`、`output/` | 临时文件和本地导出 | 否 |

## API与部署

FastAPI服务提供1、5、20、60个交易日期限的排名、预测和组合建议。服务需要独立发布的正式快照，由 `models/hk/universal/active.json` 指定。

```powershell
$env:HK_QUANT_API_KEY = "replace-with-a-long-random-key"
python -m hk_quant.api --host 0.0.0.0 --port 8000
```

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/health` | 健康检查，无需密钥 |
| GET | `/v1/model/status` | 正式模型与数据日期 |
| GET | `/v1/rankings?horizon=20&limit=50` | 市场排名 |
| GET | `/v1/stocks/{code}/forecast` | 个股预测 |
| POST | `/v1/portfolio/advice` | JSON持仓建议 |
| POST | `/v1/portfolio/advice/csv` | CSV持仓建议 |

除健康检查外，接口均需 `X-API-Key` 请求头。持仓建议以港元记账，输出整手订单建议，供使用者决策。

部署使用 Zeabur Dev 的 Python运行环境，安装命令为 `pip install -r requirements.txt`，启动命令为 `python -m hk_quant.api --host 0.0.0.0 --port $PORT`。设置 `HK_QUANT_API_KEY`，并单独挂载匹配的数据和正式快照，无需 Docker。

<a id="credentials"></a>

## 关于作者

作者 **Shuai Wu** 获得 WorldQuant Challenge **Gold Level（金级）**。

<p align="center">
  <img src="docs/assets/credentials/worldquant-challenge-gold.png" width="420" alt="Shuai Wu — WorldQuant Challenge Gold Level">
</p>

## Stars

<p align="center">
  <a href="https://github.com/Asher-S-Wu/Vectaix-Finance/stargazers">
    <img src="https://img.shields.io/github/stars/Asher-S-Wu/Vectaix-Finance?style=for-the-badge&amp;logo=github&amp;label=Stars&amp;color=00866F" alt="GitHub Stars">
  </a>
</p>
