<div align="center">

# Vectaix Finance

### 面向香港市场的可审计量化研究、预测与组合建议服务

[![GitHub stars](https://img.shields.io/github/stars/Asher-S-Wu/Vectaix-Finance?style=for-the-badge&logo=github&color=f4b942)](https://github.com/Asher-S-Wu/Vectaix-Finance/stargazers)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Zeabur](https://img.shields.io/badge/Deploy-Zeabur-6c63ff?style=for-the-badge)](https://zeabur.com/)

[English](./README.md) · **简体中文** · [日本語](./README.ja.md) · [한국어](./README.ko.md)

</div>

> 把行情、财务、公司身份、公司行动和风险约束放在同一条可追溯链路里，再输出预测与可执行的整手订单建议。

## 这是什么

Vectaix Finance 是一个面向香港股票市场的量化研究与服务项目。它根据日期对齐的行情、估值、财务和公司行动数据构建特征，训练 1、5、20、60 个交易日四种期限的预测模型，并通过 FastAPI 提供市场排名、个股预测和持仓建议。

模型只有在验收通过、当日快照完整、预测字段齐全后，才会写入 `models/hk/universal/active.json`。API 只读取正式快照，不会在请求时临时训练模型。

## 核心能力

- **多期限预测**：分数、上涨概率、预期收益和 q10/q50/q90 区间。
- **市场排名与个股查询**：支持四个预测期限，并保留不可评分状态。
- **组合建议**：同时考虑现金、费用、整手、成交参与率、仓位上限和目标年化波动率。
- **数据审计**：核对证券身份、交易单位、币种、汇率和日期证据。
- **回测验收**：覆盖 IC、概率校准、区间覆盖、组合表现和发布门槛。

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # macOS / Linux
pip install -r requirements.txt

set HK_QUANT_API_KEY=replace-with-a-long-random-key
python -m hk_quant.api --host 0.0.0.0 --port 8000
```

常用研究命令：

```bash
python scripts/train_backtest.py a
python scripts/predict.py a --as-of 2026-09-08
python scripts/make_report.py
```

## API

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查，无需密钥。 |
| `GET` | `/v1/model/status` | 当前正式模型与数据日期。 |
| `GET` | `/v1/rankings?horizon=20&limit=50` | 市场排名。 |
| `GET` | `/v1/stocks/{code}/forecast` | 个股四期限预测。 |
| `POST` | `/v1/portfolio/advice` | JSON 持仓建议。 |
| `POST` | `/v1/portfolio/advice/csv` | CSV 持仓建议。 |

除 `/health` 外都要在 `X-API-Key` 请求头中提供 `HK_QUANT_API_KEY`。组合建议的现金单位为 HKD，返回目标数量、订单数量、费用、权重、排除原因和约束状态。

## 目录与部署

`hk_quant/` 保存 API、训练、发布和组合优化代码；`data/hk/universal/` 保存研究数据；`models/hk/universal/` 保存模型与正式快照；`backtests/`、`reports/` 保存回测结果；`tests/` 保存测试。

项目部署在 Zeabur Dev 方案即可，不需要 Docker。安装命令为 `pip install -r requirements.txt`，启动命令为 `python -m hk_quant.api --host 0.0.0.0 --port $PORT`，并设置 `HK_QUANT_API_KEY`。没有有效 `active.json` 时，服务会明确拒绝提供正式预测。

## 研究边界

历史回测只代表对应数据窗口、费用和成交规则下的结果。项目仍处于研究阶段，预测不构成投资建议。

## Stars 走势

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=Asher-S-Wu/Vectaix-Finance&type=Date&theme=dark)](https://www.star-history.com/#Asher-S-Wu/Vectaix-Finance&Date)

</div>

