# A 股训练模型公开版 · 2026-09-30

本次更新发布独立 A 股源码、测试、五个已训练模型文件、实验协议及汇总研究报告。原港股源码、README 和展示文件保持原样。

## 已训练模型

- 冻结候选：`models/cn/universal/frozen/` 中的 factor、Ridge、small LightGBM、large LightGBM
- 最终选择：Ridge，10 个预设基础输入加 `log_horizon`；基础因子管线共 37 项
- 最新重训权重：`models/cn/universal/snapshots/cn-linear-latest-20260930-v1/model.pkl`
- 对应元数据和最终验收说明与权重一并保留，权重 SHA256 与原封存实验一致

冻结模型用于独立确认，最新重训模型用于当前输入研究；两者是不同模型。确认区间 20 日 rank IC 为 0.1111，95% 区间约 0.0608–0.1626。四个期限的概率 Brier skill 均为负，最新重训模型没有独立样本外评估。`eligible=false`、`execution_validated=false`。未核实的终局证券权益使组合收益、CAGR 和最大回撤不可确认。

完整说明见 [四页研究报告](delivery_report/ashare_training_report.zh-CN.pdf)、[可编辑报告](delivery_report/ashare_training_report.zh-CN.md) 和 [方法文档](docs/ashare/README.md)。

## 数据与运行范围

公开仓库不分发厂商行情输入、证券主表、风险收益矩阵、指数历史、企业行动逐条记录、个股预测快照或成交流水。使用者应自行取得数据来源授权，并按源码的数据协议准备本地输入。模型文件本身包含已学习参数和汇总校准统计，不包含训练数据表。

因此，克隆本仓库可以检查权重、阅读结果并运行自包含合成测试，但不能直接启动带真实股票排名和持仓建议的完整离线 API。公开版未提供 `active.json`，避免将缺少数据和预测的目录标成可服务快照。验收说明中的 `research_ready=true` 记录原完整研究快照的历史验证状态，不代表公开版包含完整运行数据。

`ASHARE_RELEASE.json` 是此次实际发布文件的 SHA256 清单。模型元数据、验收说明和报告中的来源哈希对应原实验证据；其中未公开的输入路径是来源记录，不是随仓库附送文件的承诺。公开版报告数据仅保留汇总，单证券原始价格记录和企业行动持仓名单不随报告发布。

## 安装与验证

建议 Python 3.12，按原仓库锁定版本安装：

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt pytest==9.1.1 httpx==0.28.1
.venv/bin/python -m pytest -q tests/test_ashare_*.py
ASHARE_SYNTHETIC_E2E=1 .venv/bin/python -m pytest -q tests/test_ashare_end_to_end.py
```

合成测试数据均在临时目录中生成，只验证接口和管线；不能用作真实收益证据。原仓库全量测试缺少部分 `tmp/` 工具及历史 HTML 样例，已知情况见 [验证记录](docs/ashare/verification.md)。

报告重建另需 `requirements-report.txt`：

```sh
.venv/bin/pip install -r requirements-report.txt
.venv/bin/python delivery_report/build_report_pdf.py --input delivery_report/report_data.json --output /tmp/ashare-report
```

此命令只读取公开汇总和随附授权字体，无需行情数据、联网或密钥。只在验证文件哈希后加载可信的自有 pickle，不能加载不可信来源的 pickle。

## 获取授权数据后复现

采集器要求使用者自己的、已获授权的凭据助手，接口约定见 `ashare_quant.collect.HelperClient`。历史默认助手路径属于原执行环境，助手本身及凭据不在本仓库；在其他环境应通过 `HelperClient(helper=...)` 指定自己的实现。完整采集、历史训练和真实 API 运行的前置条件与命令见 [A 股方法文档](docs/ashare/README.md)。不要把凭据、厂商缓存或自行生成的真实行情文件提交到公开仓库。

在已具备完整合规输入的独立本地目录中，可以复现实验、生成预测并运行显式披露步骤后，再配置 API 密钥和启用本地服务。公开结果是历史研究记录，不是收益承诺或真实下单服务。
