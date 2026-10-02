# A股研究报告文件

- `ashare_training_report.zh-CN.pdf`：四页中文报告，包含固定实验协议、留出质量、成本回放及限制
- `ashare_training_report.zh-CN.md`：可编辑文字与表格；图形由重建脚本生成
- `report_data.json`：报告使用的已完成汇总值与原实验源文件 SHA256；公开副本移除单证券原始价格记录，企业行动持仓名单改为准确数量，数值结论不变
- `build_report_pdf.py`：离线重建脚本，只读取汇总证据，不调用网络或训练模型
- `AshareReportSans-Regular.ttf` 与 `FONT-LICENSE.txt`：随附并嵌入 PDF 的中文子集字体及 SIL OFL 1.1 许可

## 离线重建

需要 Python 3.10+ 和 ReportLab（本次验证版本为 4.4.9，可通过根目录 requirements-report.txt 安装）。在本目录运行：

```
python build_report_pdf.py --input report_data.json --output .
```

已安装 ReportLab 时无需联网，不需要任何密钥或系统中文字体。随附字体仅覆盖此报告使用的字符；大幅改写中文内容时需另换获授权的完整中文 TrueType 字体并重新检查排版。

公开版的 `--input` 重建不依赖未公开的数据。报告内引用的来源路径和 SHA256 记录原完整实验；当前实际公开文件的哈希见根目录 `ASHARE_RELEASE.json`。其中 `execution_summary.json` 和本报告数据已作汇总化处理，公开副本哈希与原实验哈希分别保留，不混为同一文件。

只有自行持有完整授权输入及原实验明细时，才能从完整本地实验目录读取汇总；公开克隆不具备以下 `--repo` 模式的所有前置文件：

```
python build_report_pdf.py --repo /path/to/Vectaix-Finance --output /path/to/report
```

生成器要求最终回放 JSON 齐全且管线状态为 complete。已有数值结论改变时，需同步审核叙述和图表后重建。PDF 中的历史回测及排名属于研究证据，不是实盘业绩或投资建议。
