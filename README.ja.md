<div align="center">

# Vectaix Finance

### 香港市場向け、監査可能なクオンツ研究・予測・ポートフォリオ提案サービス

[![GitHub stars](https://img.shields.io/github/stars/Asher-S-Wu/Vectaix-Finance?style=for-the-badge&logo=github&color=f4b942)](https://github.com/Asher-S-Wu/Vectaix-Finance/stargazers)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Zeabur](https://img.shields.io/badge/Deploy-Zeabur-6c63ff?style=for-the-badge)](https://zeabur.com/)

[English](./README.md) · [简体中文](./README.zh-CN.md) · **日本語** · [한국어](./README.ko.md)

</div>

> 市場データ、財務情報、銘柄 identity、コーポレートアクション、リスク制約を追跡可能な一つの流れにまとめ、予測と実行可能な単元株注文を返します。

## 概要

Vectaix Finance は香港株式市場のクオンツ研究・サービス基盤です。日付をそろえた価格、バリュエーション、財務、コーポレートアクションから特徴量を作り、1・5・20・60取引日の予測を学習します。FastAPI で市場ランキング、銘柄予測、保有銘柄への提案を提供します。

モデル検証、当日スナップショット、必須フィールドの確認をすべて通過した場合だけ `models/hk/universal/active.json` を公開します。API は公開済みスナップショットだけを読み込み、リクエスト中に学習しません。

## 主な機能

- 4つの期間のスコア、上昇確率、期待リターン、q10/q50/q90区間。
- 市場ランキングと銘柄別予測。予測不能な銘柄の状態も保持。
- 現金、手数料、売買単位、出来高参加率、ポジション上限、目標年率ボラティリティを考慮したポートフォリオ提案。
- 銘柄 identity、取引単位、通貨、為替、日付証拠の監査。
- walk-forward 検証、IC、確率校正、区間カバレッジ、公開ゲート。

## クイックスタート

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HK_QUANT_API_KEY=replace-with-a-long-random-key
python -m hk_quant.api --host 0.0.0.0 --port 8000
```

研究用の主なコマンド：

```bash
python scripts/train_backtest.py a
python scripts/predict.py a --as-of 2026-09-08
python scripts/make_report.py
```

## API

| メソッド | パス | 用途 |
| --- | --- | --- |
| `GET` | `/health` | ヘルスチェック。認証不要。 |
| `GET` | `/v1/model/status` | 公開モデルとデータ日付。 |
| `GET` | `/v1/rankings?horizon=20&limit=50` | 市場ランキング。 |
| `GET` | `/v1/stocks/{code}/forecast` | 銘柄の4期間予測。 |
| `POST` | `/v1/portfolio/advice` | JSON 保有銘柄の提案。 |
| `POST` | `/v1/portfolio/advice/csv` | CSV 保有銘柄の提案。 |

`/health` 以外は `X-API-Key` ヘッダーが必要です。現金の単位は HKD です。

## 構成とデプロイ

`hk_quant/` に API、学習、公開、最適化を、`data/hk/universal/` に研究データを、`models/hk/universal/` にモデルとスナップショットを、`backtests/` と `reports/` に結果を、`tests/` にテストを置いています。

Zeabur Dev に Python サービスとしてデプロイできます。インストールは `pip install -r requirements.txt`、起動は `python -m hk_quant.api --host 0.0.0.0 --port $PORT`、環境変数は `HK_QUANT_API_KEY` です。公開済み `active.json` がない場合、サービスは予測を返しません。

## 研究上の注意

バックテストは指定された期間、費用、約定ルールに基づく過去の結果です。プロジェクトは研究段階にあり、予測は投資助言ではありません。

## Stars の推移

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=Asher-S-Wu/Vectaix-Finance&type=Date&theme=dark)](https://www.star-history.com/#Asher-S-Wu/Vectaix-Finance&Date)

</div>

