<div align="center">

# Vectaix Finance

중국 본토 A주와 홍콩 주식 선정 모델. 학습된 가중치, 과거 검증 결과와 거래 기록을 제공합니다.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ridge](https://img.shields.io/badge/Model-Ridge-546E7A)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-00866F)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![GitHub stars](https://img.shields.io/github/stars/Asher-S-Wu/Vectaix-Finance)](https://github.com/Asher-S-Wu/Vectaix-Finance/stargazers)

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · **한국어**

</div>

Vectaix Finance는 위안화 기준 중국 본토 A주와 홍콩달러 기준 홍콩 주식을 각각의 연구 파이프라인으로 다룹니다. A주 공개판에는 학습된 Ridge, 팩터 점수, LightGBM 모델이 포함됩니다. 홍콩 주식 부분에서는 월별 종목 선정과 체결별 재생 결과를 소개합니다.

[A주 모델](#ashare) · [홍콩 주식 백테스트](#hong-kong) · [로컬 실행](#로컬-실행)

<a id="ashare"></a>

## A주 모델

A주 파이프라인은 가격, 거래량, 시가총액, 시장 상태에 관한 37개 팩터를 계산합니다. 공통으로 점수를 계산할 수 있는 종목군에서 고정 팩터 점수, Ridge, 소형·대형 LightGBM을 비교했습니다. 2024년 개발 데이터의 20거래일 평균 일별 Rank IC로 Ridge를 선정했으며, 입력은 사전에 정한 기본 특성 10개와 `log_horizon`입니다.

2016년부터 2022년까지의 데이터로 학습하고 2023년 데이터로 별도 보정했습니다. 선정한 모델을 고정한 뒤 2025년부터 2026-09-30까지의 확인 구간에서 평가했습니다. 확인 데이터는 모델 선정이나 튜닝에 사용하지 않았습니다.

| 예측 기간(거래일) | 평균 일별 Rank IC | 95% 구간 | 상승 확률 Brier skill |
| --- | ---: | --- | ---: |
| 1 | 0.0608 | [0.0455, 0.0758] | -0.35% |
| 5 | 0.0822 | [0.0503, 0.1145] | -0.54% |
| 20 | 0.1111 | [0.0608, 0.1626] | -0.22% |
| 60 | 0.1610 | [0.0812, 0.2450] | -1.15% |

Rank IC는 매일 종목 간 점수와 이후 수익률의 Spearman 상관계수입니다. 구간은 60거래일 순환 블록 부트스트랩을 2,000회 수행해 추정했으며, 해당 과거 기간의 순위 예측 품질을 나타냅니다. 확률 Brier skill은 모든 기간에서 음수로, 과거 상승 빈도 기준선보다 낮았습니다. 별도로 공개한 2026-09-30 최신 Ridge 재학습 가중치는 현재 입력을 이용한 연구용이며 표본 외 평가는 아직 없습니다.

일부 기업행동과 증권의 최종 평가액이 확인되지 않아 포트폴리오 수익률, CAGR, 최대 낙폭을 확정할 수 없습니다. 스냅샷 상태는 `eligible=false`, `execution_validated=false`입니다. 과거 결과는 미래 성과를 보장하지 않습니다.

공개판에는 네 개의 [고정 후보 모델](models/cn/universal/frozen/)과 [최신 Ridge 가중치·메타데이터](models/cn/universal/snapshots/cn-linear-latest-20260930-v1/)가 포함됩니다. 공급업체 원본 입력과 개별 종목 예측 스냅샷은 사용자가 별도로 이용 허가를 받은 Tushare 데이터로 준비해야 합니다. 공개판에는 서비스용 전체 스냅샷이 없어 복제 직후에는 모델 확인과 합성 데이터 테스트를 실행할 수 있습니다. 실제 종목 순위에는 완전한 로컬 입력이 필요합니다.

[공개판 안내(중국어)](ASHARE_RELEASE.zh-CN.md) · [연구 보고서 PDF(중국어)](delivery_report/ashare_training_report.zh-CN.pdf) · [방법과 재현 절차](docs/ashare/README.md) · [확인 구간 지표](backtests/cn/universal/confirmation_summary.json) · [A주 소스](ashare_quant/)

<a id="hong-kong"></a>

## 홍콩 주식 백테스트

홍콩 주식 파이프라인은 가격·거래량·시장 특성 37개를 점수로 합쳐 매월 상위 30개 홍콩 주식을 선정합니다. 2024-02-01~2026-09-16 과거 데이터 백테스트에서 소형 LightGBM의 비용 차감 후 연환산 수익률은 **35.56%**였으며, 100만 홍콩달러가 **222.09만 홍콩달러**가 됐습니다.

![과거 백테스트: 모델 자산, 항셍 지수, 항셍 테크 지수 및 낙폭](docs/assets/performance.png)

### 지수와 수익 비교

기간은 **2024-02-01~2026-09-16**, 총 644거래일입니다. 네 모델은 매월 말 공통으로 점수를 계산할 수 있는 종목군에서 20거래일 점수 상위 30개를 고릅니다. 다음 거래일 종가에 모의 체결하며, 초기 자금은 100만 홍콩달러, 편도 비용은 0.25%, 거래대금 참여율 상한은 1%입니다.

| 모델 / 지수 | 연환산 수익률 | 누적 수익률 | 최대 낙폭 | 수익 포지션 비율 |
| --- | ---: | ---: | ---: | ---: |
| **소형 LightGBM** | **35.56%** | **122.09%** | -20.10% | 49.16% |
| 대형 LightGBM | 32.28% | 108.30% | -15.49% | 49.21% |
| 선형 모델 | 13.37% | 38.98% | -10.36% | 45.32% |
| 팩터 점수 | 5.83% | 16.02% | -21.38% | 51.78% |
| 항셍 지수 | 19.27% | 58.77% | -19.95% | — |
| 항셍 테크 지수 | 14.02% | 41.08% | -36.32% | — |

소형 모델은 같은 기간 항셍 지수보다 연환산 **16.28%포인트**, 누적 **63.33%포인트** 높은 수익을 기록했습니다.

수익 포지션 비율은 **청산을 마친 포지션 중 비용 차감 후 수익을 낸 비율**입니다. 소형 모델은 891개 중 438개로 49.16%입니다. 편도 비용 0.50%의 스트레스 조건에서는 연환산 수익률이 31.41%입니다. 모델에는 모의 거래 비용이 반영됐으며, 지수는 배당을 제외한 가격 지수로 비용을 차감하지 않았습니다.

[일별 자산 곡선](docs/showcase/performance_curve.csv) · [네 모델과 두 비용 조건](docs/showcase/model_comparison.csv) · [기간·조건·출처](docs/showcase/summary.json)

### 종목 순위를 만드는 입력

소형 모델은 가격 이격, 모멘텀, 변동성, 거래대금, 시가총액, 시장 내 상승 종목의 확산 정도 등 37개 입력을 사용합니다. 이를 하나의 점수로 합쳐 종목 순위를 정합니다.

![학습 시 분할 이득 기준 상위 10개 입력](docs/assets/factor_importance.png)

252일 저점과의 거리와 60일 가격 이격은 전체 입력의 분할 이득에서 각각 16.40%, 15.81%를 차지합니다. 이 차트는 학습 중 모델이 각 입력을 활용한 정도를 보여줍니다. 포트폴리오 전체 성과는 위 자산 곡선에서 확인할 수 있습니다.

[전체 입력 중요도](docs/showcase/factor_importance.csv) · [특성 계산](hk_quant/features.py) · [모델 구현](hk_quant/models.py)

### 첫 매수부터 마지막 매도까지

이번 과거 데이터 시뮬레이션은 **첫 매수 묶음의 30개 종목 전체**를 추적합니다. 월말 선정, 다음 거래일 매수, 분할 청산까지 85건의 체결과 현금 장부를 기록했습니다.

| 날짜 | 기록 |
| --- | --- |
| 2024-01-31 | 월말 점수로 30개 종목 선정. |
| 2024-02-01 | 다음 거래일 종가에 매수. 거래대금 한도로 일부 매수 금액 축소. 비용 포함 투입금은 716,281.52홍콩달러. |
| 2024-03-01 | 예정된 청산일에 매도 시작. |
| 2024-03-04~03-14 | 거래대금 한도로 남은 물량을 나눠 매도하고 3월 14일 마지막 체결 완료. |

#### 매수 직후 보유 비중

주식 평가액은 714,495.29홍콩달러, 현금은 283,718.48홍콩달러입니다. 매수 비용 차감 후 총자산은 998,213.76홍콩달러이고 현금 비중은 28.42%입니다. 가장 큰 네 포지션을 따로 표시하고 나머지 26개를 묶었습니다.

![2024년 2월 1일 매수 후 보유 종목과 현금](docs/assets/holdings.png)

[선정 당시 순위](docs/showcase/first_cycle_selection.csv) · [30개 종목의 보유액과 비중](docs/showcase/first_cycle_holdings.csv)

#### 캔들 위의 매수·매도 시점

캔들은 당시 점수 상위 두 종목인 08619.HK(비용 차감 후 −59.91%)와 02171.HK(+61.45%)를 보여줍니다. 08619.HK의 두 차례 분할 매도를 포함해 각 모의 체결에 표시를 달았습니다.

![상위 두 종목의 수정주가 캔들과 모의 매수·매도 지점](docs/assets/trade_candles.png)

| 날짜 | 종목 | 구분 | 조정 단위 수 | 조정 가격 | 체결 금액(HKD) | 비용(HKD) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2024-02-01 | 08619.HK | 매수 | 4,885.528685 | 2.307179 | 11,271.79 | 28.18 |
| 2024-02-01 | 02171.HK | 매수 | 7,742.082702 | 4.080000 | 31,587.70 | 78.97 |
| 2024-03-01 | 08619.HK | 매도 | 1,951.329001 | 1.118632 | 2,182.82 | 5.46 |
| 2024-03-01 | 02171.HK | 매도 | 7,742.082702 | 6.620000 | 51,252.59 | 128.13 |
| 2024-03-04 | 08619.HK | 매도 | 2,934.199683 | 0.804017 | 2,359.15 | 5.90 |

거래표는 소수 수량을 허용한 조정 연구 단위를 사용합니다. [전체 85건의 모의 체결](docs/showcase/first_cycle_trades.csv)에는 날짜, 종목, 방향, 수량, 가격, 금액, 비용과 해당 묶음의 현금 변화가 들어 있습니다.

#### 30개 포지션의 최종 손익

30개 중 18개 포지션이 수익을 냈습니다. 매수 30건, 매도 55건으로 총 85건입니다. 매수 비용 포함 원가는 716,281.52홍콩달러, 매도 순수입은 781,194.89홍콩달러로 **순이익은 64,913.36홍콩달러**입니다. 매수·매도 비용 3,744.12홍콩달러를 차감한 결과입니다. 투입 원가 대비 수익률은 9.06%, 초기 100만 홍콩달러에 대한 기여는 6.49%입니다.

![첫 30개 포지션 전체의 비용 차감 후 손익](docs/assets/first_cycle_pnl.png)

[포지션별 원가·수입·수익률](docs/showcase/first_cycle_positions.csv)

`cohort_cash_after`는 이 30개 포지션의 독립 현금 장부입니다. 100만 홍콩달러에서 1,064,913.36홍콩달러까지 추적하며, 후속 리밸런싱의 신규 포지션은 제외합니다. 전체 전략 계좌의 자산은 일별 자산 곡선 표에 있습니다.

### 백테스트 계산 기준

- 매도 후 사용 가능한 현금의 95%를 30개 고정 슬롯에 나눕니다. 매수는 신호일 기준 20일 평균 거래대금과 체결일 거래대금 각각의 1% 이내입니다. 미사용 금액은 현금으로 남기고 미청산 물량은 이후 거래일에 매도를 시도합니다.
- 주식 데이터는 Tushare `hk_daily_adj`, `hk_adjfactor`, 지수는 `index_global`에서 가져왔습니다. 조정된 연구용 단위를 사용하며 과거 매매 단위, 정확한 배당 입금일과 실제 주문 체결은 재현하지 않았습니다.
- 연환산 수익은 경과 달력 일수로 복리 계산하며 최대 낙폭은 일별 계좌 자산으로 계산합니다. 기말 자산에는 미청산 포지션의 참고 평가액이 포함되며, 소형 모델에는 한 종목이 남아 있습니다. 참고 평가액과 실제 체결 가능한 가격은 다를 수 있습니다. `execution_validated`는 `false`입니다.
- 소개 모델은 2023-12-29까지의 데이터로 학습했고 고정된 파라미터로 재생합니다. 이 기간은 후보 모델 비교에 사용됐으므로 독립 홀드아웃 검증이 아닌 과거 비교 백테스트입니다.

## 로컬 실행

Python 3.12와 저장소에 고정된 의존성 버전을 권장합니다. 코드, 테스트, A주 학습 가중치와 요약 보고서, 홍콩 주식 차트와 사례 CSV가 포함됩니다. 공급업체 시세와 개별 종목 예측에는 별도로 이용 허가를 받은 입력이 필요합니다. 홍콩 주식 가중치와 전체 재생 기록은 별도로 보관됩니다.

```bash
python -m venv .venv
```

Windows PowerShell에서는 `.venv\Scripts\Activate.ps1`, macOS / Linux에서는 `source .venv/bin/activate`로 활성화한 뒤 설치합니다.

```bash
pip install -r requirements.txt
```

macOS / Linux 셸에서 테스트 의존성을 설치한 뒤 자체 완결형 A주 테스트를 실행할 수 있습니다.

```bash
pip install pytest==9.1.1 httpx==0.28.1
python -m pytest -q tests/test_ashare_*.py
```

전체 합성 통합 테스트는 직접 활성화해야 하며, 명령은 [공개판 안내](ASHARE_RELEASE.zh-CN.md#安装与验证)에 있습니다. 임시 디렉터리에 데이터를 생성하고 파이프라인과 API를 확인합니다. 전체 테스트 범위와 누락된 홍콩 주식 과거 샘플은 [검증 기록](docs/ashare/verification.md)을 참고하세요.

### 홍콩 주식 결과 재생성

저장된 모델, 예측, 시세와 이번 백테스트 결과가 있으면 저장소 루트에서 차트를 다시 생성할 수 있습니다.

```bash
python -m scripts.build_showcase
```

동일한 기업행동 기록과 저장된 예측을 사용하는 재생 명령입니다. 모델을 다시 학습하지 않습니다.

```bash
python -m hk_quant.fixed_backtest --forecast-root backtests/hk/fixed_models_20260916 --data-root data/hk/research_20260916 --source-root data/hk/universal --results-root backtests/hk/fixed_execution_20260916 --terminal-actions data/hk/universal/references/terminal_actions/cash_settlements.csv --transfers data/hk/universal/references/terminal_actions/board_transfers.csv
```

로컬 입력 경로와 저장된 모델 기록 형식은 [재생 코드](hk_quant/fixed_backtest.py), 차트에 필요한 입력은 [차트 생성 코드](scripts/build_showcase.py)에 정의돼 있습니다.

## 디렉터리

| 경로 | 용도 | 배포 포함 |
| --- | --- | --- |
| `ashare_quant/` | A주 수집, 팩터, 모델, 재생, API와 포트폴리오 제안 | 예 |
| `models/cn/universal/` | 네 고정 후보와 최신 Ridge 가중치·메타데이터 | 예 |
| `backtests/cn/universal/` | 실험 규약, 선정 기록과 요약 평가 | 요약만 |
| `docs/ashare/`, `delivery_report/` | A주 방법, 검증 기록과 중국어 보고서 | 예 |
| `hk_quant/` | 홍콩 데이터 처리, 팩터, 모델, 고정 모델 재생, API와 포트폴리오 제안 | 예 |
| `scripts/build_showcase.py` | 기존 홍콩 결과에서 차트와 사례 표 생성 | 예 |
| `docs/assets/`, `docs/showcase/` | 홍콩 README 이미지와 소규모 결과 표 | 예 |
| `tests/` | A주·홍콩 주식 연구 및 서비스 테스트 | 예 |
| `legacy/` | 이전 A주·초기 홍콩 주식 스크립트와 평가 기준 | 예 |
| `data/` | 공급업체 입력과 생성 데이터 | 아니요 |
| `models/hk/`, `backtests/hk/` | 홍콩 가중치, 예측과 전체 재생 기록 | 아니요 |
| `reports/`, `tmp/`, `output/` | 로컬 보고서, 임시 파일과 내보내기 | 아니요 |

## API와 배포

두 FastAPI 서비스는 1·5·20·60거래일 예측에 대해 아래 엔드포인트를 제공합니다. 각 서비스에는 해당 시장 데이터와 완전한 예측 스냅샷이 필요합니다.

### A주 API

이용 허가를 받은 입력을 준비하고 [스냅샷과 확률 평가 공개 절차](docs/ashare/README.md#attach-observed-probability-evidence-before-serving)를 완료하세요. 환경 변수 `ASHARE_QUANT_API_KEY`를 안전하게 설정한 뒤 위안화 서비스를 시작합니다.

```bash
python -m ashare_quant.api --host 127.0.0.1 --port 8001
```

실제 종목 순위와 보유 종목 제안에는 일치하는 데이터, 예측, 활성 스냅샷 포인터가 필요합니다. 공개 가중치만으로 전체 서비스를 실행할 수 없습니다. API는 연구 결과와 주문 제안을 반환하며 주문을 전송하지 않습니다.

### 홍콩 주식 API

서비스에는 별도로 게시한 공식 스냅샷이 필요하며, `models/hk/universal/active.json`에서 지정합니다.

```powershell
$env:HK_QUANT_API_KEY = "replace-with-a-long-random-key"
python -m hk_quant.api --host 0.0.0.0 --port 8000
```

### 공통 엔드포인트

| 메서드 | 경로 | 용도 |
| --- | --- | --- |
| GET | `/health` | 상태 확인, 키 불필요 |
| GET | `/v1/model/status` | 공식 모델과 데이터 날짜 |
| GET | `/v1/rankings?horizon=20&limit=50` | 시장 순위 |
| GET | `/v1/stocks/{code}/forecast` | 개별 종목 예측 |
| POST | `/v1/portfolio/advice` | JSON 보유 내역 기반 제안 |
| POST | `/v1/portfolio/advice/csv` | CSV 보유 내역 기반 제안 |

두 서비스 모두 상태 확인 외의 API는 `X-API-Key` 헤더가 필요합니다. 홍콩 포트폴리오 제안은 홍콩달러로 계산하며, 사용자가 판단할 수 있도록 매매 단위에 맞춘 주문 제안을 반환합니다.

### 홍콩 주식 배포

Zeabur Dev의 Python 환경에 배포합니다. 설치 명령은 `pip install -r requirements.txt`, 시작 명령은 `python -m hk_quant.api --host 0.0.0.0 --port $PORT`입니다. `HK_QUANT_API_KEY`를 설정하고 일치하는 데이터와 공식 스냅샷을 별도로 마운트합니다. Docker는 필요하지 않습니다.

<a id="credentials"></a>

## 작성자 소개

작성자 **Shuai Wu**는 WorldQuant Challenge **Gold Level**을 달성했습니다.

<p align="center">
  <img src="docs/assets/credentials/worldquant-challenge-gold.png" width="420" alt="Shuai Wu — WorldQuant Challenge Gold Level">
</p>

## Star History

<p align="center">
  <a href="https://www.star-history.com/#Asher-S-Wu/Vectaix-Finance&amp;Date">
    <img src="https://api.star-history.com/svg?repos=Asher-S-Wu/Vectaix-Finance&amp;type=Date" width="800" alt="GitHub Stars 증가 추이">
  </a>
</p>
