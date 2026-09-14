<div align="center">

# Vectaix Finance

### 홍콩 시장을 위한 감사 가능한 퀀트 연구·예측·포트폴리오 제안 서비스

[![GitHub stars](https://img.shields.io/github/stars/Asher-S-Wu/Vectaix-Finance?style=for-the-badge&logo=github&color=f4b942)](https://github.com/Asher-S-Wu/Vectaix-Finance/stargazers)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Zeabur](https://img.shields.io/badge/Deploy-Zeabur-6c63ff?style=for-the-badge)](https://zeabur.com/)

[English](./README.md) · [简体中文](./README.zh-CN.md) · [日本語](./README.ja.md) · **한국어**

</div>

> 시장 데이터, 재무 정보, 종목 식별, 기업행동, 위험 제약을 추적 가능한 하나의 흐름으로 묶어 예측과 실제 주문 단위의 포트폴리오 제안을 제공합니다.

## 프로젝트 소개

Vectaix Finance는 홍콩 주식시장을 위한 퀀트 연구 및 서비스 프로젝트입니다. 날짜를 맞춘 가격·밸류에이션·재무·기업행동 데이터로 특징을 만들고, 1·5·20·60거래일 예측 모델을 학습합니다. FastAPI를 통해 시장 순위, 종목 예측, 보유 포트폴리오 제안을 제공합니다.

모델 검증, 당일 스냅샷, 필수 필드 검사를 모두 통과한 경우에만 `models/hk/universal/active.json`을 공개합니다. API는 공개된 스냅샷만 읽으며 요청 중 모델을 학습하지 않습니다.

## 주요 기능

- 네 가지 기간의 점수, 상승 확률, 기대수익률, q10/q50/q90 구간.
- 시장 순위와 종목별 예측. 예측할 수 없는 종목의 상태도 보존.
- 현금, 수수료, 거래 단위, 거래량 참여율, 포지션 상한, 목표 연환산 변동성을 반영한 포트폴리오 제안.
- 종목 식별, 거래 단위, 통화, 환율, 날짜 증거 감사.
- walk-forward 검증, IC, 확률 보정, 구간 커버리지와 공개 게이트.

## 빠른 시작

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HK_QUANT_API_KEY=replace-with-a-long-random-key
python -m hk_quant.api --host 0.0.0.0 --port 8000
```

연구용 명령:

```bash
python scripts/train_backtest.py a
python scripts/predict.py a --as-of 2026-09-08
python scripts/make_report.py
```

## API

| 메서드 | 경로 | 용도 |
| --- | --- | --- |
| `GET` | `/health` | 인증 없는 상태 확인. |
| `GET` | `/v1/model/status` | 공개 모델과 데이터 날짜 확인. |
| `GET` | `/v1/rankings?horizon=20&limit=50` | 시장 순위 조회. |
| `GET` | `/v1/stocks/{code}/forecast` | 종목의 네 기간 예측 조회. |
| `POST` | `/v1/portfolio/advice` | JSON 보유 종목 제안. |
| `POST` | `/v1/portfolio/advice/csv` | CSV 보유 종목 제안. |

`/health`를 제외한 모든 경로에는 `X-API-Key` 헤더가 필요합니다. 현금 단위는 HKD입니다.

## 구성과 배포

`hk_quant/`에는 API·학습·공개·최적화 코드가, `data/hk/universal/`에는 연구 데이터가, `models/hk/universal/`에는 모델과 스냅샷이, `backtests/`와 `reports/`에는 결과가, `tests/`에는 테스트가 있습니다.

Zeabur Dev의 Python 서비스로 배포할 수 있습니다. 설치 명령은 `pip install -r requirements.txt`, 시작 명령은 `python -m hk_quant.api --host 0.0.0.0 --port $PORT`, 환경 변수는 `HK_QUANT_API_KEY`입니다. 유효한 `active.json`이 없으면 서비스는 공식 예측을 반환하지 않습니다.

## 연구 범위

백테스트는 해당 데이터 구간, 비용, 체결 규칙에서의 과거 결과입니다. 프로젝트는 연구 단계이며 예측은 투자 조언이 아닙니다.

## Stars 추이

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=Asher-S-Wu/Vectaix-Finance&type=Date&theme=dark)](https://www.star-history.com/#Asher-S-Wu/Vectaix-Finance&Date)

</div>

