"""项目目录约定，供A股与港股的全部流程共用。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
MARKET_FOLDERS = {"a": "a", "hk2": "hk"}


def data_dir(market):
    return ROOT / "data" / MARKET_FOLDERS[market]


def model_dir(market):
    return ROOT / "models" / MARKET_FOLDERS[market]


def backtest_dir(market):
    return ROOT / "backtests" / MARKET_FOLDERS[market]


def qveris_dir(market):
    return backtest_dir(market) / "qveris"
