"""Never share writable artifact namespaces with the existing HK model."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/cn/universal'
MODELS = ROOT / 'models/cn/universal'
RESULTS = ROOT / 'backtests/cn/universal'
