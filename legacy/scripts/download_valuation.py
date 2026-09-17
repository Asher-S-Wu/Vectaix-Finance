#!/usr/bin/env python3
"""download_valuation.py a — 下载 A 股百度估值历史(多线程)
指标: 总市值 + 市盈率(TTM) + 市净率(可选). 半月频采样, 月度模型前向填充."""
import sys, os, time
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed
from project_paths import data_dir

market = sys.argv[1]
if market != "a":
    raise ValueError("旧港股估值下载已移除；本脚本仅支持 a")
uni = pd.read_csv(data_dir(market) / "reference/universe.csv", dtype=str)
outdir = data_dir(market) / "raw/valuations"
os.makedirs(outdir, exist_ok=True)

indicators = ["总市值", "市盈率(TTM)", "市净率"]
fn = ak.stock_zh_valuation_baidu
def code_of(w): return str(w)[:6]

codes = uni["windcode"].tolist()

def fetch(w):
    sym = code_of(w)
    fp = f"{outdir}/{sym}.csv"
    if os.path.exists(fp) and os.path.getsize(fp) > 50:
        return sym, "skip"
    rows = []
    for ind in indicators:
        for attempt in range(3):
            try:
                df = fn(symbol=sym, indicator=ind, period="全部")
                if df is not None and len(df):
                    rows.append(df.rename(columns={"date": "date", "value": ind})[["date", ind]])
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
    if rows:
        m = rows[0]
        for r in rows[1:]:
            m = m.merge(r, on="date", how="outer")
        m.sort_values("date").to_csv(fp, index=False)
        return sym, "ok"
    return sym, "empty"

ok = fail = skip = 0
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(fetch, w): w for w in codes}
    for i, fut in enumerate(as_completed(futs)):
        sym, st = fut.result()
        if st == "ok": ok += 1
        elif st == "skip": skip += 1
        else: fail += 1
        if (i + 1) % 50 == 0:
            print(f"{time.strftime('%H:%M:%S')} {i+1}/{len(codes)} ok={ok} skip={skip} fail={fail} el={(time.time()-t0)/60:.1f}min", flush=True)
print(f"ALLDONE ok={ok} skip={skip} fail={fail}")
