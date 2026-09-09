#!/usr/bin/env python3
"""download_prices.py <universe.csv> <outdir> — 经 agent-gw SDK 批量下载 Wind 日K线
3标的/批, 重试4次带退避, 断点续传(按批次文件存在跳过)."""
import sys, os, time, json, traceback
from io import StringIO
import pandas as pd
from agent_gw import AgentGwClient
from hk_market_data import clean_hk_price_file, filter_hk_prices, hk_calendar

universe_csv, outdir = sys.argv[1], sys.argv[2]
START, END = "2015-12-01", "2026-09-09"
os.makedirs(outdir, exist_ok=True)

uni = pd.read_csv(universe_csv, dtype=str)
codes = uni["windcode"].tolist()
IS_HK = all(code.endswith(".HK") for code in codes)
if IS_HK:
    START, END = hk_calendar()[[0, -1]].strftime("%Y-%m-%d")

def batches(lst, n=3):
    for i in range(0, len(lst), n):
        yield i // n, lst[i:i+n]

def save_files(raw):
    saved = 0
    for f in (raw.get("files") or []):
        name, content = f.get("name"), f.get("content")
        if name and content:
            os.makedirs(os.path.dirname(name), exist_ok=True)
            if IS_HK:
                frame = pd.read_csv(StringIO(str(content)), dtype={"wind_code": str})
                filter_hk_prices(frame).to_csv(name, index=False, date_format="%Y-%m-%d")
            else:
                with open(name, "w", encoding="utf-8") as fh:
                    fh.write(str(content))
            saved += 1
    return saved

total = (len(codes) + 2) // 3
ok = fail = 0
t_start = time.time()
log = open(os.path.join(outdir, "_download.log"), "a")

for bi, batch in batches(codes):
    bp = os.path.join(outdir, f"batch_{bi:04d}.csv")
    if os.path.exists(bp) and os.path.getsize(bp) > 100:
        if IS_HK:
            clean_hk_price_file(bp)
        ok += 1
        continue
    tickers = ",".join(batch)
    done = False
    for attempt in range(4):
        try:
            with AgentGwClient(timeout=120) as c:
                r = c.tools.call_data_source_tool({
                    "data_source_name": "wind", "api_name": "wind_get_price",
                    "params": {"ticker": tickers, "file_path": bp,
                               "start_date": START, "end_date": END, "price_adj": "F"}})
                raw = r.raw
            if raw.get("is_success") and save_files(raw) > 0:
                done = True
                break
            err = str(raw.get("error"))[:120]
        except Exception as e:
            err = repr(e)[:120]
        time.sleep(15 * (attempt + 1)) if "Too many" in err else time.sleep(5)
    if done:
        ok += 1
    else:
        fail += 1
        log.write(f"FAIL batch {bi} {tickers} err={err}\n"); log.flush()
    time.sleep(4.0)
    if (bi + 1) % 20 == 0:
        el = time.time() - t_start
        msg = f"{time.strftime('%H:%M:%S')} batch {bi+1}/{total} ok={ok} fail={fail} elapsed={el/60:.1f}min\n"
        log.write(msg); log.flush(); print(msg.strip())

log.write(f"ALLDONE total={total} ok={ok} fail={fail} elapsed={(time.time()-t_start)/60:.1f}min\n")
log.close()
print(f"ALLDONE total={total} ok={ok} fail={fail}")
