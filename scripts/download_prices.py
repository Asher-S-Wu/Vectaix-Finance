#!/usr/bin/env python3
"""download_prices.py <universe.csv> <outdir> — 批量下载 A 股 Wind 日K线。"""
import os
import sys
import time

import pandas as pd

from agent_gw import AgentGwClient

universe_csv, outdir = sys.argv[1], sys.argv[2]
START, END = "2015-12-01", "2026-09-09"
os.makedirs(outdir, exist_ok=True)
universe = pd.read_csv(universe_csv, dtype=str)
codes = universe["windcode"].tolist()
if any(code.endswith(".HK") for code in codes):
    raise ValueError("旧港股行情下载已移除")


def batches(items, size=3):
    for index in range(0, len(items), size):
        yield index // size, items[index:index + size]


def save_files(raw):
    saved = 0
    for item in raw.get("files") or []:
        name, content = item.get("name"), item.get("content")
        if name and content:
            os.makedirs(os.path.dirname(name), exist_ok=True)
            with open(name, "w", encoding="utf-8") as stream:
                stream.write(str(content))
            saved += 1
    return saved


total = (len(codes) + 2) // 3
ok = fail = 0
started = time.time()
log = open(os.path.join(outdir, "_download.log"), "a", encoding="utf-8")
for batch_index, batch in batches(codes):
    output = os.path.join(outdir, f"batch_{batch_index:04d}.csv")
    if os.path.exists(output) and os.path.getsize(output) > 100:
        ok += 1
        continue
    tickers = ",".join(batch)
    done = False
    for attempt in range(4):
        try:
            with AgentGwClient(timeout=120) as client:
                response = client.tools.call_data_source_tool({"data_source_name": "wind", "api_name": "wind_get_price",
                    "params": {"ticker": tickers, "file_path": output, "start_date": START, "end_date": END, "price_adj": "F"}})
            raw = response.raw
            if raw.get("is_success") and save_files(raw) > 0:
                done = True
                break
            error = str(raw.get("error"))[:120]
        except Exception as exc:
            error = repr(exc)[:120]
        time.sleep(15 * (attempt + 1) if "Too many" in error else 5)
    if done:
        ok += 1
    else:
        fail += 1
        log.write(f"FAIL batch {batch_index} {tickers} err={error}\n"); log.flush()
    time.sleep(4)
    if (batch_index + 1) % 20 == 0:
        message = f"{time.strftime('%H:%M:%S')} batch {batch_index + 1}/{total} ok={ok} fail={fail} elapsed={(time.time() - started) / 60:.1f}min\n"
        log.write(message); log.flush(); print(message.strip())
log.write(f"ALLDONE total={total} ok={ok} fail={fail} elapsed={(time.time() - started) / 60:.1f}min\n")
log.close()
print(f"ALLDONE total={total} ok={ok} fail={fail}")
