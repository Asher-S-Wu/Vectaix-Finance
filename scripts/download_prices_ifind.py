#!/usr/bin/env python3
"""download_prices_ifind.py <universe.csv> <outdir> [hk2] — 经 iFinD 下载日K线
3标的/批 × 3年分段, 重试, 断点续传. 输出列统一为: trade_date,wind_code,open,high,low,close,volume,amt(近似)"""
import sys, os, time
import pandas as pd
from agent_gw import AgentGwClient
from hk_market_data import clean_hk_price_file, filter_hk_prices, hk_calendar, hk_ifind_code

universe_csv, outdir = sys.argv[1], sys.argv[2]
IS_HK = len(sys.argv) > 3 and sys.argv[3] == "hk2"
os.makedirs(outdir, exist_ok=True)
CHUNKS = [("2015-12-01","2018-11-30"), ("2018-12-01","2021-11-30"),
          ("2021-12-01","2024-11-30"), ("2024-12-01","2026-09-09")]
if IS_HK:
    end = hk_calendar()[-1].strftime("%Y-%m-%d")
    CHUNKS = [(s, min(e, end)) for s, e in CHUNKS if s <= end]

uni = pd.read_csv(universe_csv, dtype=str)
codes = uni["windcode"].tolist()
def qcode(w):
    return hk_ifind_code(w) if IS_HK else w
def ncode(c):  # 输出规范代码(港股5位)
    if IS_HK and c.endswith(".HK"):
        return c.split(".")[0].zfill(5) + ".HK"
    return c

def batches(lst, n=3):
    for i in range(0, len(lst), n):
        yield i // n, lst[i:i+n]

log = open(os.path.join(outdir, "_download_ifind.log"), "a")
ok = fail = 0
t0 = time.time()
total = (len(codes) + 2) // 3

for bi, batch in batches(codes):
    bp = os.path.join(outdir, f"batch_{bi:04d}.csv")
    if os.path.exists(bp) and os.path.getsize(bp) > 100:
        if IS_HK:
            clean_hk_price_file(bp)
        ok += 1; continue
    tickers = ",".join(qcode(w) for w in batch)
    parts = []
    err = ""
    for (s, e) in CHUNKS:
        got = False
        for attempt in range(4):
            try:
                with AgentGwClient(timeout=120) as c:
                    r = c.tools.call_data_source_tool({
                        "data_source_name": "ifind", "api_name": "ifind_get_price",
                        "params": {"ticker": tickers, "file_path": "/tmp/_ifind_tmp.csv",
                                   "start_date": s, "end_date": e, "adjust": "forward"}})
                    raw = r.raw
                if raw.get("is_success") and raw.get("files"):
                    content = str(raw["files"][0].get("content", ""))
                    if len(content) > 100:
                        from io import StringIO
                        parts.append(pd.read_csv(StringIO(content)))
                        got = True
                        break
                err = str(raw.get("error"))[:100]
                if "EMPTY_DATA" in err:
                    break  # 上市晚于该时段, 不重试
            except Exception as ex:
                err = repr(ex)[:100]
            time.sleep(4 * (attempt + 1))
        if not got:
            log.write(f"CHUNKFAIL batch {bi} {tickers} {s} err={err}\n"); log.flush()
    if parts:
        d = pd.concat(parts).drop_duplicates(["time", "thscode"]).sort_values("time")
        d = d.rename(columns={"time": "trade_date", "thscode": "wind_code"})
        d["wind_code"] = d["wind_code"].map(ncode)
        d["trade_date"] = pd.to_datetime(d["trade_date"].astype(str).str.replace("-",""), format="%Y%m%d")
        # 成交额近似 = 量 × (开+收)/2
        d["amt"] = d["volume"] * (d["open"] + d["close"]) / 2
        d = d[["trade_date","wind_code","open","high","low","close","volume","amt"]]
        if IS_HK:
            d = filter_hk_prices(d)
        d.to_csv(bp, index=False)
        ok += 1
    else:
        fail += 1
        log.write(f"FAIL batch {bi} {tickers}\n"); log.flush()
    if (bi + 1) % 20 == 0:
        msg = f"{time.strftime('%H:%M:%S')} batch {bi+1}/{total} ok={ok} fail={fail} el={(time.time()-t0)/60:.1f}min\n"
        log.write(msg); log.flush(); print(msg.strip())
    time.sleep(0.5)
log.write(f"ALLDONE total={total} ok={ok} fail={fail}\n"); log.close()
print(f"ALLDONE total={total} ok={ok} fail={fail}")
