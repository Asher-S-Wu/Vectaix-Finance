#!/usr/bin/env python3
"""download_supervisor.py — 双通道(Wind优先,iFinD兜底)补齐全部行情批次
Wind: 单次全区间 3标的/批; iFinD: 3年分段. 断点续传, 限流自动退避."""
import os, time
from pathlib import Path
import pandas as pd
from io import StringIO
from agent_gw import AgentGwClient
from hk_market_data import clean_hk_price_file, filter_hk_prices, hk_calendar, hk_ifind_code
from project_paths import data_dir

CHUNKS = [("2015-12-01","2018-11-30"), ("2018-12-01","2021-11-30"),
          ("2021-12-01","2024-11-30"), ("2024-12-01","2026-09-09")]
START, END = "2015-12-01", "2026-09-09"

def save_files(raw, is_hk):
    n = 0
    for f in (raw.get("files") or []):
        if f.get("name") and f.get("content"):
            os.makedirs(os.path.dirname(f["name"]), exist_ok=True)
            if is_hk:
                frame = pd.read_csv(StringIO(str(f["content"])), dtype={"wind_code": str})
                filter_hk_prices(frame).to_csv(f["name"], index=False, date_format="%Y-%m-%d")
            else:
                with open(f["name"], "w") as fh:
                    fh.write(str(f["content"]))
            n += 1
    return n

def wind_batch(tickers, bp, is_hk):
    start, end = hk_calendar()[[0, -1]].strftime("%Y-%m-%d") if is_hk else (START, END)
    with AgentGwClient(timeout=120) as c:
        r = c.tools.call_data_source_tool({"data_source_name":"wind","api_name":"wind_get_price",
            "params":{"ticker":tickers,"file_path":bp,"start_date":start,"end_date":end,"price_adj":"F"}})
        raw = r.raw
    if raw.get("is_success"):
        return save_files(raw, is_hk) > 0, ""
    return False, str(raw.get("error"))[:80]

def ifind_batch(tickers_q, bp, is_hk):
    parts = []
    chunks = CHUNKS
    if is_hk:
        end = hk_calendar()[-1].strftime("%Y-%m-%d")
        chunks = [(s, min(e, end)) for s, e in CHUNKS if s <= end]
    for (s,e) in chunks:
        with AgentGwClient(timeout=120) as c:
            r = c.tools.call_data_source_tool({"data_source_name":"ifind","api_name":"ifind_get_price",
                "params":{"ticker":tickers_q,"file_path":"/tmp/_ifind_sup.csv","start_date":s,"end_date":e,"adjust":"forward"}})
            raw = r.raw
        if raw.get("is_success") and raw.get("files"):
            content = str(raw["files"][0].get("content",""))
            if len(content) > 100:
                parts.append(pd.read_csv(StringIO(content)))
        else:
            err = str(raw.get("error"))[:80]
            if "EMPTY_DATA" not in err:
                return False, err
    if not parts:
        return False, "empty"
    d = pd.concat(parts).drop_duplicates(["time","thscode"]).sort_values("time")
    d = d.rename(columns={"time":"trade_date","thscode":"wind_code"})
    if is_hk:
        d["wind_code"] = d["wind_code"].map(lambda c: c.split(".")[0].zfill(5)+".HK")
    d["trade_date"] = pd.to_datetime(d["trade_date"].astype(str).str.replace("-",""), format="%Y%m%d")
    d["amt"] = d["volume"] * (d["open"]+d["close"])/2
    if is_hk:
        d = filter_hk_prices(d)
    d[["trade_date","wind_code","open","high","low","close","volume","amt"]].to_csv(bp, index=False)
    return True, ""

def run_market(uni_csv, outdir, is_hk):
    uni = pd.read_csv(uni_csv, dtype=str)
    codes = uni["windcode"].tolist()
    batches = [(i//3, codes[i:i+3]) for i in range(0, len(codes), 3)]
    if is_hk:
        for bi, _ in batches:
            path = Path(outdir) / f"batch_{bi:04d}.csv"
            if path.exists() and path.stat().st_size > 100:
                clean_hk_price_file(path)
    missing = [(bi,b) for bi,b in batches
               if not (os.path.exists(f"{outdir}/batch_{bi:04d}.csv") and os.path.getsize(f"{outdir}/batch_{bi:04d}.csv")>100)]
    print(f"{outdir}: {len(batches)-len(missing)}/{len(batches)} 已存在, 缺 {len(missing)}", flush=True)
    ok=fail=0
    for bi, batch in missing:
        bp = f"{outdir}/batch_{bi:04d}.csv"
        tickers = ",".join(batch)
        tickers_q = ",".join(hk_ifind_code(w) if is_hk else w for w in batch)
        done = False
        # 先试 Wind(单call全区间)
        try:
            done, err = wind_batch(tickers, bp, is_hk)
        except Exception as ex:
            err = repr(ex)[:80]
        if not done:
            time.sleep(6)
            for att in range(3):
                try:
                    done, err = ifind_batch(tickers_q, bp, is_hk)
                except Exception as ex:
                    err = repr(ex)[:80]
                if done: break
                time.sleep(10*(att+1))
        if done: ok+=1
        else:
            fail+=1
            print(f"  FAIL batch {bi} {tickers} {err}", flush=True)
        time.sleep(3)
    print(f"{outdir} 本轮 ok={ok} fail={fail}", flush=True)
    return fail

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:  # 单市场模式: uni_csv outdir is_hk
        uni_csv, outdir, is_hk = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
        for rnd in range(8):
            f = run_market(uni_csv, outdir, is_hk)
            if f == 0:
                print("ALL COMPLETE", flush=True); break
            print(f"轮次 {rnd+1} 结束, 剩余失败={f}, 休息90s", flush=True)
            time.sleep(90)
    else:
        for rnd in range(8):  # 最多8轮补漏
            f1 = run_market(data_dir("a") / "reference/universe.csv", data_dir("a") / "raw/prices", False)
            f2 = run_market(data_dir("hk2") / "reference/universe.csv", data_dir("hk2") / "raw/prices", True)
            if f1==0 and f2==0:
                print("ALL COMPLETE", flush=True); break
            print(f"轮次 {rnd+1} 结束, 剩余失败 a={f1} hk={f2}, 休息90s", flush=True)
            time.sleep(90)
