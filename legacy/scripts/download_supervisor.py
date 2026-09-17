#!/usr/bin/env python3
"""download_supervisor.py — 补齐 A 股行情批次，Wind 优先、iFinD 补充。"""
import os
import time
import argparse
from io import StringIO
from pathlib import Path

import pandas as pd

from project_paths import data_dir

CHUNKS = [("2015-12-01", "2018-11-30"), ("2018-12-01", "2021-11-30"),
          ("2021-12-01", "2024-11-30"), ("2024-12-01", "2026-09-09")]
START, END = "2015-12-01", "2026-09-09"


def save_files(raw):
    count = 0
    for item in raw.get("files") or []:
        if item.get("name") and item.get("content"):
            os.makedirs(os.path.dirname(item["name"]), exist_ok=True)
            with open(item["name"], "w", encoding="utf-8") as stream:
                stream.write(str(item["content"]))
            count += 1
    return count


def wind_batch(tickers, output):
    from agent_gw import AgentGwClient
    with AgentGwClient(timeout=120) as client:
        response = client.tools.call_data_source_tool({"data_source_name": "wind", "api_name": "wind_get_price",
            "params": {"ticker": tickers, "file_path": output, "start_date": START, "end_date": END, "price_adj": "F"}})
    raw = response.raw
    return (save_files(raw) > 0, "") if raw.get("is_success") else (False, str(raw.get("error"))[:80])


def ifind_batch(tickers, output):
    from agent_gw import AgentGwClient
    parts = []
    for start, end in CHUNKS:
        with AgentGwClient(timeout=120) as client:
            response = client.tools.call_data_source_tool({"data_source_name": "ifind", "api_name": "ifind_get_price",
                "params": {"ticker": tickers, "file_path": "/tmp/_ifind_sup.csv", "start_date": start, "end_date": end, "adjust": "forward"}})
        raw = response.raw
        if raw.get("is_success") and raw.get("files"):
            content = str(raw["files"][0].get("content", ""))
            if len(content) > 100:
                parts.append(pd.read_csv(StringIO(content)))
        elif "EMPTY_DATA" not in str(raw.get("error")):
            return False, str(raw.get("error"))[:80]
    if not parts:
        return False, "empty"
    data = pd.concat(parts).drop_duplicates(["time", "thscode"]).sort_values("time")
    data = data.rename(columns={"time": "trade_date", "thscode": "wind_code"})
    data["trade_date"] = pd.to_datetime(data["trade_date"].astype(str).str.replace("-", ""), format="%Y%m%d")
    data["amt"] = data["volume"] * (data["open"] + data["close"]) / 2
    data[["trade_date", "wind_code", "open", "high", "low", "close", "volume", "amt"]].to_csv(output, index=False)
    return True, ""


def run_market(universe_file, output_dir):
    universe = pd.read_csv(universe_file, dtype=str)
    codes = universe["windcode"].tolist()
    if any(code.endswith(".HK") for code in codes):
        raise ValueError("旧港股行情下载已移除")
    batches = [(index // 3, codes[index:index + 3]) for index in range(0, len(codes), 3)]
    missing = [(index, batch) for index, batch in batches if not (os.path.exists(f"{output_dir}/batch_{index:04d}.csv") and os.path.getsize(f"{output_dir}/batch_{index:04d}.csv") > 100)]
    print(f"{output_dir}: {len(batches) - len(missing)}/{len(batches)} 已存在, 缺 {len(missing)}", flush=True)
    ok = fail = 0
    for index, batch in missing:
        output, tickers = f"{output_dir}/batch_{index:04d}.csv", ",".join(batch)
        try:
            done, error = wind_batch(tickers, output)
        except Exception as exc:
            done, error = False, repr(exc)[:80]
        if not done:
            time.sleep(6)
            for attempt in range(3):
                try:
                    done, error = ifind_batch(tickers, output)
                except Exception as exc:
                    done, error = False, repr(exc)[:80]
                if done:
                    break
                time.sleep(10 * (attempt + 1))
        if done:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL batch {index} {tickers} {error}", flush=True)
        time.sleep(3)
    print(f"{output_dir} 本轮 ok={ok} fail={fail}", flush=True)
    return fail


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="补齐 A 股行情批次；可选指定 A 股 universe.csv 和输出目录。")
    parser.add_argument("paths", nargs="*", metavar="PATH",
                        help="可选：<universe.csv> <outdir>")
    args = parser.parse_args(argv)
    if len(args.paths) not in {0, 2}:
        parser.error("只接受两个可选位置参数：<universe.csv> <outdir>；旧 is_hk 参数已移除")
    if not args.paths:
        return data_dir("a") / "reference/universe.csv", data_dir("a") / "raw/prices"
    return Path(args.paths[0]), Path(args.paths[1])


if __name__ == "__main__":
    universe_file, output_dir = parse_arguments()
    for round_number in range(8):
        failures = run_market(universe_file, output_dir)
        if failures == 0:
            print("ALL COMPLETE", flush=True)
            break
        print(f"轮次 {round_number + 1} 结束, 剩余失败={failures}, 休息90s", flush=True)
        time.sleep(90)
