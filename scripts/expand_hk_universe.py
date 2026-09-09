#!/usr/bin/env python3
"""通过 Wind search_stocks 按市值分段拉取港股扩池清单(绕过单次100条上限)。"""
import argparse, csv, json, os, re, subprocess, time
from pathlib import Path
from project_paths import data_dir

CLI = ["node", "skills/wind-mcp-skill/scripts/cli.mjs",
       "call", "stock_data", "search_stocks"]  # 相对路径+cwd, 绝对路径会导致0字节输出

# 市值分段(亿港元), 每段目标<100只
BANDS = [
    (1500, None), (800, 1500), (400, 800), (250, 400),
    (150, 250), (100, 150), (70, 100), (50, 70),
]

def norm_code(w: str) -> str:
    """0659.HK -> 00659.HK (5位存储约定)"""
    num, ex = w.split(".")
    return f"{int(num):05d}.{ex}"

def _raw(question: str, plugin_dir: Path) -> str:
    tmp = f"/tmp/wind_search_{os.getpid()}.json"
    with open(tmp, "w") as f:
        subprocess.run(CLI + [question], stdout=f, stderr=subprocess.DEVNULL,
                       timeout=180, cwd=plugin_dir)
    return open(tmp).read()

def search(question: str, plugin_dir: Path) -> list:
    # Wind CLI 间歇性 0 字节输出: 空结果重试至多4次
    qj = json.dumps({"question": question.replace(" ", "")}, ensure_ascii=False)
    out = ""
    for attempt in range(4):
        out = _raw(qj, plugin_dir)
        if out.strip():
            break
        time.sleep(15)
    rows = []
    # 提取 data_preview 的 CSV 行: code,简称,市值[,币种]
    for m in re.finditer(r"(\d{1,5}\.HK),([^,\\]+),([\d.]+)", out):
        rows.append((norm_code(m.group(1)), m.group(2), float(m.group(3))))
    return rows

def main():
    parser = argparse.ArgumentParser(description="通过已安装的 Wind 插件更新港股股票池")
    parser.add_argument("--wind-plugin", type=Path, required=True, help="Wind 插件目录，包含 skills/wind-mcp-skill/scripts/cli.mjs")
    args = parser.parse_args()
    plugin_dir = args.wind_plugin.resolve()
    seen = {}
    for lo, hi in BANDS:
        if hi is None:
            q = f"筛选港股市场总市值超过{lo}亿港元的股票"
        else:
            q = f"筛选港股市场总市值{lo}亿到{hi}亿港元的股票"
        for attempt in range(3):
            try:
                rows = search(q, plugin_dir)
                break
            except Exception as e:
                print(f"[retry{attempt}] {q}: {e}", flush=True)
                time.sleep(20)
        else:
            rows = []
        n_new = 0
        for code, name, cap in rows:
            if code not in seen:
                seen[code] = (name, cap)
                n_new += 1
        print(f"{q} -> 返回{len(rows)}只, 新增{n_new}, 累计{len(seen)}", flush=True)
        if len(rows) >= 100:
            print(f"  !! 该段可能触顶100, 需细分", flush=True)
        time.sleep(3)  # 防限流
    out = data_dir("hk2") / "reference/universe_expanded.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["windcode", "name", "mktcap_hkd_yi"])
        for code, (name, cap) in sorted(seen.items(), key=lambda x: -x[1][1]):
            w.writerow([code, name, cap])
    print(f"saved {len(seen)} -> {out}")
    with (data_dir("hk2") / "reference/universe_index.csv").open(newline="") as f:
        names = {row["windcode"]: row["name"] for row in csv.DictReader(f)}
    names.update({code: name for code, (name, _) in seen.items()})
    with (data_dir("hk2") / "reference/universe.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["windcode", "name"])
        writer.writerows(sorted(names.items()))

if __name__ == "__main__":
    main()
