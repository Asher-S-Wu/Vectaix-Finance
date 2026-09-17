#!/usr/bin/env python3
# 当前模型评价标准，命中率按跑赢池内中位数统计。
import json, sys

THRESHOLDS = {
    "ic_mean":        {"min": 0.03,  "desc": "月度Rank IC均值"},
    "icir":           {"min": 0.30,  "desc": "Rank ICIR"},
    "ic_pos_ratio":   {"min": 0.55,  "desc": "IC为正月份占比"},
    "ann_excess":     {"min": 0.08,  "desc": "TopN年化超额(扣费后)"},
    "info_ratio":     {"min": 0.50,  "desc": "超额信息比率"},
    "monthly_win":    {"min": 0.55,  "desc": "组合相对基准月胜率"},
    "hit_rate_med":   {"min": 0.55,  "desc": "选股命中率(TopN跑赢中位数比例)"},
    "turnover_ann":   {"max": 30.0,  "desc": "年化换手(倍)"},
    "oos_months":     {"min": 36,    "desc": "样本外月份数"},
}

def check(metrics):
    results = []
    for k, rule in THRESHOLDS.items():
        v = metrics.get(k)
        if v is None:
            results.append((k, rule["desc"], None, False, "缺失"))
            continue
        ok = (v >= rule["min"]) if "min" in rule else (v <= rule["max"])
        results.append((k, rule["desc"], v, ok, f"门槛 {rule.get('min', rule.get('max'))}"))
    return results

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        m = json.load(f)
    results = check(m)
    allpass = all(r[3] for r in results)
    print(f"文件: {sys.argv[1]}")
    for k, desc, v, ok, thr in results:
        vs = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc} ({k}) = {vs}  ({thr})")
    hm = m.get("hit_rate")
    if hm is not None:
        print(f"  [INFO] 命中率(vs均值, 参考) = {hm:.4f}")
    print(f"总体: {'PASS' if allpass else 'FAIL'}")
    sys.exit(0 if allpass else 1)
