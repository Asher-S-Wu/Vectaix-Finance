#!/usr/bin/env python3
"""train_backtest.py <market:a|hk2> [config_json]
多因子 walk-forward 训练 + 样本外回测 + 指标输出
防未来函数: 信号=T月末, 成交=T+1收盘, 训练样本剔除标签实现日晚于信号日的月份(purge 1个月)
"""
import sys, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

from project_paths import ROOT, data_dir, model_dir, backtest_dir
from factor_lib import compute_factors, forward_labels, month_ends
from hk_universe import load_memberships, quarterly_pool, complete_factors, require_full_portfolio


market = sys.argv[1]
cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
TOPN = cfg.get("topn", 15 if market == "hk2" else 30)
LIQ_GATE = cfg.get("liq_gate", 0.0)      # 流动性门槛: amt_log_20 截面分位数下限
TRAIN_MIN = cfg.get("train_min", 48)     # 最少训练月数
COST = cfg.get("cost", 0.003)            # 双边成本
HOLDOUT_M = cfg.get("holdout_months", 12)
LGB_PARAMS = cfg.get("lgb", dict(num_leaves=31, learning_rate=0.05, n_estimators=300,
                                 subsample=0.8, colsample_bytree=0.8, min_child_samples=50))
LGB_PARAMS.setdefault("n_jobs", 4)
TARGET = cfg.get("target", "gaussian_rank")
MONTH_BALANCED = cfg.get("month_balanced", False)
TRAIN_WINDOW = cfg.get("train_window_months", 0)
if TARGET not in {"gaussian_rank", "percentile_rank", "above_median", "rank_quintile", "hybrid_rank"}:
    raise ValueError(f"未知训练目标: {TARGET}")

t0 = time.time()
processed = data_dir(market) / "processed"
results = backtest_dir(market)
training_results = results / "training"
models = model_dir(market)
for directory in (processed, training_results, models):
    directory.mkdir(parents=True, exist_ok=True)
compression = "gzip" if market == "a" else None
px = pd.read_pickle(processed / "prices.pkl", compression=compression)
val = pd.read_pickle(processed / "valuations.pkl", compression=compression)
close, amt, vol = px["close"], px["amt"], px["volume"]

print(f"[{market}] 面板: {close.shape[1]} 只股票, {close.shape[0]} 交易日")

if "factor_file" in cfg:
    factors = pd.read_pickle(ROOT / cfg["factor_file"])
else:
    factors = compute_factors(close, amt, vol, val_wide=val)
    tradable = close.gt(0) & amt.gt(0) & vol.gt(0)
    eligible = pd.concat([
        pd.DataFrame({"date": d, "code": tradable.columns[tradable.loc[d]]})
        for d in month_ends(close.index)
    ], ignore_index=True)
    factors = factors.merge(eligible, on=["date", "code"], how="inner")
    factors.to_pickle(processed / "factors.pkl")
label_prices = close.where(amt.gt(0) & vol.gt(0)) if market == 'hk2' else close
labels = forward_labels(label_prices)
df = factors.merge(labels, on=["date", "code"], how="left")
FEATS = [c for c in factors.columns if c not in ("date", "code")]
if cfg.get("feats"):
    FEATS = [f for f in cfg["feats"] if f in FEATS]
RECENCY_TAU = cfg.get("recency_tau", 0)  # >0 时按 exp 半衰期(月)加权训练样本
print(f"因子数: {len(FEATS)}, 样本: {len(df)}, 月份: {df.date.nunique()}")

# ---------- 预处理: 截面 MAD 去极值 + zscore ----------
def cs_norm(g):
    for c in FEATS:
        x = g[c]
        med = x.median()
        mad = (x - med).abs().median()
        if mad > 0:
            x = x.clip(med - 5 * 1.4826 * mad, med + 5 * 1.4826 * mad)
        sd = x.std()
        g[c] = (x - x.mean()) / sd if sd > 0 else 0.0
    # 各目标仅从已完整兑现的同月收益构造，训练时仍按标签完成日筛选。
    r = g["fwd_ret"].rank()
    n = r.notna().sum()
    if n <= 5:
        g["y"] = np.nan
    elif TARGET == "gaussian_rank":
        g["y"] = stats.norm.ppf((r - 0.5) / n)
    elif TARGET == "percentile_rank":
        g["y"] = (r - 0.5) / n
    elif TARGET == "above_median":
        g["y"] = (g["fwd_ret"] > g["fwd_ret"].median()).astype(float).where(g["fwd_ret"].notna())
    elif TARGET == "rank_quintile":
        g["y"] = np.floor(5 * (r - 0.5) / n)
    elif TARGET == "hybrid_rank":
        median_side = (g["fwd_ret"] > g["fwd_ret"].median()).astype(float).where(g["fwd_ret"].notna()) * 2 - 1
        g["y"] = 0.5 * stats.norm.ppf((r - 0.5) / n) + 0.5 * median_side
    return g

if market != 'hk2':
    df = pd.concat([cs_norm(g.copy()) for _, g in df.groupby("date")], ignore_index=True)

dates = sorted(df["date"].unique())
realized_dates = sorted(df.loc[df["label_end"].le(close.index[-1]), "date"].unique())
if market == 'hk2':
    memberships = load_memberships()
    # 新股票池的评价期明确从 2024 年起；更早数据只用于当时已知候选股的训练。
    dates = [d for d in dates if d >= pd.Timestamp('2024-01-01')]
    realized_dates = [d for d in realized_dates if d >= pd.Timestamp('2024-01-01')]
print(f"预处理后月份: {len(dates)}, 首个: {pd.Timestamp(dates[0]).date()}, 最后: {pd.Timestamp(dates[-1]).date()}")

# ---------- 单因子 IC(全样本, 供参考) ----------
ic_rows = []
for c in FEATS:
    ics = df.groupby("date").apply(lambda g: g[c].corr(g["fwd_ret"], method="spearman") if g[c].notna().sum() > 10 else np.nan)
    ics = ics.dropna()
    ic_rows.append({"factor": c, "ic_mean": ics.mean(), "icir": ics.mean() / (ics.std() + 1e-12),
                    "ic_pos": (ics > 0).mean()})
ic_df = pd.DataFrame(ic_rows).sort_values("ic_mean", key=abs, ascending=False)
ic_df.to_csv(results / "factor_ic.csv", index=False)
print("单因子 IC Top8:")
print(ic_df.head(8).to_string(index=False))

class LGBEnsemble:
    """多种子 LightGBM 集成, 提升预测稳定性"""
    def __init__(self, params, seeds=(42, 7, 2024)):
        self.params, self.seeds = params, seeds
    def fit(self, X, y, w=None, group=None):
        self.models_ = []
        for s in self.seeds:
            if TARGET == "rank_quintile":
                m = lgb.LGBMRanker(**self.params, random_state=s, verbose=-1)
                m.fit(X, y.astype(int), sample_weight=w, group=group)
            else:
                m = lgb.LGBMRegressor(**self.params, random_state=s, verbose=-1)
                m.fit(X, y, sample_weight=w)
            self.models_.append(m)
        return self
    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models_], axis=0)

# ---------- walk-forward ----------
def sample_weights(frame, as_of):
    if not MONTH_BALANCED and RECENCY_TAU <= 0:
        return None
    weights = pd.Series(1.0, index=frame.index)
    if MONTH_BALANCED:
        weights /= frame.groupby("date")["date"].transform("size")
        weights *= len(frame) / frame["date"].nunique()
    if RECENCY_TAU > 0:
        age = (as_of.year - frame["date"].dt.year) * 12 + (as_of.month - frame["date"].dt.month)
        weights *= np.power(0.5, age.clip(lower=0) / RECENCY_TAU)
    return weights.to_numpy()

holdout_start = realized_dates[-HOLDOUT_M]
preds = []
quarter_frames = {}


def training_frame(as_of):
    if market != 'hk2':
        return df
    quarter = pd.Timestamp(as_of).to_period('Q')
    if quarter not in quarter_frames:
        pool = quarterly_pool(close, amt, vol, memberships, as_of)
        codes = pool.loc[pool.selected, 'code']
        frame = df[df.code.isin(codes) & df.date.le(quarter.end_time)]
        frame = frame[complete_factors(frame, FEATS)]
        quarter_frames[quarter] = pd.concat([cs_norm(g.copy()) for _, g in frame.groupby('date')], ignore_index=True)
    return quarter_frames[quarter]


for i, dtest in enumerate(dates):
    current = training_frame(dtest)
    tr = current[current["date"].lt(dtest) & current["label_end"].le(dtest) & current["y"].notna()]
    if TRAIN_WINDOW > 0:
        tr = tr[tr["date"].isin(tr["date"].drop_duplicates().iloc[-TRAIN_WINDOW:])]
    if tr["date"].nunique() < TRAIN_MIN:
        continue
    te = current[current["date"] == dtest]
    require_full_portfolio(te, TOPN)
    Xtr, ytr = tr[FEATS], tr["y"]
    wtr = sample_weights(tr, pd.Timestamp(dtest))
    Xte = te[FEATS]
    model = LGBEnsemble(LGB_PARAMS).fit(Xtr, ytr, wtr, tr.groupby("date", sort=False).size().to_numpy())
    p = model.predict(Xte)
    out = te[["date", "code", "fwd_ret", "amt_log_20", "label_start", "label_end"]].copy()
    out["pred"] = p
    out["is_holdout"] = dtest >= holdout_start
    preds.append(out)
    if len(preds) % 12 == 0:
        print(f"  walk-forward {len(preds)} 个月, 当前 {pd.Timestamp(dtest).date()}, {time.time()-t0:.0f}s")

pred = pd.concat(preds)
pred.to_pickle(results / "predictions.pkl")

# ---------- 回测 ----------
completed_pred = pred[pred["label_end"].le(close.index[-1])]
bench = completed_pred.groupby("date")["fwd_ret"].mean()  # 池内等权基准
recs = []
prev_hold = None
for d, g in completed_pred.groupby("date"):
    gsel = g
    if LIQ_GATE > 0:
        thr = g["amt_log_20"].quantile(LIQ_GATE)
        gsel = g[g["amt_log_20"] >= thr]
    top = gsel.nlargest(TOPN, "pred")
    cur = set(top["code"])
    if prev_hold is None:
        turn = 1.0
    else:
        turn = 1 - len(cur & prev_hold) / TOPN
    prev_hold = cur
    cost = turn * COST
    r_top = top["fwd_ret"].mean() - cost if top["fwd_ret"].notna().all() else np.nan
    r_b = bench.loc[d]
    r_med = g["fwd_ret"].median()
    hit = (top["fwd_ret"] > r_b).mean()
    hit_med = (top["fwd_ret"] > r_med).mean()
    recs.append({"date": d, "r_top": r_top, "r_bench": r_b, "excess": r_top - r_b,
                 "hit_rate": hit, "hit_rate_med": hit_med, "turnover": turn,
                 "selected_stocks": len(top), "missing_return_stocks": int(top["fwd_ret"].isna().sum()),
                 "is_holdout": bool(g["is_holdout"].iloc[0]),
                 "ic": g["pred"].corr(g["fwd_ret"], method="spearman")})
bt = pd.DataFrame(recs).set_index("date")
bt.to_pickle(training_results / "returns.pkl")

def metrics_of(sub):
    n = len(sub)
    if sub["r_top"].isna().any():
        return {"status": "missing_execution_prices", "oos_months": int(n),
                "missing_return_months": int(sub["r_top"].isna().sum()),
                "missing_return_stock_months": int(sub["missing_return_stocks"].sum())}
    ann_excess = sub["excess"].mean() * 12
    ir = sub["excess"].mean() / (sub["excess"].std() + 1e-12) * np.sqrt(12)
    return {
        "ic_mean": float(sub["ic"].mean()),
        "icir": float(sub["ic"].mean() / (sub["ic"].std() + 1e-12)),
        "ic_pos_ratio": float((sub["ic"] > 0).mean()),
        "ann_excess": float(ann_excess),
        "info_ratio": float(ir),
        "monthly_win": float((sub["excess"] > 0).mean()),
        "hit_rate": float(sub["hit_rate"].mean()),
        "hit_rate_med": float(sub["hit_rate_med"].mean()),
        "turnover_ann": float(sub["turnover"].mean() * 12),
        "oos_months": int(n),
        "ann_ret_top": float(sub["r_top"].mean() * 12),
        "ann_ret_bench": float(sub["r_bench"].mean() * 12),
        "max_dd_top": float(((1 + sub["r_top"]).cumprod() / (1 + sub["r_top"]).cumprod().cummax() - 1).min()),
        "max_dd_bench": float(((1 + sub["r_bench"]).cumprod() / (1 + sub["r_bench"]).cumprod().cummax() - 1).min()),
    }

m_all = metrics_of(bt)
m_ho = metrics_of(bt[bt["is_holdout"]])
json.dump(m_all, open(training_results / "metrics.json", "w"), indent=2)
json.dump(m_ho, open(training_results / "metrics_recent.json", "w"), indent=2)

# ---------- 全量终训, 保存活体模型(用于实盘选股) ----------
current = training_frame(close.index[-1])
training = current[current["label_end"].le(close.index[-1]) & current["y"].notna()].copy()
if TRAIN_WINDOW > 0:
    training = training[training["date"].isin(training["date"].drop_duplicates().iloc[-TRAIN_WINDOW:])]
w_full = sample_weights(training, close.index[-1])
final_model = LGBEnsemble(LGB_PARAMS).fit(training[FEATS], training["y"], w_full,
                                       training.groupby("date", sort=False).size().to_numpy())
pd.to_pickle({"model": final_model, "features": FEATS, "trained_at": str(pd.Timestamp.now()),
              "market": market, "params": LGB_PARAMS, "config": cfg, "target": TARGET,
              "month_balanced": MONTH_BALANCED, "data_as_of": str(close.index[-1].date()),
              "last_training_signal": str(training["date"].max().date()),
              "latest_label_end": str(training["label_end"].max().date()),
              "universe_policy": "HSCI quarterly, max 500, 60-day amount >= HKD 10m, two-year history" if market == 'hk2' else None,
              "library_version": lgb.__version__, "selection_history": "historically tuned; not a fresh holdout"},
             open(models / "model.pkl", "wb"))
# 特征重要性(首个seed)
imp = pd.Series(final_model.models_[0].feature_importances_, index=FEATS).sort_values(ascending=False)
imp.to_csv(models / "importance.csv")
print("活体模型已保存, 特征重要性 Top8:")
print(imp.head(8).to_string())
print(f"\n=== {market} 全样本外 ({m_all['oos_months']}月) ===")
for k, v in m_all.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
print("=== 最后12个月 ===", json.dumps(m_ho, ensure_ascii=False))
print(f"耗时 {time.time()-t0:.0f}s")
