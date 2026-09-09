"""将各期已保存的预测合成组合分数，供历史回测和最新信号共用。"""
def zscore(values):
    sd = values.std()
    return (values - values.mean()) / sd if sd > 0 else values * 0


def blend_scores(predictions, factors, blend_factor=None, blend_w=0.0,
                 normalize_predictions=False):
    out = predictions.copy()
    blend = bool(blend_factor and blend_w)
    out["base_score"] = (out.groupby("date")["pred"].transform(zscore)
                         if blend or normalize_predictions else out["pred"])
    if blend:
        values = factors[["date", "code", blend_factor]].rename(columns={blend_factor: "blend_raw"})
        out = out.merge(values, on=["date", "code"], how="left",
                        validate="one_to_one", indicator="factor_match")
        missing = out["factor_match"].ne("both")
        if missing.any():
            raise ValueError(f"缺少混合因子对应记录: {out.loc[missing, ['date', 'code']].to_dict('records')[:5]}")
        out = out.drop(columns="factor_match")
        out["base_score"] += blend_w * out.groupby("date")["blend_raw"].transform(zscore)
    return out


def smooth_scores(scores, weight):
    out = scores.sort_values(["code", "date"]).copy()
    out["score"] = out["base_score"]
    if weight > 0:
        previous = out.groupby("code")["base_score"].shift(1)
        out["score"] = (1 - weight) * out["base_score"] + weight * previous
    return out
