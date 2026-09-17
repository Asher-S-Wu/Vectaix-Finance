"""旧港股实验模型的反序列化兼容类型。"""
import numpy as np


class HKEnsemble:
    """仅用于读取已保存模型并生成其历史预测，不提供训练或选股流程。"""

    def __init__(self, params, target, seeds=(42, 7, 2024), recency_half_life=0):
        self.params, self.target, self.seeds = params, target, seeds
        self.recency_half_life = recency_half_life

    def predict(self, values):
        if self.target == "linear":
            valid = values.notna().all(axis=1)
            result = np.full(len(values), np.nan)
            result[valid] = self.models_[0].predict(values.loc[valid])
            return result
        return np.mean([model.predict(values) for model in self.models_], axis=0)
