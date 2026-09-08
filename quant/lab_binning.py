"""Batch the existing weighted quantiles without changing tree bin boundaries."""

# Adapted from scikit-learn 1.9.0:
# sklearn/ensemble/_hist_gradient_boosting/binning.py
# sklearn/utils/stats.py
# The finite one-dimensional positive-weight branch computes the original
# percentile searches together. Sorting, CDF accumulation, averaging, duplicate
# removal and clipping retain the original rules. Other binning branches are
# unchanged. Unsupported weighted inputs raise an error; none are substituted.
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2007-2026 The scikit-learn developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import sklearn


SUPPORTED_SKLEARN_VERSION = "1.9.0"


def _positive_weighted_percentiles(col_data, sample_weight, percentiles):
    """Vectorize the pinned weighted-percentile algorithm for real 1-D inputs."""
    if col_data.ndim != 1 or col_data.dtype.kind not in "fibu" or sample_weight.shape != col_data.shape:
        raise ValueError("批量加权分箱要求一维实数特征及同形状权重。")
    floating_dtype = col_data.dtype if col_data.dtype.kind == "f" else np.asarray(0.0).dtype
    array = np.asarray(col_data, dtype=floating_dtype).reshape(-1, 1)
    weights = np.asarray(sample_weight, dtype=floating_dtype).reshape(-1, 1)
    ranks = np.asarray(percentiles, dtype=floating_dtype)
    if not np.isfinite(array).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("批量加权分箱要求剔除缺失值后的特征有限，且全部权重有限并严格为正。")

    # Keep the original second sort in the same N-by-1 shape and axis. Its tie
    # order also determines the floating-point accumulation order of weights.
    sorted_idx = np.argsort(array, axis=0, stable=False)
    sorted_weights = np.take_along_axis(weights, sorted_idx, axis=0)
    weight_cdf = np.cumsum(sorted_weights.T, axis=1)[0]
    if not np.isfinite(weight_cdf).all():
        raise ValueError("批量加权分箱的累计权重必须有限。")
    adjusted_ranks = ranks / 100 * weight_cdf[-1]
    zero_ranks = adjusted_ranks == 0
    adjusted_ranks[zero_ranks] = np.nextafter(
        adjusted_ranks[zero_ranks], adjusted_ranks[zero_ranks] + 1,
    )
    indices = np.searchsorted(weight_cdf, adjusted_ranks)
    max_idx = len(array) - 1
    indices = np.clip(indices, 0, max_idx)
    next_indices = np.clip(indices + 1, 0, max_idx)
    values = array[sorted_idx[indices, 0], 0]
    next_values = array[sorted_idx[next_indices, 0], 0]
    fraction_above = weight_cdf[indices] - adjusted_ranks
    return np.where(
        fraction_above > np.finfo(floating_dtype).eps,
        values,
        (values + next_values) / 2,
    )


def batch_binning_thresholds(col_data, max_bins, sample_weight=None):
    """Compute the same thresholds, sharing sorting and CDF across percentiles."""
    from sklearn.ensemble._hist_gradient_boosting.common import ALMOST_INF, X_DTYPE

    missing_mask = np.isnan(col_data)
    any_missing = missing_mask.any()
    if any_missing:
        col_data = col_data[~missing_mask]
    if sample_weight is not None:
        if any_missing:
            sample_weight = sample_weight[~missing_mask]
        nnz_sw = sample_weight != 0
        col_data = col_data[nnz_sw]
        sample_weight = sample_weight[nnz_sw]

    sort_idx = np.argsort(col_data)
    col_data = col_data[sort_idx]
    if sample_weight is not None:
        sample_weight = sample_weight[sort_idx]
    distinct_values = np.unique(col_data).astype(X_DTYPE)
    if len(distinct_values) == 1:
        return np.asarray([])
    if len(distinct_values) <= max_bins:
        bin_thresholds = sliding_window_view(distinct_values, 2).mean(axis=1)
    elif sample_weight is None:
        percentiles = np.linspace(0, 100, num=max_bins + 1)[1:-1]
        bin_thresholds = np.percentile(
            col_data, percentiles, method="averaged_inverted_cdf",
        )
        assert bin_thresholds.shape[0] == max_bins - 1
    else:
        percentiles = np.linspace(0, 100, num=max_bins + 1)[1:-1]
        bin_thresholds = _positive_weighted_percentiles(col_data, sample_weight, percentiles)
        assert bin_thresholds.shape[0] == max_bins - 1
    unique_bin_values = np.unique(bin_thresholds)
    if unique_bin_values.shape[0] != bin_thresholds.shape[0]:
        bin_thresholds = unique_bin_values
    np.clip(bin_thresholds, a_min=None, a_max=ALMOST_INF, out=bin_thresholds)
    return bin_thresholds


def install_batch_binning() -> None:
    """Install once at training startup, before creating any model workers."""
    if sklearn.__version__ != SUPPORTED_SKLEARN_VERSION:
        raise RuntimeError(
            f"批量分箱只适用于 scikit-learn {SUPPORTED_SKLEARN_VERSION}，"
            f"当前为 {sklearn.__version__}。",
        )
    from sklearn.ensemble._hist_gradient_boosting import binning

    binning._find_binning_thresholds = batch_binning_thresholds
