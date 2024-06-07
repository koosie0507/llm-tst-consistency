from math import isnan, erf

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _compute_areas_to_target(a: list[float], b: list[float], target: list[float]) -> tuple[float, float]:
    area_a = np.trapz(np.abs(np.array(a) - np.array(target)))
    area_b = np.trapz(np.abs(np.array(b) - np.array(target)))
    return area_a, area_b


def _compute_euclidian_norms_to_target(a: list[float], b: list[float], target: list[float]) -> tuple[float, float]:
    distance_a = np.linalg.norm(np.array(a) - np.array(target))
    distance_b = np.linalg.norm(np.array(b) - np.array(target))
    return distance_a, distance_b


def _is_diff_significant(x: float, y: float, confidence: float = 0.95) -> bool:
    c = abs(x - y)
    std_dev = np.std([x, y])
    mu = np.mean([x, y])
    z_score = c - mu / std_dev

    # need to double-check this formula
    p_value = 1 - 0.5 * (1 + erf(z_score / np.sqrt(2)))
    if isnan(p_value):
        return False

    threshold = 1 - confidence
    return p_value < threshold


def make_report(features, ds_stats, df):
    euclid = {}
    area = {}
    ks = {}
    for feature in features:
        diff_col = f"{feature}_diff"
        is_closer_col = f"is_{feature}_closer"
        is_different_col = f"is_{feature}_different"
        target = [ds_stats[feature].mean]*len(df)
        baseline = df[f"baseline_{feature}"].values
        hlf = df[f"hlf_{feature}"].values

        # Kolmogorov-Smirnov test
        ks_bh = ks_2samp(baseline, hlf)
        ks_bt = ks_2samp(baseline, target)
        ks_ht = ks_2samp(hlf, target)
        ks[diff_col] = ks_bt.statistic - ks_ht.statistic
        ks[is_closer_col] = ks_bt.statistic > ks_ht.statistic
        ks[is_different_col] = ks_bh.pvalue < 0.25

        # Euclidian norm
        dist_a, dist_b = _compute_euclidian_norms_to_target(baseline, hlf, target)
        euclid[diff_col] = dist_a - dist_b
        euclid[is_closer_col] = dist_a > dist_b
        euclid[is_different_col] = _is_diff_significant(dist_a, dist_b)

        # Area between curves
        area_a, area_b = _compute_areas_to_target(baseline, hlf, target)
        area[diff_col] = area_a - area_b
        area[is_closer_col] = area_a > area_b
        area[is_different_col] = _is_diff_significant(area_a, area_b)

    return pd.DataFrame(
        columns=list(euclid.keys()),
        data=[euclid, area, ks],
        index=["by euclidian norm", "by area", "k-s test"]
    )
