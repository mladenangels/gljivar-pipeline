"""Model features, kept identical to backtest/model2.py so the job does not need the backtest."""
import numpy as np

RAIN_BINS = [(1, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 18), (19, 22), (23, 30), (31, 45), (46, 60)]


def features(RR, TG):
    """RR: [n, lags 0..60] mm, TG: [n, lags 0..30] deg C."""
    f = [np.log1p(RR[:, a:b + 1].sum(1)) for a, b in RAIN_BINS]
    names = [f"rain_{a}_{b}" for a, b in RAIN_BINS]
    t7, t30 = TG[:, 0:7].mean(1), TG[:, 7:31].mean(1)
    f += [t7, t7 ** 2, t30, t30 ** 2, TG[:, 0:14].min(1)]
    names += ["t7", "t7_sq", "t8_30", "t8_30_sq", "tmin_14"]
    return np.column_stack(f), names
