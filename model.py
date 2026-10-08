"""Model features, kept identical to backtest/model2.py so the job does not need the backtest."""
import numpy as np

DD_LAGS = 14  # night dewpoint depression lags 0..13 (see backtest/dew_test.py)
RAIN_BINS = [(1, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 18), (19, 22), (23, 30), (31, 45), (46, 60)]


def features(RR, TG):
    """RR: [n, lags 0..60] mm, TG: [n, lags 0..30] deg C."""
    f = [np.log1p(RR[:, a:b + 1].sum(1)) for a, b in RAIN_BINS]
    names = [f"rain_{a}_{b}" for a, b in RAIN_BINS]
    t7, t30 = TG[:, 0:7].mean(1), TG[:, 7:31].mean(1)
    f += [t7, t7 ** 2, t30, t30 ** 2, TG[:, 0:14].min(1)]
    names += ["t7", "t7_sq", "t8_30", "t8_30_sq", "tmin_14"]
    return np.column_stack(f), names


def dew_features(DD, kind):
    """DD: [n, lags 0..13] night dewpoint depression (deg C). Same as backtest/dew_test.py.

    wet:  humid nights (dd < 1 C) in the last 3 / 7 / 14 days
    mean: mean dd over days 0-6 and 7-13
    """
    wet = DD < 1.0
    f = []
    if kind in ("wet", "both"):
        f += [wet[:, :3].sum(1), wet[:, :7].sum(1), wet[:, :14].sum(1)]
    if kind in ("mean", "both"):
        f += [DD[:, :7].mean(1), DD[:, 7:14].mean(1)]
    return np.column_stack(f).astype("float64")


def season_weight(table, day):
    """Weight for `day` from 12 mid-month values, linear between the 15th of each month."""
    import pandas as pd
    day = pd.Timestamp(day)
    mids = [pd.Timestamp(day.year + dy, m, 15) for dy in (-1, 0, 1) for m in range(1, 13)]
    vals = list(table) * 3
    k = np.searchsorted(np.array(mids, dtype="datetime64[ns]"), np.datetime64(day, "ns"))
    a, b = mids[k - 1], mids[k]
    w = (day - a) / (b - a)
    return (1 - w) * vals[k - 1] + w * vals[k]


def autumn_share(day):
    """Weight of a group's autumn part: 0 until 15 Aug, 1 from 15 Sep, back to 0 during February."""
    import pandas as pd
    d = pd.Timestamp(day).dayofyear
    if d < 60:
        return float(np.clip((60 - d) / 31, 0, 1))
    return float(np.clip((d - 227) / 31, 0, 1))


def _percentile(m, X, DD=None):
    mu = np.array(m["mu"])
    if m.get("dew"):
        nb = X.shape[1]
        if DD is None:
            F = np.tile(mu[nb:], (len(X), 1))
        else:
            # a cell whose dew record has a hole scores as an average-humidity cell
            F = dew_features(np.nan_to_num(DD, nan=99.0), m["dew"])
            F[~np.isfinite(DD).all(1)] = mu[nb:]
        X = np.hstack([X, F])
    u = ((X - mu) / np.array(m["sd"])) @ np.array(m["beta"])
    ref = np.array(m["ref_quantiles"])
    return np.interp(u, ref, np.linspace(0, 100, len(ref)))


def score(model, season, RR, TG, day, med, DD=None):
    """Final 0-100 score per group for cells with lag matrices RR, TG and zone flags `med`.

    A group is either one fit, or (schemaVersion 2) a `summer` and an `autumn` fit, each with
    its own percentile reference, blended by `autumn_share`. One fit learned its temperature
    optimum mostly from summer finds and rated real October finds at 28-38 (see export_model.py).
    """
    X, _ = features(RR, TG)
    out = {}
    for name, m in model["groups"].items():
        if "parts" in m:
            a = autumn_share(day)
            pct = (1 - a) * _percentile(m["parts"]["summer"], X, DD) + a * _percentile(m["parts"]["autumn"], X, DD)
        else:
            pct = _percentile(m, X, DD)
        w = np.where(med, season_weight(season["med"][name], day), season_weight(season["cont"][name], day))
        out[name] = np.clip(np.round(pct * w), 0, 100).astype("uint8")
    return out
