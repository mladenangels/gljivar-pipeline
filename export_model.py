"""Fit the production model per group and write out/model.json.

Same features and conditional logit as backtest/model2.py ("within" design), but fitted
on all core years (AT+SI+HR 2011-2025) instead of 2011-2019. The replication region
(CH+S-DE) stays out of the fit and is reported as a check.

Score shown in the app = percentile of the model's log-odds among control days in the fit,
so 80 means "better than 80 % of ordinary days at the same places".

Groups in SEASONAL get two fits (schemaVersion 2): `summer` on finds outside AUTUMN_MONTHS and
`autumn` on finds in them, each with its own percentile reference; model.score blends them by
date. Checked 2026-10-08 after FB feedback ("Pohorje excellent, score low"): with one fit, real
vrganj finds in Oct-Nov above 600 m scored a median 33 (season weight included). Fitted on
2011-2019 and scored on 2020-2025, the seasonal vrganj model raises that to 60 and the AUC
from 0.634 to 0.642 (CH/DE 0.653 -> 0.650). The same split made sunčanica and lisičarka worse
(0.671 -> 0.643, 0.595 -> 0.579), so they keep one fit.

Fog/dew (DEW, added 2026-10-09 after FB feedback "vrganji grow with little rain"): night dewpoint
depression features from ERA5-Land d2m (backtest/dew_test.py, dew_paired.py). Fitted on
2011-2019, tested on 2020-2025 / CH+DE, paired bootstrap of the AUC gain:
  lisičarka "both"      +0.020 [+0.006, +0.035] / +0.011 [+0.004, +0.017]  -> used
  vrganj autumn "wet"   +0.019 [-0.008, +0.043] / -0.001 [-0.009, +0.008]  -> NOT used: live on 2026-10-09
                        it took Pohorje from 69 to 39 after a fortnight of dry nights (ERA5 and ECMWF
                        agree), the very place pickers reported excellent on 10-08. Night dd on
                        ridges sits above the valley inversion and misses the fog below.
  sunčanica             +0.000 / -0.002, smrčak -0.028 / -0.002            -> not used
"""
import json
import sys

import numpy as np
import pandas as pd

import config as C

sys.path.insert(0, str(C.BACKTEST))
from analyze import D, Matched  # noqa: E402
from dew_test import dew_lags  # noqa: E402
from model2 import RAIN_BINS, features, fit_clogit, padded_sets  # noqa: E402
from model import DD_LAGS, dew_features  # noqa: E402

N_QUANTILES = 1001
SEASONAL = {"vrganj"}
AUTUMN_MONTHS = [9, 10, 11, 12, 1]
DEW = {("lisicarka", "all"): "both"}


def fit_part(s, X, mask, gm, repl, dew=None):
    """One conditional-logit fit on the finds in `mask`, percentile reference from its controls."""
    mu, sd = X[mask].mean(0), X[mask].std(0) + 1e-9
    Xs = (X - mu) / sd
    m_tr = Matched(s, "within", mask)
    beta = fit_clogit(Xs, padded_sets(m_tr))
    score = Xs @ beta
    ref = np.quantile(score[m_tr.ctl], np.linspace(0, 1, N_QUANTILES))
    auc_tr, auc_rep = m_tr.auc(score), Matched(s, "within", gm & repl).auc_ci(score)
    part = {"mu": mu.tolist(), "sd": sd.tolist(), "beta": beta.tolist(), "ref_quantiles": ref.tolist(),
            "auc_train": round(auc_tr, 3), "auc_replication": round(auc_rep[0], 3)}
    if dew:
        part["dew"] = dew
    return part, auc_tr, auc_rep


def main():
    s = pd.read_parquet(D / "samples.parquet")
    L = np.load(D / "lags.npz")
    RR, TG = L["rr"].astype("float64"), L["tg"].astype("float64")
    X, names = features(RR, TG)
    DD = dew_lags(s, DD_LAGS)
    core, repl = s.region.values == "core", s.region.values == "repl"

    out = {"schemaVersion": 2, "features": names, "rain_bins": RAIN_BINS,
           "rr_lags": C.RR_LAGS, "tg_lags": C.TG_LAGS, "dd_lags": DD_LAGS,
           "trained_on": "GBIF CC0/CC BY finds AT+SI+HR 2011-2025, ERA5-Land weather",
           "groups": {}}
    autumn = np.isin(s.event_date.dt.month.values, AUTUMN_MONTHS)
    for g in C.GROUPS:
        gm = s.group.values == g
        tr = gm & core
        parts = {"all": tr} if g not in SEASONAL else {"summer": tr & ~autumn, "autumn": tr & autumn}
        fitted = {}
        for pname, mask in parts.items():
            kind = DEW.get((g, pname))
            Xp = np.hstack([X, dew_features(DD, kind)]) if kind else X
            fitted[pname], auc_tr, auc_rep = fit_part(s, Xp, mask, gm & (autumn if pname == "autumn" else
                                                                        ~autumn if pname == "summer" else True), repl, kind)
            print(f"{g:10s} {pname:6s} train {auc_tr:.3f} | replication {auc_rep[0]:.3f} "
                  f"[{auc_rep[1]:.3f}-{auc_rep[2]:.3f}] n={auc_rep[3]}")
        out["groups"][g] = fitted["all"] if "all" in fitted else {"parts": fitted}

    C.OUT.mkdir(exist_ok=True)
    (C.OUT / "model.json").write_text(json.dumps(out))
    print("wrote", C.OUT / "model.json")


if __name__ == "__main__":
    main()
