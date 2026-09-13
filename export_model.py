"""Fit the production model per group and write out/model.json.

Same features and conditional logit as backtest/model2.py ("within" design), but fitted
on all core years (AT+SI+HR 2011-2025) instead of 2011-2019. The replication region
(CH+S-DE) stays out of the fit and is reported as a check.

Score shown in the app = percentile of the model's log-odds among control days in the fit,
so 80 means "better than 80 % of ordinary days at the same places".
"""
import json
import sys

import numpy as np
import pandas as pd

import config as C

sys.path.insert(0, str(C.BACKTEST))
from analyze import D, Matched  # noqa: E402
from model2 import RAIN_BINS, features, fit_clogit, padded_sets  # noqa: E402

N_QUANTILES = 1001


def main():
    s = pd.read_parquet(D / "samples.parquet")
    L = np.load(D / "lags.npz")
    RR, TG = L["rr"].astype("float64"), L["tg"].astype("float64")
    X, names = features(RR, TG)
    core, repl = s.region.values == "core", s.region.values == "repl"

    out = {"schemaVersion": 1, "features": names, "rain_bins": RAIN_BINS,
           "rr_lags": C.RR_LAGS, "tg_lags": C.TG_LAGS,
           "trained_on": "GBIF CC0/CC BY finds AT+SI+HR 2011-2025, ERA5-Land weather",
           "groups": {}}
    for g in C.GROUPS:
        gm = s.group.values == g
        tr = gm & core
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xs = (X - mu) / sd
        m_tr = Matched(s, "within", tr)
        beta = fit_clogit(Xs, padded_sets(m_tr))
        score = Xs @ beta
        ref = np.quantile(score[m_tr.ctl], np.linspace(0, 1, N_QUANTILES))
        m_rep = Matched(s, "within", gm & repl)
        auc_tr, auc_rep = m_tr.auc(score), m_rep.auc_ci(score)
        out["groups"][g] = {"mu": mu.tolist(), "sd": sd.tolist(), "beta": beta.tolist(),
                            "ref_quantiles": ref.tolist(),
                            "auc_train": round(auc_tr, 3), "auc_replication": round(auc_rep[0], 3)}
        print(f"{g:10s} train {auc_tr:.3f} | replication {auc_rep[0]:.3f} "
              f"[{auc_rep[1]:.3f}-{auc_rep[2]:.3f}] n={auc_rep[3]}")

    C.OUT.mkdir(exist_ok=True)
    (C.OUT / "model.json").write_text(json.dumps(out))
    print("wrote", C.OUT / "model.json")


if __name__ == "__main__":
    main()
