"""Readers for the result files (see results/README.md for the record formats).

Policy ratios are stored under the keys of the original experiment code: 'F-MLE' (MLE policy), 'F-CDR' / 'F-CDR(cv)'
(CDR policy), 'random' (random policy); the subgradient policy is stored in separate files ('fig3_sub*.pkl',
'fig4_sub*.pkl'). The LP files of the stored results also contain fields 'F-Sub', 'F-Sub(cv)', 'c_sub' and row 0 of
'cv_scores' from an earlier index rule (random tie-breaking, no re-solved fluid LP); these are superseded by the
subgradient files and not read here, except for the single beta = 0 history, where no repair is possible and all
policies coincide.
"""
import glob
import os
import pickle
from collections import defaultdict


def _load(results_dir, pattern):
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, pattern))):
        with open(path, "rb") as f:
            records += pickle.load(f)
    return records


def load_fig3(results_dir):
    """Figure 3: {(K, seed): {'mle', 'random', 'cdr': {c: ratio}, 'sub': {c: ratio}}}."""
    out = defaultdict(lambda: {"cdr": {}, "sub": {}})
    for r in _load(results_dir, "fig3_lp*.pkl"):
        d = out[(r["K"], r["seed"])]
        d["mle"], d["random"] = float(r["F-MLE"]), float(r["random"])
        d["cdr"].update({c: float(v["F-CDR"]) for c, v in r["curves"].items()})
    for r in _load(results_dir, "fig3_sub*.pkl"):
        out[(r["K"], r["seed"])]["sub"].update({c: float(v) for c, v in r["curves"].items()})
    return dict(out)


def load_fig4(results_dir):
    """Figure 4: list of dicts with K, seed, b, mle, random, cdr, c_cdr, sub, c_sub (one per history)."""
    sub = {(r["K"], r["seed"], round(r["b"], 4)): r for r in _load(results_dir, "fig4_sub*.pkl")}
    rows = []
    for r in _load(results_dir, "fig4_lp*.pkl"):
        k = (r["K"], r["seed"], round(r["b"], 4))
        if k in sub:
            s, c_s = float(sub[k]["ratio"]), float(sub[k]["c_sub"])
        else:  # only the stored beta = 0 history: no repairs, every policy coincides with the true-kernel policy
            s, c_s = float(r["F-Sub(cv)"]), float(r["c_sub"])
        rows.append(dict(K=r["K"], seed=r["seed"], b=r["b"], mle=float(r["F-MLE"]), random=float(r["random"]),
                         cdr=float(r["F-CDR(cv)"]), c_cdr=float(r["c_cdr"]), sub=s, c_sub=c_s))
    return rows


def load(results_dir, name):
    """All records of the files ``name*.pkl`` (e.g. 'table2_certificates', 'text_controls', 'valid_radius')."""
    return _load(results_dir, name + "*.pkl")
