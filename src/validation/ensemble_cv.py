"""Paired comparison of ensembles on the per-trial cross-validation scores of cv_dump.py, column by column.

    python src/validation/ensemble_cv.py <reference> <candidate> [<candidate> ...]
A specification is   name=member+member            (same members in both columns)
                or   name=std:member+...,gen:member+...   (other members per column, as in a per-cell system)
Members: "<dump name>" (ordinary network) or "<dump name>#gx" (gender experts). Member scores of a trial are averaged with equal
weights; the EER is computed per fold-seed and column and compared with the reference fold-seed by fold-seed (mean difference, t).
std = any wrong face (no-gender cells), gen = same-gender wrong face (gender cells).
"""
import os, sys, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import common

DUMP = os.path.join(common.PRECOMPUTED, "cv_dumps"); _C = {}


def member(m):
    f = m.split("#")[0]
    if f not in _C: _C[f] = pickle.load(open(os.path.join(DUMP, f + ".pkl"), "rb"))
    return _C[f]["trials"], _C[f]["scores"][m]


def run(spec):
    name, body = spec.split("=", 1)
    cols = dict(p.split(":") for p in body.split(",")) if ":" in body else {"std": body, "gen": body}
    out = {}
    for c in ("std", "gen"):
        L = [member(m) for m in cols[c].split("+")]; T0 = L[0][0]
        keys = sorted(k for k in T0 if k[2] == c and all(k in sc for _, sc in L))
        for T, _ in L[1:]:
            assert all(np.array_equal(T[k][0], T0[k][0]) and np.array_equal(T[k][1], T0[k][1]) for k in keys), "different trials"
        out[c] = {k[:2]: common.eer(T0[k][2], np.mean([sc[k] for _, sc in L], 0)) for k in keys}
    return name, out


R = [run(s) for s in sys.argv[1:]]; rn, ref = R[0]
for n, r in R:
    ks = sorted(set(r["std"]) & set(ref["std"]))
    line = f"{n:<32}"
    for c in ("std", "gen"):
        d = np.array([r[c][k] - ref[c][k] for k in ks]); se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0
        line += f" {c} {np.mean([r[c][k] for k in ks]):6.2f} ({d.mean():+.2f}, t {d.mean() / se if se else 0:+.1f})"
    print(line + f"  [{len(ks)} fold-seeds, reference {rn}]", flush=True)
