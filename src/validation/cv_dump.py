"""Per-trial scores of the 7-fold cross-validation for one model configuration (input of ensemble_cv.py).

    python src/validation/cv_dump.py <name> --faces clip --voices wavlm,w2vag,ecapa,redim --recipe tau003_p05_ft200 [--seeds 5] [--experts]
Folds, seeds and trials are those of model.cv. With --experts also the gender-expert member: two networks per fold, each trained on
one gender, mixed per trial by p(gender | voice) from a logistic regression fitted on the fold's training voices.
Members stored: "<name>" and, with --experts, "<name>#gx". Output: precomputed/cv_dumps/<name>.pkl.
"""
import os, sys, pickle, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
import common, model

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--faces", default="clip"); ap.add_argument("--voices", required=True)
ap.add_argument("--recipe", required=True); ap.add_argument("--corpora", default="vox1,vox2"); ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--experts", action="store_true")
a = ap.parse_args()
F0, V0 = a.faces.split(","), a.voices.split(","); cfg, ft = common.recipe(a.recipe)
D = model.Data([model.src_mav()] + [model.src_vox(x, F0, V0) for x in a.corpora.split(",")], F0, V0)
mav = np.where(D.src == "mav")[0]; G = D.gender; GS = sorted(set(G[mav].tolist())); idw = model.id_weights(D, cfg)


def last(m, ids_mask, seed, sid):
    return m if not ft else model.train(model.ft_cfg(cfg), D, ids_mask, seed=seed, model=m, steps=ft["steps"], lr=ft["lr"], stat_ids=sid)


T, SC = {}, {a.name: {}, a.name + "#gx": {}}
for sd in range(a.seeds):
    perm = np.random.default_rng(100 + sd).permutation(mav)
    for k in range(7):
        va = perm[k::7]; mask = np.ones(D.n_id, bool); mask[va] = False
        trials = model.val_trials(D, list(va), np.random.default_rng(1000 * sd + k))
        sid = np.setdiff1d(mav, va); mm = np.zeros(D.n_id, bool); mm[mav] = True; mm[va] = False; seed = sd * 100 + k
        m = last(model.train(cfg, D, mask, seed=seed, id_w=idw, stat_ids=sid), mm, seed, sid)
        for cell, (vi, fi, y) in trials.items():
            T[(sd, k, cell)] = (vi.astype(np.int32), fi.astype(np.int32), y.astype(np.int8))
            SC[a.name][(sd, k, cell)] = model.score([m], D.Xf[fi], D.Xv[vi]).astype(np.float32)
        if a.experts:
            me = {g: last(model.train(cfg, D, mask & (G == g), seed=seed, id_w=idw, stat_ids=sid), mm & (G == g), seed, sid) for g in GS}
            rows = np.where(np.isin(D.yv, np.where(mask)[0]))[0]; rows = np.random.default_rng(0).choice(rows, min(30000, len(rows)), replace=False)
            lr = LogisticRegression(max_iter=300, C=1.0).fit(D.Xv[rows], (G[D.yv[rows]] == GS[1]).astype(int))
            for cell, (vi, fi, y) in trials.items():
                p1 = lr.predict_proba(D.Xv[vi])[:, 1]
                SC[a.name + "#gx"][(sd, k, cell)] = (p1 * model.score([me[GS[1]]], D.Xf[fi], D.Xv[vi]) + (1 - p1) * model.score([me[GS[0]]], D.Xf[fi], D.Xv[vi])).astype(np.float32)
        torch.cuda.empty_cache()
    print(f"seed {sd} done", flush=True)
out = os.path.join(common.PRECOMPUTED, "cv_dumps"); os.makedirs(out, exist_ok=True)
pickle.dump({"trials": T, "scores": {k: v for k, v in SC.items() if v}}, open(os.path.join(out, a.name + ".pkl"), "wb"))
for mem, sc in SC.items():
    if sc:
        r = {c: np.mean([common.eer(T[(sd, k, c)][2], sc[(sd, k, c)]) for sd in range(a.seeds) for k in range(7)]) for c in ("std", "gen")}
        print(f"{mem}: CV std {r['std']:.2f} gen {r['gen']:.2f}", flush=True)
