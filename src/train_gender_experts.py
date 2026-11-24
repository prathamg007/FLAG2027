"""Gender-expert member for the gender cells.

Per seed two networks, each trained only on the people of one gender (main stage, then the last stage on the challenge people of
that gender). Per trial the two are mixed by p(gender | voice) from a logistic regression on the voice inputs of the training
people. Used in the gender cells only.

    python src/train_gender_experts.py <tag> --faces clip,pel --voices ecapa,redim,redimm,wsr293 [--recipe tau003_p05_ft200]
Writes models/<tag>.ckpt.pt, models/<tag>.scores.json and the DEV2-short scores (precomputed/dev2short_base_scores.pkl).
"""
import os, sys, json, time, argparse
os.environ.setdefault("FV_KEEP_OVERLAP", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
import common, model, dev2
from score_dev2short import store

ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("--faces", default="clip"); ap.add_argument("--voices", required=True)
ap.add_argument("--recipe", default="tau003_p05_ft200"); ap.add_argument("--corpora", default="vox1,vox2")
ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--seed0", type=int, default=0)
a = ap.parse_args()
F0, V0, ext = a.faces.split(","), a.voices.split(","), [x for x in a.corpora.split(",") if x]
cfg, ft = common.recipe(a.recipe)
D = model.Data([model.src_mav()] + [model.src_vox(e, F0, V0) for e in ext], F0, V0)
mav = D.src == "mav"; G = D.gender; GS = sorted(set(G[mav].tolist())); sid = np.where(mav)[0]; idw = model.id_weights(D, cfg)
info = {"tag": a.tag, "ext": ext, "F": F0, "V": V0, "cfg": cfg, "ft": ft, "seeds": a.seeds, "seed0": a.seed0, "recipe": a.recipe,
        "genders": [int(g) for g in GS], "people per gender": [int((G == g).sum()) for g in GS]}
print(json.dumps(info), flush=True); t0 = time.time()
M = {g: [] for g in GS}
for s in range(a.seed0, a.seed0 + a.seeds):
    for g in GS:
        m = model.train(cfg, D, G == g, seed=s, id_w=idw, stat_ids=sid)
        if ft: m = model.train(model.ft_cfg(cfg), D, mav & (G == g), seed=s, model=m, steps=ft["steps"], lr=ft["lr"], stat_ids=sid)
        M[g].append(m.eval())
    print(f"seed {s} done [{time.time() - t0:.0f}s]", flush=True)
rows = np.random.default_rng(0).choice(len(D.yv), min(30000, len(D.yv)), replace=False)
LR = LogisticRegression(max_iter=300, C=1.0).fit(D.Xv[rows], (G[D.yv[rows]] == GS[1]).astype(int))


def mix(ff, vf):
    p1 = LR.predict_proba(vf)[:, 1]
    return (1 - p1) * model.score(M[GS[0]], ff, vf) + p1 * model.score(M[GS[1]], ff, vf)


T = dev2.trials(); s2 = {k: mix(model.feats("mavold", Fp, F0), model.feats("mavolds", Vp, V0)) for k, (Vp, Fp, Y) in T.items()}
print(f"{a.tag}: DEV2-short {dev2.eer4(s2)}", flush=True); store({(a.tag, "mavolds"): s2})
out = {}
for cell, trl in common.eval_trials().items():
    out[cell] = ([t[0] for t in trl], mix(model.feats("eval", [t[2] for t in trl], F0), model.feats("eval", [t[1] for t in trl], V0)))
torch.save({"info": info, "state_dicts": {g: [{k: v.cpu() for k, v in m.state_dict().items()} for m in M[g]] for g in GS},
            "gender_lr": {"coef": LR.coef_, "intercept": LR.intercept_}}, os.path.join(common.MODELS, a.tag + ".ckpt.pt"))
json.dump({"info": info, "scores": {f"{k[0]}/{k[1]}": {"ids": v[0], "s": [float(x) for x in v[1]]} for k, v in out.items()}},
          open(os.path.join(common.MODELS, a.tag + ".scores.json"), "w"))
print("DONE", a.tag, flush=True)
