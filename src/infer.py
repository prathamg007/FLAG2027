"""Inference from trained checkpoints: score every member of the final system on the evaluation trials and write the
submission.

    python src/infer.py [--spec configs/final_system.json] [--out final] [--check]

Members (per cell, with weights, in the spec):
  base model     models/<tag>.ckpt.pt, 10 networks -> mean cosine of face and voice embeddings
  gender expert  models/<tag>.ckpt.pt, 10 networks per gender + voice gender classifier -> p(gender | voice)-weighted mix
  anchor         anchor consistency term computed from the anchor set (anchor.py)
Each cell's score is the weighted mean of its members; the zip stores -score (lower = same person).
--check compares every member with models/<tag>.scores.json.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import common, model

CELLS = {f"{t}/{l}": (t, l) for t, l in common.LISTS}


def heads(state_dicts, cfg):
    out = []
    for sd in state_dicts:
        m = model.Head(sd["face.1.weight"].shape[1], sd["voice.1.weight"].shape[1], cfg); m.load_state_dict(sd); out.append(m.to(common.DEVICE).eval())
    return out


def member_scores(tag, trials):
    """{cell: scores} of one member on the evaluation trials."""
    if tag == "anchor":
        import anchor
        A = anchor.load_anchors()
        return {c: anchor.anchor_score("eval", [t[1] for t in trials[k]], [t[2] for t in trials[k]], A) for c, k in CELLS.items()}
    ck = torch.load(os.path.join(common.MODELS, tag + ".ckpt.pt"), map_location="cpu", weights_only=False); i = ck["info"]
    get = lambda k: (model.feats("eval", [t[2] for t in trials[k]], i["F"]), model.feats("eval", [t[1] for t in trials[k]], i["V"]))
    if "gender_lr" not in ck:
        ms = heads(ck["state_dicts"], i["cfg"])
        return {c: model.score(ms, *get(k)) for c, k in CELLS.items()}
    GS = sorted(ck["state_dicts"])
    w = np.asarray(ck["gender_lr"]["coef"]).reshape(-1); b = float(np.asarray(ck["gender_lr"]["intercept"]).reshape(-1)[0])
    M = {g: heads(ck["state_dicts"][g], i["cfg"]) for g in GS}
    out = {}
    for c, k in CELLS.items():
        ff, vf = get(k)
        p1 = 1.0 / (1.0 + np.exp(-(vf.astype(np.float64) @ w + b)))     # p(voice has gender GS[1])
        out[c] = (1 - p1) * model.score(M[GS[0]], ff, vf) + p1 * model.score(M[GS[1]], ff, vf)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.path.join(common.CONFIGS, "final_system.json"))
    ap.add_argument("--out", default="final"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    spec = json.load(open(a.spec)); trials = common.eval_trials()
    cache, final = {}, {}
    for c, mix in spec.items():
        for tag, _ in mix:
            if tag not in cache:
                cache[tag] = member_scores(tag, trials)
                if a.check:
                    st = json.load(open(os.path.join(common.MODELS, tag + ".scores.json")))["scores"]
                    d = max(float(np.abs(np.asarray(cache[tag][cc]) - np.asarray(st[cc]["s"])).max()) for cc in CELLS)
                    print(f"{tag:<34} largest difference to stored scores: {d:.2e}", flush=True)
        k = CELLS[c]
        final[k] = ([t[0] for t in trials[k]], sum(wt * np.asarray(cache[t][c]) for t, wt in mix) / sum(wt for _, wt in mix))
    print("ZIP", model.write_zip(final, a.out), flush=True)
