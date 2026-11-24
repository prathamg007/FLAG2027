"""Train one base model (one recipe, one input set, several seeds) on all training identities.

    python src/train_base.py <tag> --faces clip,pel --voices wavlm,w2vag,ecapa,redim,wsr293 --recipe tau003_p05_ft200
                             [--corpora vox1,vox2] [--seeds 10] [--seed0 0]
Writes models/<tag>.ckpt.pt (networks + settings) and models/<tag>.scores.json (scores of the evaluation trials).
Recipes: configs/recipes.json.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import common, model, dev2


if __name__ == "__main__":
    os.environ.setdefault("FV_KEEP_OVERLAP", "1")             # final models train on every external identity
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("--faces", default="clip"); ap.add_argument("--voices", required=True)
    ap.add_argument("--recipe", required=True); ap.add_argument("--corpora", default="vox1,vox2")
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--seed0", type=int, default=0)
    a = ap.parse_args()
    F0, V0, ext = a.faces.split(","), a.voices.split(","), [x for x in a.corpora.split(",") if x]
    cfg, ft = common.recipe(a.recipe)
    D = model.Data([model.src_mav()] + [model.src_vox(e, F0, V0) for e in ext], F0, V0)
    info = {"tag": a.tag, "ext": ext, "F": F0, "V": V0, "cfg": cfg, "ft": ft, "seeds": a.seeds, "seed0": a.seed0, "recipe": a.recipe,
            "ids": int(D.n_id), "ids_by_source": {s: int((D.src == s).sum()) for s in sorted(set(D.src))},
            "faces": int(len(D.yf)), "voices": int(len(D.yv))}
    print(json.dumps(info), flush=True)
    models = model.train_full(cfg, D, seeds=range(a.seed0, a.seed0 + a.seeds), ft=ft)
    info["dev2"] = dev2.eer4(dev2.scores(models, F0, V0)); print("DEV2", json.dumps(info["dev2"]), flush=True)
    os.makedirs(common.MODELS, exist_ok=True)
    torch.save({"info": info, "state_dicts": [{k: v.cpu() for k, v in m.state_dict().items()} for m in models]},
               os.path.join(common.MODELS, a.tag + ".ckpt.pt"))
    ev = model.score_eval(models, F0, V0)
    json.dump({"info": info, "scores": {f"{k[0]}/{k[1]}": {"ids": v[0], "s": [float(x) for x in v[1]]} for k, v in ev.items()}},
              open(os.path.join(common.MODELS, a.tag + ".scores.json"), "w"))
    print("DONE", a.tag, flush=True)
