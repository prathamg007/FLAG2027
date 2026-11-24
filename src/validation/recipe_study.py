"""Measure training recipes on the three training-side rulers and store the results in configs/recipes.json ("rulers"):
  dev2   MAV-Celeb v2 validation set, full clips (4 cells)
  dev2s  the same with voices cut to challenge clip lengths (DEV2-short)
  cv     7-fold cross-validation by identity over the 70 challenge training people (std / gen columns, per fold values)
Fixed for every recipe: CLIP faces; WavLM + age/gender + ECAPA + ReDimNet-B6 voices; challenge set + VoxCeleb1 + VoxCeleb2 with
the identities of the validation set removed; 3 networks for DEV2, CV seeds from $CV_SEEDS (default 3).

    python src/validation/recipe_study.py [recipe ...]
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common, model, dev2

F0, V0 = ["clip"], ["wavlm", "w2vag", "ecapa", "redim"]
P = os.path.join(common.CONFIGS, "recipes.json")
R = json.load(open(P))
D = model.Data([model.src_mav()] + [model.src_vox(x, F0, V0) for x in ("vox1", "vox2")], F0, V0)
for name in sys.argv[1:] or list(R):
    cfg, ft = common.recipe(name)
    ms = model.train_full(cfg, D, seeds=range(3), ft=ft)
    res = {"dev2": dev2.eer4(dev2.scores(ms, F0, V0)), "dev2s": dev2.eer4(dev2.scores(ms, F0, V0, vds="mavolds")),
           "cv": model.cv(cfg, D, seeds=tuple(range(int(os.environ.get("CV_SEEDS", "3")))), ft=ft)}
    R = json.load(open(P)); R[name]["rulers"] = res; json.dump(R, open(P, "w"), indent=1)
    print(name, "DEV2", res["dev2"]["avg"], "DEV2-short", res["dev2s"]["avg"], "CV", res["cv"]["avg"], flush=True)
