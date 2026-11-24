"""Combine stored member scores into the submission: per cell, the weighted mean of the members' scores.

    python src/combine.py <tag> --spec configs/final_system.json
        members and weights as given (reproduces the submission)
    python src/combine.py <tag> --families "ngE=<tags>|ngB=<tags>|gE=<tags>|gB=<tags>"
        equal weights for the base models + the anchor consistency term `anchor` with the weight rule:
        w = 3 x sd(mean base score of the cell's family) / sd(mean base score of the reference family), both on DEV2-short.
        The reference family (learned temperature) sets the scale; a sharp fixed temperature narrows the score range, and the
        rule keeps the balance between base models and anchor term constant. The weight is a constant per cell from validation data.
Writes models/<tag>.zip and models/<tag>.spec.json.
"""
import os, sys, json, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import common, model

KEY = {"ngE": "no_gender/English", "ngB": "no_gender/Bangla", "gE": "gender/English", "gB": "gender/Bangla"}
REF = ["e3c_light_b6", "e3e_light_lean", "e3f_light_b6m"]; LAM_REF = 3.0


def rule_spec(families):
    short = pickle.load(open(os.path.join(common.PRECOMPUTED, "dev2short_base_scores.pkl"), "rb"))
    sd = lambda tags: float(np.mean([np.std(sum(np.asarray(short[(t, "mavolds")][k]) for t in tags) / len(tags)) for k in short[(tags[0], "mavolds")]]))
    spec = {}
    for c, tl in (x.split("=") for x in families.split("|")):
        tags = tl.split(","); lam = LAM_REF * sd(tags) / sd(REF)
        spec[KEY[c]] = [[t, 1.0] for t in tags] + [["anchor", round(len(tags) * lam, 3)]]
    return spec


def assemble(tag, spec):
    cache, out = {}, {}
    def load(t):
        if t not in cache: cache[t] = json.load(open(os.path.join(common.MODELS, t + ".scores.json")))["scores"]
        return cache[t]
    for track, lang in common.LISTS:
        key = f"{track}/{lang}"; mix = spec[key]; ids = load(mix[0][0])[key]["ids"]
        assert all(load(t)[key]["ids"] == ids for t, _ in mix)
        out[(track, lang)] = (ids, sum(w * np.array(load(t)[key]["s"]) for t, w in mix) / sum(w for _, w in mix))
    json.dump(spec, open(os.path.join(common.MODELS, tag + ".spec.json"), "w"), indent=1)
    return model.write_zip(out, tag)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--spec"); ap.add_argument("--families")
    a = ap.parse_args()
    spec = json.load(open(a.spec)) if a.spec else rule_spec(a.families)
    print("ZIP", assemble(a.tag, spec), flush=True)
