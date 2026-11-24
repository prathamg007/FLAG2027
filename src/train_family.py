"""Train a family of base models: one recipe, one model per voice-input set, then score each on DEV2-short.

    python src/train_family.py --prefix e3rcpel_ --sets "c=wavlm,w2vag,ecapa,redim,wsr293;e=..." --recipe tau003_p05_ft200 --faces clip,pel
Model tags: <prefix><set>_<recipe>. Existing models are skipped.
"""
import os, sys, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", required=True); ap.add_argument("--sets", required=True); ap.add_argument("--recipe", required=True)
ap.add_argument("--faces", default="clip"); ap.add_argument("--corpora", default="vox1,vox2")
ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--seed0", type=int, default=0)
a = ap.parse_args()
for item in a.sets.split(";"):
    s, voices = item.split("=")
    tag = f"{a.prefix}{s}_{a.recipe}"
    if os.path.exists(os.path.join(common.MODELS, tag + ".scores.json")):
        print("exists:", tag, flush=True); continue
    subprocess.check_call([sys.executable, os.path.join(HERE, "train_base.py"), tag, "--faces", a.faces, "--voices", voices, "--recipe", a.recipe,
                           "--corpora", a.corpora, "--seeds", str(a.seeds), "--seed0", str(a.seed0)])
    subprocess.check_call([sys.executable, os.path.join(HERE, "score_dev2short.py"), tag])
