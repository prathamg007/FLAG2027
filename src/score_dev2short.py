"""Score trained models on DEV2-short and store the scores in precomputed/dev2short_base_scores.pkl
(input of the anchor-weight rule in combine.py).

    python src/score_dev2short.py <tag> [<tag> ...]
"""
import os, sys, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import common, model, dev2

PKL = os.path.join(common.PRECOMPUTED, "dev2short_base_scores.pkl")


def store(new):
    """Merge {(tag, 'mavolds'): {cell: scores}} into the pickle (safe when several jobs write)."""
    while True:
        try: os.mkdir(PKL + ".lock"); break
        except FileExistsError: time.sleep(0.3)
    try:
        old = pickle.load(open(PKL, "rb")) if os.path.exists(PKL) else {}
        old.update(new); tmp = PKL + f".{os.getpid()}.tmp"; pickle.dump(old, open(tmp, "wb")); os.replace(tmp, PKL)
    finally: os.rmdir(PKL + ".lock")


def load_heads(tag):
    ck = torch.load(os.path.join(common.MODELS, tag + ".ckpt.pt"), map_location="cpu", weights_only=False)
    ms = []
    for sd in ck["state_dicts"]:
        m = model.Head(sd["face.1.weight"].shape[1], sd["voice.1.weight"].shape[1], ck["info"]["cfg"]); m.load_state_dict(sd)
        ms.append(m.to(common.DEVICE).eval())
    return ms, ck["info"]


if __name__ == "__main__":
    new = {}
    for tag in sys.argv[1:]:
        ms, info = load_heads(tag)
        s = dev2.scores(ms, info["F"], info["V"], vds="mavolds"); new[(tag, "mavolds")] = s
        print(f"{tag}: DEV2-short {dev2.eer4(s)}", flush=True)
    store(new)
