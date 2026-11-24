"""Paths, challenge trial lists and the EER metric.

Every file is addressed by a fixed key path (Windows form, prefix KEY_ROOT) on every machine; feature tables, manifests and
trial lists store these keys. real() maps a key to the file on disk:
    KEY_ROOT\\train_set\\..., KEY_ROOT\\eval_set\\...  ->  $FLAG_DATA/...
    KEY_ROOT\\ext_data\\...                         ->  $FLAG_EXT/...
File lists of the external corpora (manifests, genders, overlap lists) are read from data/manifests.
"""
import os, random, ntpath
os.environ.setdefault("FV_VOX2_OLD", "1")       # VoxCeleb2 file list of the final models and the anchor set
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_ROOT = r"D:\FLAG27"
KJ, KSEP = ntpath.join, "\\"
DATA = os.environ.get("FLAG_DATA") or os.path.join(REPO, "data")
EXT_DIR = os.environ.get("FLAG_EXT") or os.path.join(DATA, "ext_data")
FEAT = os.environ.get("FLAG_FEAT") or os.path.join(REPO, "features")
MODELS = os.environ.get("FLAG_MODELS") or os.path.join(REPO, "models")
CONFIGS = os.path.join(REPO, "configs")
MANIFESTS = os.environ.get("FLAG_MANIFESTS") or os.path.join(REPO, "data", "manifests")   # file lists, genders, overlap lists
PRECOMPUTED = os.path.join(REPO, "precomputed")
DEVICE = os.environ.get("FV_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

TRAIN = KJ(KEY_ROOT, "train_set")
EVAL = KJ(KEY_ROOT, "eval_set", "evaluation_phase")
LISTS = [("no_gender", "English"), ("no_gender", "Bangla"), ("gender", "English"), ("gender", "Bangla")]
SUBNAME = {"English": "sub_score_v4_English_heard.txt", "Bangla": "sub_score_v4_Bangla_unheard.txt"}


def real(p):
    """Key path -> file on disk."""
    if not p.startswith(KEY_ROOT):
        return p
    rel = p[len(KEY_ROOT):].strip(KSEP).split(KSEP)
    return os.path.join(EXT_DIR, *rel[1:]) if rel[0] == "ext_data" else os.path.join(DATA, *rel)


def key(p):
    """File on disk under DATA -> key path."""
    return p if not p.startswith(DATA) else KJ(KEY_ROOT, *p[len(DATA):].strip(os.sep).split(os.sep))


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def l2n(X):
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def train_index():
    """Rows of the challenge training list: voice, face, identity, class index."""
    rows = []
    with open(real(KJ(TRAIN, "train_set", "train_English.txt")), encoding="utf-8") as f:
        for line in f:
            t = line.split()
            if len(t) == 6:
                rows.append({"voice": KJ(TRAIN, "train_set", t[2].replace("/", KSEP)),
                             "face": KJ(TRAIN, "train_set", t[3].replace("/", KSEP)), "id": t[4], "cls": int(t[5])})
    return rows


def gender_meta():
    """Identity -> gender (0 = male, 1 = female) of the challenge training people."""
    m = {}
    for line in open(real(KJ(TRAIN, "meta_file_train_set.csv")), encoding="utf-8"):
        if line.strip():
            i, g = line.strip().split(",")
            m[i] = 0 if g.strip().lower() == "m" else 1
    return m


def eval_trials():
    """{(track, language): [(pair id, voice key, face key), ...]} of the evaluation lists.
    One list writes 'faces 00000.jpg' with a space for the slash; the face path is rebuilt from the last token."""
    out = {}
    for track, lang in LISTS:
        tr = []
        for line in open(real(KJ(EVAL, track, f"{lang}_valid.txt")), encoding="utf-8"):
            t = line.split()
            if len(t) < 3:
                continue
            v = KJ(EVAL, track, t[1].replace("/", KSEP))
            f = t[2] if len(t) == 3 else t[2].rstrip("/") + "/" + t[3]
            tr.append((t[0], v, KJ(EVAL, track, f.replace("/", KSEP))))
        out[(track, lang)] = tr
    return out


def recipe(name):
    """Training settings of a recipe in configs/recipes.json: (cfg, last stage or None)."""
    import json
    r = json.load(open(os.path.join(CONFIGS, "recipes.json")))[name]
    return dict(r["cfg"]), ({"steps": r["ft"][0], "lr": r["ft"][1]} if r.get("ft") else None)


def eer(y, s):
    """Equal error rate in percent; s: higher = same person."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2 * 100)
