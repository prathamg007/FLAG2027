"""DEV2-short result of a complete system: per cell the mean of the base models' DEV2-short scores + w x anchor term, w by the
weight rule of combine.py. On DEV2-short the anchor set excludes MAV-Celeb v2 (the validation people themselves).

    python src/validation/dev2short_system.py "ngE=<tags>|ngB=<tags>|gE=<tags>|gB=<tags>"
Base scores: precomputed/dev2short_base_scores.pkl (score_dev2short.py, train_gender_experts.py);
anchor-term scores: precomputed/dev2short_anchor_scores.pkl.
"""
import os, sys, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import common, dev2
from combine import REF, LAM_REF

b = pickle.load(open(os.path.join(common.PRECOMPUTED, "dev2short_base_scores.pkl"), "rb"))
anc = pickle.load(open(os.path.join(common.PRECOMPUTED, "dev2short_anchor_scores.pkl"), "rb"))
avg = lambda ts: {k: sum(np.asarray(b[(t, "mavolds")][k]) for t in ts) / len(ts) for k in b[(ts[0], "mavolds")]}
sd = lambda s: float(np.mean([np.std(v) for v in s.values()]))
ORDER = [("std", "heard"), ("std", "unheard"), ("gen", "heard"), ("gen", "unheard")]   # = ngE, ngB, gE, gB
fams = [f.split("=")[1].split(",") for f in sys.argv[1].split("|")]
base, full = {}, {}
for k in b[(REF[0], "mavolds")]:
    f = fams[ORDER.index(tuple(k))]; w = LAM_REF * sd(avg(f)) / sd(avg(REF))
    base[k] = avg(f)[k]; full[k] = base[k] + w * np.asarray(anc[k])
print("base alone", dev2.eer4(base)); print("with anchor term", dev2.eer4(full))
