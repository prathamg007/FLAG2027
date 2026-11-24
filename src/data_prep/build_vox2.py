"""VoxCeleb2 manifest of the clips sampled by fetch_vox2.py: up to CAP_V voice clips per speaker, all face frames.
    python src/data_prep/build_vox2.py
Writes $FLAG_EXT/vox2_manifest.tsv and vox2_gender.json.
"""
import os, glob, json, collections
EXT = os.environ.get("FLAG_EXT", "ext_data")
g = {}
for i, line in enumerate(open(os.path.join(EXT, "vox2_meta.csv"), encoding="utf-8-sig")):
    t = [x.strip() for x in line.split(",")]
    if i and len(t) >= 4: g[t[0]] = 0 if t[2] == "m" else 1
CAP_V = 24                                   # voice clips per speaker (all faces are kept)
import numpy as np
have = set()
fp = os.path.join(os.environ.get("FLAG_FEAT", "features"), "vox2__ecapa.npz")   # clips that already have features are kept first
if os.path.exists(fp):
    with np.load(fp, allow_pickle=True) as z: have = {str(x) for x in z["paths"]}
rows, nv = [], collections.Counter()
clips = collections.defaultdict(list)
for j in sorted(glob.glob(os.path.join(EXT, "vox2", "*", "*.jpg"))):
    a = j[:-4] + ".flac"
    if os.path.exists(a) and os.path.getsize(a) > 1000:
        clips[j.split(os.sep)[-2]].append((j, a))
for ident in sorted(clips):
    cl = clips[ident]
    order = sorted(range(len(cl)), key=lambda i: (cl[i][1] not in have, bin(i)[::-1], i))   # extracted first, then spread
    keepv = set(order[:CAP_V]) | {i for i in range(len(cl)) if cl[i][1] in have}
    for i, (j, a) in enumerate(cl):
        stem = os.path.basename(j)[:-4]
        rows.append(("face", ident, stem, j))
        if i in keepv: rows.append(("voice", ident, stem, a))
with open(os.path.join(EXT, "vox2_manifest.tsv"), "w", encoding="utf-8") as f:
    for r in rows: f.write("\t".join(r) + "\n")
json.dump(g, open(os.path.join(EXT, "vox2_gender.json"), "w"))
ids = {r[1] for r in rows}
print("VOX2 MANIFEST faces", sum(r[0] == "face" for r in rows), "voices", sum(r[0] == "voice" for r in rows), "ids", len(ids), "with gender", len(ids & set(g)), flush=True)
