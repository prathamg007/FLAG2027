"""VoxCeleb1 manifest. Faces: VGG cross-modal biometrics release `zippedFaces.tar.gz` (up to 2 crops per video, extracted
with the `faces` argument); voices: $FLAG_EXT/vox1_audio (see download_vox1_audio.py).
    python src/data_prep/build_vox1.py [faces]
Writes $FLAG_EXT/vox1_manifest.tsv (kind, identity, video, path) and vox1_gender.json (0 = male, 1 = female).
"""
import os, sys, tarfile, json, glob, collections
EXT = os.environ.get("FLAG_EXT", "ext_data")
meta = {}
for i, line in enumerate(open(os.path.join(EXT, "vox1_meta.csv"), encoding="utf-8")):
    if i == 0: continue
    vid, name, g, nat, split = line.rstrip("\n").split("\t")
    meta[name] = (vid, 0 if g == "m" else 1, nat, split)
out_dir = os.path.join(EXT, "vox1_faces")
rows = []
if "faces" in sys.argv[1:]:
    per = collections.Counter(); n = 0
    with tarfile.open(os.path.join(EXT, "zippedFaces.tar.gz"), "r|gz") as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith(".jpg")): continue
            parts = m.name.split("/")           # unzippedFaces/<Name>/1.6/<video>/<frame>.jpg
            if len(parts) != 5 or parts[1] not in meta: continue
            key = (parts[1], parts[3])
            if per[key] >= 2: continue
            per[key] += 1
            d = os.path.join(out_dir, meta[parts[1]][0], parts[3]); os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, os.path.basename(parts[4])), "wb") as f:
                f.write(tf.extractfile(m).read())
            n += 1
            if n % 5000 == 0: print("faces", n, flush=True)
    print("FACES EXTRACTED", n, flush=True)
for p in sorted(glob.glob(os.path.join(out_dir, "*", "*", "*.jpg"))):
    ident, video = p.split(os.sep)[-3:-1]
    rows.append(("face", ident, video, p))
cap = next((int(a[4:]) for a in sys.argv[1:] if a.startswith("cap=")), None)   # max voices per identity
byid = collections.defaultdict(list)
for p in sorted(glob.glob(os.path.join(EXT, "vox1_audio", "audio_*", "*.wav"))):
    b = os.path.basename(p)[:-4]                # id10001_<video>_<utt>; video ids may contain '_'
    ident, rest = b[:7], b[8:]
    byid[ident].append(("voice", ident, rest.rsplit("_", 1)[0], p))
for ident in sorted(byid):
    v = sorted(byid[ident], key=lambda r: (r[2], r[3]))
    if cap and len(v) > cap:                    # evenly spaced over the video-sorted list
        v = [v[round(i * (len(v) - 1) / (cap - 1))] for i in range(cap)]
    rows += v
with open(os.path.join(EXT, "vox1_manifest.tsv"), "w", encoding="utf-8") as f:
    for r in rows: f.write("\t".join(r) + "\n")
json.dump({v[0]: v[1] for v in meta.values()}, open(os.path.join(EXT, "vox1_gender.json"), "w"))
c = collections.Counter(r[0] for r in rows)
print("MANIFEST", dict(c), "face ids", len({r[1] for r in rows if r[0] == "face"}), "voice ids", len({r[1] for r in rows if r[0] == "voice"}), flush=True)
