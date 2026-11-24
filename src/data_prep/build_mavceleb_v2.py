"""MAV-Celeb v2 (English + Hindi, organisers' release): up to `per_id` faces and voices per identity, spread over
languages and videos.
    python src/data_prep/build_mavceleb_v2.py <mavceleb_v2_complete.zip> v2 [per_id=120]
Extracts to $FLAG_EXT/mavold/v2/ and writes $FLAG_EXT/mavold_manifest.tsv.
"""
import os, sys, zipfile, collections, glob
EXT = os.environ.get("FLAG_EXT", "ext_data"); OUT = os.path.join(EXT, "mavold")
def sample(zp, tag, per_id):
    z = zipfile.ZipFile(zp)
    groups = collections.defaultdict(lambda: collections.defaultdict(list))   # (kind, id) -> (lang, video) -> names
    for n in z.namelist():
        if not n.lower().endswith((".jpg", ".wav")): continue
        parts = n.split("/")
        k = next((i for i, p in enumerate(parts) if p in ("faces", "voices")), None)
        if k is None or len(parts) - k < 4: continue
        kind = "face" if parts[k] == "faces" else "voice"
        groups[(kind, parts[k + 1])]["/".join(parts[k + 2:-1])].append(n)
    n_out = 0
    for (kind, ident), vids in sorted(groups.items()):
        lists = [sorted(v) for _, v in sorted(vids.items())]
        # spread within each video too: reorder each list so early picks are far apart
        lists = [[l[i] for i in sorted(range(len(l)), key=lambda i: (bin(i)[::-1], i))] for l in lists]
        pick, r = [], 0
        while len(pick) < per_id and any(len(l) > r for l in lists):
            pick += [l[r] for l in lists if len(l) > r][:per_id - len(pick)]; r += 1
        for n in pick:
            parts = n.split("/"); k = next(i for i, p in enumerate(parts) if p in ("faces", "voices"))
            dst = os.path.join(OUT, tag, *parts[k:])
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as f: f.write(z.read(n))
            n_out += 1
    print(tag, "identities", len({i for _, i in groups}), "files kept", n_out, flush=True)
if len(sys.argv) > 2:
    sample(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 120)
rows = []
for tag in sorted(os.listdir(OUT)):
    for kind, sub, ext in (("face", "faces", "jpg"), ("voice", "voices", "wav")):
        for p in sorted(glob.glob(os.path.join(OUT, tag, sub, "*", "**", "*." + ext), recursive=True)):
            rel = os.path.relpath(p, os.path.join(OUT, tag, sub)).split(os.sep)
            rows.append((kind, f"{tag}_{rel[0]}", "/".join(rel[1:-1]), p))
with open(os.path.join(EXT, "mavold_manifest.tsv"), "w", encoding="utf-8") as f:
    for r in rows: f.write("\t".join(r) + "\n")
c = collections.Counter(r[0] for r in rows)
print("MAVOLD MANIFEST", dict(c), "ids", len({r[1] for r in rows}), flush=True)
