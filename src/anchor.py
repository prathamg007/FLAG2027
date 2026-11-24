"""Anchor consistency term: a set of anchors from public corpora, each a pair of averaged embeddings stored under the
corpus's anonymous ID: a face anchor F_i (mean ArcFace embedding) and a voice anchor V_i (mean ReDimNet-B6 embedding).

For a trial with face f and voice v, X = anchor whose face anchor is closest to f, Y = anchor whose voice anchor is closest to v:
    m = sig((cos(f, F_X) - T_F) / TAU) * (cos(v, V_X) - OFF_V) + sig((cos(v, V_Y) - T_V) / TAU) * (cos(f, F_Y) - OFF_F)
If the face is close to a face anchor, the voice is checked against that pair's voice anchor (bonus or penalty); likewise from
the voice side. Otherwise both gates stay closed and m is close to 0. Nothing is trained. Constants were set on the MAV-Celeb v2
validation set; the voice gate was also checked for false openings on the challenge training identities. The score of a trial
depends only on its own face and voice.

    python src/anchor.py anchor      # writes models/anchor.scores.json (evaluation trials)
"""
import os, sys, json, collections
os.environ["FV_KEEP_OVERLAP"] = "1"          # the anchor set keeps every identity of its corpora
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common, model

CORPORA = ["vox1", "vox2", "mavold"]
T_F, T_V, TAU, OFF_V, OFF_F = 0.50, 0.65, 0.05, 0.45, 0.30
VOICE_ENC = "redim"                          # speaker encoder of the voice anchors and the voice gate


def anchor_set(corpora, venc="ecapa"):
    """(ids, face anchors, voice anchors of `venc`). The list of anchors is defined by the arc + ecapa feature files."""
    ids, GF, GV = [], [], []
    for ds in corpora:
        s = model.src_vox(ds, ["arc"], ["ecapa"])
        fa, va = model.table(ds, "arc"), model.table(ds, venc)
        F, V = collections.defaultdict(list), collections.defaultdict(list)
        for i, p in zip(s["fid"], s["fp"]): F[i].append(fa[p])
        for i, p in zip(s["vid"], s["vp"]): V[i].append(va[p])
        for i in sorted(set(F) & set(V)):
            f, v = np.mean(F[i], 0), np.mean(V[i], 0)
            ids.append(ds + ":" + i); GF.append(f / np.linalg.norm(f)); GV.append(v / np.linalg.norm(v))
    return ids, np.stack(GF).astype(np.float32), np.stack(GV).astype(np.float32)


def load_anchors(corpora=CORPORA):
    ids, GF, _ = anchor_set(corpora)
    ids_v, _, GV = anchor_set(corpora, VOICE_ENC)
    assert ids == ids_v
    return ids, GF, GV


def anchor_score(dataset, vpaths, fpaths, A, chunk=2000, vds=None):
    """Anchor consistency term m of each (voice, face) trial."""
    _, GF, GV = A
    fa, va = model.table(dataset, "arc"), model.table(vds or dataset, VOICE_ENC)
    offv = np.full(len(vpaths), OFF_V)
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    out = []
    for a in range(0, len(vpaths), chunk):
        Sf = np.stack([fa[p] for p in fpaths[a:a + chunk]]) @ GF.T
        Vc = np.stack([va[p] for p in vpaths[a:a + chunk]])
        Sg = Vc @ GV.T
        r = np.arange(len(Sf)); X = Sf.argmax(1); Y = Sg.argmax(1)
        out.append(sig((Sf[r, X] - T_F) / TAU) * ((Vc * GV[X]).sum(1) - offv[a:a + chunk]) + sig((Sg[r, Y] - T_V) / TAU) * (Sf[r, Y] - OFF_F))
    return np.concatenate(out)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "anchor"; A = load_anchors(); out = {}
    for (track, lang), trl in common.eval_trials().items():
        m = anchor_score("eval", [t[1] for t in trl], [t[2] for t in trl], A)
        out[f"{track}/{lang}"] = {"ids": [t[0] for t in trl], "s": [float(x) for x in m]}
    json.dump({"info": {"tag": tag, "anchors": len(A[0]), "corpora": CORPORA, "gates": [T_F, T_V, TAU, OFF_V, OFF_F],
                        "voice_encoder": VOICE_ENC}, "scores": out}, open(os.path.join(common.MODELS, tag + ".scores.json"), "w"))
    print("WROTE", tag, "anchors", len(A[0]), flush=True)
