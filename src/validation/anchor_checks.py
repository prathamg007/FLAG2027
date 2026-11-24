"""Training-side checks of the anchor consistency term.
  overlap   challenge training identities whose ArcFace face centroid matches a face anchor (cosine > 0.5)
  gates     share of faces / voices of challenge training people without an anchor for which a gate opens

    python src/validation/anchor_checks.py
"""
import os, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import common, model, anchor

ids, GF, GV = anchor.load_anchors()
mv = model.src_mav(); fa = model.table("mav", "arc"); va = model.table("mav", anchor.VOICE_ENC)
by = collections.defaultdict(list)
for i, p in zip(mv["fid"], mv["fp"]): by[i].append(fa[p])
C = np.stack([np.mean(by[i], 0) for i in sorted(by)]); C /= np.linalg.norm(C, axis=1, keepdims=True)
S = C @ GF.T
known = {i for i, s in zip(sorted(by), S.max(1)) if s > 0.5}
print("challenge training identities with an anchor:", sorted(known))
uf = np.array([i not in known for i in mv["fid"]]); uv = np.array([i not in known for i in mv["vid"]])
Sf = (np.stack([fa[p] for p in mv["fp"]]) @ GF.T).max(1); Sv = (np.stack([va[p] for p in mv["vp"]]) @ GV.T).max(1)
print(f"face gate opens on {100 * (Sf[uf] > anchor.T_F).mean():.3f} % of faces and voice gate on {100 * (Sv[uv] > anchor.T_V).mean():.3f} % "
      f"of voices of the {len(by) - len(known)} challenge training people without an anchor")
