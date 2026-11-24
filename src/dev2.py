"""DEV2: four-cell validation set from MAV-Celeb v2 (83 identities not in the challenge training set, English and Hindi).
    heard = English voices, unheard = Hindi voices; std = any wrong face, gen = same-gender wrong face.
Per voice one true face and one wrong face (fixed seed); EER pooled per cell. DEV2-short uses the same trials with the
voices cut to the challenge clip-length distribution (feature set `mavolds`)."""
import numpy as np
import common, model
_TR = None


def trials():
    global _TR
    if _TR is None:
        s = model.src_vox("mavold")
        rng = np.random.default_rng(0)
        fby = {}
        for p, i in zip(s["fp"], s["fid"]): fby.setdefault(i, []).append(p)
        ids = sorted(fby); g = s["gender"]
        out = {}
        for vp, vi in zip(s["vp"], s["vid"]):
            lang = "heard" if common.KSEP + "English" + common.KSEP in vp else "unheard"
            for cell in ("std", "gen"):
                oth = [k for k in ids if k != vi and (cell == "std" or g[k] == g[vi])]
                V, Fc, Y = out.setdefault((cell, lang), ([], [], []))
                V.append(vp); Fc.append(fby[vi][rng.integers(len(fby[vi]))]); Y.append(1)
                b = oth[rng.integers(len(oth))]
                V.append(vp); Fc.append(fby[b][rng.integers(len(fby[b]))]); Y.append(0)
        _TR = {k: (v[0], v[1], np.array(v[2])) for k, v in out.items()}
    return _TR


def scores(models, fnames, vnames, vds="mavold"):
    """vds="mavolds": voices cut to challenge-like clip lengths (DEV2-short)."""
    return {k: model.score(models, model.feats("mavold", F, fnames), model.feats(vds, V, vnames)) for k, (V, F, Y) in trials().items()}


def eer4(sc):
    T = trials()
    r = {f"{c}_{l}": round(common.eer(T[(c, l)][2], sc[(c, l)]), 2) for c in ("std", "gen") for l in ("heard", "unheard")}
    r["avg"] = round(float(np.mean(list(r.values()))), 2)
    return r
