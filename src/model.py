"""Feature tables, training data, the two-tower model, training, cross-validation and scoring.

Every score is a function of one (face, voice) pair: models run in eval mode, feature standardisation uses frozen
training statistics, and ensembles are plain means of raw cosine scores.
"""
import os, json, zipfile, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import common
from common import FEAT, MANIFESTS, DEVICE

_T = {}
_SPLIT = {"w2vag": ("w2vag", slice(0, 1024), True), "vattr": ("w2vag", slice(1024, 1028), False)}   # block -> (file, columns, L2-normalise)


def table(dataset, name):
    """Key path -> embedding (float32) of one feature block."""
    k = (dataset, name)
    if k not in _T:
        enc, sl, norm = _SPLIT.get(name, (name, slice(None), True))
        z = np.load(os.path.join(FEAT, f"{dataset}__{enc}.npz"), allow_pickle=True)
        E = z["emb"].astype(np.float32)[:, sl]
        if norm:
            E = common.l2n(E)
        _T[k] = {str(p): E[i] for i, p in enumerate(z["paths"])}
    return _T[k]


def feats(dataset, paths, names):
    """Concatenated feature blocks for a list of key paths."""
    return np.concatenate([np.stack([table(dataset, n)[p] for p in paths]) for n in names], 1)


# ------------------------------------------------------------------------------------------------------ data sources
def src_mav():
    """Challenge training set: row i is face i and voice i of one identity."""
    rows = common.train_index(); gm = common.gender_meta()
    ids = [r["id"] for r in rows]
    return {"name": "mav", "dataset": "mav", "vp": [r["voice"] for r in rows], "vid": ids,
            "fp": [r["face"] for r in rows], "fid": ids, "gender": {i: gm[i] for i in set(ids)}}


def src_vox(dataset, fnames=("clip",), vnames=("ecapa", "wavlm", "w2vag")):
    """External corpus from data/manifests/<dataset>_manifest.tsv (kind, identity, video, key path).
    Only files with features for every requested block are used.
      FV_KEEP_OVERLAP=1 keeps every identity; otherwise identities listed in <dataset>_overlap_train.json (people of the
                        MAV-Celeb v2 validation set) are dropped. mavold always drops its overlap with the challenge set.
      FV_VOX2_OLD=1     restricts VoxCeleb2 to the file list <dataset>_manifest_pass1.tsv (the list the models were trained on).
      FV_VOX1_M16=1     restricts VoxCeleb1 voices to the first 16 per identity when ReDimNet-M is an input (first generation)."""
    vp, vid, fp, fid = [], [], [], []
    for line in open(os.path.join(MANIFESTS, dataset + "_manifest.tsv"), encoding="utf-8"):
        k, ident, video, path = line.rstrip(chr(10)).split(chr(9))
        (vp if k == "voice" else fp).append(path); (vid if k == "voice" else fid).append(ident)
    g = json.load(open(os.path.join(MANIFESTS, dataset + "_gender.json")))
    dup = os.path.join(MANIFESTS, dataset + "_overlap_train.json")
    if os.path.exists(dup) and (dataset == "mavold" or os.environ.get("FV_KEEP_OVERLAP") != "1"):
        for k in json.load(open(dup)):
            g.pop(k, None)
    okv = set.intersection(*[set(table(dataset, n)) for n in vnames])
    snap = os.path.join(MANIFESTS, dataset + "_manifest_pass1.tsv")
    old = {l.rstrip(chr(10)).split(chr(9))[3] for l in open(snap, encoding="utf-8")} if (os.environ.get("FV_VOX2_OLD") == "1" and os.path.exists(snap)) else None
    if old is not None: okv &= old
    if os.environ.get("FV_VOX1_M16") == "1" and dataset == "vox1" and "redimm" in vnames:
        cnt, first = {}, set()
        for p_, i_ in zip(vp, vid):
            if cnt.get(i_, 0) < 16: first.add(p_); cnt[i_] = cnt.get(i_, 0) + 1
        okv &= first
    okf = set.intersection(*[set(table(dataset, n)) for n in fnames])
    if old is not None: okf &= old
    kv = [i for i, p in enumerate(vp) if p in okv]; kf = [i for i, p in enumerate(fp) if p in okf]
    both = {vid[i] for i in kv} & {fid[i] for i in kf} & set(g)
    kv = [i for i in kv if vid[i] in both]; kf = [i for i in kf if fid[i] in both]
    return {"name": dataset, "dataset": dataset, "vp": [vp[i] for i in kv], "vid": [vid[i] for i in kv],
            "fp": [fp[i] for i in kf], "fid": [fid[i] for i in kf], "gender": {i: g[i] for i in both}}


class Data:
    """All training rows of the chosen sources on the GPU, grouped by identity."""

    def __init__(self, sources, fnames, vnames):
        self.fnames, self.vnames = fnames, vnames
        Xf, Xv, yf, yv, self.ids, self.src, self.gender = [], [], [], [], [], [], []
        for s in sources:
            base = len(self.ids)
            loc = sorted(s["gender"]); m = {x: base + i for i, x in enumerate(loc)}
            self.ids += [f"{s['name']}:{x}" for x in loc]; self.src += [s["name"]] * len(loc)
            self.gender += [s["gender"][x] for x in loc]
            Xf.append(feats(s["dataset"], s["fp"], fnames)); yf += [m[x] for x in s["fid"]]
            Xv.append(feats(s["dataset"], s["vp"], vnames)); yv += [m[x] for x in s["vid"]]
        self.Xf, self.Xv = np.concatenate(Xf), np.concatenate(Xv)
        self.yf, self.yv = np.array(yf), np.array(yv)
        self.gender = np.array(self.gender); self.src = np.array(self.src)
        self.rsf, self.rsv = self.src[self.yf], self.src[self.yv]
        self.tf = torch.from_numpy(self.Xf).to(DEVICE); self.tv = torch.from_numpy(self.Xv).to(DEVICE)
        self.n_id = len(self.ids)

        def group(y):
            order = np.argsort(y, kind="stable"); cnt = np.bincount(y, minlength=self.n_id)
            return order, np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt
        self.of, self.sf, self.cf = group(self.yf)
        self.ov, self.sv, self.cv = group(self.yv)

    def sample(self, rng, ids, K, order, start, cnt):
        """K random rows of each identity in ids."""
        r = rng.random((len(ids), K))
        return order[(start[ids][:, None] + np.floor(r * cnt[ids][:, None]).astype(np.int64))].reshape(-1)


# ------------------------------------------------------------------------------------------------------------- model
class Head(nn.Module):
    """Two towers (input dropout -> Linear -> BatchNorm -> ReLU -> dropout -> Linear) into a shared L2-normalised space.
    Standardisation statistics of the challenge training set are frozen buffers."""

    def __init__(self, df, dv, cfg, mu_f=None, sd_f=None, mu_v=None, sd_v=None):
        super().__init__()
        dim, hid, p_in, p = cfg.get("dim", 128), cfg.get("hid", 256), cfg.get("p_in", 0.8), cfg.get("p", 0.3)

        def tower(i):
            return nn.Sequential(nn.Dropout(p_in), nn.Linear(i, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Dropout(p), nn.Linear(hid, dim))
        self.face = tower(df)
        self.voice = tower(dv)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))   # used only without a fixed temperature
        for n, a, d in (("mu_f", mu_f, df), ("sd_f", sd_f, df), ("mu_v", mu_v, dv), ("sd_v", sd_v, dv)):
            self.register_buffer(n, torch.zeros(d) if a is None and n[0] == "m" else (torch.ones(d) if a is None else torch.as_tensor(a)))

    def proj(self, f, v):                        # standardised inputs -> unit embeddings
        return F.normalize(self.face(f)), F.normalize(self.voice(v))

    def embed(self, f, v):                       # raw inputs
        return self.proj((f - self.mu_f) / self.sd_f, (v - self.mu_v) / self.sd_v)

    @torch.no_grad()
    def score(self, f, v):                       # cosine; higher = same person
        ef, ev = self.embed(f, v)
        return (ef * ev).sum(1)


def ft_cfg(cfg):
    """Last stage on the challenge people: 48 people x 4 samples per batch, cosine schedule."""
    return dict(cfg, cos=True, P=min(cfg.get("P", 256), 48), K=4)


def mp_nce(L, pos):
    """Multi-positive InfoNCE over the rows of L."""
    lp = L - torch.logsumexp(L, 1, keepdim=True)
    return -((lp * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


def train(cfg, D, id_mask, seed=0, on_step=None, model=None, steps=None, lr=None, id_w=None, stat_ids=None):
    """Train (or continue training `model`) on the identities where id_mask is True.
    Each corpus is standardised with its own statistics; the challenge statistics come from the challenge training
    identities `stat_ids` and are frozen into the model."""
    common.seed_all(seed)
    rng = np.random.default_rng(seed)
    ids = np.where(id_mask & (D.cf > 0) & (D.cv > 0))[0]
    sid = ids if stat_ids is None else stat_ids
    TF, TV = D.tf.clone(), D.tv.clone(); st = {}
    for sname in sorted(set(D.src)):
        for X, T, y, rs, k in ((D.Xf, TF, D.yf, D.rsf, "f"), (D.Xv, TV, D.yv, D.rsv, "v")):
            rows = rs == sname
            use = rows & np.isin(y, sid) if sname == "mav" else rows & np.isin(y, ids)
            if use.sum() < 50:
                use = rows & np.isin(y, sid)
            mu, sd = X[use].mean(0), X[use].std(0) + 1e-3
            r = torch.from_numpy(np.where(rows)[0]).to(DEVICE)
            T[r] = (T[r] - torch.from_numpy(mu).to(DEVICE)) / torch.from_numpy(sd).to(DEVICE)
            if sname == "mav":
                st["mu_" + k], st["sd_" + k] = mu, sd
    if model is None:
        model = Head(D.Xf.shape[1], D.Xv.shape[1], cfg, **st).to(DEVICE)
    steps = steps or cfg.get("steps", 1500); lr = lr or cfg.get("lr", 3e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.get("wd", 0.05))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1) if cfg.get("cos", True) else None
    P, K = min(cfg.get("P", 256), len(ids)), cfg.get("K", 2)
    w = np.ones(len(ids)) if id_w is None else id_w[ids].astype(np.float64)
    w = w / w.sum()
    tyf = torch.from_numpy(D.yf).to(DEVICE); tyv = torch.from_numpy(D.yv).to(DEVICE)
    CH = 64                                      # sampled index batches are copied to the GPU 64 steps at a time
    for step in range(1, steps + 1):
        if (step - 1) % CH == 0:
            FI, VI = [], []
            for _ in range(min(CH, steps - step + 1)):
                cs = rng.choice(ids, P, replace=False, p=w)               # P identities, K faces and K voices each, drawn independently
                FI.append(D.sample(rng, cs, K, D.of, D.sf, D.cf)); VI.append(D.sample(rng, cs, K, D.ov, D.sv, D.cv))
            FI = torch.from_numpy(np.stack(FI)).to(DEVICE); VI = torch.from_numpy(np.stack(VI)).to(DEVICE)
        model.train()
        fi, vi = FI[(step - 1) % CH], VI[(step - 1) % CH]
        ef, ev = model.proj(TF[fi], TV[vi])
        yf, yv = tyf[fi], tyv[vi]
        scale = model.logit_scale.exp().clamp(max=100) if not cfg.get("tau") else 1.0 / cfg["tau"]
        L = scale * ev @ ef.T
        pos = (yv[:, None] == yf[None, :]).float()
        loss = 1.0 * 0.5 * (mp_nce(L, pos) + mp_nce(L.T, pos.T))     # symmetric: voices -> faces and faces -> voices
        opt.zero_grad(); loss.backward(); opt.step()
        if sched: sched.step()
        if on_step: on_step(step, model)
    model.eval()
    return model


def id_weights(D, cfg):
    """Sampling weight per identity: challenge identities are drawn mav_w times as often."""
    w = np.ones(D.n_id)
    w[D.src == "mav"] = cfg.get("mav_w", 1.0)
    return w


def train_full(cfg, D, seeds, ft=None):
    """One network per seed on all identities, optionally followed by the last stage on the challenge identities."""
    out = []
    mav = D.src == "mav"
    for s in seeds:
        sid = np.where(mav)[0]
        m = train(cfg, D, np.ones(D.n_id, bool), seed=s, id_w=id_weights(D, cfg), stat_ids=sid)
        if ft:
            m = train(ft_cfg(cfg), D, mav.copy(), seed=s, model=m, steps=ft["steps"], lr=ft["lr"], stat_ids=sid)
        out.append(m)
    return out


@torch.no_grad()
def score(models, Xf, Xv):
    """Plain mean of the networks' cosine scores; one score per row."""
    tf = torch.from_numpy(Xf).float().to(DEVICE); tv = torch.from_numpy(Xv).float().to(DEVICE)
    out = 0.0
    for m in models:
        m.eval(); out = out + m.score(tf, tv).cpu().numpy()
    return out / len(models)


# ------------------------------------------------------------------------------------------------------- validation
def val_trials(D, val_ids, rng, per_row=2):
    """Balanced trials among held-out identities: std (any wrong face) and gen (same-gender wrong face)."""
    val_ids = [i for i in val_ids if D.cf[i] > 1 and D.cv[i] > 0]
    out = {}
    vrows = np.where(np.isin(D.yv, val_ids))[0]
    for cell in ("std", "gen"):
        V, Fc, Y = [], [], []
        for i in vrows:
            a = D.yv[i]
            others = [k for k in val_ids if k != a and (cell == "std" or D.gender[k] == D.gender[a])]
            if not others:
                continue
            for _ in range(per_row):
                V.append(i); Fc.append(D.of[D.sf[a] + rng.integers(D.cf[a])]); Y.append(1)
                b = others[rng.integers(len(others))]
                V.append(i); Fc.append(D.of[D.sf[b] + rng.integers(D.cf[b])]); Y.append(0)
        out[cell] = (np.array(V), np.array(Fc), np.array(Y))
    return out


def cv(cfg, D, folds=7, seeds=(0,), ft=None):
    """7-fold cross-validation by identity over the challenge training people; external identities always train.
    Returns {'std', 'gen', 'avg', 'per'} averaged over folds and seeds."""
    mav = np.where(D.src == "mav")[0]
    res = collections.defaultdict(list)
    for sd in seeds:
        perm = np.random.default_rng(100 + sd).permutation(mav)
        for k in range(folds):
            va = perm[k::folds]
            mask = np.ones(D.n_id, bool); mask[va] = False
            trials = val_trials(D, list(va), np.random.default_rng(1000 * sd + k))
            sid = np.setdiff1d(mav, va)
            m = train(cfg, D, mask, seed=sd * 100 + k, id_w=id_weights(D, cfg), stat_ids=sid)
            if ft:
                mm = np.zeros(D.n_id, bool); mm[mav] = True; mm[va] = False
                m = train(ft_cfg(cfg), D, mm, seed=sd * 100 + k, model=m, steps=ft["steps"], lr=ft["lr"], stat_ids=sid)
            for cell, (vi, fi, y) in trials.items():
                res[cell].append(common.eer(y, score([m], D.Xf[fi], D.Xv[vi])))
    s, g = float(np.mean(res["std"])), float(np.mean(res["gen"]))
    return {"std": round(s, 2), "gen": round(g, 2), "avg": round((s + g) / 2, 2),
            "per": {c: [round(float(x), 3) for x in res[c]] for c in ("std", "gen")}}


# ------------------------------------------------------------------------------------------------------- evaluation
def score_eval(models, fnames, vnames):
    """{(track, language): (pair ids, scores)} for the evaluation lists."""
    out = {}
    for cell, tr in common.eval_trials().items():
        out[cell] = ([t[0] for t in tr], score(models, feats("eval", [t[2] for t in tr], fnames), feats("eval", [t[1] for t in tr], vnames)))
    return out


def write_zip(cell_scores, tag):
    """Submission archive models/<tag>.zip; each line is '<pair id> <-similarity>' (lower = same person)."""
    root = os.path.join(common.MODELS, tag); trials = common.eval_trials()
    for (track, lang), (ids, s) in cell_scores.items():
        assert ids == [t[0] for t in trials[(track, lang)]] and np.all(np.isfinite(s))
        d = os.path.join(root, track); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, common.SUBNAME[lang]), "w", encoding="utf-8", newline="\n") as f:
            for pid, v in zip(ids, s):
                f.write(f"{pid} {-float(v):.6f}\n")
    zp = os.path.join(common.MODELS, f"{tag}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for track, lang in common.LISTS:
            z.write(os.path.join(root, track, common.SUBNAME[lang]), f"{track}/{common.SUBNAME[lang]}")
    with zipfile.ZipFile(zp) as z:
        for (track, lang), tr in trials.items():
            got = [l.split()[0] for l in z.read(f"{track}/{common.SUBNAME[lang]}").decode().strip().split("\n")]
            assert got == [t[0] for t in tr]
    return zp
