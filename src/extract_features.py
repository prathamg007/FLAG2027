"""Per-file feature extraction with frozen public encoders. Each embedding is a function of one file.

    python src/extract_features.py <dataset> <encoder>[,<encoder>...]

Datasets: mav (challenge training files), eval (evaluation files, scored only), vox1, vox2, mavold (lists in data/manifests),
          mavolds (MAV-Celeb v2 voices cut to the challenge clip-length distribution, for DEV2-short).
Voice encoders (mono 16 kHz, first 10 s):
  wavlm   WavLM-base-plus, mean over time of the layer-averaged hidden states (768)
  w2vag   wav2vec2-large-robust age/gender, mean over time of the last hidden state (1024) + age and 3 gender outputs
  ecapa   SpeechBrain ECAPA-TDNN, VoxCeleb (192)
  redim   ReDimNet-B6, VoxCeleb2 (192)          redimm  ReDimNet-M, VoxBlink2 + VoxCeleb2 + CN-Celeb (192)
Face encoders (224 x 224 crops):
  clip    CLIP ViT-L/14 image embedding (768)
  arc     InsightFace buffalo_l (SCRFD + ArcFace R50), 512, L2-normalised; used by the anchor consistency term
Output: $FLAG_FEAT/<dataset>__<encoder>.npz with {paths: key paths, emb: float16}; resumable.
"""
import os, sys, time, glob, zlib
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

DEV = "cuda"
VOICE = {"wavlm", "w2vag", "ecapa", "redim", "redimm"}
FACE = {"clip", "arc"}
MAXS = 160000                                    # first 10 s
SHORT, _DUR = False, None


def file_list(dataset, kind):
    """Key paths of the files of one dataset."""
    global SHORT, _DUR
    if dataset == "mavolds":
        SHORT = kind == "voice"; _DUR = np.load(os.path.join(common.MANIFESTS, "v4_train_durations.npy")); dataset = "mavold"
    ext = ".wav" if kind == "voice" else ".jpg"
    if dataset == "mav":
        base = common.KJ(common.KEY_ROOT, "train_set", "train_set")
        return sorted(common.key(q) for q in glob.glob(os.path.join(common.real(base), "**", "*" + ext), recursive=True))
    if dataset == "eval":
        out = set()
        for tr in common.eval_trials().values():
            out.update(t[1] if kind == "voice" else t[2] for t in tr)
        return sorted(out)
    if dataset in ("vox1", "vox2", "mavold"):
        return [l.rstrip("\n").split("\t")[3] for l in open(os.path.join(common.MANIFESTS, dataset + "_manifest.tsv"), encoding="utf-8")
                if l.split("\t")[0] == kind]
    raise KeyError(dataset)


def load_audio(p):
    """Mono 16 kHz, first 10 s; for mavolds cut to a length drawn deterministically per file from the challenge training
    clip-length distribution; at least 0.25 s."""
    import soundfile as sf
    y, sr = sf.read(common.real(p), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    if sr != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    y = y[:MAXS]
    if SHORT:
        q = (zlib.crc32(p.encode()) % 10000) / 10000.0
        y = y[:int(16000 * float(np.quantile(_DUR, q)))]
    if len(y) < 4000:
        y = np.pad(y, (0, 4000 - len(y)))
    return y


def make_voice(enc):
    if enc in ("wavlm", "w2vag"):
        from transformers import AutoConfig, AutoFeatureExtractor, WavLMModel, Wav2Vec2Model
        from huggingface_hub import hf_hub_download
        mid, cls, prefix = {"wavlm": ("microsoft/wavlm-base-plus", WavLMModel, "wavlm."),
                            "w2vag": ("audeering/wav2vec2-large-robust-24-ft-age-gender", Wav2Vec2Model, "wav2vec2.")}[enc]
        m = cls(AutoConfig.from_pretrained(mid))
        sd = torch.load(hf_hub_download(mid, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        m.load_state_dict({k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}, strict=False)
        m = m.to(DEV).eval()
        fe = AutoFeatureExtractor.from_pretrained(mid)
        if enc == "w2vag":                       # age and gender heads of the checkpoint: dense -> tanh -> out_proj
            hd = {n: [sd[f"{n}.{k}"].float().to(DEV) for k in ("dense.weight", "dense.bias", "out_proj.weight", "out_proj.bias")]
                  for n in ("age", "gender")}

        @torch.no_grad()
        def f(y):
            x = fe(y, sampling_rate=16000, return_tensors="pt")["input_values"].to(DEV)
            if enc == "w2vag":
                h = m(x).last_hidden_state.mean(1)
                ex = [torch.tanh(h @ W1.T + b1) @ W2.T + b2 for W1, b1, W2, b2 in hd.values()]
                return torch.cat([h] + ex, 1).squeeze(0).float().cpu().numpy()
            o = m(x, output_hidden_states=True)
            return torch.stack(o.hidden_states).mean(0).mean(1).squeeze(0).float().cpu().numpy()
        return f
    if enc == "ecapa":
        from speechbrain.inference.speaker import EncoderClassifier
        m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir=os.path.join(common.REPO, "pretrained", "sb_ecapa"), run_opts={"device": DEV})

        @torch.no_grad()
        def f(y):
            return m.encode_batch(torch.from_numpy(y)[None].to(DEV)).squeeze().float().cpu().numpy()
        return f
    if enc in ("redim", "redimm"):
        name, tt, dsn = {"redim": ("b6", "ft_lm", "vox2"), "redimm": ("M", "ft_mix", "vb2+vox2+cnc")}[enc]
        m = torch.hub.load("IDRnD/ReDimNet", "ReDimNet", model_name=name, train_type=tt, dataset=dsn, trust_repo=True).to(DEV).eval()

        @torch.no_grad()
        def f(y):
            return m(torch.from_numpy(y)[None].to(DEV)).squeeze().float().cpu().numpy()
        return f
    raise KeyError(enc)


def make_face(enc):
    """Function: list of key paths -> embeddings."""
    from PIL import Image
    if enc == "clip":
        from transformers import CLIPModel, CLIPProcessor
        from concurrent.futures import ThreadPoolExecutor
        m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEV).eval().half()
        pr = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        pool = ThreadPoolExecutor(8)

        @torch.no_grad()
        def g(paths):
            ims = list(pool.map(lambda p: Image.open(common.real(p)).convert("RGB"), paths))
            x = pr(images=ims, return_tensors="pt")["pixel_values"].to(DEV).half()
            e = m.get_image_features(pixel_values=x)
            e = e if torch.is_tensor(e) else e.pooler_output
            return e.float().cpu().numpy()
        return g
    if enc == "arc":
        import cv2
        if hasattr(os, "add_dll_directory"): os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", root=os.environ.get("INSIGHTFACE_ROOT", os.path.expanduser("~/.insightface")),
                           allowed_modules=["detection", "recognition"], providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(224, 224))
        rec = app.models["recognition"]

        def g(paths):                            # largest detected face; without a detection the crop itself is the face chip
            out = []
            for p in paths:
                img = cv2.imread(common.real(p))
                fs = app.get(img)
                if fs:
                    e = max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])).normed_embedding
                else:
                    e = rec.get_feat(cv2.resize(img, (112, 112))).flatten(); e = e / (np.linalg.norm(e) + 1e-9)
                out.append(np.asarray(e, dtype=np.float32))
            return np.stack(out)
        return g
    raise KeyError(enc)


def run(dataset, enc):
    kind = "voice" if enc in VOICE else "face"
    paths = file_list(dataset, kind)
    os.makedirs(common.FEAT, exist_ok=True)
    out = os.path.join(common.FEAT, f"{dataset}__{enc}.npz")
    done = {}
    if os.path.exists(out):
        with np.load(out, allow_pickle=True) as z:
            done = {str(p): e for p, e in zip(z["paths"], np.array(z["emb"]))}
    todo = [p for p in paths if p not in done]
    print(f"[{dataset}/{enc}] {len(paths)} files, {len(todo)} to do", flush=True)
    if not todo:
        return

    def save():
        ks = [p for p in paths if p in done]
        np.savez(out + ".tmp.npz", paths=np.array(ks), emb=np.stack([done[k] for k in ks]).astype(np.float16))
        os.replace(out + ".tmp.npz", out)

    t0 = last = time.time(); bad = 0
    if kind == "voice":
        f = make_voice(enc)
        for i, p in enumerate(todo):
            try:
                done[p] = f(load_audio(p))
            except Exception as e:
                if "CUDA" in repr(e): save(); raise
                bad += 1; print("BAD", p, repr(e)[:120], flush=True)
            if time.time() - last > 300: save(); last = time.time()
    else:
        f = make_face(enc); B = 64
        for i in range(0, len(todo), B):
            ch = todo[i:i + B]
            try:
                done.update(zip(ch, f(ch)))
            except Exception:
                for p in ch:                     # retry one by one to isolate an unreadable file
                    try: done[p] = f([p])[0]
                    except Exception as e2:
                        if "CUDA" in repr(e2): save(); raise
                        bad += 1; print("BAD", p, repr(e2)[:120], flush=True)
            if time.time() - last > 600: save(); last = time.time()
    save()
    print(f"SAVED {dataset}/{enc} n={len(done)} failed={bad} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for enc in sys.argv[2].split(","):
        run(sys.argv[1], enc)
        torch.cuda.empty_cache()
