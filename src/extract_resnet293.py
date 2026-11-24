"""WeSpeaker ResNet293-LM voice features (VoxCeleb, ONNX): 80-dim Kaldi fbank (25 ms / 10 ms, Hamming) with mean
subtraction, as in WeSpeaker. Each clip (mono 16 kHz, first 10 s) is repeat-padded to whole seconds.
Files: those of $FLAG_FEAT/<dataset>__ecapa.npz (run extract_features.py with ecapa first).

    python src/extract_resnet293.py mav,eval,mavold,mavolds,vox1,vox2 [--part k/n]
    python src/extract_resnet293.py vox2 --merge n           # join the parts 0..n-1 of a split run
Output: $FLAG_FEAT/<dataset>__wsr293.npz
"""
import os, sys, time, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import common, extract_features

ap = argparse.ArgumentParser(); ap.add_argument("datasets"); ap.add_argument("--part"); ap.add_argument("--merge", type=int)
a = ap.parse_args()


def merge(ds, n):
    parts = [np.load(os.path.join(common.FEAT, f"{ds}__wsr293.part{k}of{n}.npz"), allow_pickle=True) for k in range(n)]
    dst = os.path.join(common.FEAT, f"{ds}__wsr293.npz")
    np.savez(dst + ".tmp.npz", paths=np.concatenate([p["paths"] for p in parts]), emb=np.concatenate([p["emb"] for p in parts]))
    os.replace(dst + ".tmp.npz", dst); print("MERGED", ds, flush=True)


def encoder():
    import onnxruntime as ort, torchaudio
    from huggingface_hub import snapshot_download
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    onnx = glob.glob(os.path.join(snapshot_download("Wespeaker/wespeaker-voxceleb-resnet293-LM"), "*.onnx"))[0]
    sess = ort.InferenceSession(onnx, so, providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    def f(y):
        y = np.resize(y, int(np.ceil(len(y) / 16000)) * 16000)
        fb = torchaudio.compliance.kaldi.fbank(torch.from_numpy(y)[None] * 32768.0, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                                               sample_frequency=16000, window_type="hamming", use_energy=False)
        fb = fb - fb.mean(0, keepdim=True)
        return np.asarray(sess.run(None, {iname: fb[None].numpy().astype(np.float32)})[-1]).reshape(-1)
    return f


def run(ds, f):
    extract_features.SHORT = ds == "mavolds"
    if extract_features.SHORT: extract_features._DUR = np.load(os.path.join(common.MANIFESTS, "v4_train_durations.npy"))
    paths = [str(p) for p in np.load(os.path.join(common.FEAT, f"{ds}__ecapa.npz"), allow_pickle=True)["paths"]]
    paths = [p for p in paths if os.path.exists(common.real(p))]
    suffix = ""
    if a.part:
        k, n = map(int, a.part.split("/")); paths = paths[k::n]; suffix = f".part{k}of{n}"
    out = os.path.join(common.FEAT, f"{ds}__wsr293{suffix}.npz")
    if os.path.exists(out): print(f"[{ds}] exists", flush=True); return
    E, ok, t0 = [], [], time.time()
    for p in paths:
        try:
            E.append(np.asarray(f(np.ascontiguousarray(extract_features.load_audio(p), dtype=np.float32)), dtype=np.float32)); ok.append(p)
        except Exception as ex:
            if "CUDA" in repr(ex): raise
            print("BAD", p, repr(ex)[:100], flush=True)
    np.savez(out + ".tmp.npz", paths=np.array(ok), emb=np.stack(E).astype(np.float16)); os.replace(out + ".tmp.npz", out)
    print(f"SAVED {ds}/wsr293{suffix} n={len(ok)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if a.merge:
        for ds in a.datasets.split(","): merge(ds, a.merge)
    else:
        f = encoder()
        for ds in a.datasets.split(","): run(ds, f)
