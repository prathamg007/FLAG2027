"""PE-Core-L/14-336 face features (timm `vit_pe_core_large_patch14_336.fb`, pooled embedding, fp16).
Files: those of $FLAG_FEAT/<dataset>__clip.npz (run extract_features.py with clip first).

    python src/extract_pecore.py mav,eval,mavold,vox1,vox2
Output: $FLAG_FEAT/<dataset>__pel.npz
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import common

DEV, B = "cuda", 128


@torch.no_grad()
def run(ds, m, tf, pool):
    out = os.path.join(common.FEAT, f"{ds}__pel.npz")
    paths = [str(p) for p in np.load(os.path.join(common.FEAT, f"{ds}__clip.npz"), allow_pickle=True)["paths"]]
    if os.path.exists(out) and len(np.load(out, allow_pickle=True)["paths"]) == len(paths):
        print(f"[{ds}] complete", flush=True); return
    E, t0 = [], time.time()
    for i in range(0, len(paths), B):
        ims = list(pool.map(lambda p: Image.open(common.real(p)).convert("RGB"), paths[i:i + B]))
        x = torch.stack(list(pool.map(tf, ims))).to(DEV)
        E.append(m(x.half()).float().cpu().numpy())
    np.savez(out + ".tmp.npz", paths=np.array(paths), emb=np.concatenate(E).astype(np.float16)); os.replace(out + ".tmp.npz", out)
    print(f"SAVED {ds}/pel n={len(paths)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    import timm
    m = timm.create_model("vit_pe_core_large_patch14_336.fb", pretrained=True, num_classes=0).to(DEV).eval().half()
    tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=m))
    pool = ThreadPoolExecutor(6)
    for ds in sys.argv[1].split(","):
        run(ds, m, tf, pool)
