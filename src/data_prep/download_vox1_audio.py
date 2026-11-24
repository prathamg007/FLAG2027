"""Download the VoxCeleb1 dev audio mirror (Hugging Face `humanify/voxceleb1_dev`) into $FLAG_EXT/vox1_audio.
    python src/data_prep/download_vox1_audio.py
"""
import os, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download
D = os.path.join(os.environ.get("FLAG_EXT", "ext_data"), "vox1_audio")
files = sorted(f for f in HfApi().list_repo_files("humanify/voxceleb1_dev", repo_type="dataset") if f.endswith(".wav"))


def get(f):
    if os.path.exists(os.path.join(D, f.replace("/", os.sep))): return 1
    for a in range(8):
        try:
            hf_hub_download("humanify/voxceleb1_dev", f, repo_type="dataset", local_dir=D); return 1
        except Exception:
            time.sleep(20 * (a + 1))
    return 0


with ThreadPoolExecutor(12) as ex:
    n = sum(ex.map(get, files))
print("DONE", n, "of", len(files), flush=True)
