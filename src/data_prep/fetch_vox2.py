"""Sample VoxCeleb2 (external training data) from the public HF parquet mirror
`blueskyheaven/voxceleb2-mp4-binary` without downloading all 278 GB.

Each parquet row group holds 100 consecutive clips (speakers are contiguous). For every
speaker we fetch the ONE row group holding most of their clips and keep up to N_PER clips,
evenly spaced. Per kept clip: the middle video frame (224x224 face-track crop) as JPEG and
the audio track as 16 kHz mono FLAC (first 10 s).

    python src/data_prep/fetch_vox2.py index      # speaker column of all shards -> ext_data/vox2_index.json
    python src/data_prep/fetch_vox2.py fetch [threads]

Output: ext_data/vox2/<speaker>/<shard>_<rg>_<row>.jpg|.flac ; resumable (ext_data/vox2_done.txt).
"""
import os, sys, io, json, time, random, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ROOT = os.environ.get("FLAG_EXT", "ext_data")
REPO = "datasets/blueskyheaven/voxceleb2-mp4-binary/data/train-%05d-of-00056.parquet"
INDEX = os.path.join(ROOT, "vox2_index.json")
PASS = int(os.environ.get("VOX2_PASS", "1"))     # 2 = a second row group per speaker (other videos of the same people)
DONE = os.path.join(ROOT, "vox2_done.txt" if PASS == 1 else "vox2_done2.txt")
WANT = {}                                        # pass 2: (shard, rg) -> speakers to take from it
OUT = os.path.join(ROOT, "vox2")
N_PER = 12
lock = threading.Lock()


_MD = {}


def open_pf(shard, block=None):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    kw = {"block_size": block} if block else {}
    return pq.ParquetFile(HfFileSystem().open(REPO % shard, "rb", **kw))


class RangeFile:
    """File-like view of a remote parquet file backed by ONE pre-fetched byte range."""

    def __init__(self, size, start, buf):
        self.size, self.start, self.buf, self.pos = size, start, buf, 0
        self.closed = False
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def tell(self): return self.pos
    def readable(self): return True
    def seekable(self): return True
    def writable(self): return False
    def flush(self): pass
    def close(self): self.closed = True
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        a = self.pos - self.start
        if a < 0 or a + n > len(self.buf):
            raise IOError(f"read outside prefetched range: pos={self.pos} n={n}")
        self.pos += n
        return self.buf[a:a + n]


def read_videos(shard, rg):
    """All 'video' blobs of one row group with a single exact HTTP range request."""
    import pyarrow.parquet as pq, requests
    from huggingface_hub import hf_hub_url, get_token
    name = "data/train-%05d-of-00056.parquet" % shard
    with lock:
        if shard not in _MD:
            _MD[shard] = (open_pf(shard).metadata, HfFileSystemSize(shard))
    md, size = _MD[shard]
    col = md.row_group(rg).column(0)
    assert col.path_in_schema == "video"
    start = col.dictionary_page_offset if col.has_dictionary_page else col.data_page_offset
    end = start + col.total_compressed_size
    url = hf_hub_url("blueskyheaven/voxceleb2-mp4-binary", name, repo_type="dataset")
    h = {"Range": f"bytes={start}-{end - 1}"}
    tok = get_token()
    if tok: h["Authorization"] = f"Bearer {tok}"
    r = requests.get(url, headers=h, timeout=(20, 180))
    r.raise_for_status()
    assert len(r.content) == end - start, (len(r.content), end - start)
    pf = pq.ParquetFile(RangeFile(size, start, r.content), metadata=md)
    return pf.read_row_group(rg, columns=["video"]).column(0)


def HfFileSystemSize(shard):
    from huggingface_hub import HfFileSystem
    return HfFileSystem().info(REPO % shard)["size"]


def index_shard(shard):
    for attempt in range(5):
        try:
            pf = open_pf(shard, 1 << 16)
            out = []
            for rg in range(pf.metadata.num_row_groups):
                out.append(pf.read_row_group(rg, columns=["speaker_id"]).column(0).to_pylist())
            print("indexed shard", shard, len(out), flush=True)
            return shard, out
        except Exception as e:
            print("index retry", shard, repr(e)[:150], flush=True); time.sleep(10)
    return shard, None


def build_index():
    idx = json.load(open(INDEX)) if os.path.exists(INDEX) else {}
    todo = [s for s in range(56) if str(s) not in idx]
    with ThreadPoolExecutor(12) as ex:
        for shard, out in ex.map(index_shard, todo):
            if out is not None:
                idx[str(shard)] = out
                json.dump(idx, open(INDEX, "w"))
    print("INDEX DONE", len(idx), "shards", flush=True)


def plan():
    idx = json.load(open(INDEX))
    best = {}                                   # speaker -> (count, shard, rg)
    for s, groups in idx.items():
        for rg, spk in enumerate(groups):
            for sp in set(spk):
                c = spk.count(sp)
                if sp not in best or c > best[sp][0]:
                    best[sp] = (c, int(s), rg)
    groups = sorted({(s, rg) for c, s, rg in best.values() if c >= 4})
    if PASS == 2:                               # the row group with the second-most clips of each speaker
        second = {}
        for s, gr in idx.items():
            for rg, spk in enumerate(gr):
                for sp in set(spk):
                    c = spk.count(sp)
                    if (int(s), rg) != best[sp][1:] and (sp not in second or c > second[sp][0]):
                        second[sp] = (c, int(s), rg)
        for sp, (c, s, rg) in second.items():
            if c >= 4: WANT.setdefault((s, rg), set()).add(sp)
        groups = sorted(WANT)
    random.Random(0).shuffle(groups)            # unbiased partial coverage if interrupted
    return idx, groups, len(best)


def decode(blob):
    import av
    c = av.open(io.BytesIO(blob))
    vs = c.streams.video[0]
    n = vs.frames or 0
    frame = None
    for i, fr in enumerate(c.decode(video=0)):
        frame = fr
        if i >= max(0, n // 2):
            break
    img = frame.to_image()
    c.seek(0)
    rs = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    for fr in c.decode(audio=0):
        for o in rs.resample(fr):
            chunks.append(o.to_ndarray().reshape(-1))
    c.close()
    y = np.concatenate(chunks)[:160000].astype(np.int16)
    return img, y


def fetch_group(args):
    import soundfile as sf
    idx, s, rg = args
    spk = idx[str(s)][rg]
    keep = []
    for sp in sorted(set(spk)):
        rows = [i for i, x in enumerate(spk) if x == sp]
        if len(rows) < 4 or (PASS == 2 and sp not in WANT.get((s, rg), ())):
            continue
        sel = sorted(set(np.linspace(0, len(rows) - 1, min(N_PER, len(rows))).round().astype(int)))
        keep += [rows[i] for i in sel]
    for attempt in range(6):
        try:
            tb = read_videos(s, rg)
            n = 0
            for r in keep:
                try:
                    img, y = decode(tb[r].as_py())
                    if len(y) < 16000:
                        continue
                    d = os.path.join(OUT, spk[r]); os.makedirs(d, exist_ok=True)
                    stem = os.path.join(d, f"{s:02d}_{rg:03d}_{r:03d}")
                    img.save(stem + ".jpg", quality=92)
                    sf.write(stem + ".flac", y, 16000, subtype="PCM_16")
                    n += 1
                except Exception as e:
                    print("clip err", s, rg, r, repr(e)[:100], flush=True)
            with lock:
                open(DONE, "a").write(f"{s} {rg} {n}\n")
            return n
        except Exception as e:
            print("group retry", s, rg, attempt, repr(e)[:150], flush=True); time.sleep(15 * (attempt + 1))
    return 0


def fetch(threads):
    idx, groups, nspk = plan()
    done = set()
    if os.path.exists(DONE):
        done = {tuple(map(int, l.split()[:2])) for l in open(DONE) if l.strip()}
    todo = [g for g in groups if g not in done]
    print(f"speakers {nspk}, groups {len(groups)}, todo {len(todo)}", flush=True)
    t0 = time.time(); k = 0
    with ThreadPoolExecutor(threads) as ex:
        for n in ex.map(fetch_group, [(idx, s, rg) for s, rg in todo]):
            k += 1
            if k % 50 == 0:
                print(f"groups {k}/{len(todo)}  {time.time()-t0:.0f}s  speakers so far {len(os.listdir(OUT))}", flush=True)
    print("VOX2 FETCH DONE", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "index":
        build_index()
    else:
        fetch(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
