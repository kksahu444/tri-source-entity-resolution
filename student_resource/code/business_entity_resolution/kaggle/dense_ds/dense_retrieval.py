"""C4 dense channel, Aditya's kaggle/dense kernel with the data read from a private Kaggle dataset.

The original fetches the release inside the kernel through a GITHUB_TOKEN secret; here the seven
source/ground-truth .tsv.gz files come from dataset ashuwhy/amlc26-raw instead, so no secret is needed.
Everything after fetch() is unchanged. Output: /kaggle/working/dense_{split}.parquet (s1, cand, score).
"""
import gzip
import json
import os
import re
import shutil
import sys
import time
import unicodedata

import numpy as np
import pandas as pd
import torch

WORK, DATA = "/kaggle/working", "/kaggle/tmp/data"
K = 30
BUDGET_S = 3 * 3600
WS = re.compile(r"\s+")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gpu_profile():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        log(f"GPU{i}: {p.name} {p.total_memory / 2**30:.1f} GB cc {p.major}.{p.minor}")
        assert p.major == 7 and p.minor == 5, "spec 5: expected a T4 (compute capability 7.5)"


def fetch():
    import glob
    os.makedirs(DATA, exist_ok=True)
    for src in glob.glob("/kaggle/input/**/*.tsv.gz", recursive=True):
        dst = os.path.join(DATA, os.path.basename(src)[:-3])
        with gzip.open(src, "rb") as s, open(dst, "wb") as o:
            shutil.copyfileobj(s, o, 1 << 24)
        log("unpacked", os.path.basename(src))


def read(split, k):
    p = os.path.join(DATA, f"{split}_source{k}.tsv")
    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    with open(p, "rb") as f:
        assert len(df) == sum(1 for _ in f) - 1
    va = lambda s: WS.sub(" ", unicodedata.normalize("NFKC", s)).strip()
    df["text"] = [f"{va(n)}, {va(a)}" for n, a in zip(df["business_name"], df["business_address"])]
    return df[["entity_id", "country", "text"]]


def load_model(name, device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
    return tok, model


@torch.no_grad()
def encode(texts, name, device, bs=512, max_len=64, prefix=""):
    tok, model = load_model(name, device)
    order = np.argsort([len(t) for t in texts])          # length-sorted batches
    out = np.zeros((len(texts), model.config.hidden_size), np.float16)
    for s in range(0, len(texts), bs):
        idx = order[s:s + bs]
        b = tok([prefix + texts[i] for i in idx], padding=True, truncation=True, max_length=max_len,
                return_tensors="pt").to(device)
        h = model(**b).last_hidden_state
        if "e5" in name:
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1)
        else:
            e = h[:, 0]
        out[idx] = torch.nn.functional.normalize(e.float(), dim=-1).half().cpu().numpy()
    return out


def encode_2gpu(texts, name, prefix, tag):
    """One model per GPU in two threads (CUDA kernels release the GIL); halves of the rows each."""
    from concurrent.futures import ThreadPoolExecutor
    half = len(texts) // 2
    with ThreadPoolExecutor(2) as ex:
        parts = list(ex.map(lambda a: encode(a[0], name, f"cuda:{a[1]}", prefix=prefix),
                            [(texts[:half], 0), (texts[half:], 1)]))
    return np.concatenate(parts)


@torch.no_grad()
def topk(q, r, device, k=K, qchunk=4096, rshard=1_000_000):
    """Exact inner-product top-k. Index shards stream through the GPU once (shard-outer loop);
    the running best k per query stays on the GPU."""
    best_v = torch.full((len(q), k), -1e4, dtype=torch.float16, device=device)
    best_i = torch.zeros((len(q), k), dtype=torch.int64, device=device)
    for start in range(0, len(r), rshard):
        sh = torch.from_numpy(r[start:start + rshard]).to(device)
        kk = min(k, sh.shape[0])
        for s in range(0, len(q), qchunk):
            qt = torch.from_numpy(q[s:s + qchunk]).to(device)
            v, i = (qt @ sh.T).topk(kk, dim=1)
            cv = torch.cat([best_v[s:s + qchunk], v], 1)
            ci = torch.cat([best_i[s:s + qchunk], i + start], 1)
            nv, pos = cv.float().topk(k, dim=1)
            best_v[s:s + qchunk] = nv.half()
            best_i[s:s + qchunk] = torch.gather(ci, 1, pos)
        del sh
        torch.cuda.empty_cache()
    return best_v.cpu().numpy(), best_i.cpu().numpy().astype(np.int32)


def precision_check(name):
    texts = ["Garza Empire LLC, 2739 Pebble Glen Circle, Millcreek, UT"] * 8 + [f"Shop {i}, {i} Main Road" for i in range(504)]
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m32 = AutoModel.from_pretrained(name).cuda().eval()
    b = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to("cuda")
    with torch.no_grad():
        h32 = m32(**b).last_hidden_state.float()
        h16 = m32.half()(**b).last_hidden_state.float()
    ok = torch.isfinite(h16).all().item()
    diff = (h32 - h16).abs().max().item()
    log(f"precision check {name}: finite={ok} max_abs_diff={diff:.4f}")
    assert ok, "fp16 produced NaN/Inf"
    del m32
    torch.cuda.empty_cache()


def main():
    gpu_profile()
    fetch()
    frames = {(s, k): read(s, k) for s in ("test", "train") for k in (1, 2, 3)}
    n_texts = sum(len(v) for v in frames.values())
    name = "BAAI/bge-m3"
    sample = frames[("test", 2)]["text"].tolist()[:20000]
    t = time.time()
    encode(sample, name, "cuda:0")
    proj = (time.time() - t) / len(sample) * n_texts / 2
    log(f"bge-m3 1% benchmark: projected {proj / 3600:.2f} h on 2 GPUs")
    if proj > BUDGET_S:
        name = "intfloat/multilingual-e5-small"
    precision_check(name)
    qp, pp = ("query: ", "passage: ") if "e5" in name else ("", "")
    log("model", name)
    for split in ("test", "train"):
        s1 = frames[(split, 1)]
        rec = pd.concat([frames[(split, 2)], frames[(split, 3)]], ignore_index=True)
        qe = encode_2gpu(s1["text"].tolist(), name, qp, f"{split}_q")
        re_ = encode_2gpu(rec["text"].tolist(), name, pp, f"{split}_r")
        log(split, "encoded", qe.shape, re_.shape)
        out = []
        for n, c in enumerate(sorted(s1["country"].unique())):
            qi = np.flatnonzero(s1["country"].to_numpy() == c)
            ri = np.flatnonzero(rec["country"].to_numpy() == c)
            v, i = topk(qe[qi], re_[ri], f"cuda:{n % 2}")
            out.append(pd.DataFrame({"s1": np.repeat(s1["entity_id"].to_numpy()[qi], K),
                                     "cand": rec["entity_id"].to_numpy()[ri][i.ravel()],
                                     "score": v.ravel()}))
            log(split, c, "top-k done", len(qi))
        pd.concat(out, ignore_index=True).to_parquet(f"{WORK}/dense_{split}.parquet", index=False, compression="zstd")
        json.dump({"model": name}, open(f"{WORK}/dense_meta.json", "w"))
    log("done")


if __name__ == "__main__":
    main()
