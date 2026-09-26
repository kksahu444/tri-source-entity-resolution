"""Cross-encoder on Kaggle 2x T4: fine-tune on out-of-dev train pairs, then score pair files.

Input dataset (private, attached as a data source): ce_train.parquet from `python -m src.ce_pairs`
(s1_id, cand_id, label, val, text_a, text_b), and any number of score_<name>.parquet (s1_id, cand_id)
with texts_<name>.parquet (entity_id, text): the pruned dev/test pairs from `python -m src.ce_lane_export`.
Texts are stored once per record, not per pair, and pairs are scored in chunks of CHUNK.
If the dataset holds no ce_train.parquet, the model comes from a previous version of this kernel
(ce_model/ in its output, attached as a kernel source) and only scoring runs.

Backbone: xlm-roberta-base (MIT, 278M), pair input "name | address" [SEP] "name | address".
Output in /kaggle/working: ce_<name>.parquet (s1_id, cand_id, ce_p), ce_model/, ce_report.json.
"""
import glob
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch import nn

WORK = "/kaggle/working"
BACKBONE = "FacebookAI/xlm-roberta-base"
MAX_LEN = 96
EPOCHS = 1
BATCH = 64
LR = 2e-5
SEED = 2026
CHUNK = 2_000_000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find(pattern):
    return sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))


def f05(y, p):
    order = np.argsort(-p)
    tp = np.cumsum(y[order])
    k = np.arange(1, len(y) + 1)
    prec, rec = tp / k, tp / max(y.sum(), 1)
    f = 1.25 * prec * rec / np.maximum(0.25 * prec + rec, 1e-9)
    b = int(np.argmax(f))
    return float(f[b]), float(p[order][b])


def auc(y, p):
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def train(tok, df):
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    torch.manual_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(BACKBONE, num_labels=1).cuda()
    net = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * math.ceil(len(df) / BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.BCEWithLogitsLoss()
    a, b, y = df["text_a"].tolist(), df["text_b"].tolist(), df["label"].to_numpy(np.float32)
    rng = np.random.default_rng(SEED)
    step, t0 = 0, time.time()
    net.train()
    for ep in range(EPOCHS):
        order = rng.permutation(len(df))
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            enc = tok([a[i] for i in idx], [b[i] for i in idx], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                logit = net(**enc).logits.squeeze(-1)
            loss = loss_fn(logit.float(), torch.from_numpy(y[idx]).cuda())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 500 == 0:
                log(f"step {step}/{steps} loss {loss.item():.4f} {(time.time() - t0) / step * (steps - step) / 60:.0f} min left")
    return model.eval()


@torch.no_grad()
def score_on(model, tok, a, b, device, bs=512):
    model = model.to(device)
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
    out = np.zeros(len(a), np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([a[i] for i in idx], [b[i] for i in idx], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            out[idx] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    return out


def score(model, tok, df):
    import copy
    a, b = df["text_a"].tolist(), df["text_b"].tolist()
    if torch.cuda.device_count() < 2 or len(a) < 10_000:
        return score_on(model, tok, a, b, "cuda:0")
    half = len(a) // 2
    twin = copy.deepcopy(model).to("cuda:1")
    with ThreadPoolExecutor(2) as ex:
        parts = list(ex.map(lambda m: score_on(*m),
                            [(model, tok, a[:half], b[:half], "cuda:0"), (twin, tok, a[half:], b[half:], "cuda:1")]))
    return np.concatenate(parts)


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        log(f"GPU{i}: {p.name} {p.total_memory / 2**30:.1f} GB")
    report = {"backbone": BACKBONE, "max_len": MAX_LEN, "epochs": EPOCHS, "batch": BATCH, "lr": LR}
    train_files = find("ce_train.parquet")
    if train_files:
        tok = AutoTokenizer.from_pretrained(BACKBONE)
        df = pd.read_parquet(train_files[0])
        tr, va = df[df["val"] == 0], df[df["val"] == 1]
        log(f"train {len(tr):,} pairs (pos {tr['label'].mean():.3f}), val {len(va):,}")
        model = train(tok, tr)
        model.save_pretrained(f"{WORK}/ce_model")
        tok.save_pretrained(f"{WORK}/ce_model")
        p = score(model, tok, va)
        y = va["label"].to_numpy()
        report["val"] = {"pairs": int(len(va)), "auc": auc(y, p), "best_pair_f05": f05(y, p)}
        for c in sorted(va["country"].unique()):
            m = (va["country"] == c).to_numpy()
            report["val"][c] = {"auc": auc(y[m], p[m]), "best_pair_f05": f05(y[m], p[m])}
        # the stack only sees pairs blocking found; score that slice separately
        m = (va["union_rank"] < 99).to_numpy()
        report["val"]["in_blocking"] = {"auc": auc(y[m], p[m]), "best_pair_f05": f05(y[m], p[m])}
        log("val", json.dumps(report["val"]))
        va.assign(ce_p=p)[["s1_id", "cand_id", "label", "union_rank", "country", "ce_p"]].to_parquet(
            f"{WORK}/ce_val.parquet", index=False)
    else:
        path = os.path.dirname(find("ce_model/config.json")[0])
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).cuda().eval()
        report["model_from"] = path
    for f in find("score_*.parquet"):
        name = os.path.basename(f)[len("score_"):-len(".parquet")]
        pairs = pd.read_parquet(f)
        text = pd.read_parquet(os.path.join(os.path.dirname(f), f"texts_{name}.parquet")).set_index("entity_id")["text"]
        t = time.time()
        ce_p = np.empty(len(pairs), np.float32)
        for s in range(0, len(pairs), CHUNK):
            c = pairs.iloc[s:s + CHUNK]
            ce_p[s:s + CHUNK] = score(model, tok, pd.DataFrame({"text_a": text.reindex(c["s1_id"]).to_numpy(),
                                                                "text_b": text.reindex(c["cand_id"]).to_numpy()}))
            log(name, f"{s + len(c):,}/{len(pairs):,}", f"{time.time() - t:.0f}s")
        pairs.assign(ce_p=ce_p).to_parquet(f"{WORK}/ce_{name}.parquet", index=False)
        report[f"scored_{name}"] = {"pairs": int(len(pairs)), "seconds": round(time.time() - t)}
    json.dump(report, open(f"{WORK}/ce_report.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
