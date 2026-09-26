"""Work order T6: cross-encoder on the uncertain band, Kaggle 2x T4.

Input dataset: dev_pairs.parquet (text_a, text_b, y, ce_fold 0-2, in_band, keys), test_pairs.parquet.
1. GPU profile (T4 expected) and precision check (fp16 vs fp32 on 512 dev pairs: finite, max |dlogit| < 0.05).
2. Bake-off on fold 0: mDeBERTa-v3-base on GPU0, XLM-R-base on GPU1 (same data, seed, budget).
   Winner = lower held-out band log-loss (the full decision-layer comparison happens locally).
3. Winner trains folds 1 and 2 (GPU0, GPU1).
4. Outputs: out-of-fold logits for every dev pair (each pair scored by the model that did not train on its
   fold), test-band logits = mean of the fold-1 and fold-2 models (never a full-data refit),
   bakeoff.csv, loss-scale log.
"""
import glob
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

MODELS = json.loads(os.environ.get("CE_MODELS", "null")) or {"mdeberta": "microsoft/mdeberta-v3-base",
                                                           "xlmr": "FacebookAI/xlm-roberta-base"}
W, TMP = os.environ.get("CE_OUT", "/kaggle/working"), os.environ.get("CE_TMP", "/kaggle/tmp")
IN = os.environ.get("CE_IN", "/kaggle/input")
SEED, LR, WD, WARM, EPOCHS, BATCH = 2026, 2e-5, 0.01, 0.06, 1, 64


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find(pat):
    return sorted(glob.glob(f"{IN}/**/{pat}", recursive=True), key=len)[0]


# ------------------------------------------------------------------ worker (one GPU)
def worker(kind, fold, out_prefix, score_test):
    import torch
    from torch.nn.functional import binary_cross_entropy_with_logits as bce
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    name = MODELS[kind]
    tok = AutoTokenizer.from_pretrained(name)
    d = pd.read_parquet(find("dev_pairs.parquet"))
    max_len = int(json.load(open(f"{TMP}/max_len.json"))["max_len"])
    tr, va = d[d.ce_fold != fold].reset_index(drop=True), d[d.ce_fold == fold].reset_index(drop=True)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * math.ceil(len(tr) / BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(WARM * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    lens = (tr.text_a.str.len() + tr.text_b.str.len()).to_numpy()

    def batches(frame, order, bs):
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = tok(frame.text_a.values[idx].tolist(), frame.text_b.values[idx].tolist(), truncation=True,
                      max_length=max_len, padding=True, return_tensors="pt")
            yield idx, {k: v.to(dev) for k, v in enc.items() if k != "token_type_ids"}

    rng = np.random.default_rng(SEED + fold)
    step, t0 = 0, time.time()
    model.train()
    for ep in range(EPOCHS):
        # length-sorted buckets of 64 batches, shuffled bucket order
        order = np.argsort(lens + rng.random(len(lens)))
        chunks = [order[i:i + BATCH * 64] for i in range(0, len(order), BATCH * 64)]
        rng.shuffle(chunks)
        for ch in chunks:
            for idx, enc in batches(tr, ch, BATCH):
                y = torch.tensor(tr.y.values[idx], dtype=torch.float32, device=dev)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logit = model(**enc).logits.squeeze(-1)
                loss = bce(logit.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                if step % 100 == 0:
                    log(f"{kind} fold{fold} step {step}/{steps} loss {loss.item():.4f} scale {scaler.get_scale():.0f} "
                        f"{step * BATCH / (time.time() - t0):.0f} pairs/s")
                    if amp and scaler.get_scale() < 1:
                        raise RuntimeError("fp16 loss scale collapsed below 1: switch to fp32")
    model.eval()

    @torch.no_grad()
    def score(frame):
        out = np.zeros(len(frame), np.float32)
        order = np.argsort((frame.text_a.str.len() + frame.text_b.str.len()).to_numpy())
        for idx, enc in batches(frame, order, 256):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out[idx] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        return out

    va_logit = score(va)
    p = 1 / (1 + np.exp(-va_logit))
    eps = 1e-6
    ll = lambda m: float(-np.mean(va.y[m] * np.log(p[m] + eps) + (1 - va.y[m]) * np.log(1 - p[m] + eps)))
    band = va.in_band.to_numpy()
    from sklearn.metrics import roc_auc_score
    res = {"kind": kind, "fold": fold, "n_train": len(tr), "n_val": len(va), "logloss_all": ll(np.ones(len(va), bool)),
           "logloss_band": ll(band), "auc_band": float(roc_auc_score(va.y[band], p[band])),
           "train_s": round(time.time() - t0)}
    pd.DataFrame({"country": va.country, "qi": va.qi, "ri": va.ri, "logit": va_logit}).to_parquet(f"{out_prefix}_oof.parquet")
    if score_test:
        te = pd.read_parquet(find("test_pairs.parquet"))
        pd.DataFrame({"country": te.country, "qi": te.qi, "ri": te.ri, "logit": score(te)}).to_parquet(f"{out_prefix}_test.parquet")
    json.dump(res, open(f"{out_prefix}.json", "w"))
    log("done", json.dumps(res))


# ------------------------------------------------------------------ orchestration
def precision_check():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    d = pd.read_parquet(find("dev_pairs.parquet")).sample(512, random_state=SEED)
    out = {}
    for kind, name in MODELS.items():
        tok = AutoTokenizer.from_pretrained(name)
        m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).cuda().eval()
        enc = tok(d.text_a.tolist(), d.text_b.tolist(), truncation=True, max_length=128, padding=True, return_tensors="pt")
        enc = {k: v.cuda() for k, v in enc.items() if k != "token_type_ids"}
        with torch.no_grad():
            l32 = torch.cat([m(**{k: v[i:i + 64] for k, v in enc.items()}).logits for i in range(0, 512, 64)]).float()
            with torch.autocast("cuda", dtype=torch.float16):
                l16 = torch.cat([m(**{k: v[i:i + 64] for k, v in enc.items()}).logits for i in range(0, 512, 64)]).float()
        ok = bool(torch.isfinite(l16).all())
        diff = float((l32 - l16).abs().max())
        out[kind] = {"finite": ok, "max_abs_logit_diff": round(diff, 5), "pass": ok and diff < 0.05}
        del m
        torch.cuda.empty_cache()
    log("precision check", json.dumps(out))
    return out


def max_length():
    from transformers import AutoTokenizer
    d = pd.read_parquet(find("dev_pairs.parquet")).sample(20000, random_state=SEED)
    lens = []
    for name in MODELS.values():
        tok = AutoTokenizer.from_pretrained(name)
        enc = tok(d.text_a.tolist(), d.text_b.tolist(), truncation=False)
        lens.append(np.percentile([len(x) for x in enc["input_ids"]], 95))
    ml = int(min(128, max(lens)))
    json.dump({"max_len": ml, "p95": [float(x) for x in lens]}, open(f"{TMP}/max_len.json", "w"))
    log("max_len", ml, lens)
    return ml


def run_pair(jobs):
    procs = []
    for gpu, (kind, fold, prefix, score_test) in enumerate(jobs):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        cmd = [sys.executable, __file__, "worker", kind, str(fold), prefix, str(int(score_test))]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"worker failed: {codes}")


def main():
    import torch
    os.makedirs(TMP, exist_ok=True)
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        log(f"GPU{i}: {p.name} {p.total_memory / 2**30:.1f} GB cc {p.major}.{p.minor}")
        assert (p.major, p.minor) == (7, 5), "expected a T4"
    pc = precision_check()
    json.dump(pc, open(f"{W}/precision_check.json", "w"))
    max_length()
    candidates = [k for k in MODELS if pc[k]["pass"]]
    assert candidates, "no backbone passed the fp16 precision check"
    # bake-off on fold 0
    run_pair([(k, 0, f"{TMP}/{k}_f0", False) for k in candidates][:2])
    res = [json.load(open(f"{TMP}/{k}_f0.json")) for k in candidates]
    pd.DataFrame(res).to_csv(f"{W}/bakeoff.csv", index=False)
    win = min(res, key=lambda r: r["logloss_band"])["kind"]
    log("bake-off winner", win, json.dumps(res))
    # winner on folds 1 and 2, each also scores the test band
    run_pair([(win, 1, f"{TMP}/{win}_f1", True), (win, 2, f"{TMP}/{win}_f2", True)])
    oof = pd.concat([pd.read_parquet(f"{TMP}/{win}_f{f}_oof.parquet") for f in range(3)], ignore_index=True)
    oof["model"] = win
    oof.to_parquet(f"{W}/ce_dev_oof.parquet", index=False)
    te = [pd.read_parquet(f"{TMP}/{win}_f{f}_test.parquet") for f in (1, 2)]
    t = te[0].copy()
    t["logit"] = (te[0]["logit"].to_numpy() + te[1]["logit"].to_numpy()) / 2
    t["model"] = win
    t.to_parquet(f"{W}/ce_test.parquet", index=False)
    json.dump({"winner": win, "folds": [json.load(open(f"{TMP}/{win}_f{f}.json")) for f in range(3)]},
              open(f"{W}/ce_summary.json", "w"), indent=1)
    log("all done")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(sys.argv[2], int(sys.argv[3]), sys.argv[4], bool(int(sys.argv[5])))
    else:
        main()
