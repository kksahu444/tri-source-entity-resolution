"""Stage `ce` (work order T6): cross-encoder on the uncertain band, 2x T4, inside the pipeline.

Dev: band = stage-A OOF p in [lo, hi]. Training pairs = band + a small sample of confident pairs; folds =
3 groups by (country, sorted core name of the S1) so a text model cannot memorise names across folds.
Precision check (fp16 vs fp32, 512 pairs) -> bake-off on fold 0 (mDeBERTa-v3-base GPU0, XLM-R-base GPU1,
lower held-out band log-loss wins) -> winner trains folds 1 and 2. Every band pair gets an out-of-fold logit.
Test: band = stage-A test p in [lo, hi]; logit = mean of the winner's fold-1 and fold-2 models (never a
full-data refit). Only band pairs carry a logit (dev and test alike); stage B uses it as a feature.
Any failure leaves a marker and the pipeline continues without the cross-encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .blocking.union import block_dir
from .featurize import countries
from .stack import pred_dir
from .utils import Timer, get_logger

log = get_logger()
MODELS = json.loads(os.environ.get("CE_MODELS", "null")) or {"mdeberta": "microsoft/mdeberta-v3-base",
                                                           "xlmr": "FacebookAI/xlm-roberta-base"}
SEED, LR, WD, WARM, EPOCHS, BATCH = 2026, 2e-5, 0.01, 0.06, 1, 64


def ce_dir(cfg):
    d = os.path.join(cfg["paths"]["artifacts"], "ce")
    os.makedirs(d, exist_ok=True)
    return d


def ce_logits(cfg, split, c) -> pd.DataFrame | None:
    """Band logits of one country (columns qi, ri, logit), or None when the stage is off or failed."""
    d = os.path.join(cfg["paths"]["artifacts"], "ce")
    path = os.path.join(d, f"{split}_logits.parquet")
    if not (cfg.get("cross_encoder") or {}).get("enabled", False) or os.path.exists(os.path.join(d, "FAILED"))             or not os.path.exists(path):
        return None
    t = pd.read_parquet(path)
    return t[t["country"] == c].reset_index(drop=True)


def stack_variant(cfg) -> str:
    """Stage-B variant chosen by `decide` on dev ("base" unless the cross-encoder variant won)."""
    path = os.path.join(cfg["paths"]["artifacts"], "decision.pkl")
    if not os.path.exists(path):
        return "base"
    return pickle.load(open(path, "rb")).get("stack_variant", "base")


def _fold(country, cores):
    return np.array([int(hashlib.md5((country + "|" + " ".join(sorted(set(c.split())))).encode()).hexdigest()[:8], 16) % 3
                     for c in cores], np.int8)


def export(cfg, split):
    cc = cfg.get("cross_encoder") or {}
    lo, hi, conf = cc.get("band_lo", 0.005), cc.get("band_hi", 0.995), cc.get("confident_share", 0.02)
    rng = np.random.default_rng(cfg["seed"])
    frames = []
    for c in countries(cfg, split):
        d = pd.read_parquet(os.path.join(pred_dir(cfg), f"{split}_{c}_stageA.parquet"))
        p = d["p"].to_numpy()
        band = (p >= lo) & (p <= hi)
        keep = band | (rng.random(len(p)) < conf) if split == "dev" else band
        d = d[keep].reset_index(drop=True)
        qt = pq.read_table(os.path.join(block_dir(cfg), f"{split}_{c}_q.parquet"), columns=["name_a", "addr_a", "core"])
        rt = pq.read_table(os.path.join(block_dir(cfg), f"{split}_{c}_r.parquet"), columns=["name_a", "addr_a"])
        qi, ri = d["qi"].to_numpy(), d["ri"].to_numpy()
        txt = lambda t, i: [f"name: {n} | address: {a}" for n, a in zip(t["name_a"].take(pa.array(i)).to_pylist(),
                                                                       t["addr_a"].take(pa.array(i)).to_pylist())]
        out = pd.DataFrame({"country": c, "qi": qi.astype(np.int32), "ri": ri.astype(np.int32),
                            "text_a": txt(qt, qi), "text_b": txt(rt, ri), "in_band": band[keep]})
        if split == "dev":
            out["y"] = d["y"].to_numpy().astype(np.int8)
            out["ce_fold"] = _fold(c, qt["core"].take(pa.array(qi)).to_pylist())
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    path = os.path.join(ce_dir(cfg), f"{split}_pairs.parquet")
    df.to_parquet(path, index=False)
    return path, df


# ------------------------------------------------------------------ worker (one GPU per process)
def worker(kind, pairs_path, out_prefix, fold=None, model_dir=None, max_len=128):
    import torch
    from torch.nn.functional import binary_cross_entropy_with_logits as bce
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    name = model_dir or MODELS[kind]
    tok = AutoTokenizer.from_pretrained(name)
    d = pd.read_parquet(pairs_path)

    def batches(frame, order, bs):
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = tok(frame.text_a.values[idx].tolist(), frame.text_b.values[idx].tolist(), truncation=True,
                      max_length=max_len, padding=True, return_tensors="pt")
            yield idx, {k: v.to(dev) for k, v in enc.items() if k != "token_type_ids"}

    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, ignore_mismatched_sizes=True).to(dev)
    res = {"kind": kind, "fold": fold}
    if model_dir is None:                                   # train on the other folds
        tr = d[d.ce_fold != fold].reset_index(drop=True)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        steps = EPOCHS * math.ceil(len(tr) / BATCH)
        sched = get_linear_schedule_with_warmup(opt, int(WARM * steps), steps)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        lens = (tr.text_a.str.len() + tr.text_b.str.len()).to_numpy()
        rng = np.random.default_rng(SEED + fold)
        step, t0, scales = 0, time.time(), []
        model.train()
        for _ in range(EPOCHS):
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
                        scales.append(scaler.get_scale())
                        print(f"{kind} fold{fold} step {step}/{steps} loss {loss.item():.4f} scale {scales[-1]:.0f} "
                              f"{step * BATCH / (time.time() - t0):.0f} pairs/s", flush=True)
                        if amp and scales[-1] < 1:
                            raise RuntimeError("fp16 loss scale collapsed below 1")
        res.update({"n_train": len(tr), "train_s": round(time.time() - t0), "min_loss_scale": min(scales) if scales else None})
        model.save_pretrained(f"{out_prefix}_model")
        tok.save_pretrained(f"{out_prefix}_model")
        d = d[d.ce_fold == fold].reset_index(drop=True)      # score the held-out fold
    model.eval()
    out = np.zeros(len(d), np.float32)
    order = np.argsort((d.text_a.str.len() + d.text_b.str.len()).to_numpy())
    with torch.no_grad():
        for idx, enc in batches(d, order, 256):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out[idx] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
    pd.DataFrame({"country": d.country, "qi": d.qi, "ri": d.ri, "logit": out}).to_parquet(f"{out_prefix}_logits.parquet")
    if "y" in d.columns and fold is not None:
        p = 1 / (1 + np.exp(-out))
        b = d.in_band.to_numpy()
        yb = d.y.to_numpy()[b]
        res["logloss_band"] = float(-np.mean(yb * np.log(p[b] + 1e-6) + (1 - yb) * np.log(1 - p[b] + 1e-6)))
        res["n_val_band"] = int(b.sum())
    json.dump(res, open(f"{out_prefix}.json", "w"))
    print("worker done", json.dumps(res), flush=True)


def _launch(jobs):
    """Run up to one worker per GPU in parallel; jobs = list of argument lists."""
    procs = []
    for gpu, args in enumerate(jobs):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu % max(1, _n_gpus()))}
        procs.append(subprocess.Popen([sys.executable, "-u", "-m", "src.cross_encoder", "worker", *args], env=env,
                                      cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"cross-encoder worker failed: {codes}")


def _n_gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def precision_and_length(pairs: pd.DataFrame) -> tuple[dict, int]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    sample = pairs.sample(min(512, len(pairs)), random_state=SEED)
    lsample = pairs.sample(min(20000, len(pairs)), random_state=SEED)
    out, p95 = {}, []
    for kind, name in MODELS.items():
        tok = AutoTokenizer.from_pretrained(name)
        enc_all = tok(lsample.text_a.tolist(), lsample.text_b.tolist(), truncation=False)
        p95.append(float(np.percentile([len(x) for x in enc_all["input_ids"]], 95)))
        if not torch.cuda.is_available():
            out[kind] = {"pass": True, "note": "no GPU (smoke test)"}
            continue
        m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, ignore_mismatched_sizes=True).cuda().eval()
        enc = tok(sample.text_a.tolist(), sample.text_b.tolist(), truncation=True, max_length=128, padding=True, return_tensors="pt")
        enc = {k: v.cuda() for k, v in enc.items() if k != "token_type_ids"}
        with torch.no_grad():
            l32 = torch.cat([m(**{k: v[i:i + 64] for k, v in enc.items()}).logits for i in range(0, len(sample), 64)]).float()
            with torch.autocast("cuda", dtype=torch.float16):
                l16 = torch.cat([m(**{k: v[i:i + 64] for k, v in enc.items()}).logits for i in range(0, len(sample), 64)]).float()
        finite, diff = bool(torch.isfinite(l16).all()), float((l32 - l16).abs().max())
        out[kind] = {"finite": finite, "max_abs_logit_diff": round(diff, 5), "pass": finite and diff < 0.05}
        del m
        torch.cuda.empty_cache()
    return out, int(min(128, max(p95)))


def run(cfg: dict, split: str = "dev") -> dict:
    t = Timer()
    cc = cfg.get("cross_encoder") or {}
    if not cc.get("enabled", False):
        return {"stage": "ce", "notes": "disabled"}
    d = ce_dir(cfg)
    marker = os.path.join(d, "FAILED")
    if split == "dev":
        for f in ("FAILED", "summary.json", "dev_logits.parquet", "test_logits.parquet"):
            if os.path.exists(os.path.join(d, f)):
                os.remove(os.path.join(d, f))
    elif os.path.exists(marker) or not os.path.exists(os.path.join(d, "summary.json")):
        return {"stage": "ce", "notes": "skipped: no trained cross-encoder"}
    elif stack_variant(cfg) != "ce":
        return {"stage": "ce", "notes": "skipped: decide kept the stage-B variant without the cross-encoder"}
    try:
        if split == "dev":
            path, pairs = export(cfg, "dev")
            path = os.path.abspath(path)
            pc, max_len = precision_and_length(pairs)
            kinds = [k for k in MODELS if pc[k]["pass"]]
            if not kinds:
                raise RuntimeError(f"no backbone passed the precision check: {pc}")
            log.info("ce: %s dev pairs (%s in band), precision %s, max_len %d (%.0fs)", f"{len(pairs):,}",
                     f"{int(pairs.in_band.sum()):,}", json.dumps(pc), max_len, t())
            d = os.path.abspath(d)
            _launch([[k, path, os.path.join(d, f"{k}_f0"), "--fold", "0", "--max_len", str(max_len)] for k in kinds[:2]])
            bake = [json.load(open(os.path.join(d, f"{k}_f0.json"))) for k in kinds[:2]]
            for k in kinds[:2]:                                  # fold-0 logits are kept, the models are not needed
                shutil.rmtree(os.path.join(d, f"{k}_f0_model"), ignore_errors=True)
            win = min(bake, key=lambda r: r["logloss_band"])["kind"]
            log.info("ce bake-off %s -> %s (%.0fs)", json.dumps(bake), win, t())
            _launch([[win, path, os.path.join(d, f"{win}_f{f}"), "--fold", str(f), "--max_len", str(max_len)] for f in (1, 2)])
            logits = pd.concat([pd.read_parquet(os.path.join(d, f"{win}_f{f}_logits.parquet")) for f in range(3)], ignore_index=True)
            band = pairs[pairs.in_band][["country", "qi", "ri"]]
            logits = band.merge(logits, on=["country", "qi", "ri"], how="left")
            logits.to_parquet(os.path.join(d, "dev_logits.parquet"), index=False)
            summary = {"winner": win, "max_len": max_len, "precision": pc, "bakeoff": bake,
                       "folds": [json.load(open(os.path.join(d, f"{win}_f{f}.json"))) for f in range(3)],
                       "dev_pairs": len(pairs), "dev_band": int(pairs.in_band.sum())}
            json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=1)
            pd.DataFrame(bake).to_csv(os.path.join(cfg["paths"]["runs"], "bakeoff.csv"), index=False)
            return {"stage": "ce", "runtime_s": t(), "notes": json.dumps(summary)}
        summary = json.load(open(os.path.join(d, "summary.json")))
        win, max_len = summary["winner"], summary["max_len"]
        path, pairs = export(cfg, "test")
        path, d = os.path.abspath(path), os.path.abspath(d)
        _launch([[win, path, os.path.join(d, f"test_f{f}"), "--model_dir", os.path.join(d, f"{win}_f{f}_model"),
                  "--max_len", str(max_len)] for f in (1, 2)])
        a, b = (pd.read_parquet(os.path.join(d, f"test_f{f}_logits.parquet")) for f in (1, 2))
        assert (a["qi"].to_numpy() == b["qi"].to_numpy()).all() and (a["ri"].to_numpy() == b["ri"].to_numpy()).all()
        a["logit"] = (a["logit"].to_numpy() + b["logit"].to_numpy()) / 2
        a.to_parquet(os.path.join(d, "test_logits.parquet"), index=False)
        return {"stage": "ce", "runtime_s": t(), "notes": json.dumps({"test_band": len(a), "winner": win})}
    except Exception:
        open(marker, "w").write(traceback.format_exc())
        log.error("cross-encoder failed; continuing without it:\n%s", traceback.format_exc())
        return {"stage": "ce", "runtime_s": t(), "notes": "FAILED (pipeline continues without cross-encoder)"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("kind")
    ap.add_argument("pairs")
    ap.add_argument("out_prefix")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--max_len", type=int, default=128)
    a = ap.parse_args()
    worker(a.kind, a.pairs, a.out_prefix, a.fold, a.model_dir, a.max_len)
