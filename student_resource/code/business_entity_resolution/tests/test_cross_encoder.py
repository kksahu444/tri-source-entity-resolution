import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.blocking.union import address_only_channel  # noqa: E402


def _rand_csr(n, d, seed):
    x = sp.random(n, d, density=0.2, format="csr", random_state=seed, dtype=np.float32)
    x = sp.csr_matrix(x.multiply(1 / np.sqrt(x.multiply(x).sum(1) + 1e-9)))
    x.sort_indices()
    return x.astype(np.float32)


def test_address_only_channel():
    qa = _rand_csr(20, 40, 0)
    ra = sp.vstack([_rand_csr(30, 40, 1), qa[[3, 11]]], format="csr")      # records 30, 31 copy S1 3 and 11's address
    ra.sort_indices()
    qi, ri = address_only_channel(qa, ra, k=2, min_cos=0.99, generate=10, cap=10**6, chunk=7)
    assert {(3, 30), (11, 31)} <= set(zip(qi.tolist(), ri.tolist()))
    assert all((qi == q).sum() <= 2 for q in np.unique(qi))
    # a feature shared by every record is dropped from generation (df > cap), exact scores still decide
    qi2, ri2 = address_only_channel(qa, ra, k=2, min_cos=0.99, generate=10, cap=1, chunk=7)
    assert set(zip(qi2.tolist(), ri2.tolist())) <= set(zip(qi.tolist(), ri.tolist()))


def test_ce_feature_join(tmp_path, monkeypatch):
    from src import stack_b
    lg = pd.DataFrame({"qi": [0, 2, 5], "ri": [7, 1, 3], "logit": [1.5, -2.0, 0.25]})
    monkeypatch.setattr(stack_b, "ce_logits", lambda cfg, split, c: lg)
    qi = np.array([2, 0, 0, 5, 9])
    ri = np.array([1, 7, 8, 3, 3])
    out = stack_b.ce_feature({}, "dev", "US", qi, ri)
    assert np.allclose(out[[0, 1, 3]], [-2.0, 1.5, 0.25]) and np.isnan(out[[2, 4]]).all()


@pytest.mark.skipif(os.environ.get("ER_SKIP_CE_SMOKE") == "1", reason="needs transformers and a model download")
def test_ce_stage_smoke(tmp_path, monkeypatch):
    """Dev: export -> precision/length -> bake-off -> folds 1-2 -> OOF band logits; test: fold-1/2 average.
    Runs on the CPU with a tiny random model (CE_MODELS), in worker subprocesses like on Kaggle."""
    pytest.importorskip("transformers")
    tiny = '{"a": "hf-internal-testing/tiny-random-bert", "b": "hf-internal-testing/tiny-random-bert"}'
    monkeypatch.setenv("CE_MODELS", tiny)
    import importlib

    from src import cross_encoder
    importlib.reload(cross_encoder)
    art = tmp_path / "art"
    (art / "block").mkdir(parents=True)
    (art / "pred").mkdir()
    (tmp_path / "runs").mkdir()
    rng = np.random.default_rng(0)
    for split, n_q, n_r in (("dev", 60, 80), ("test", 40, 50)):
        q = pd.DataFrame({"name_a": [f"acme {i}" for i in range(n_q)], "addr_a": [f"{i} main st" for i in range(n_q)],
                          "core": [f"acme {i % 7}" for i in range(n_q)]})
        r = pd.DataFrame({"name_a": [f"acme {i % n_q}" for i in range(n_r)], "addr_a": [f"{i} main st" for i in range(n_r)]})
        q.to_parquet(art / "block" / f"{split}_US_q.parquet")
        r.to_parquet(art / "block" / f"{split}_US_r.parquet")
        pd.DataFrame({"qi": [0]}).to_parquet(art / "block" / f"{split}_US_pairs.parquet")
        qi = np.repeat(np.arange(n_q), 4)
        ri = (qi + np.tile([0, 3, 17, 29], n_q)) % n_r                    # unique (qi, ri) pairs
        p = rng.random(len(qi))
        pd.DataFrame({"qi": qi, "ri": ri, "y": (ri % n_q == qi).astype(int), "p": p}).to_parquet(
            art / "pred" / f"{split}_US_stageA.parquet")
    cfg = {"seed": 1, "paths": {"artifacts": str(art), "runs": str(tmp_path / "runs")},
           "cross_encoder": {"enabled": True, "band_lo": 0.1, "band_hi": 0.9, "confident_share": 0.5}}
    res = cross_encoder.run(cfg, "dev")
    assert "FAILED" not in res["notes"], open(art / "ce" / "FAILED").read() if (art / "ce" / "FAILED").exists() else ""
    dev = cross_encoder.ce_logits(cfg, "dev", "US")
    stage_a = pd.read_parquet(art / "pred" / "dev_US_stageA.parquet")
    assert len(dev) == int(((stage_a.p >= 0.1) & (stage_a.p <= 0.9)).sum()) and dev["logit"].notna().all()
    assert not any(n.endswith("_f0_model") for n in os.listdir(art / "ce"))
    # test pass is skipped unless decide chose the cross-encoder variant
    assert "skipped" in cross_encoder.run(cfg, "test")["notes"]
    pickle.dump({"stack_variant": "ce"}, open(art / "decision.pkl", "wb"))
    res = cross_encoder.run(cfg, "test")
    assert "FAILED" not in res["notes"]
    te = cross_encoder.ce_logits(cfg, "test", "US")
    tp = pd.read_parquet(art / "pred" / "test_US_stageA.parquet")
    assert len(te) == int(((tp.p >= 0.1) & (tp.p <= 0.9)).sum()) and np.isfinite(te["logit"]).all()


def _write_dev(tmp, n_s1=1500, seed=0):
    """Synthetic dev run: every true record retrieved, 5 unique negatives per S1; the "ce" variant's
    probabilities are sharper than the base variant's."""
    rng = np.random.default_rng(seed)
    for d in ("block", "feat", "pred"):
        os.makedirs(f"{tmp}/{d}", exist_ok=True)
    n_true = rng.choice([0, 1, 2, 3], n_s1, p=[0.3, 0.4, 0.2, 0.1])
    qi = np.concatenate([np.repeat(np.arange(n_s1), n_true), np.repeat(np.arange(n_s1), 5)])
    y = np.r_[np.ones(n_true.sum(), int), np.zeros(5 * n_s1, int)]
    ri = np.arange(len(qi))
    base = np.clip(0.3 * y + 0.7 * rng.random(len(y)), 0, 1)
    sharp = np.clip(0.8 * y + 0.2 * rng.random(len(y)), 0, 1)
    pd.DataFrame({"entity_id": [f"S{i}" for i in range(n_s1)]}).to_parquet(f"{tmp}/block/dev_US_q.parquet")
    pd.DataFrame({"qi": [0]}).to_parquet(f"{tmp}/block/dev_US_pairs.parquet")
    pd.DataFrame({"src3": rng.integers(0, 2, len(qi)), "addr_r_empty": rng.integers(0, 2, len(qi)),
                  "n_parts_q": rng.integers(1, 4, len(qi)), "q_name_city_crowd": rng.integers(1, 5, len(qi))}
                 ).to_parquet(f"{tmp}/feat/dev_US.parquet")
    for name, p in (("dev_US.parquet", base), ("dev_US_base.parquet", base), ("dev_US_ce.parquet", sharp)):
        pd.DataFrame({"qi": qi, "ri": ri, "y": y, "p": p}).to_parquet(f"{tmp}/pred/{name}")
    uni = pd.DataFrame({"entity_id": [f"S{i}" for i in range(n_s1)], "n_true": n_true, "fold": np.arange(n_s1) % 5})
    return uni, sharp


def test_decide_picks_stack_variant(tmp_path, monkeypatch):
    from src import decide
    uni, sharp = _write_dev(str(tmp_path))
    monkeypatch.setattr(decide, "build_dev_universe", lambda cfg: uni)
    cfg = {"seed": 1, "paths": {"artifacts": str(tmp_path), "runs": str(tmp_path)}, "flags": {"HARD_ASSIGNMENT": True}}
    res = decide.run(cfg, "dev")
    dec = pickle.load(open(tmp_path / "decision.pkl", "rb"))
    assert dec["stack_variant"] == "ce", res["notes"]
    assert np.allclose(pd.read_parquet(tmp_path / "pred" / "dev_US.parquet")["p"], sharp)
    notes = __import__("json").loads(res["notes"])
    assert notes["gain_vs_base"]["ci"][0] > 0 and set(notes["variants"]) == {"base", "ce"}
    # without the cross-encoder predictions only the base variant is fitted
    os.remove(tmp_path / "pred" / "dev_US_ce.parquet")
    decide.run(cfg, "dev")
    assert pickle.load(open(tmp_path / "decision.pkl", "rb"))["stack_variant"] == "base"


def test_stackb_test_uses_ce_variant(tmp_path):
    import lightgbm as lgb

    from src import stack_b
    tmp = str(tmp_path)
    for d in ("block", "feat", "pred", "ce"):
        os.makedirs(f"{tmp}/{d}")
    n = 120
    rng = np.random.default_rng(0)
    pd.DataFrame({"qi": [0]}).to_parquet(f"{tmp}/block/test_US_pairs.parquet")
    pd.DataFrame({"core": [f"name {i % 7}" for i in range(n)], "addr_core": [f"{i % 5} road" for i in range(n)],
                  "house": [str(i % 5) for i in range(n)]}).to_parquet(f"{tmp}/block/test_US_r.parquet")
    df = pd.DataFrame({"qi": np.arange(n) // 6, "ri": np.arange(n), "a": rng.random(n), "c1_score": rng.random(n)})
    df.to_parquet(f"{tmp}/feat/test_US.parquet")
    pa = rng.random(n)
    pd.DataFrame({"qi": df.qi, "ri": df.ri, "p": pa}).to_parquet(f"{tmp}/pred/test_US_stageA.parquet")
    band = (pa > 0.2) & (pa < 0.8)
    pd.DataFrame({"country": "US", "qi": df.qi[band], "ri": df.ri[band], "logit": 3 * (pa[band] - 0.5)}).to_parquet(
        f"{tmp}/ce/test_logits.parquet")
    keys = stack_b.record_keys(pd.read_parquet(f"{tmp}/block/test_US_r.parquet"))
    ctx = stack_b.context_features(df.qi.to_numpy(), df.ri.to_numpy(), pa, keys)
    cols = ["a", "c1_score"] + list(ctx.columns) + [stack_b.CE_COL]
    X = np.hstack([df[["a", "c1_score"]].to_numpy(np.float32), ctx.to_numpy(np.float32),
                   np.where(band, 3 * (pa - 0.5), np.nan)[:, None].astype(np.float32)])
    y = (pa > 0.5).astype(int)
    ms = [lgb.train({"objective": "binary", "verbose": -1, "min_data_in_leaf": 5}, lgb.Dataset(X, y), 5) for _ in range(5)]
    pickle.dump({"models": ms, "cols": cols, "full_model": None}, open(f"{tmp}/feat/stageB_ce.pkl", "wb"))
    pickle.dump({"stack_variant": "ce"}, open(f"{tmp}/decision.pkl", "wb"))
    cfg = {"seed": 1, "paths": {"artifacts": tmp, "runs": tmp}, "features": {"consistency": False},
           "cross_encoder": {"enabled": True}, "stack_b_test_context": "average"}
    res = stack_b.run(cfg, "test")
    notes = __import__("json").loads(res["notes"])
    assert notes["variant"] == "ce" and abs(notes["ce_band_share_test_US"] - band.mean()) < 1e-3
    assert pd.read_parquet(f"{tmp}/pred/test_US.parquet")["p"].between(0, 1).all()


def test_stackb_dev_trains_both_variants(tmp_path):
    from src import stack_b
    tmp = str(tmp_path)
    for d in ("block", "feat", "pred", "ce"):
        os.makedirs(f"{tmp}/{d}")
    n = 3000
    rng = np.random.default_rng(1)
    qi, ri = np.arange(n) // 6, np.arange(n)
    y = (rng.random(n) < 0.3).astype(int)
    pa = np.clip(0.4 * y + 0.6 * rng.random(n), 0, 1)
    pd.DataFrame({"qi": [0]}).to_parquet(f"{tmp}/block/dev_US_pairs.parquet")
    pd.DataFrame({"core": [f"name {i % 7}" for i in range(n)], "addr_core": [f"{i % 5} road" for i in range(n)],
                  "house": [str(i % 5) for i in range(n)]}).to_parquet(f"{tmp}/block/dev_US_r.parquet")
    pd.DataFrame({"qi": qi, "ri": ri, "y": y, "fold": qi % 5, "a": rng.random(n), "c1_score": rng.random(n)}
                 ).to_parquet(f"{tmp}/feat/dev_US.parquet")
    pd.DataFrame({"qi": qi, "ri": ri, "y": y, "p": pa}).to_parquet(f"{tmp}/pred/dev_US_stageA.parquet")
    band = (pa > 0.1) & (pa < 0.9)
    pd.DataFrame({"country": "US", "qi": qi[band], "ri": ri[band], "logit": 4 * y[band] - 2 + rng.normal(0, 1, band.sum())}
                 ).to_parquet(f"{tmp}/ce/dev_logits.parquet")
    cfg = {"seed": 1, "paths": {"artifacts": tmp, "runs": tmp}, "features": {"consistency": False},
           "cross_encoder": {"enabled": True}}
    res = stack_b.run(cfg, "dev")
    notes = __import__("json").loads(res["notes"])
    assert notes["variants"] == ["base", "ce"] and stack_b.CE_COL in notes["top_features_ce"]
    assert os.path.exists(f"{tmp}/feat/stageB.pkl") and os.path.exists(f"{tmp}/feat/stageB_ce.pkl")
    b, c = (pd.read_parquet(f"{tmp}/pred/dev_US_{v}.parquet") for v in ("base", "ce"))
    assert np.allclose(pd.read_parquet(f"{tmp}/pred/dev_US.parquet")["p"], b["p"])
    ll = lambda p: -np.mean(y * np.log(p + 1e-6) + (1 - y) * np.log(1 - p + 1e-6))
    assert ll(c["p"].to_numpy()) < ll(b["p"].to_numpy())                  # the informative logit helps
    # cross-encoder off -> one variant
    cfg["cross_encoder"]["enabled"] = False
    assert __import__("json").loads(stack_b.run(cfg, "dev")["notes"])["variants"] == ["base"]
