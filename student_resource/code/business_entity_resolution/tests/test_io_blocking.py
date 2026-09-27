import os
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import io_utils  # noqa: E402
from src.blocking.crowd import crowd_channel, key_channel  # noqa: E402
from src.blocking.tfidf import pair_cos, topk  # noqa: E402


def test_read_tsv_quoted_names(tmp_path):
    p = tmp_path / "s.tsv"
    p.write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\t"""ehpad Club SAS"\t1 Rue X\tFrance\n'
                 "S1-2\tA\t\tUS\n", encoding="utf-8")
    df = io_utils.read_tsv(str(p))
    assert df["business_name"].tolist() == ['"ehpad Club SAS', "A"] and df["business_address"].iloc[1] == ""


def test_write_submission_order_and_subset(tmp_path):
    m, c = str(tmp_path / "m.tsv"), str(tmp_path / "c.tsv")
    stats = io_utils.write_submission(m, c, ["S1-9", "S1-1"], {"S2-1", "S3-2", "S2-5"},
                                      np.array(["S1-1"]), np.array(["S3-2"]),
                                      np.array(["S1-1", "S1-1", "S1-1"]), np.array(["S3-2", "S2-1", "S3-2"]))
    lines = open(m, encoding="utf-8").read().split("\n")
    assert lines[0] == "source1_entity_id\tmatched_entity_ids" and lines[1] == "S1-9\t" and lines[2] == "S1-1\tS3-2"
    assert open(c, encoding="utf-8").read().split("\n")[2] == "S1-1\tS3-2,S2-1"
    assert stats["n_candidates"] == 2
    with pytest.raises(AssertionError):
        io_utils.write_submission(m, c, ["S1-1"], {"S2-1", "S3-2"}, np.array(["S1-1"]), np.array(["S2-1"]),
                                  np.array(["S1-1"]), np.array(["S3-2"]))


def _rand_csr(n, d, seed):
    x = sp.random(n, d, density=0.2, format="csr", random_state=seed, dtype=np.float32)
    x = sp.csr_matrix(x.multiply(1 / np.sqrt(x.multiply(x).sum(1) + 1e-9)))
    x.sort_indices()
    return x.astype(np.float32)


def test_topk_exact_and_pair_cos():
    q, r = _rand_csr(30, 50, 0), _rand_csr(200, 50, 1)
    qi, ri, s = topk(q, r.T.tocsr(), 5, threshold=0.0, chunk=7)
    dense = (q @ r.T).toarray()
    for i in range(30):
        got = sorted(s[qi == i].tolist(), reverse=True)
        exp = sorted(dense[i][dense[i] > 0].tolist(), reverse=True)[:5]
        assert np.allclose(got, exp, atol=1e-5)
    assert np.allclose(pair_cos(q, r, qi, ri), s, atol=1e-5)


def test_hash_tfidf_matches_sklearn():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from src.blocking.tfidf import HashTfidf
    docs = ["garza empire", "garza emprie", "bethel grade", "classic safe minerals", "classic safe",
            "modern construction", "modern constrcucon", "garza empire", "bethel grade"] * 3
    h = HashTfidf("char", min_df=2, max_df=0.9, n_jobs=1)
    xh = h.fit_transform(docs)
    xs = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, min_df=2, max_df=0.9).fit_transform(docs)
    assert np.allclose((xh @ xh.T).toarray(), (xs @ xs.T).toarray(), atol=1e-5)


def test_second_hop_channel_and_keys():
    from src.blocking.crowd import second_hop_channel
    from src.blocking.union import record_key_codes
    r = pd.DataFrame({"core": ["indo inds", "indo inds", "indo inds", "other", ""],
                      "house": ["34", "34", "19", "34", "34"], "city": ["haliyal"] * 5, "addr_core": ["x"] * 5})
    codes = record_key_codes(r, ["name+house", "name+city"])
    assert codes["name+house"][0] == codes["name+house"][1] != codes["name+house"][2] and codes["name+house"][4] == -1
    # anchor (q=7, r=0): name+house adds record 1; name+city adds records 1 and 2; nothing else
    qi, ri = second_hop_channel(np.array([7]), np.array([0]), codes, cap=20)
    assert set(zip(qi.tolist(), ri.tolist())) == {(7, 0), (7, 1), (7, 2)}
    qi, ri = second_hop_channel(np.array([7]), np.array([0]), codes, cap=2)      # name+city key has 3 records
    assert set(zip(qi.tolist(), ri.tolist())) == {(7, 0), (7, 1)}


def test_reverse_name_channel():
    from src.blocking.union import reverse_name_channel
    qn = _rand_csr(40, 30, 3)                    # 40 S1 name vectors
    rn = sp.vstack([qn[[5, 17, 33]], _rand_csr(10, 30, 4)], format="csr")   # records 0-2 copy S1 5, 17, 33
    rn.sort_indices()
    qi, ri = reverse_name_channel(np.array([0, 1, 2]), rn, qn, k=3, min_cos=0.3, chunk=2)
    got = set(zip(qi.tolist(), ri.tolist()))
    assert {(5, 0), (17, 1), (33, 2)} <= got and set(ri.tolist()) <= {0, 1, 2}
    assert all((qi[ri == j]).size <= 3 for j in (0, 1, 2))


def test_reverse_all_channel_every_record_gets_its_s1():
    from src.blocking.union import reverse_all_channel
    qn, qa = _rand_csr(40, 30, 3), _rand_csr(40, 30, 3)
    rn = sp.vstack([qn[[5, 17, 33]], _rand_csr(6, 30, 4)], format="csr")    # records 0-2 copy S1 5, 17, 33
    ra = sp.vstack([qa[[5, 17, 33]], _rand_csr(6, 30, 4)], format="csr")
    rn.sort_indices(); ra.sort_indices()
    qi, ri = reverse_all_channel(np.arange(9), qn, qa, rn, ra, k=2, generate=10, cap=10**6, chunk=4)
    q2, r2 = reverse_all_channel(np.array([1]), qn, qa, rn, ra, k=2, generate=10, cap=10**6, chunk=4)
    assert (17, 1) in set(zip(q2.tolist(), r2.tolist())) and set(r2.tolist()) == {1}
    got = set(zip(qi.tolist(), ri.tolist()))
    assert {(5, 0), (17, 1), (33, 2)} <= got
    assert np.bincount(ri, minlength=9).max() <= 2          # at most k S1s per record


def test_c9_recovers_misspelt_record_in_a_name_crowd():
    from src.blocking.union import block_country
    rng = np.random.default_rng(0)
    words = [f"w{i:03d}x" for i in range(400)]
    streets = [f"rue {words[i]} {words[i + 200]}" for i in range(150)]
    n = 150                                     # 150 S1s, one name crowd on distinct streets, plus filler
    q = pd.DataFrame({"entity_id": [f"S1-{i}" for i in range(n)], "core": ["lille club"] * 40 + [f"{words[i]} shop" for i in range(40, n)],
                      "addr_core": [f"{i + 1} {streets[i]} lille" for i in range(n)], "house": [str(i + 1) for i in range(n)],
                      "street": streets[:n], "city": ["lille"] * n, "src": np.int8(1)})
    # 5 exact copies per S1 on the S1's street make every S1's top-k full of its own street's records
    rows = [(f"{q.core[i]}", f"{i + 1} {streets[i]} lille", str(i + 1), streets[i]) for i in range(n) for _ in range(5)]
    target = 7                                  # a misspelt copy of S1 7 ("lille club", 8 rue ...)
    typo = streets[target].replace("x", "z")
    rows.append(("lille club", f"{target + 1} {typo} lille", str(target + 1), typo))
    r = pd.DataFrame({"entity_id": [f"S2-{i}" for i in range(len(rows))], "core": [x[0] for x in rows],
                      "addr_core": [x[1] for x in rows], "house": [x[2] for x in rows], "street": [x[3] for x in rows],
                      "city": "lille", "src": np.int8(2)})
    bc = {"k_tfidf": 3, "crowd_cap": 2, "key_block_cap": 1, "key_keep": 1, "query_chunk": 100, "c1_generate": 5}
    pairs, _ = block_country({"blocking": bc}, q, r)
    t = len(r) - 1
    assert not ((pairs.ri == t) & (pairs.qi == target)).any()          # lost without C9
    bc["reverse_all"] = {"enabled": True, "k": 3, "generate": 50, "min_score": 0.0}
    pairs, info = block_country({"blocking": bc}, q, r)
    assert ((pairs.ri == t) & (pairs.qi == target) & (pairs.in_c9 == 1)).any()


def test_c7_forward_pass_survives_query_competition():
    from src.blocking.union import block_country
    n = 60
    r = pd.DataFrame({"entity_id": [f"S2-{i}" for i in range(n + 2)],
                      "core": [f"vendor{i:03d} mart" for i in range(n)] + ["zylofonic traderz"] * 2,
                      "addr_core": [f"{i} lane{i % 7} pune" for i in range(n)] + ["", ""],
                      "house": [str(i) for i in range(n)] + ["", ""], "street": [f"lane{i % 7}" for i in range(n)] + ["", ""],
                      "city": ["pune"] * n + ["", ""], "src": np.int8(2)})
    # query 0 copies the records' name exactly and takes each record's single reverse slot from query 1
    q = pd.DataFrame({"entity_id": ["S1-0", "S1-1"], "core": ["zylofonic traderz", "zylophonic traders"],
                      "addr_core": ["4 mg road pune", "12 mg road pune"], "house": ["4", "12"], "street": ["mg", "mg"],
                      "city": ["pune", "pune"], "src": np.int8(1)})
    bc = {"k_tfidf": 30, "crowd_cap": 10, "key_block_cap": 200, "key_keep": 10, "query_chunk": 100, "c1_generate": 100,
          "reverse_empty": {"enabled": True, "k": 1, "min_cos": 0.3}}
    c7 = lambda p: set(zip(p.loc[p["in_c7"] == 1, "qi"], p.loc[p["in_c7"] == 1, "ri"]))
    pairs, _ = block_country({"blocking": bc}, q, r)
    assert c7(pairs) == {(0, n), (0, n + 1)}
    bc["reverse_empty"]["forward_k"] = 2
    pairs, _ = block_country({"blocking": bc}, q, r)
    assert {(1, n), (1, n + 1)} <= c7(pairs)


def test_crowd_and_key_channels():
    qk = pd.Series(["a b", "c", ""])
    rk = pd.Series(["a b"] * 3 + ["c"] * 12 + [""])
    qa = sp.identity(3, format="csr", dtype=np.float32)
    ra = sp.csr_matrix(np.ones((16, 3), np.float32) / np.sqrt(3))
    qi, ri, rs = crowd_channel(qk, rk, qa, ra, cap=10)
    assert sorted(ri[qi == 0].tolist()) == [0, 1, 2] and (qi == 1).sum() == 10 and (qi == 2).sum() == 0
    qi, ri, skipped = key_channel(qk, rk, cap=10)
    assert set(qi.tolist()) == {0} and skipped == 1
