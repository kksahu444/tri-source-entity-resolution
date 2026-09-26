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


def test_crowd_and_key_channels():
    qk = pd.Series(["a b", "c", ""])
    rk = pd.Series(["a b"] * 3 + ["c"] * 12 + [""])
    qa = sp.identity(3, format="csr", dtype=np.float32)
    ra = sp.csr_matrix(np.ones((16, 3), np.float32) / np.sqrt(3))
    qi, ri, rs = crowd_channel(qk, rk, qa, ra, cap=10)
    assert sorted(ri[qi == 0].tolist()) == [0, 1, 2] and (qi == 1).sum() == 10 and (qi == 2).sum() == 0
    qi, ri, skipped = key_channel(qk, rk, cap=10)
    assert set(qi.tolist()) == {0} and skipped == 1
