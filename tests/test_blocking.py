"""Blocking: content duplicates, the per-source cut, stop-keys and country partitions."""
import pandas as pd

from entity_resolution.blocking import build_text_index, generate_candidates
from entity_resolution.config import BlockingConfig, Config
from entity_resolution.normalize import normalize_records
from entity_resolution.pipeline import prepare
from entity_resolution.synthetic import generate


def records(rows):
    df = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])
    df["source"] = df["entity_id"].str[1].astype(int)
    return normalize_records(df)


def block(rows, **overrides):
    cfg = BlockingConfig(**overrides)
    recs = records(rows)
    return generate_candidates(recs, build_text_index(recs, cfg, for_features=False), cfg)


def test_content_duplicates_are_all_listed():
    cands = block([
        ("S1-1", "Acme Robotics Inc", "500 Market St, San Jose", "US"),
        ("S2-1", "ACME ROBOTICS INC.", "500 Market St, San Jose", "US"),
        ("S2-2", "Acme Robotics Inc", "500 Market St., San Jose", "US"),
        ("S2-3", "Blue Ridge Bakery", "12 Oak Ave, Austin", "US"),
    ])
    dup = cands[cands["cand_id"].isin(["S2-1", "S2-2"])]
    assert len(dup) == 2
    assert (dup["blk_dup"] == 2).all()
    assert dup["blk_score"].nunique() == 1  # one representative, one score


def test_cut_counts_representatives_per_source():
    rows = [("S1-1", "Sai Traders", "Station Road, Pune", "India")]
    rows += [(f"S2-{i}", f"Sai Traders {i}", f"{i} Station Road, Pune", "India") for i in range(8)]
    rows += [(f"S3-{i}", f"Sai Traders {i}", f"{i} MG Road, Pune", "India") for i in range(8)]
    cands = block(rows, max_per_source=3, reverse_k=0)
    assert cands.groupby("cand_src").size().to_dict() == {2: 3, 3: 3}
    assert sorted(cands["blk_rank"].unique()) == [0, 1, 2]
    assert len(block(rows, max_per_source=0)) == 16


def test_reverse_keep_is_bounded_by_forward_rank():
    rows = [("S1-1", "Sai Traders", "Station Road, Pune", "India")]
    rows += [(f"S2-{i}", f"Sai Traders {i}", f"{i} Station Road, Pune", "India") for i in range(12)]
    cands = block(rows, max_per_source=2, reverse_k=1)
    assert len(cands) == 6  # every target's best S1 is S1-1, but only its top 3 x 2 survive


def test_name_crowd_ranked_by_address():
    rows = [("S1-1", "Zenith Plumbing", "14 Pine St, Reno", "US")]
    rows += [(f"S2-{i}", "Zenith Plumbing", f"{i} Elm St, Boise", "US") for i in range(2, 6)]
    rows += [("S2-1", "Zenith Plumbing", "14 Pine Street, Reno", "US")]
    crowd_only = dict(full_char_k=0, addr_word_k=0, score_weights={}, reverse_k=0, max_key_block=0)
    cands = block(rows, crowd_k=1, **crowd_only)
    assert cands["cand_id"].tolist() == ["S2-1"]
    assert cands["blk_mask"].tolist() == [64]
    assert block(rows, crowd_k=10, max_crowd=3, **crowd_only).empty  # crowd of 5 > 3: skipped


def test_exact_address_key():
    rows = [("S1-1", "Zenith Plumbing", "1400 Pine Street, Reno", "US"),
            ("S2-1", "ZP Holdings", "1400 Pine Street, Reno", "US")]
    cands = block(rows, full_char_k=0, addr_word_k=0, crowd_k=0, score_weights={})
    assert cands["blk_mask"].tolist() == [16]


def test_no_cross_country_candidates(tmp_path):
    data = generate(tmp_path / "data", n_train=200, n_test=50, seed=5)
    prep = prepare(Config(data_dir=str(data)), "train", with_features=False)
    country = prep.recs.set_index("entity_id")["country"]
    cands = prep.cands
    assert len(cands) > 0
    assert (country.loc[cands["s1_id"]].to_numpy() == country.loc[cands["cand_id"]].to_numpy()).all()


def test_train_s1_sample_keeps_every_target(tmp_path):
    data = generate(tmp_path / "data", n_train=400, n_test=50, seed=6)
    cfg = Config(data_dir=str(data))
    cfg.blocking.train_s1_sample = 0.25
    prep = prepare(cfg, "train", with_features=False)
    src = prep.recs["source"]
    assert 50 < (src == 1).sum() < 150
    assert set(prep.data.ground_truth) == set(prep.data.s1_ids)
    assert set(prep.cands["s1_id"]) <= set(prep.data.s1_ids)
    full = prepare(Config(data_dir=str(data)), "train", with_features=False)
    assert (src != 1).sum() == (full.recs["source"] != 1).sum()


def test_name_token_and_city_key():
    rows = [("S1-1", "Zenith Plumbing", "14 Pine St, Reno, NV", "US"),
            ("S2-1", "Zenith Heating and Air", "Reno, NV", "US"),
            ("S2-2", "Zenith Plumbing", "8 Oak Ave, Boise, ID", "US")]
    cands = block(rows, full_char_k=0, addr_word_k=0, crowd_k=0, score_weights={})
    assert cands["cand_id"].tolist() == ["S2-1"]
    assert cands["blk_mask"].tolist() == [16]


def test_blank_address_targets_found_by_name():
    rows = [("S1-1", "Ei Network Private Limited", "Flat 3, Venu Apartments, Model Colony, Pune", "India"),
            ("S2-1", "Ei Network Pirvdcae Limited", "", "India"),
            ("S2-2", "Orbit Traders", "", "India")]
    cands = block(rows, full_char_k=0, addr_word_k=0, crowd_k=0, score_weights={}, max_key_block=0,
                  blank_name_k=5)
    assert cands["cand_id"].tolist() == ["S2-1"]
    assert cands["blk_mask"].tolist() == [128]


def test_house_number_and_city_key():
    rows = [("S1-1", "Lotus Foods Private Limited", "1008, Promenade 3, Lbs Road, Mumbai, Maharashtra", "India"),
            ("S2-1", "लोटस फूड्स प्राइवेट लिमिटेड", "1008, Mumbai, MH", "India"),
            ("S2-2", "Lotus Foods Private Limited", "1008, Pune, MH", "India")]
    cands = block(rows, full_char_k=0, addr_word_k=0, crowd_k=0, score_weights={}, blank_name_k=0)
    assert cands["cand_id"].tolist() == ["S2-1"]


def test_indic_script_targets_matched_on_skeleton():
    rows = [("S1-1", "Galaxy Consultancy Private Limited", "E-29, Nehru Ground, Faridabad, Haryana", "India"),
            ("S2-1", "गैलेक्सी कंसल्टेंसी प्राइवेट लिमिटेड", "E-29, Faridabad, HR", "India"),
            ("S2-2", "Galaxy Consultancy Private Limited", "E-29, Faridabad, HR", "India")]
    cands = block(rows, full_char_k=0, addr_word_k=0, crowd_k=0, score_weights={}, max_key_block=0,
                  blank_name_k=0, script_k=5)
    assert cands["cand_id"].tolist() == ["S2-1"]  # the Latin-script twin is left to full_char
    assert cands["blk_mask"].tolist() == [256]
