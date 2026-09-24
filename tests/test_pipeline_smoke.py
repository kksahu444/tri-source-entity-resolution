"""End-to-end run on a small synthetic dataset: train -> predict -> valid submission files."""
from entity_resolution.config import Config
from entity_resolution.pipeline import predict, train
from entity_resolution.synthetic import generate
from entity_resolution.validate import validate_outputs


def test_end_to_end(tmp_path):
    data = generate(tmp_path / "data", n_train=300, n_test=200, seed=1)
    cfg = Config(data_dir=str(data), artifacts_dir=str(tmp_path / "art"), output_dir=str(tmp_path / "out"))
    cfg.model.n_folds = 3
    report = train(cfg)
    assert report["blocking"]["pair_recall"] > 0.9
    assert report["decision"]["oof_macro_f"] > 0.8

    summary = predict(cfg)
    assert summary["validation_errors"] == []
    errors, warnings = validate_outputs(tmp_path / "out" / "matching_results.tsv",
                                        tmp_path / "out" / "candidate_pairs.tsv", data / "test")
    assert errors == [] and warnings == []
