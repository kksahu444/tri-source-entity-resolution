"""Pipeline configuration.

Every tunable number lives here so experiments are reproducible: the resolved config is
saved next to the trained model in ``artifacts/`` and reloaded at prediction time.
Override any value with a JSON file (see ``configs/default.json``) via ``--config``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class BlockingConfig:
    """Candidate generation. All ``*_k`` values are neighbours per Source-1 record per target source."""

    name_char_k: int = 15  # char 2-4gram TF-IDF on the legal-suffix-free name
    name_word_k: int = 10  # word + phonetic-skeleton TF-IDF on the name (transliterations)
    full_char_k: int = 15  # char 3-4gram TF-IDF on name + address
    addr_word_k: int = 8  # word 1-2gram TF-IDF on address (same address, different trade name)
    min_sim: float = 0.05  # neighbours below this cosine are discarded
    max_key_block: int = 30  # exact-key blocks larger than this are treated as stop-keys
    max_candidates: int = 0  # optional cap per Source-1 record after the union (0 = no cap)
    use_embeddings: bool = False  # dense blocker + feature (needs requirements-embeddings.txt)
    embed_model: str = "intfloat/multilingual-e5-small"  # MIT licence, ~118M params
    embed_k: int = 15


@dataclass
class ModelConfig:
    """Pairwise matcher (LightGBM, grouped K-fold by Source-1 entity)."""

    n_folds: int = 5
    seed: int = 42
    num_boost_round: int = 3000
    early_stopping_rounds: int = 100
    stack: bool = True  # second stage on group-relative stage-1 scores; kept only if OOF improves
    lgb_params: dict = field(
        default_factory=lambda: {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 40,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbose": -1,
        }
    )


@dataclass
class DecisionConfig:
    """How pair probabilities become per-entity match lists. Tuned on out-of-fold scores when ``tune``."""

    strategy: str = "threshold"  # "threshold" | "expected_f"
    threshold: float = 0.5  # probability cut (or probability floor for expected_f)
    exclusive: bool = True  # each S2/S3 record is assigned to at most one S1 entity
    tune: bool = True


@dataclass
class Config:
    data_dir: str = "dataset"
    artifacts_dir: str = "artifacts"
    output_dir: str = "output"
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Build a Config from a (possibly partial) nested dict; unknown keys raise."""
        sections = {"blocking": BlockingConfig, "model": ModelConfig, "decision": DecisionConfig}
        kwargs = {}
        for key, value in data.items():
            if key in sections:
                valid = {f.name for f in fields(sections[key])}
                unknown = set(value) - valid
                if unknown:
                    raise KeyError(f"unknown {key} config keys: {sorted(unknown)}")
                kwargs[key] = sections[key](**value)
            elif key in {f.name for f in fields(cls)}:
                kwargs[key] = value
            else:
                raise KeyError(f"unknown config key: {key}")
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
