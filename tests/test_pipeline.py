"""Unit tests for the defense pipeline (Detection / oracle → decision)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.attacks.oracle import OracleAssertion  # noqa: E402
from counterusv.defense.consistency import ConsistencyScorer  # noqa: E402
from counterusv.defense.pipeline import (  # noqa: E402
    ASSOCIATION_ASSUMPTION,
    DefensePipeline,
    PipelineConfig,
    decide_action,
    load_pipeline_config,
)
from counterusv.kinematics.behavior_model import (  # noqa: E402
    EnvelopeModel,
    FittedSubspace,
    MultiHorizonEnvelope,
)
from counterusv.models.detector import Detection  # noqa: E402

CORE = [
    "sog_med", "sog_p95", "sog_std",
    "loiter_frac", "straightness", "accel_mean_abs",
]


def _toy_envelope(name: str, window_s: int, center: np.ndarray) -> EnvelopeModel:
    rng = np.random.default_rng(abs(hash(name)) % (2**31))
    X = center + rng.normal(0, 0.25, size=(80, 6))
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    gmm = GaussianMixture(n_components=1, random_state=0).fit(Xs)
    scores = -gmm.score_samples(Xs)
    fit = FittedSubspace(
        subspace="core",
        feature_names=list(CORE),
        scaler=scaler,
        model_name="gmm",
        model=gmm,
        n_components=1,
        val_loglik=float(gmm.score(Xs)),
        n_train=len(X),
        n_val=len(X),
        thresholds={"far_0.05": float(np.percentile(scores, 95))},
    )
    return EnvelopeModel(
        name=name,
        members={"canonical_class": [name]},
        subspaces={"core": {"gmm": fit}},
        core_features=list(CORE),
        course_features=[],
        window_s=float(window_s),
        primary="gmm",
    )


def _toy_pipeline() -> DefensePipeline:
    rec_center = np.array([6.0, 9.0, 1.5, 0.1, 0.85, 0.05])
    fish_center = np.array([2.0, 4.0, 1.0, 0.5, 0.4, 0.05])
    envelopes = {
        "recreational": MultiHorizonEnvelope(
            name="recreational",
            members={"canonical_class": ["recreational"]},
            primary_window_s=300,
            core_features=list(CORE),
            course_features=[],
            horizons={300: _toy_envelope("recreational", 300, rec_center)},
            primary="gmm",
        ),
        "fishing": MultiHorizonEnvelope(
            name="fishing",
            members={"canonical_class": ["fishing"]},
            primary_window_s=300,
            core_features=list(CORE),
            course_features=[],
            horizons={300: _toy_envelope("fishing", 300, fish_center)},
            primary="gmm",
        ),
    }
    eo_map = {
        "recreational": {"policy": "score", "envelope": "recreational"},
        "fishing": {"policy": "score", "envelope": "fishing"},
        "usv": {"policy": "abstain", "reason": "hostile"},
    }
    scorer = ConsistencyScorer(envelopes, eo_map, far_target=0.05)
    return DefensePipeline(scorer, load_pipeline_config())


def _benign_feats(**kw) -> dict:
    base = {
        "sog_med": 6.0,
        "sog_p95": 9.0,
        "sog_std": 1.5,
        "loiter_frac": 0.1,
        "straightness": 0.85,
        "accel_mean_abs": 0.05,
    }
    base.update(kw)
    return base


def test_load_pipeline_config():
    cfg = load_pipeline_config()
    assert cfg.far_target == pytest.approx(0.05)
    assert cfg.feature_arm == "kinematics_only"
    assert cfg.association_note
    assert "given" in ASSOCIATION_ASSUMPTION.lower()


def test_pass_benign_recreational():
    pipe = _toy_pipeline()
    d = pipe.evaluate("recreational", _benign_feats())
    assert d.action == "pass"
    assert d.consistency.status == "scored"
    assert d.is_flagged is False
    assert decide_action(d.consistency) == "pass"


def test_flag_high_sog_as_fishing():
    pipe = _toy_pipeline()
    attack = _benign_feats(
        sog_med=35.0, sog_p95=36.0, sog_std=1.0,
        loiter_frac=0.0, straightness=0.99, accel_mean_abs=0.2,
    )
    d = pipe.evaluate(
        "fishing",
        attack,
        purpose="eval",
        track_meta={"role": "hostile", "source": "synth"},
    )
    assert d.action == "flag"
    assert d.consistency.is_inconsistent is True


def test_detection_and_oracle_paths():
    pipe = _toy_pipeline()
    feats = _benign_feats()
    det = Detection(
        box_xyxy=(0.0, 0.0, 10.0, 10.0),
        score=0.9,
        class_name="recreational",
        class_id=2,
        role="benign",
    )
    d1 = pipe.evaluate(det, feats)
    assert d1.source == "detection"
    assert d1.action == "pass"

    oracle = OracleAssertion(
        contact_id=42,
        true_class="usv",
        asserted_class="recreational",
        condition="perfect_disguise_oracle",
        score=1.0,
        role="benign",
        box_xyxy=(0.0, 0.0, 10.0, 10.0),
        class_id=2,
    )
    d2 = pipe.evaluate(oracle, feats, purpose="eval")
    assert d2.source == "oracle"
    assert d2.contact_id == 42
    assert d2.true_class == "usv"
    assert d2.oracle_condition == "perfect_disguise_oracle"
    assert d2.action == "pass"
    assert "given" in d2.association.lower() or "out of scope" in d2.association.lower()


def test_abstain_on_usv_class():
    pipe = _toy_pipeline()
    d = pipe.evaluate("usv", _benign_feats())
    assert d.action == "abstain"
    assert d.consistency.status == "abstain"


def test_kinematics_geometry_arm_requires_freeze():
    pipe = _toy_pipeline()
    cfg = pipe.config
    missing = PipelineConfig(
        version=cfg.version,
        frozen=cfg.frozen,
        far_target=cfg.far_target,
        feature_arm="kinematics_geometry",
        freeze_path=REPO_ROOT / "results" / "behavior_model_geometry" / "DOES_NOT_EXIST.json",
        verify_digests=False,
        primary_model=cfg.primary_model,
        association_note=cfg.association_note,
        path=cfg.path,
    )
    with pytest.raises(FileNotFoundError, match="kinematics_geometry freeze"):
        DefensePipeline(pipe.scorer, missing)


def test_kinematics_geometry_arm_loads_when_freeze_present():
    freeze = REPO_ROOT / "results" / "behavior_model_geometry" / "FROZEN.json"
    if not freeze.is_file():
        pytest.skip("geometry freeze not present (run fit_geometry_model.py)")
    cfg = load_pipeline_config()
    geo = PipelineConfig(
        version=cfg.version,
        frozen=cfg.frozen,
        far_target=cfg.far_target,
        feature_arm="kinematics_geometry",
        freeze_path=freeze,
        verify_digests=False,
        primary_model=cfg.primary_model,
        association_note=cfg.association_note,
        path=cfg.path,
    )
    pipe = DefensePipeline(_toy_pipeline().scorer, geo)
    assert pipe.config.feature_arm == "kinematics_geometry"


def test_arm_freeze_path_from_pipeline_yaml():
    cfg = load_pipeline_config()
    assert "behavior_model" in str(cfg.freeze_path)
    # Default arm is kinematics_only → kinematics freeze.
    assert cfg.feature_arm == "kinematics_only"
    assert cfg.freeze_path.name == "FROZEN.json"
