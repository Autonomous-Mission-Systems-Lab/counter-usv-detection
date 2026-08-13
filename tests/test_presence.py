"""Unit tests for the presence-only cross-check and shared defense harness."""

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
from counterusv.defense.harness import (  # noqa: E402
    DefenseBackend,
    evaluate_contact,
)
from counterusv.defense.pipeline import DefensePipeline, load_pipeline_config  # noqa: E402
from counterusv.defense.presence import (  # noqa: E402
    PresenceObservation,
    PresenceOnlyDefense,
    decide_presence,
    load_presence_config,
    presence_for_disguise,
    presence_for_evasion,
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


def test_load_presence_config():
    cfg = load_presence_config()
    assert "missed" in cfg.flag_outcomes
    assert "mislabeled" in cfg.flag_outcomes
    assert "detected" in cfg.pass_outcomes
    assert cfg.association_note


def test_flag_evasion_missed():
    action, result = decide_presence(
        PresenceObservation(track_present=True, eo_outcome="missed")
    )
    assert action == "flag"
    assert result.is_inconsistent is True
    assert result.status == "scored"


def test_flag_evasion_mislabeled():
    action, result = decide_presence(
        PresenceObservation(track_present=True, eo_outcome="mislabeled")
    )
    assert action == "flag"
    assert result.is_inconsistent is True


def test_pass_disguise_detected():
    action, result = decide_presence(
        PresenceObservation(
            track_present=True,
            eo_outcome="detected",
            asserted_class="fishing",
        )
    )
    assert action == "pass"
    assert result.is_inconsistent is False
    assert "cannot catch disguise" in (result.note or "")


def test_abstain_no_track():
    action, result = decide_presence(
        PresenceObservation(track_present=False, eo_outcome="missed")
    )
    assert action == "abstain"
    assert result.status == "abstain_no_track"
    assert result.is_inconsistent is None


def test_presence_defense_helpers():
    defense = PresenceOnlyDefense.from_config()
    assert defense.kind == "presence_only"

    d_ev = defense.evaluate(presence=presence_for_evasion(contact_id=1))
    assert d_ev.action == "flag"
    assert d_ev.defense_kind == "presence_only"
    assert d_ev.feature_arm is None
    assert d_ev.consistency is None
    assert d_ev.presence is not None
    assert d_ev.is_flagged is True

    oracle = OracleAssertion(
        contact_id=7,
        true_class="usv",
        asserted_class="recreational",
        condition="perfect_disguise_oracle",
        score=1.0,
        role="benign",
        box_xyxy=(0.0, 0.0, 5.0, 5.0),
        class_id=2,
    )
    d_di = defense.evaluate(oracle, purpose="eval")
    assert d_di.action == "pass"
    assert d_di.contact_id == 7
    assert d_di.true_class == "usv"
    assert d_di.source == "oracle"

    det = Detection(
        box_xyxy=(0.0, 0.0, 5.0, 5.0),
        score=0.9,
        class_name="fishing",
        class_id=1,
        role="benign",
    )
    assert presence_for_disguise(det).eo_outcome == "detected"
    assert defense.evaluate(det).action == "pass"


def test_presence_requires_input():
    defense = PresenceOnlyDefense.from_config()
    with pytest.raises(ValueError, match="requires presence"):
        defense.evaluate()


def test_harness_protocol_and_swap():
    presence = PresenceOnlyDefense.from_config()
    assert isinstance(presence, DefenseBackend)

    # Toy consistency backend (no freeze) for swap parity.
    rng = np.random.default_rng(0)
    center = np.array([6.0, 9.0, 1.5, 0.1, 0.85, 0.05])
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
    env = EnvelopeModel(
        name="recreational",
        members={"canonical_class": ["recreational"]},
        subspaces={"core": {"gmm": fit}},
        core_features=list(CORE),
        course_features=[],
        window_s=300.0,
        primary="gmm",
    )
    scorer = ConsistencyScorer(
        {
            "recreational": MultiHorizonEnvelope(
                name="recreational",
                members={"canonical_class": ["recreational"]},
                primary_window_s=300,
                core_features=list(CORE),
                course_features=[],
                horizons={300: env},
                primary="gmm",
            )
        },
        {"recreational": {"policy": "score", "envelope": "recreational"}},
        far_target=0.05,
    )
    consistency = DefensePipeline(scorer, load_pipeline_config())
    assert isinstance(consistency, DefenseBackend)
    assert consistency.kind == "consistency"

    feats = {
        "sog_med": 6.0,
        "sog_p95": 9.0,
        "sog_std": 1.5,
        "loiter_frac": 0.1,
        "straightness": 0.85,
        "accel_mean_abs": 0.05,
    }
    oracle = OracleAssertion(
        contact_id=9,
        true_class="usv",
        asserted_class="recreational",
        condition="perfect_disguise_oracle",
        score=1.0,
        role="benign",
        box_xyxy=(0.0, 0.0, 5.0, 5.0),
        class_id=2,
    )
    # Same disguise contact: presence passes (EO detected); consistency may pass
    # on benign kinematics — the gap measurement lives in detection-rate work.
    p = evaluate_contact(
        presence,
        assertion=oracle,
        presence=presence_for_disguise(oracle),
        purpose="eval",
    )
    c = evaluate_contact(
        consistency, assertion=oracle, features=feats, purpose="eval"
    )
    assert p.action == "pass"
    assert p.defense_kind == "presence_only"
    assert c.defense_kind == "consistency"
    assert c.action in ("flag", "pass", "abstain")
    assert set(p.as_dict()) >= {"action", "defense_kind", "presence", "consistency"}
