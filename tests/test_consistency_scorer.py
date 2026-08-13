"""Unit tests for ConsistencyScorer — interface, abstain policy, firewall."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from counterusv.defense import (
    ConsistencyScorer,
    FirewallError,
    assert_benign_train_allowed,
    filter_benign_training,
)
from counterusv.kinematics.behavior_model import (
    EnvelopeModel,
    FittedSubspace,
    MultiHorizonEnvelope,
)
from counterusv.models.detector import Detection

CORE = [
    "sog_med", "sog_p95", "sog_std",
    "loiter_frac", "straightness", "accel_mean_abs",
]


def _toy_envelope(name: str = "recreational", window_s: int = 300) -> EnvelopeModel:
    """Tiny 1-component GMM on a tight cluster around typical recreational motion."""
    rng = np.random.default_rng(0)
    # [sog_med, sog_p95, sog_std, loiter, straight, accel]
    center = np.array([6.0, 9.0, 1.5, 0.1, 0.85, 0.05])
    X = center + rng.normal(0, 0.3, size=(80, 6))
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    gmm = GaussianMixture(n_components=1, random_state=0).fit(Xs)
    scores = -gmm.score_samples(Xs)
    thr05 = float(np.percentile(scores, 95))
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
        thresholds={"far_0.05": thr05, "far_0.01": float(np.percentile(scores, 99))},
    )
    return EnvelopeModel(
        name=name,
        members={"canonical_class": [name]},
        subspaces={"core": {"gmm": fit}},
        core_features=list(CORE),
        course_features=["turn_rate_mean_dps", "cog_circ_std_deg"],
        window_s=float(window_s),
        primary="gmm",
    )


def _toy_bundle() -> MultiHorizonEnvelope:
    h120 = _toy_envelope("recreational", 120)
    h300 = _toy_envelope("recreational", 300)
    return MultiHorizonEnvelope(
        name="recreational",
        members={"canonical_class": ["recreational"]},
        primary_window_s=300,
        core_features=list(CORE),
        course_features=["turn_rate_mean_dps", "cog_circ_std_deg"],
        horizons={120: h120, 300: h300},
        primary="gmm",
    )


def _toy_scorer() -> ConsistencyScorer:
    eo_map = {
        "recreational": {"policy": "score", "envelope": "recreational"},
        "small_craft": {
            "policy": "score",
            "envelope": "small_craft_class_b_proxy",
            "reason": "proxy",
        },
        "usv": {"policy": "abstain", "reason": "hostile"},
        "military": {"policy": "abstain", "reason": "hostile"},
        "benign_unspecified": {"policy": "abstain", "reason": "coarse_only"},
        "static_aid": {"policy": "abstain", "reason": "non_target"},
    }
    # small_craft maps to a missing artifact on purpose for one test — add proxy
    proxy = MultiHorizonEnvelope(
        name="small_craft_class_b_proxy",
        members={"transceiver_class": ["B"]},
        primary_window_s=300,
        core_features=list(CORE),
        course_features=["turn_rate_mean_dps", "cog_circ_std_deg"],
        horizons={300: _toy_envelope("small_craft_class_b_proxy", 300)},
        primary="gmm",
    )
    return ConsistencyScorer(
        {"recreational": _toy_bundle(), "small_craft_class_b_proxy": proxy},
        eo_map,
        far_target=0.05,
        primary_model="gmm",
    )


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


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def test_score_benign_consistent():
    scorer = _toy_scorer()
    res = scorer.score("recreational", _benign_feats())
    assert res.status == "scored"
    assert res.envelope_used == "recreational"
    assert res.score is not None
    assert res.threshold is not None
    assert res.is_inconsistent is False
    assert res.window_s == 300
    assert res.far_target == 0.05


def test_score_anomalous_flagged():
    scorer = _toy_scorer()
    # High-SOG straight dash — far from the recreational cluster.
    res = scorer.score(
        "recreational",
        _benign_feats(sog_med=40.0, sog_p95=45.0, sog_std=5.0, loiter_frac=0.0,
                      straightness=0.99, accel_mean_abs=2.0),
    )
    assert res.status == "scored"
    assert res.is_inconsistent is True
    assert res.score > res.threshold


def test_far_target_knob_changes_threshold():
    scorer = _toy_scorer()
    feats = _benign_feats()
    loose = scorer.score("recreational", feats, far_target=0.05)
    tight = scorer.score("recreational", feats, far_target=0.01)
    assert loose.threshold is not None and tight.threshold is not None
    # 1% FAR → higher anomaly threshold (fewer flags).
    assert tight.threshold >= loose.threshold


def test_score_detection_uses_class_name():
    scorer = _toy_scorer()
    det = Detection(
        box_xyxy=(0, 0, 10, 10),
        score=0.9,
        class_name="recreational",
        class_id=3,
        role="benign",
    )
    res = scorer.score_detection(det, _benign_feats())
    assert res.asserted_class == "recreational"
    assert res.status == "scored"


def test_result_as_dict():
    scorer = _toy_scorer()
    d = scorer.score("recreational", _benign_feats()).as_dict()
    assert "score" in d and "is_inconsistent" in d and "envelope_used" in d


# ---------------------------------------------------------------------------
# Abstain policy
# ---------------------------------------------------------------------------

def test_abstain_hostile_and_coarse():
    scorer = _toy_scorer()
    for cls in ("usv", "military", "benign_unspecified", "static_aid"):
        res = scorer.score(cls, _benign_feats())
        assert res.status == "abstain", cls
        assert res.envelope_used is None
        assert res.is_inconsistent is None
        assert res.score is None


def test_unknown_class():
    scorer = _toy_scorer()
    res = scorer.score("not_a_real_class", _benign_feats())
    assert res.status == "unknown_class"
    assert res.is_inconsistent is None


def test_small_craft_uses_proxy_envelope():
    scorer = _toy_scorer()
    res = scorer.score("small_craft", _benign_feats())
    assert res.status == "scored"
    assert res.envelope_used == "small_craft_class_b_proxy"


def test_scoreable_and_abstain_lists():
    scorer = _toy_scorer()
    assert "recreational" in scorer.scoreable_classes()
    assert "usv" in scorer.abstain_classes()
    assert "usv" not in scorer.scoreable_classes()


# ---------------------------------------------------------------------------
# Multi-horizon
# ---------------------------------------------------------------------------

def test_multihorizon_prefers_longest_complete():
    scorer = _toy_scorer()
    feats = _benign_feats()
    res = scorer.score(
        "recreational",
        features_by_window={120: feats, 300: feats},
        complete_windows={120, 300},
    )
    assert res.window_s == 300


def test_multihorizon_falls_back_to_short():
    scorer = _toy_scorer()
    feats = _benign_feats()
    res = scorer.score(
        "recreational",
        features_by_window={120: feats, 300: feats},
        complete_windows={120},  # 300 incomplete
    )
    assert res.status == "scored"
    assert res.window_s == 120


def test_no_complete_window():
    scorer = _toy_scorer()
    res = scorer.score(
        "recreational",
        features_by_window={120: _benign_feats()},
        complete_windows=set(),
    )
    assert res.status == "no_window"
    assert res.score is None


# ---------------------------------------------------------------------------
# Firewall
# ---------------------------------------------------------------------------

def test_firewall_blocks_hostile_train():
    with pytest.raises(FirewallError):
        assert_benign_train_allowed({"role": "hostile", "source": "synth"})


def test_firewall_blocks_usv_source():
    with pytest.raises(FirewallError):
        assert_benign_train_allowed({"role": "benign", "source": "usv"})


def test_firewall_blocks_usv_canonical():
    with pytest.raises(FirewallError):
        assert_benign_train_allowed({"role": "benign", "canonical_class": "usv"})


def test_firewall_allows_benign():
    assert_benign_train_allowed({"role": "benign", "source": "ais"})


def test_score_purpose_train_requires_meta_and_blocks_hostile():
    scorer = _toy_scorer()
    with pytest.raises(FirewallError):
        scorer.score("recreational", _benign_feats(), purpose="train")
    with pytest.raises(FirewallError):
        scorer.score(
            "recreational", _benign_feats(),
            purpose="train",
            track_meta={"role": "hostile"},
        )
    # Benign train path OK (still scores; firewall only guards the meta).
    res = scorer.score(
        "recreational", _benign_feats(),
        purpose="train",
        track_meta={"role": "benign", "source": "ais"},
    )
    assert res.status == "scored"


def test_score_purpose_eval_allows_hostile_meta():
    scorer = _toy_scorer()
    res = scorer.score(
        "recreational",
        _benign_feats(sog_med=40.0, sog_p95=45.0),
        purpose="eval",
        track_meta={"role": "hostile", "source": "synth"},
    )
    assert res.status == "scored"


def test_filter_benign_training_drops_blocked():
    df = pd.DataFrame([
        {"role": "benign", "source": "ais", "canonical_class": "fishing"},
        {"role": "hostile", "source": "synth", "canonical_class": "usv"},
        {"role": "benign", "source": "usv", "canonical_class": "recreational"},
        {"role": "non_target", "source": "ais", "canonical_class": "static_aid"},
    ])
    out = filter_benign_training(df)
    assert len(out) == 1
    assert out.iloc[0]["canonical_class"] == "fishing"
    with pytest.raises(FirewallError):
        filter_benign_training(df, raise_on_blocked=True)


# ---------------------------------------------------------------------------
# Envelope override (pooled ablation / envelope_override)
# ---------------------------------------------------------------------------

def test_envelope_override_routes_scoreable_class():
    scorer = _toy_scorer()
    pooled = MultiHorizonEnvelope(
        name="pooled_benign",
        members={"canonical_class": ["fishing", "recreational"]},
        primary_window_s=300,
        core_features=list(CORE),
        course_features=["turn_rate_mean_dps", "cog_circ_std_deg"],
        horizons={300: _toy_envelope("pooled_benign", 300)},
        primary="gmm",
    )
    scorer.envelopes["pooled_benign"] = pooled
    res = scorer.score(
        "recreational", _benign_feats(), envelope_override="pooled_benign"
    )
    assert res.status == "scored"
    assert res.envelope_used == "pooled_benign"
    policy, env, reason = scorer.resolve_envelope(
        "recreational", envelope_override="pooled_benign"
    )
    assert policy == "score"
    assert env == "pooled_benign"
    assert reason == "envelope_override"


def test_envelope_override_ignored_on_abstain():
    scorer = _toy_scorer()
    scorer.envelopes["pooled_benign"] = MultiHorizonEnvelope(
        name="pooled_benign",
        members={},
        primary_window_s=300,
        core_features=list(CORE),
        course_features=["turn_rate_mean_dps", "cog_circ_std_deg"],
        horizons={300: _toy_envelope("pooled_benign", 300)},
        primary="gmm",
    )
    res = scorer.score("usv", _benign_feats(), envelope_override="pooled_benign")
    assert res.status == "abstain"
    assert res.envelope_used is None


def test_envelope_override_missing_envelope():
    scorer = _toy_scorer()
    res = scorer.score(
        "recreational", _benign_feats(), envelope_override="pooled_benign"
    )
    assert res.status == "missing_envelope"
    assert res.envelope_used == "pooled_benign"


def test_attach_envelope(tmp_path):
    from counterusv.kinematics.behavior_model import save_envelope

    scorer = _toy_scorer()
    assert "pooled_benign" not in scorer.envelopes
    bundle = MultiHorizonEnvelope(
        name="pooled_benign",
        members={"canonical_class": ["fishing"]},
        primary_window_s=300,
        core_features=list(CORE),
        course_features=["turn_rate_mean_dps", "cog_circ_std_deg"],
        horizons={300: _toy_envelope("pooled_benign", 300)},
        primary="gmm",
    )
    path = tmp_path / "pooled_benign.joblib"
    save_envelope(bundle, path)
    scorer.attach_envelope("pooled_benign", path)
    assert "pooled_benign" in scorer.envelopes
    res = scorer.score(
        "recreational", _benign_feats(), envelope_override="pooled_benign"
    )
    assert res.status == "scored"
    assert res.envelope_used == "pooled_benign"
