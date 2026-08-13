"""Tests for real-track label-swap helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.eval.label_swap import (  # noqa: E402
    admissible_splits,
    attack_run_mask,
    load_envelope_map,
    load_label_swap_config,
    matrix_from_records,
    swap_pairs,
    thin_n_label,
    touches_proxy_pool,
)


def test_attack_run_mask_gates():
    df = pd.DataFrame({
        "sog_p95": [35.0, 20.0, 40.0, 32.0],
        "straightness": [0.99, 0.99, 0.90, 0.96],
        "window_complete": [True, True, True, False],
        "role": ["benign", "benign", "benign", "benign"],
    })
    m = attack_run_mask(df)
    assert m.tolist() == [True, False, False, False]


def test_hygiene_recreational_to_fishing_allows_train():
    cfg = load_label_swap_config()
    emap = load_envelope_map()
    splits = admissible_splits("recreational", "fishing", cfg=cfg, emap=emap)
    assert "train" in splits
    assert "val" in splits and "test" in splits


def test_hygiene_recreational_to_small_craft_forbids_train():
    cfg = load_label_swap_config()
    emap = load_envelope_map()
    assert touches_proxy_pool("recreational", "small_craft", cfg=cfg, emap=emap)
    splits = admissible_splits("recreational", "small_craft", cfg=cfg, emap=emap)
    assert "train" not in splits
    assert splits == {"val", "test"}


def test_matched_class_held_out_only():
    cfg = load_label_swap_config()
    emap = load_envelope_map()
    splits = admissible_splits("recreational", "recreational", cfg=cfg, emap=emap)
    assert splits == {"val", "test"}


def test_hygiene_unchanged_by_pooled_routing_intent():
    """Pooled scoring must not change admissible_splits (score-time override only)."""
    cfg = load_label_swap_config()
    emap = load_envelope_map()
    # Even if config mentions pooled routing, hygiene stays map-driven.
    cfg_pooled = {**cfg, "envelope_routing": "pooled", "envelope_override": "pooled_benign"}
    assert admissible_splits("recreational", "fishing", cfg=cfg_pooled, emap=emap) == (
        admissible_splits("recreational", "fishing", cfg=cfg, emap=emap)
    )
    assert admissible_splits("recreational", "small_craft", cfg=cfg_pooled, emap=emap) == (
        admissible_splits("recreational", "small_craft", cfg=cfg, emap=emap)
    )


def test_swap_pairs_off_diagonal():
    pairs = swap_pairs(["recreational", "fishing"], ["fishing", "recreational", "sailing"])
    assert ("recreational", "recreational") not in pairs
    assert ("recreational", "fishing") in pairs
    assert ("fishing", "sailing") in pairs


def test_matrix_from_records():
    records = [
        {"source_class": "recreational", "asserted_class": "fishing",
         "flag_rate": 0.8, "n_scored": 50},
        {"source_class": "recreational", "asserted_class": "sailing",
         "flag_rate": 0.9, "n_scored": 50},
        {"source_class": "passenger_ferry", "asserted_class": "fishing",
         "flag_rate": 1.0, "n_scored": 10},
    ]
    rates, ns = matrix_from_records(records)
    assert rates.loc["recreational", "fishing"] == pytest.approx(0.8)
    assert int(ns.loc["passenger_ferry", "fishing"]) == 10
    assert thin_n_label(10) == "thin-n"
    assert thin_n_label(50) is None
