"""Tests for the weight-free release bundler."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "release"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

from build_bundle import (  # noqa: E402
    RESULTS_EXACT,
    drop_mmsi,
    find_redacted,
    strip_mcships_coco,
    strip_mcships_csv,
    verify_stage,
    write_checksums,
    hash_stage,
    write_release,
    write_artifact_readme,
)
from package_derived import DERIVED_ARTIFACTS  # noqa: E402


def test_derived_inventory_excludes_points_and_mcships() -> None:
    assert "tracks/tracks_ais_points.parquet" not in DERIVED_ARTIFACTS
    assert "annotations/mcships.coco.json" not in DERIVED_ARTIFACTS
    assert "behavior/benign_train_manifest.parquet" in DERIVED_ARTIFACTS
    assert "defense/placements.parquet" in DERIVED_ARTIFACTS


def test_results_exact_includes_pooled_ablation() -> None:
    """Pooled-vs-conditional summaries must ship with the data/results deposit."""
    for rel in (
        "results/label_swap_pooled/label_swap_summary.json",
        "results/label_swap_pooled/label_swap_report.md",
        "results/label_swap_pooled/label_swap_cells.parquet",
        "results/oracle_ddr_pooled/oracle_ddr_summary.json",
        "results/oracle_ddr_pooled/oracle_ddr_report.md",
        "results/oracle_ddr_pooled/oracle_ddr_cells.parquet",
    ):
        assert rel in RESULTS_EXACT, rel
    # Envelope joblibs stay withheld (weights policy).
    assert not any("envelopes/" in r or r.endswith(".joblib") for r in RESULTS_EXACT)


def test_drop_mmsi_parquet_and_csv(tmp_path: Path) -> None:
    pq = tmp_path / "t.parquet"
    pd.DataFrame({"trip_id": [1, 2], "mmsi": [10, 20], "x": [0.1, 0.2]}).to_parquet(
        pq, index=False
    )
    out_pq = tmp_path / "out.parquet"
    drop_mmsi(pq, out_pq)
    got = pd.read_parquet(out_pq)
    assert "mmsi" not in got.columns
    assert list(got["trip_id"]) == [1, 2]

    csv = tmp_path / "t.csv"
    pd.DataFrame({"trip_id": [1], "mmsi": [99], "split": ["train"]}).to_csv(
        csv, index=False
    )
    out_csv = tmp_path / "out.csv"
    drop_mmsi(csv, out_csv)
    got_c = pd.read_csv(out_csv)
    assert "mmsi" not in got_c.columns


def test_strip_mcships_coco_and_csv(tmp_path: Path) -> None:
    coco = {
        "images": [
            {"id": 1, "source": "seaships", "file_name": "a.jpg"},
            {"id": 2, "source": "mcships", "file_name": "b.jpg"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1},
            {"id": 2, "image_id": 2, "category_id": 2},
        ],
        "categories": [{"id": 1, "name": "fishing"}],
    }
    src = tmp_path / "master.json"
    src.write_text(json.dumps(coco))
    dst = tmp_path / "out.json"
    strip_mcships_coco(src, dst)
    out = json.loads(dst.read_text())
    assert [i["id"] for i in out["images"]] == [1]
    assert [a["id"] for a in out["annotations"]] == [1]

    csv = tmp_path / "audit.csv"
    pd.DataFrame(
        {"image_id": [1, 2], "source": ["seaships", "mcships"]}
    ).to_csv(csv, index=False)
    out_csv = tmp_path / "audit_out.csv"
    strip_mcships_csv(csv, out_csv)
    got = pd.read_csv(out_csv)
    assert list(got["source"]) == ["seaships"]


def test_redaction_gate_catches_planted_weight(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    (stage / "data").mkdir(parents=True)
    (stage / "data" / "ok.txt").write_text("fine")
    planted = stage / "results" / "detector_baselines" / "yolo11s" / "weights"
    planted.mkdir(parents=True)
    (planted / "best.pt").write_bytes(b"not-a-real-weight")
    bad = find_redacted(stage)
    assert any(p.endswith("best.pt") for p in bad)


def test_verify_stage_checksums_and_rejects_weight(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "ok.txt").write_text("hello")
    write_artifact_readme(stage)
    rows = hash_stage(stage)
    write_checksums(stage, rows)
    write_release(stage, rows, freezes_ok=False)
    assert verify_stage(stage) == 0

    (stage / "leak.joblib").write_bytes(b"nope")
    # checksums won't list leak.joblib, but redaction gate must fail
    assert verify_stage(stage) == 1
