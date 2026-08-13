"""Tests for the regenerable paper figure pipeline."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "scripts" / "report"
sys.path.insert(0, str(REPORT))

from _common import (  # noqa: E402
    FIG1_SRC,
    copy_fig1,
    sha256,
    verify_freezes,
)


def test_verify_freezes_passes_on_current_pins() -> None:
    result = verify_freezes(skip=False)
    assert result["verified"] is True
    assert result["n_checked"] > 10
    assert result["mismatches"] == []


def test_verify_freezes_catches_tamper(tmp_path: Path) -> None:
    """Mutate a *copy* of a pinned file and assert verify fails on that tree.

    Never symlink into live ``results/`` — unlink through a symlink would
    delete the real artifact.
    """
    import _common as common

    freeze = json.loads((REPO_ROOT / "results" / "defense" / "FROZEN.json").read_text())
    entry = freeze["evaluation"]["artifacts"]["label_swap_summary.json"]
    rel = entry["path"]
    src = REPO_ROOT / rel
    assert src.is_file(), "label_swap summary must exist for this test"

    shadow = tmp_path / "repo"
    # Minimal freeze that only pins the one file we mutate.
    (shadow / "results" / "attacks").mkdir(parents=True)
    (shadow / "results" / "defense").mkdir(parents=True)
    (shadow / Path(rel).parent).mkdir(parents=True, exist_ok=True)

    attacks_payload = {
        "configs": {},
        "results": {},
    }
    defense_payload = {
        "configs": {},
        "evaluation": {
            "artifacts": {
                "label_swap_summary.json": {
                    "path": rel,
                    "sha256": entry["sha256"],
                    "bytes": entry["bytes"],
                }
            }
        },
        "arms": {},
    }
    (shadow / "results" / "attacks" / "FROZEN.json").write_text(
        json.dumps(attacks_payload) + "\n"
    )
    (shadow / "results" / "defense" / "FROZEN.json").write_text(
        json.dumps(defense_payload) + "\n"
    )

    # Copy then mutate — never touch the live file.
    shutil.copy2(src, shadow / rel)
    mutated = shadow / rel
    mutated.write_bytes(mutated.read_bytes() + b"\n")

    # Point the helper at the shadow tree.
    old = {
        "REPO_ROOT": common.REPO_ROOT,
        "ATTACKS_FREEZE": common.ATTACKS_FREEZE,
        "DEFENSE_FREEZE": common.DEFENSE_FREEZE,
    }
    common.REPO_ROOT = shadow
    common.ATTACKS_FREEZE = shadow / "results" / "attacks" / "FROZEN.json"
    common.DEFENSE_FREEZE = shadow / "results" / "defense" / "FROZEN.json"
    try:
        with pytest.raises(SystemExit, match="digest verification FAILED"):
            common.verify_freezes(skip=False)
    finally:
        common.REPO_ROOT = old["REPO_ROOT"]
        common.ATTACKS_FREEZE = old["ATTACKS_FREEZE"]
        common.DEFENSE_FREEZE = old["DEFENSE_FREEZE"]


def test_copy_fig1_requires_svg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import _common as common

    monkeypatch.setattr(common, "FIG1_SRC", tmp_path / "missing.svg")
    with pytest.raises(SystemExit, match="static Fig. 1 missing"):
        common.copy_fig1(out_dir=tmp_path)


def test_copy_fig1_ok(tmp_path: Path) -> None:
    assert FIG1_SRC.is_file()
    meta = copy_fig1(out_dir=tmp_path)
    assert (tmp_path / "fig1_system.svg").is_file()
    assert meta["sha256"] == sha256(FIG1_SRC)
    assert meta["static"] is True


def test_captions_cover_every_figure(tmp_path: Path) -> None:
    """Every emitted figure needs a caption record, with panels where it has panels."""
    from captions import SCHEMA, write

    out = write(tmp_path)
    payload = json.loads(out["json"].read_text())
    assert payload["schema"] == SCHEMA
    assert out["markdown"].read_text().startswith("# Figure captions")

    by_id = {r["id"]: r for r in payload["figures"]}
    assert set(by_id) == {
        "fig1_system",
        "fig2_rq1_feasibility",
        "fig3_rq2_ddr_gap",
        "fig4_label_swap",
        "fig5_rq3_cost_warning",
        "fig6_far",
        "figS1_severity",
        "figS2_pooled_gap",
    }
    for rec in payload["figures"]:
        assert rec["short_title"].strip(), rec["id"]
        assert len(rec["caption"]) > 200, rec["id"]
        assert rec["sources"], rec["id"]
        # Titles belong in the caption, never on the figure.
        assert not rec["short_title"].startswith("Fig"), rec["id"]

    # Multi-panel figures must describe each panel they draw.
    assert set(by_id["fig3_rq2_ddr_gap"]["panels"]) == {"a", "b"}
    assert set(by_id["fig5_rq3_cost_warning"]["panels"]) == {"a", "b"}
    assert set(by_id["fig6_far"]["panels"]) == {"a", "b"}
    assert set(by_id["figS1_severity"]["panels"]) == {"a", "b", "c"}
    assert set(by_id["figS2_pooled_gap"]["panels"]) == {"a", "b"}


def test_caption_numbers_match_source_artifacts(tmp_path: Path) -> None:
    """Quoted values must equal the pinned artifacts, not a stale transcription."""
    from _common import load_json
    from captions import write

    payload = json.loads(write(tmp_path)["json"].read_text())
    by_id = {r["id"]: r for r in payload["figures"]}

    oracle = load_json("results/oracle_ddr/oracle_ddr_summary.json")
    cons = {
        (r["mimicked_class"], r["arm"]): round(float(r["ddr"]) * 100.0, 1)
        for r in oracle["consistency_ddr"]
    }
    kv3 = by_id["fig3_rq2_ddr_gap"]["key_values"]
    assert kv3["ddr_recreational_kinematics_pct"] == cons[
        ("recreational", "kinematics_only")
    ]
    assert kv3["ddr_recreational_geometry_pct"] == cons[
        ("recreational", "kinematics_geometry")
    ]
    assert kv3["operating_far_pct"]["kinematics_only"] == round(
        oracle["operating_far"]["kinematics_only"] * 100.0, 1
    )

    swap = load_json("results/label_swap/label_swap_summary.json")
    kv4 = by_id["fig4_label_swap"]["key_values"]
    assert kv4["n_band_trips"] == swap["kinematics"]["n_band_trips"]
    assert kv4["n_thin_cells"] <= kv4["n_cells"]

    # Fig. S1 asserts access level dominates severity — hold the claim to the data.
    assert by_id["figS1_severity"]["key_values"]["access_ordering_holds_across_ladder"]

    # Fig. S2: pooling shrinks the recreational→fishing gap and drops fishing kin DDR.
    kv_s2 = by_id["figS2_pooled_gap"]["key_values"]
    assert kv_s2["pooling_shrinks_rec_fish_gap"]
    assert kv_s2["pooling_drops_fishing_kin_ddr"]
    oracle_p = load_json("results/oracle_ddr_pooled/oracle_ddr_summary.json")
    fish_p = next(
        r
        for r in oracle_p["consistency_ddr"]
        if r["mimicked_class"] == "fishing" and r["arm"] == "kinematics_only"
    )
    assert kv_s2["ddr_fishing_kinematics_pooled_pct"] == round(
        float(fish_p["ddr"]) * 100.0, 1
    )

def test_build_all_smoke() -> None:
    sys.path.insert(0, str(REPORT))
    from build_all import run

    with tempfile.TemporaryDirectory(prefix="paper_test_") as tmp:
        out = Path(tmp)
        result = run(out, skip_verify=True)
        assert (out / "fig1_system.svg").is_file()
        for stem in (
            "fig2_rq1_feasibility",
            "fig3_rq2_ddr_gap",
            "fig4_label_swap",
            "fig5_rq3_cost_warning",
            "fig6_far",
            "figS1_severity",
            "figS2_pooled_gap",
        ):
            assert (out / f"{stem}.png").is_file(), stem
            assert (out / f"{stem}.pdf").is_file(), stem
        assert (out / "CAPTIONS.json").is_file()
        assert (out / "CAPTIONS.md").is_file()
        prov = out / "PROVENANCE.json"
        assert prov.is_file()
        payload = json.loads(prov.read_text())
        assert "figures" in payload
        assert "git_sha" in payload
        assert "fig1_system" in payload["figures"]
        assert result["verification"]["skipped"] is True
