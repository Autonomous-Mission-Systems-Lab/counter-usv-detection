#!/usr/bin/env python3
"""Freeze the defense bundle after RQ evaluation (both feature arms).

Pins SHA-256 digests over:

* both arm freezes (``results/behavior_model/``, ``results/behavior_model_geometry/``)
* adversary-motion sweep freeze (evaluation-only synth tracks)
* headline RQ2/RQ3 result summaries (oracle DDR, adaptive cost, label-swap)
* non-git data pins nested in arm freezes (envelopes, placements)

Shared defense configs are recorded by **path only** (git is source of truth).
Re-attests the benign-only training firewall and strips leftover config digests
from arm freezes. Envelope digests must still match — a mismatch aborts rather
than silently updating weights.

Writes ``results/defense/FROZEN.json`` + ``MODEL_CARD.md``, then smoke-loads
both consistency arms and the presence-only comparator.

Usage
-----
    python scripts/defense/freeze_defense.py
    python scripts/defense/freeze_defense.py --skip-smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_OUT = REPO_ROOT / "results" / "defense"
KIN_FREEZE = REPO_ROOT / "results" / "behavior_model" / "FROZEN.json"
GEO_FREEZE = REPO_ROOT / "results" / "behavior_model_geometry" / "FROZEN.json"
MOTION_FREEZE = REPO_ROOT / "results" / "adversary_motion" / "FROZEN_SWEEP.json"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "behavior" / "benign_train_manifest.parquet"
DEFAULT_CORPUS = REPO_ROOT / "data" / "behavior" / "benign_corpus_summary.json"

SHARED_CONFIGS = [
    "configs/defense/pipeline.yaml",
    "configs/defense/presence.yaml",
    "configs/defense/scorer_features.yaml",
    "configs/defense/class_envelope_map.yaml",
    "configs/defense/behavior_model.yaml",
    "configs/defense/behavior_model_geometry.yaml",
    "configs/defense/engagement_geometry.yaml",
    "configs/defense/oracle_ddr.yaml",
    "configs/defense/adaptive_cost.yaml",
    "data/defense/placements.parquet",
    "data/defense/placements_digest.json",
]

EVAL_ARTIFACTS = [
    "results/oracle_ddr/oracle_ddr_summary.json",
    "results/oracle_ddr/oracle_ddr_report.md",
    "results/oracle_ddr_pooled/oracle_ddr_summary.json",
    "results/oracle_ddr_pooled/oracle_ddr_report.md",
    "results/adaptive_cost/adaptive_cost_summary.json",
    "results/adaptive_cost/adaptive_cost_report.md",
    "results/label_swap/label_swap_summary.json",
    "results/label_swap/label_swap_report.md",
    "results/label_swap_pooled/label_swap_summary.json",
    "results/label_swap_pooled/label_swap_report.md",
    "results/adversary_motion/FROZEN_SWEEP.json",
]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "nogit"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _digest_entry(rel: str, *, required: bool = True) -> dict[str, Any] | None:
    path = REPO_ROOT / rel
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"freeze requires {rel}")
        return None
    return {
        "path": rel,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _path_entry(rel: str, *, required: bool = True) -> dict[str, Any] | None:
    """Record a config path without digesting (git owns configs)."""
    path = REPO_ROOT / rel
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"freeze requires {rel}")
        return None
    return {"path": rel}


def _is_yaml_config_path(rel: str) -> bool:
    return rel.startswith("configs/")


def reattest_arm_freeze(freeze_path: Path) -> dict[str, Any]:
    """Strip YAML config digests; verify envelopes and non-git data pins."""
    data = _load_json(freeze_path)
    stripped: list[str] = []

    for label, block in (data.get("configs") or {}).items():
        rel = block.get("path")
        if not rel:
            continue
        path = REPO_ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"arm freeze config missing: {rel}")
        if _is_yaml_config_path(rel):
            if "sha256" in block or "bytes" in block:
                stripped.append(label)
                block.pop("sha256", None)
                block.pop("bytes", None)
            continue
        # Non-git data pins nested under configs (e.g. placements.parquet)
        got = _sha256(path)
        exp = block.get("sha256")
        if exp and got != exp:
            raise SystemExit(
                f"data pin digest drift in {_rel(freeze_path)} / {label}: {rel}"
            )
        block["sha256"] = got
        block["bytes"] = path.stat().st_size

    env_mismatches: list[str] = []
    for name, block in (data.get("envelopes") or {}).items():
        rel = block.get("path")
        path = REPO_ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"arm freeze envelope missing: {rel}")
        got = _sha256(path)
        exp = block.get("sha256")
        if exp and got != exp:
            env_mismatches.append(f"{name}: {rel}")
    if env_mismatches:
        raise SystemExit(
            f"envelope digest drift in {_rel(freeze_path)} — re-fit required:\n  "
            + "\n  ".join(env_mismatches)
        )

    note = {
        "path": _rel(freeze_path),
        "re_attested_utc": datetime.now(timezone.utc).isoformat(),
        "config_digests_stripped": stripped,
        "envelopes_verified": True,
        "n_envelopes": len(data.get("envelopes") or {}),
    }
    data.pop("config_reattestation", None)
    if stripped:
        data["config_path_only"] = {
            "utc": note["re_attested_utc"],
            "stripped": stripped,
            "note": (
                "YAML config digests removed; git is the source of truth. "
                "Envelope and data-pin digests were re-verified unchanged."
            ),
        }
    freeze_path.write_text(json.dumps(data, indent=2) + "\n")
    if stripped:
        print(
            f"[freeze] stripped {len(stripped)} config digest(s) in "
            f"{_rel(freeze_path)}"
        )
    else:
        print(f"[freeze] arm envelopes OK: {_rel(freeze_path)}")
    return note


def attest_firewall() -> dict[str, Any]:
    """Re-check the benign train manifest; refuse hostile / usv / non_target."""
    import pandas as pd

    from counterusv.defense import FirewallError, filter_benign_training

    manifest = DEFAULT_MANIFEST
    corpus = _load_json(DEFAULT_CORPUS)
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing benign train manifest: {manifest}")

    df = pd.read_parquet(manifest)
    n = len(df)
    n_role_bad = 0
    if "role" in df.columns:
        role = df["role"].astype(str).str.lower()
        n_role_bad = int((role != "benign").sum())
    n_usv_src = (
        int(df["source"].astype(str).str.lower().eq("usv").sum())
        if "source" in df.columns
        else 0
    )
    n_hostile_cls = 0
    if "canonical_class" in df.columns:
        canon = df["canonical_class"].astype(str).str.lower()
        n_hostile_cls = int(canon.isin({"usv", "military"}).sum())

    try:
        kept = filter_benign_training(df, raise_on_blocked=True)
    except FirewallError as e:
        raise SystemExit(f"firewall attestation FAILED: {e}") from e
    if len(kept) != n:
        raise SystemExit(
            f"firewall attestation FAILED: filter dropped {n - len(kept)} "
            f"of {n} rows (expected 0)."
        )
    ok = n_role_bad == 0 and n_usv_src == 0 and n_hostile_cls == 0
    if not ok:
        raise SystemExit(
            "firewall attestation FAILED: "
            f"role_non_benign={n_role_bad}, source_usv={n_usv_src}, "
            f"canonical_usv_or_military={n_hostile_cls}"
        )

    return {
        "pass": True,
        "attested_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": _rel(manifest),
        "n_benign_train": int(n),
        "n_role_non_benign": n_role_bad,
        "n_source_usv": n_usv_src,
        "n_canonical_usv_or_military": n_hostile_cls,
        "corpus_n_benign_train": corpus.get("n_benign_train"),
        "statement": (
            "Benign train manifest is role==benign only; 0 hostile / 0 "
            "non_target / 0 usv / 0 military rows. Hostile and adaptive "
            "trajectories are evaluation-only and never enter scorer training "
            "or calibration."
        ),
    }


def _ddr_headlines(oracle: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for r in oracle.get("consistency_ddr") or []:
        if r.get("defense_kind") != "consistency":
            continue
        rows.append(
            {
                "mimicked_class": r.get("mimicked_class"),
                "arm": r.get("arm"),
                "n": r.get("n"),
                "ddr": r.get("ddr"),
            }
        )
    presence = {
        r.get("mimicked_class"): r.get("ddr")
        for r in (oracle.get("presence_ddr") or [])
        if r.get("mimicked_class")
    }
    return {
        "condition": oracle.get("condition"),
        "n_trips_scored": oracle.get("n_trips_scored"),
        "operating_far": oracle.get("operating_far"),
        "consistency_ddr": rows,
        "presence_ddr_by_class": presence,
        "unconstrained_ddr": oracle.get("unconstrained_ddr"),
    }


def _cost_headlines(cost: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_trips": cost.get("n_trips"),
        "n_joined_rows": cost.get("n_joined_rows"),
        "operating_far": cost.get("operating_far"),
        "recreational_delta_t_zero": cost.get("recreational_delta_t_zero"),
        "warning_time": {
            "stride_s": (cost.get("warning_time") or {}).get("stride_s"),
            "n_trips": (cost.get("warning_time") or {}).get("n_trips"),
        },
        "freeze_ddr_numbers_present": (cost.get("freeze") or {}).get(
            "ddr_numbers_present"
        ),
    }


def write_model_card(path: Path, payload: dict[str, Any]) -> None:
    fw = payload["firewall_attestation"]
    kin = payload["arms"]["kinematics_only"]
    geo = payload["arms"]["kinematics_geometry"]
    ddr = payload["evaluation"]["oracle_ddr_headlines"]
    cost = payload["evaluation"]["adaptive_cost_headlines"]
    lines = [
        "# Defense model card",
        "",
        f"Freeze version: `{payload['version']}`  ·  "
        f"frozen {payload['frozen_utc']}  ·  git `{payload['git_sha']}`",
        "",
        "Machine pin + SHA-256 digests: **`FROZEN.json`**. "
        "Arm-level detail stays in "
        "`results/behavior_model/` and `results/behavior_model_geometry/`.",
        "",
        "## Intended use",
        "",
        "Shore-based counter-USV **consistency defense**: given a claimed "
        "benign class (EO detection or perfect-disguise oracle) and a "
        "world-frame track, score whether the motion matches real vessels of "
        "that class. Two feature arms are reported separately:",
        "",
        "1. **`kinematics_only`** — speed / loiter / straightness / turn "
        "(frozen kinematics-only envelopes).",
        "2. **`kinematics_geometry`** — same plus asset-relative engagement "
        "geometry (closing rate, bearing stability, CPA, …).",
        "",
        "Presence-only is the RQ2 comparator (track present ∧ EO miss/"
        "mislabeled), not a consistency model. PercepGuard is differentiated "
        "by argument, not shipped as a baseline.",
        "",
        "## Training data & firewall",
        "",
        "- Source: MarineCadastre AIS, **train ∩ role==benign** only.",
        f"- Tracks: **{fw['n_benign_train']:,}** (attested).",
        f"- Firewall: {fw['statement']}",
        "- AIS is **offline only** — never a runtime input.",
        "- Hostile / adaptive tracks are **evaluation-only** "
        "(adversary motion sweep); they never enter training or calibration.",
        "",
        "## Arms",
        "",
        "| Arm | Freeze | Primary window | FAR@5% (operating) | Envelopes |",
        "|---|---|---|---|---|",
        f"| `kinematics_only` | `{kin['freeze_path']}` | "
        f"{kin.get('primary_window_s')} s | "
        f"{(ddr.get('operating_far') or {}).get('kinematics_only')} | "
        f"{kin['n_envelopes']} |",
        f"| `kinematics_geometry` | `{geo['freeze_path']}` | "
        f"{geo.get('primary_window_s')} s | "
        f"{(ddr.get('operating_far') or {}).get('kinematics_geometry')} | "
        f"{geo['n_envelopes']} |",
        "",
        "## Headline evaluation (pinned summaries)",
        "",
        "Perfect-disguise oracle DDR (non-NC, FAR-paired) — see "
        "`results/oracle_ddr/`:",
        "",
    ]
    for r in ddr.get("consistency_ddr") or []:
        lines.append(
            f"- `{r.get('mimicked_class')}` / `{r.get('arm')}`: "
            f"DDR **{r.get('ddr')}** (n={r.get('n')})"
        )
    lines += [
        "",
        "Presence DDR is **0%** on disguise by construction "
        "(gap ≡ consistency).",
        "",
        "Adaptive cost (RQ3) — see `results/adaptive_cost/`:",
        f"- Joined rows: **{cost.get('n_joined_rows')}** "
        f"({cost.get('n_trips')} trips × 2 arms).",
        f"- Recreational unconstrained (Δt≈0): kinematics DDR "
        f"**{(cost.get('recreational_delta_t_zero') or {}).get('kinematics_only', {}).get('ddr')}**; "
        f"geometry "
        f"**{(cost.get('recreational_delta_t_zero') or {}).get('kinematics_geometry', {}).get('ddr')}**.",
        f"- Motion freeze carries **no** DDR payload "
        f"(`ddr_numbers_present="
        f"{cost.get('freeze_ddr_numbers_present')}`).",
        "",
        "Real-track label-swap control — see `results/label_swap/` "
        "(pinned in this freeze): synthesis-free flag rates under a "
        "wrong asserted class at frozen FAR@5% thresholds.",
        "",
        "Pooled-vs-class-conditional ablation (discriminability at matched "
        "FAR) — pinned alongside the conditional runs:",
        "- `results/label_swap_pooled/` — same real-track swap under a "
        "single `pooled_benign` kinematics envelope.",
        "- `results/oracle_ddr_pooled/` — same oracle grid under "
        "`pooled_benign` on both arms.",
        "The kinematics arm freeze lists `pooled_benign` in its envelope "
        "roster (`pooled_ablation`); joblib weights are not redistributed.",
        "",
        "## Load",
        "",
        "```python",
        "from counterusv.defense import DefensePipeline",
        "from counterusv.defense.harness import load_defense",
        "",
        "# kinematics arm (default pipeline.yaml)",
        "pipe = DefensePipeline.from_freeze(verify_digests=True)",
        "",
        "# geometry arm",
        "pipe_geo = DefensePipeline.from_freeze(",
        '    "results/behavior_model_geometry/FROZEN.json",',
        "    verify_digests=True,",
        ")",
        "",
        'presence = load_defense("presence_only")',
        "```",
        "",
        "## Known limitations",
        "",
        "- Cooperative-AIS FAR ≠ operational FAR (no video FAR campaign).",
        "- No measured PercepGuard head-to-head (argument-only).",
        "- TMSR≈0 ⇒ no patch-conditioned DDR table; claim = oracle × patch "
        "reliability.",
        "- Geometry arm needs a designated defended asset and world-frame "
        "range/bearing; ferry over-flags; thin envelope n for some classes.",
        "- Synth hostile tracks at evaluation only — stated wherever DDR "
        "appears.",
        "",
        "## Ethical / dual-use",
        "",
        "See `docs/DUAL_USE.md`. This freeze is the defended scoring bundle "
        "and evaluation pin — not an attack. The adversary motion model ships "
        "as a summary-feature / trajectory generator with a frozen sweep "
        "spec, not a mission planner.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def smoke(verify_digests: bool = True) -> None:
    from counterusv.attacks.oracle import PerfectDisguiseOracle
    from counterusv.defense.pipeline import DefensePipeline
    from counterusv.defense.presence import (
        PresenceOnlyDefense,
        presence_for_disguise,
        presence_for_evasion,
    )
    from counterusv.models.detector import Detection

    def feats(sog: float = 6.0) -> dict[str, float]:
        return {
            "sog_med": sog,
            "sog_p95": sog * 1.2,
            "sog_std": max(0.5, sog * 0.1),
            "loiter_frac": 0.1 if sog < 15 else 0.0,
            "straightness": 0.85 if sog < 15 else 0.99,
            "accel_mean_abs": 0.05,
            "turn_rate_mean_dps": 0.1,
            "cog_circ_std_deg": 5.0,
        }

    print("[freeze] smoke: kinematics_only …")
    pipe_k = DefensePipeline.from_freeze(
        KIN_FREEZE, verify_digests=verify_digests
    )
    d = pipe_k.evaluate(
        Detection(
            box_xyxy=(10.0, 20.0, 80.0, 60.0),
            score=0.9,
            class_name="recreational",
            class_id=2,
            role="benign",
        ),
        feats(6.0),
        purpose="defense",
    )
    assert d.action in ("pass", "flag", "abstain"), d
    print(f"  kin recreational/benign → {d.action}")

    print("[freeze] smoke: kinematics_geometry …")
    pipe_g = DefensePipeline.from_freeze(
        GEO_FREEZE, verify_digests=verify_digests
    )
    oracle = PerfectDisguiseOracle.from_config()
    assertion = oracle.assert_class(
        "fishing", contact_id="freeze-smoke", box_xyxy=(10.0, 20.0, 80.0, 60.0)
    )
    d2 = pipe_g.evaluate(
        assertion,
        feats(35.0),
        purpose="eval",
        track_meta={"role": "hostile", "source": "synth"},
    )
    print(
        f"  geo oracle fishing + 35 kn → {d2.action} "
        f"(status={d2.consistency.status})"
    )
    assert d2.source == "oracle"

    print("[freeze] smoke: presence_only …")
    presence = PresenceOnlyDefense.from_config()
    d_ev = presence.evaluate(presence=presence_for_evasion(), purpose="eval")
    d_di = presence.evaluate(
        presence=presence_for_disguise("recreational"),
        purpose="eval",
    )
    print(f"  presence evasion → {d_ev.action} (expect flag)")
    print(f"  presence disguise → {d_di.action} (expect pass)")
    assert d_ev.action == "flag"
    assert d_di.action == "pass"
    print("[freeze] smoke: OK")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument(
        "--no-patch-arms",
        action="store_true",
        help="verify arm envelopes/data pins only; do not rewrite arm freezes",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not KIN_FREEZE.is_file() or not GEO_FREEZE.is_file():
        raise SystemExit(
            "Both arm freezes are required: "
            f"{_rel(KIN_FREEZE)} and {_rel(GEO_FREEZE)}"
        )

    print("[freeze] re-attest arm freezes (envelopes + data pins) …")
    if args.no_patch_arms:
        for fp in (KIN_FREEZE, GEO_FREEZE):
            data = _load_json(fp)
            for label, block in (data.get("configs") or {}).items():
                rel = block.get("path") or ""
                if _is_yaml_config_path(rel):
                    continue
                if not block.get("sha256"):
                    continue
                got = _sha256(REPO_ROOT / rel)
                if got != block["sha256"]:
                    raise SystemExit(
                        f"data pin digest drift in {_rel(fp)} / {label}"
                    )
            for name, block in (data.get("envelopes") or {}).items():
                got = _sha256(REPO_ROOT / block["path"])
                if block.get("sha256") and got != block["sha256"]:
                    raise SystemExit(
                        f"envelope digest drift in {_rel(fp)} / {name}"
                    )
        kin_note = {"path": _rel(KIN_FREEZE), "patched": False}
        geo_note = {"path": _rel(GEO_FREEZE), "patched": False}
    else:
        kin_note = reattest_arm_freeze(KIN_FREEZE)
        geo_note = reattest_arm_freeze(GEO_FREEZE)

    print("[freeze] firewall attestation …")
    firewall = attest_firewall()
    print(f"  PASS — n_benign_train={firewall['n_benign_train']:,}")

    kin = _load_json(KIN_FREEZE)
    geo = _load_json(GEO_FREEZE)
    oracle = _load_json(REPO_ROOT / "results" / "oracle_ddr" / "oracle_ddr_summary.json")
    cost = _load_json(
        REPO_ROOT / "results" / "adaptive_cost" / "adaptive_cost_summary.json"
    )

    configs: dict[str, Any] = {}
    for rel in SHARED_CONFIGS:
        if rel.startswith("configs/"):
            entry = _path_entry(rel, required=True)
        else:
            entry = _digest_entry(rel, required=True)
        assert entry is not None
        configs[Path(rel).name] = entry

    results: dict[str, Any] = {}
    for rel in EVAL_ARTIFACTS:
        entry = _digest_entry(rel, required=True)
        assert entry is not None
        results[Path(rel).name] = entry

    version = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    payload: dict[str, Any] = {
        "version": version,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "scope": (
            "Defense bundle after RQ evaluation: both consistency arms, "
            "presence comparator, adversary-motion sweep pin, oracle DDR + "
            "adaptive-cost + real-track label-swap headlines. EO attack "
            "library freezes separately under results/attacks/."
        ),
        "arms": {
            "kinematics_only": {
                "freeze_path": _rel(KIN_FREEZE),
                "sha256": _sha256(KIN_FREEZE),
                "primary_window_s": kin.get("primary_window_s"),
                "windows_s": kin.get("windows_s"),
                "primary_model": kin.get("primary_model"),
                "n_envelopes": len(kin.get("envelopes") or {}),
                "far_floor": kin.get("far_floor"),
                "reattestation": kin.get("config_path_only") or kin_note,
            },
            "kinematics_geometry": {
                "freeze_path": _rel(GEO_FREEZE),
                "sha256": _sha256(GEO_FREEZE),
                "primary_window_s": geo.get("primary_window_s"),
                "windows_s": geo.get("windows_s"),
                "primary_model": geo.get("primary_model"),
                "n_envelopes": len(geo.get("envelopes") or {}),
                "far_floor": geo.get("far_floor"),
                "reattestation": geo.get("config_path_only") or geo_note,
            },
        },
        "configs": configs,
        "evaluation": {
            "artifacts": results,
            "oracle_ddr_headlines": _ddr_headlines(oracle),
            "adaptive_cost_headlines": _cost_headlines(cost),
            "motion_freeze": {
                "path": _rel(MOTION_FREEZE),
                "sha256": _sha256(MOTION_FREEZE),
                "n_cells": _load_json(MOTION_FREEZE).get("n_cells"),
                "ddr_numbers_present": False,
                "note": _load_json(MOTION_FREEZE).get("note"),
            },
        },
        "firewall_attestation": firewall,
        "baselines": {
            "presence_only": "configs/defense/presence.yaml",
            "percepguard": "argument_only",
            "apricot": "out_of_scope_v1",
        },
        "interface": {
            "consistency": "counterusv.defense.DefensePipeline.from_freeze",
            "presence": 'counterusv.defense.harness.load_defense("presence_only")',
            "shared_decision": "DefenseDecision (flag|pass|abstain)",
        },
        "artifacts": {
            "frozen_json": _rel(args.out / "FROZEN.json"),
            "model_card": _rel(args.out / "MODEL_CARD.md"),
        },
    }

    out_json = args.out / "FROZEN.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    write_model_card(args.out / "MODEL_CARD.md", payload)
    print(f"[freeze] wrote {out_json}")
    print(f"[freeze] wrote {args.out / 'MODEL_CARD.md'}")

    if not args.skip_smoke:
        smoke(verify_digests=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
