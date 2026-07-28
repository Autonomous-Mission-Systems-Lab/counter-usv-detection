#!/usr/bin/env python3
"""Freeze the benign-behavior model + ConsistencyScorer for downstream.

Pins fitted multi-horizon envelopes, feature/envelope-map configs, FAR
calibration headlines, and a firewall attestation (no hostile / ``usv`` /
non_target in training). Writes ``FROZEN.json`` (machine pin + digests) and
``MODEL_CARD.md`` (human prose). Fit/FAR detail stays in ``fit_report.md`` /
``far_report.md`` — not re-copied into a second freeze markdown.

Envelope ``.joblib`` files stay on disk (gitignored under ``results/``);
integrity is via SHA-256 digests in ``FROZEN.json``.

Usage
-----
    python scripts/behavior/freeze_behavior_model.py
    python scripts/behavior/freeze_behavior_model.py --skip-smoke
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

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_ROOT = REPO_ROOT / "results" / "behavior_model"
DEFAULT_ENVELOPE_DIR = DEFAULT_ROOT / "envelopes"
DEFAULT_MAP = REPO_ROOT / "configs" / "defense" / "class_envelope_map.yaml"
DEFAULT_FEATURES = REPO_ROOT / "configs" / "defense" / "scorer_features.yaml"
DEFAULT_MODEL_CFG = REPO_ROOT / "configs" / "defense" / "behavior_model.yaml"
DEFAULT_CORPUS = REPO_ROOT / "data" / "behavior" / "benign_corpus_summary.json"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "behavior" / "benign_train_manifest.parquet"
DEFAULT_FIT = DEFAULT_ROOT / "fit_summary.json"
DEFAULT_FAR = DEFAULT_ROOT / "far_summary.json"


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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def attest_firewall(manifest: Path, corpus: dict[str, Any]) -> dict[str, Any]:
    """Re-check the benign train manifest; pair with corpus summary counts."""
    import pandas as pd

    from counterusv.defense import FirewallError, filter_benign_training

    if not manifest.is_file():
        raise FileNotFoundError(f"Missing benign train manifest: {manifest}")

    df = pd.read_parquet(manifest)
    n = len(df)
    # Hard counts
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

    # filter_benign_training with raise_on_blocked must keep all rows
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
        "corpus_summary": {
            "n_benign_train": corpus.get("n_benign_train"),
            "n_excluded_train": corpus.get("n_excluded_train"),
            "excluded_train_by_role": corpus.get("excluded_train_by_role"),
            "excluded_train_by_class": corpus.get("excluded_train_by_class"),
            "benign_train_per_class": corpus.get("benign_train_per_class"),
            "benign_train_per_transceiver": corpus.get(
                "benign_train_per_transceiver"
            ),
        },
        "statement": (
            "Benign train manifest is role==benign only; 0 hostile / 0 "
            "non_target / 0 usv / 0 military rows. Hostile and adaptive "
            "trajectories are evaluation-only and never enter scorer training "
            "or calibration."
        ),
    }


def headline_far(far: dict[str, Any]) -> dict[str, Any]:
    test = (far.get("by_split") or {}).get("test") or {}
    overall = test.get("overall") or {}
    at = overall.get("at_calibrated") or {}
    per_env: dict[str, Any] = {}
    for name, block in (test.get("envelopes") or {}).items():
        cal = block.get("at_calibrated") or {}
        per_env[name] = {
            "n_scored": block.get("n_scored"),
            "coverage": block.get("coverage"),
            "far_at_0.01": cal.get("far_0.01"),
            "far_at_0.05": cal.get("far_0.05"),
            "far_at_0.1": cal.get("far_0.1"),
        }
    return {
        "primary_window_s": far.get("primary_window_s"),
        "windows_s": far.get("windows_s"),
        "default_far": far.get("default_far"),
        "test_n_scored": overall.get("n_scored"),
        "test_far_at_0.01": at.get("far_0.01"),
        "test_far_at_0.05": at.get("far_0.05"),
        "test_far_at_0.1": at.get("far_0.1"),
        "per_envelope_test": per_env,
        "far_report": "results/behavior_model/far_report.md",
        "far_summary": "results/behavior_model/far_summary.json",
    }


def envelope_roster(
    envelope_dir: Path,
    emap: dict[str, Any],
    fit: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    fit_envs = fit.get("envelopes") or {}
    for name in emap.get("envelopes") or {}:
        path = envelope_dir / f"{name}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Missing envelope bundle: {path}")
        entry_fit = fit_envs.get(name) or {}
        subspaces = (entry_fit.get("subspaces") or {}).get("core") or {}
        gmm = subspaces.get("gmm") or {}
        k_by_h: dict[str, Any] = {}
        windows_meta = entry_fit.get("horizons") or {}
        if isinstance(windows_meta, dict):
            for ws, wblock in windows_meta.items():
                if not isinstance(wblock, dict):
                    continue
                if wblock.get("gmm_k") is not None:
                    k_by_h[str(ws)] = wblock.get("gmm_k")
                else:
                    core = (
                        ((wblock.get("subspaces") or {}).get("core") or {}).get(
                            "gmm"
                        )
                        or {}
                    )
                    if core.get("n_components") is not None:
                        k_by_h[str(ws)] = core.get("n_components")

        out[name] = {
            "path": _rel(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "primary_gmm_n_components": gmm.get("n_components")
            or k_by_h.get(str(entry_fit.get("primary_window_s") or 300)),
            "primary_n_train": gmm.get("n_train"),
            "primary_n_val": gmm.get("n_val"),
            "gmm_k_by_horizon": k_by_h or None,
            "members": entry_fit.get("members"),
        }
    return out


def policy_tables(emap: dict[str, Any]) -> dict[str, Any]:
    scoreable: list[dict[str, Any]] = []
    abstain: list[dict[str, Any]] = []
    for cls, entry in (emap.get("eo_class_map") or {}).items():
        policy = str(entry.get("policy") or "abstain")
        row = {
            "eo_class": cls,
            "policy": policy,
            "envelope": entry.get("envelope"),
            "reason": entry.get("reason"),
        }
        if policy == "score":
            scoreable.append(row)
        else:
            abstain.append(row)
    return {"scoreable": scoreable, "abstain": abstain}


def declared_interface() -> dict[str, Any]:
    return {
        "api": "counterusv.defense.ConsistencyScorer",
        "load": "ConsistencyScorer.from_freeze()",
        "score_inputs": [
            "asserted_class (str) or Detection.class_name",
            "track features (Mapping / Series) or features_by_window",
        ],
        "score_outputs": [
            "ConsistencyResult.status",
            "score (anomaly; higher = more inconsistent)",
            "is_inconsistent (vs FAR threshold)",
            "envelope_used",
            "window_s",
            "subspace",
        ],
        "far_target_knob": True,
        "default_far": 0.05,
        "multi_horizon": "longest complete among windows_s",
        "firewall": (
            "purpose='train' / assert_benign_train_allowed / "
            "filter_benign_training refuse hostile|usv|non_target; "
            "purpose='defense'|'eval' may score any contact"
        ),
        "consumers": ["Phase 5 defense wiring", "Phase 6 evaluation"],
    }


def write_model_card(path: Path, payload: dict[str, Any]) -> None:
    pol = payload["policy"]
    far = payload["far_floor"]
    fw = payload["firewall_attestation"]
    feat = payload["features"]
    lines = [
        "# Benign-behavior model card",
        "",
        f"Freeze version: `{payload['version']}`  ·  "
        f"frozen {payload['frozen_utc']}  ·  git `{payload['git_sha']}`",
        "",
        "Machine pin + SHA-256 digests: **`FROZEN.json`** "
        "(load via `ConsistencyScorer.from_freeze()`). Fit/FAR detail: "
        "`fit_report.md`, `far_report.md` — not duplicated here.",
        "",
        "## Intended use",
        "",
        "Class–kinematics consistency scoring for a shore-based counter-USV "
        "defense: given an EO-asserted vessel class and a contact's track "
        "features, return an anomaly score against a **one-class benign "
        "envelope** learned from real AIS. Downstream: Phase 5 defense "
        "wiring and Phase 6 evaluation (RQ2 FAR axis; DDR under "
        "perfect-disguise oracle and patch conditions).",
        "",
        "## Training data",
        "",
        "- Source: MarineCadastre AIS (US national, 2023-06-01…07), "
        "**train ∩ role==benign** only.",
        f"- Tracks / vessels: **{fw['n_benign_train']:,}** "
        f"(see corpus summary).",
        "- **Firewall:** no hostile / `usv` / non_target / military in "
        "training or calibration (attested at freeze).",
        "- AIS is **offline only** — never a runtime input to the defense.",
        "",
        "## Scoreable vs abstain",
        "",
        "| EO class | policy | envelope |",
        "|---|---|---|",
    ]
    for row in pol["scoreable"]:
        lines.append(
            f"| `{row['eo_class']}` | score | `{row.get('envelope')}` |"
        )
    for row in pol["abstain"]:
        lines.append(f"| `{row['eo_class']}` | abstain | — |")
    lines += [
        "",
        "`small_craft` has **no AIS ship-type code**; it uses the "
        "`small_craft_class_b_proxy` envelope (Class-B ∩ "
        "{recreational, fishing, sailing}). Do not equate the proxy with a "
        "labeled AIS class.",
        "",
        "## Features & horizons",
        "",
        f"- Core: `{feat.get('core')}`",
        f"- Course (COG-gated non-null): `{feat.get('course')}`",
        f"- Excluded: `{feat.get('excluded')}`",
        f"- Windows (s): **{far.get('windows_s')}** "
        f"(primary **{far.get('primary_window_s')}**); score-time policy = "
        "longest complete window.",
        "- Model: primary **GMM** (val-loglik knee for `k`); baselines "
        "Mahalanobis + IsolationForest fitted but not used by the default "
        "scorer path.",
        "",
        "## FAR floor (held-out test)",
        "",
        f"- Overall FAR @1/5/10%: "
        f"**{100 * far['test_far_at_0.01']:.1f}% / "
        f"{100 * far['test_far_at_0.05']:.1f}% / "
        f"{100 * far['test_far_at_0.1']:.1f}%** "
        f"(n_scored={far['test_n_scored']:,}).",
        "- Thresholds calibrated on **val** only; test is vessel-disjoint.",
        "- Coverage ~61–90% of test contacts by envelope (short tracks "
        "abstain if no complete window).",
        "",
        "## Known biases & limitations",
        "",
        "- **AIS carriage / self-report bias** — FAR is on cooperative AIS; "
        "non-cooperative (SMD) check is deferred to evaluation (image-plane "
        "only; world-frame calibration failed).",
        "- **Weak-disguise surface** — `recreational` is kinematically "
        "permissive (speedboat-like); a high-SOG straight profile may also "
        "fit `passenger_ferry` (HSC-like). Report per-class discriminability; "
        "do not hide it.",
        "- **Region / season** — single June 2023 week; large regions near "
        "5% FAR band, tiny Pacific cells are noise.",
        "- **Heading** — Class-B heading often missing; heading-derived "
        "features excluded; course features gated on non-null COG.",
        "- **No hostile DDR in this freeze** — separability preview only; "
        "detection rate is an evaluation claim.",
        "- **SMD tracks** were not used to train or calibrate this model.",
        "",
        "## Ethical / dual-use",
        "",
        "See `docs/DUAL_USE.md`. This artifact is a benign-behavior model "
        "and scoring API — not an attack. Release centers on the evaluation "
        "harness and defended models.",
        "",
        "## Load",
        "",
        "```python",
        "from counterusv.defense import ConsistencyScorer",
        "scorer = ConsistencyScorer.from_freeze()",
        "```",
        "",
    ]
    path.write_text("\n".join(lines))


def smoke_check(payload: dict[str, Any]) -> None:
    from counterusv.defense import ConsistencyScorer, FirewallError

    print("[freeze] smoke: ConsistencyScorer.from_freeze() …")
    scorer = ConsistencyScorer.from_freeze(
        freeze_path=DEFAULT_ROOT / "FROZEN.json",
        verify_digests=True,
    )
    assert len(scorer.envelopes) == len(payload["envelopes"])
    # Abstain path
    r = scorer.score("usv", {"sog_med": 1.0})
    assert r.status == "abstain", r
    # Firewall train refusal
    try:
        scorer.score(
            "fishing",
            {"sog_med": 1.0},
            purpose="train",
            track_meta={"role": "hostile", "canonical_class": "military"},
        )
        raise AssertionError("expected FirewallError")
    except FirewallError:
        print("  firewall train refusal OK")
    # Eval path accepts hostile meta
    r2 = scorer.score(
        "fishing",
        {
            "sog_med": 35.0,
            "sog_p95": 36.0,
            "sog_std": 1.0,
            "loiter_frac": 0.0,
            "straightness": 0.99,
            "accel_mean_abs": 0.2,
        },
        purpose="eval",
        track_meta={"role": "hostile", "source": "synth"},
    )
    assert r2.status in {"scored", "nan_features", "no_window"}, r2
    print(f"  score path status={r2.status} OK")
    print("[freeze] smoke: passed")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--envelope-dir", type=Path, default=DEFAULT_ENVELOPE_DIR)
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    emap = _load_yaml(DEFAULT_MAP)
    feat_yaml = _load_yaml(DEFAULT_FEATURES)
    model_cfg = _load_yaml(DEFAULT_MODEL_CFG)
    corpus = _load_json(DEFAULT_CORPUS)
    fit = _load_json(DEFAULT_FIT)
    far = _load_json(DEFAULT_FAR)

    print("[freeze] firewall attestation …")
    attestation = attest_firewall(DEFAULT_MANIFEST, corpus)
    print(f"  PASS — n_benign_train={attestation['n_benign_train']:,}")

    envelopes = envelope_roster(args.envelope_dir, emap, fit)
    print(f"[freeze] {len(envelopes)} envelope bundles digested")

    # Feature names from model cfg / fit
    core = list(
        (model_cfg.get("features") or {}).get("core")
        or fit.get("core_features")
        or []
    )
    course = list(
        (model_cfg.get("features") or {}).get("course")
        or fit.get("course_features")
        or []
    )
    excluded = list(
        (feat_yaml.get("excluded") or {}).keys()
        if isinstance(feat_yaml.get("excluded"), dict)
        else (feat_yaml.get("excluded") or [])
    )

    version = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    payload: dict[str, Any] = {
        "version": version,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "primary_model": (model_cfg.get("models") or {}).get("primary", "gmm"),
        "windows_s": far.get("windows_s")
        or (model_cfg.get("windows") or {}).get("windows_s")
        or [120, 180, 300],
        "primary_window_s": far.get("primary_window_s") or 300,
        "configs": {
            "class_envelope_map": {
                "path": _rel(DEFAULT_MAP),
                "sha256": _sha256(DEFAULT_MAP),
            },
            "scorer_features": {
                "path": _rel(DEFAULT_FEATURES),
                "sha256": _sha256(DEFAULT_FEATURES),
            },
            "behavior_model": {
                "path": _rel(DEFAULT_MODEL_CFG),
                "sha256": _sha256(DEFAULT_MODEL_CFG),
            },
        },
        "features": {
            "core": core,
            "course": course,
            "excluded": excluded,
            "contract": _rel(DEFAULT_FEATURES),
        },
        "policy": policy_tables(emap),
        "envelopes": envelopes,
        "firewall_attestation": attestation,
        "far_floor": headline_far(far),
        "interface": declared_interface(),
        "artifacts": {
            "fit_report": "results/behavior_model/fit_report.md",
            "fit_summary": "results/behavior_model/fit_summary.json",
            "far_report": "results/behavior_model/far_report.md",
            "far_summary": "results/behavior_model/far_summary.json",
            "corpus_report": "results/behavior_model/corpus_report.md",
            "envelope_map_report": "results/behavior_model/envelope_map_report.md",
            "model_card": "results/behavior_model/MODEL_CARD.md",
        },
    }

    args.root.mkdir(parents=True, exist_ok=True)
    out_json = args.root / "FROZEN.json"
    out_card = args.root / "MODEL_CARD.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    write_model_card(out_card, payload)
    print(f"[freeze] wrote {out_json}")
    print(f"[freeze] wrote {out_card}")

    if not args.skip_smoke:
        smoke_check(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
