#!/usr/bin/env python3
"""Build / validate the EO-class → benign-envelope map + coverage table.

Reads ``configs/defense/class_envelope_map.yaml`` and the benign train
manifest, resolves each envelope's member set, and writes:

  * ``data/behavior/envelope_coverage.json`` — per-envelope track counts
  * ``results/behavior_model/envelope_map_report.md`` — human summary

Does not fit models — that is a later step. Re-run after regenerating the
benign corpus.

Usage
-----
    python scripts/behavior/build_class_envelope_map.py
    python scripts/behavior/build_class_envelope_map.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP = REPO_ROOT / "configs" / "defense" / "class_envelope_map.yaml"
DEFAULT_CORPUS = REPO_ROOT / "data" / "behavior" / "benign_train_manifest.parquet"
DEFAULT_OUT_JSON = REPO_ROOT / "data" / "behavior" / "envelope_coverage.json"
DEFAULT_OUT_MD = REPO_ROOT / "results" / "behavior_model" / "envelope_map_report.md"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def select_members(df: pd.DataFrame, members: dict) -> pd.DataFrame:
    """Filter corpus rows by envelope ``members`` predicates."""
    out = df
    for col, allowed in (members or {}).items():
        if col not in out.columns:
            raise KeyError(f"envelope member key {col!r} not in corpus columns")
        allowed_set = {str(x) for x in allowed}
        out = out.loc[out[col].astype(str).isin(allowed_set)]
    return out


def coverage_table(
    corpus: pd.DataFrame, mapping: dict,
) -> tuple[dict[str, Any], dict[str, pd.Index]]:
    """Per-envelope n_tracks / n_vessels / class×tx breakdown + row indices."""
    envelopes = mapping.get("envelopes") or {}
    cov: dict[str, Any] = {}
    indices: dict[str, pd.Index] = {}
    for name, spec in envelopes.items():
        sub = select_members(corpus, spec.get("members") or {})
        indices[name] = sub.index
        per_cls = (
            sub.groupby("canonical_class").size()
            .sort_values(ascending=False).to_dict()
            if len(sub) else {}
        )
        per_tx = (
            sub.groupby("transceiver_class").size().to_dict()
            if "transceiver_class" in sub.columns and len(sub) else {}
        )
        cov[name] = {
            "n_tracks": int(len(sub)),
            "n_vessels": int(sub["mmsi"].nunique()) if len(sub) else 0,
            "per_class": {str(k): int(v) for k, v in per_cls.items()},
            "per_transceiver": {str(k): int(v) for k, v in per_tx.items()},
            "members": spec.get("members") or {},
            "description": spec.get("description") or "",
        }
    return cov, indices


def validate_map(mapping: dict, detector_names: dict[int, str] | None) -> list[str]:
    """Return list of warnings / raise on hard errors."""
    warnings: list[str] = []
    envelopes = set((mapping.get("envelopes") or {}).keys())
    eo_map = mapping.get("eo_class_map") or {}
    scoreable = set(mapping.get("scoreable_eo_classes") or [])
    abstain = set(mapping.get("abstain_eo_classes") or [])

    for cls, entry in eo_map.items():
        pol = entry.get("policy")
        if pol == "score":
            env = entry.get("envelope")
            if env not in envelopes:
                raise KeyError(f"{cls}: envelope {env!r} not defined")
            if cls not in scoreable:
                warnings.append(f"{cls} is scoreable but missing from scoreable_eo_classes")
        elif pol == "abstain":
            if cls not in abstain:
                warnings.append(f"{cls} abstains but missing from abstain_eo_classes")
        else:
            raise ValueError(f"{cls}: unknown policy {pol!r}")

    mapped = set(eo_map)
    if scoreable | abstain != mapped:
        warnings.append(
            f"roll-up mismatch: scoreable∪abstain={sorted(scoreable|abstain)} "
            f"vs eo_class_map={sorted(mapped)}"
        )

    if detector_names:
        det = set(detector_names.values())
        for cls in det:
            if cls not in eo_map:
                raise KeyError(f"detector class {cls!r} missing from eo_class_map")
        # working_service may be in map but not in detector — fine.
    return warnings


def write_report(
    path: Path,
    *,
    mapping: dict,
    cov: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# EO-class → benign-envelope map")
    lines.append("")
    lines.append(f"Generated: {payload['timestamp']}")
    lines.append("")
    lines.append(
        "Maps each EO-asserted canonical class to a named one-class kinematic "
        "envelope (or **abstain**). Envelopes train on subsets of "
        "`data/behavior/benign_train_manifest.parquet`. Config: "
        "`configs/defense/class_envelope_map.yaml`."
    )
    lines.append("")
    lines.append("## Scoreable classes")
    lines.append("")
    lines.append("| EO class | envelope | train tracks | vessels | notes |")
    lines.append("|---|---|---:|---:|---|")
    eo_map = mapping.get("eo_class_map") or {}
    for cls in mapping.get("scoreable_eo_classes") or []:
        entry = eo_map[cls]
        env = entry["envelope"]
        c = cov[env]
        note = (entry.get("reason") or "").strip().replace("\n", " ")
        if not note:
            note = (mapping["envelopes"][env].get("notes") or "").strip().split("\n")[0]
        note = note[:80] + ("…" if len(note) > 80 else "")
        lines.append(
            f"| `{cls}` | `{env}` | {c['n_tracks']:,} | {c['n_vessels']:,} | {note} |"
        )
    lines.append("")
    lines.append("## Abstain (no envelope)")
    lines.append("")
    lines.append("| EO class | reason |")
    lines.append("|---|---|")
    for cls in mapping.get("abstain_eo_classes") or []:
        reason = (eo_map[cls].get("reason") or "").strip().replace("\n", " ")
        lines.append(f"| `{cls}` | {reason} |")
    lines.append("")
    lines.append("## Envelope member breakdown")
    lines.append("")
    for env, c in cov.items():
        lines.append(f"### `{env}` — {c['n_tracks']:,} tracks")
        lines.append("")
        lines.append(f"Members filter: `{c['members']}`")
        lines.append("")
        if c["per_class"]:
            lines.append("| source class | n |")
            lines.append("|---|---:|")
            for cls, n in c["per_class"].items():
                lines.append(f"| {cls} | {n:,} |")
            lines.append("")
        if c["per_transceiver"]:
            lines.append(f"Transceiver: {c['per_transceiver']}")
            lines.append("")
    lines.append("## Design notes")
    lines.append("")
    lines.append(
        "- **`small_craft` → `small_craft_class_b_proxy`:** Class-B ∩ "
        "{recreational, fishing, sailing}. Excludes Class-B ferry/cargo/"
        "working_service. Report FAR for this envelope separately."
    )
    lines.append(
        "- **`working_service`:** own AIS envelope; detector currently does not "
        "assert it (`exclude_classes`), so EO-side scoring is inactive today."
    )
    lines.append(
        "- **`benign_unspecified` / hostile / non_target:** abstain — no "
        "fabricated envelope."
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def load_detector_names(data_dir: Path) -> dict[int, str] | None:
    dy = data_dir / "eo_views" / "yolo" / "data.yaml"
    if not dy.is_file():
        return None
    d = _load_yaml(dy)
    names = d.get("names") or {}
    return {int(k): str(v) for k, v in names.items()}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", type=Path, default=DEFAULT_MAP)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mapping = _load_yaml(args.map)
    if not args.corpus.is_file():
        raise FileNotFoundError(
            f"Missing {args.corpus}. Run scripts/behavior/build_benign_corpus.py first."
        )
    corpus = pd.read_parquet(args.corpus)
    print(f"[envelope] corpus={len(corpus):,} tracks from {args.corpus.name}")

    det_names = load_detector_names(args.data_dir)
    warnings = validate_map(mapping, det_names)
    for w in warnings:
        print(f"[envelope] WARNING: {w}")

    cov, _ = coverage_table(corpus, mapping)
    for name, c in cov.items():
        print(f"[envelope] {name:28s}  {c['n_tracks']:6,} tracks / "
              f"{c['n_vessels']:5,} vessels  {c['per_class']}")

    # Empty envelope is a hard error (can't fit a model).
    empty = [n for n, c in cov.items() if c["n_tracks"] == 0]
    if empty:
        raise RuntimeError(f"Empty envelopes (no train tracks): {empty}")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "map": str(args.map.relative_to(REPO_ROOT)),
        "corpus": str(args.corpus.relative_to(REPO_ROOT)),
        "n_corpus": int(len(corpus)),
        "scoreable_eo_classes": list(mapping.get("scoreable_eo_classes") or []),
        "abstain_eo_classes": list(mapping.get("abstain_eo_classes") or []),
        "envelopes": cov,
        "eo_class_map": mapping.get("eo_class_map"),
        "warnings": warnings,
    }

    if args.dry_run:
        print("[envelope] dry-run — no files written")
        return 0

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    write_report(args.out_md, mapping=mapping, cov=cov, payload=payload)
    print(f"[envelope] wrote {args.out_json}")
    print(f"[envelope] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
