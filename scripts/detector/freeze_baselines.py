#!/usr/bin/env python3
"""Freeze the detector baselines for downstream attacks / defense.

Pins the ≥3 trained families as a versioned set: config + weight checksums,
clean-mAP + USV-capability headline metrics, transfer roles, and the
grey-box / white-box / black-box access assumptions. Weights stay on disk
(gitignored); the freeze records their SHA-256 so integrity can be verified
later. Human-facing detail stays in ``report.md`` / ``usv_capability.md`` —
do not duplicate it in a second freeze markdown.

Also smoke-checks that the canonical inference wrapper loads each family and
that the held-out target refuses attack crafting.

Usage
-----
    python scripts/detector/freeze_baselines.py
    python scripts/detector/freeze_baselines.py --skip-smoke
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
sys.path.insert(0, str(REPO_ROOT / "scripts" / "detector"))

from train_detector import load_family_config, roster_families  # noqa: E402

DEFAULT_WEIGHTS_ROOT = REPO_ROOT / "results" / "detector_baselines"
DEFAULT_CLEAN_MAP = DEFAULT_WEIGHTS_ROOT / "clean_map"


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
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return out or "nogit"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def resolve_weights(weights_root: Path, family: str) -> Path:
    for p in (
        weights_root / family / "weights" / "best.pt",
        weights_root / family / "best.pt",
    ):
        if p.is_file():
            return p
    raise FileNotFoundError(f"No best.pt for {family!r} under {weights_root}")


def access_assumptions() -> dict[str, Any]:
    """Grey-box / white-box / black-box assumptions (see docs/THREAT_MODEL.md)."""
    return {
        "default": "grey_box",
        "grey_box": (
            "Detector architecture family known, weights unknown — the realistic "
            "case for attack crafting against the surrogates."
        ),
        "white_box_upper_bound": (
            "Full model access (weights + grads) allowed for upper-bound attack "
            "and defense evaluation on any family, including the held-out target."
        ),
        "black_box_transfer": (
            "Attacks are crafted ONLY on surrogates (yolo11s, yolo11l) and "
            "transferred to the held-out target (rtdetr_l). The held-out family "
            "is sequestered from attack optimization; see "
            "docs/TRANSFER_PROTOCOL.md and configs/detector/families.yaml."
        ),
        "held_out_sequestered": True,
        "docs": ["docs/THREAT_MODEL.md", "docs/TRANSFER_PROTOCOL.md"],
    }


def headline_from_clean(clean: dict | None) -> dict[str, Any]:
    if not clean:
        return {}
    shore = (clean.get("slices") or {}).get("shore_operational") or {}
    usv_src = (clean.get("slices") or {}).get("source:usv") or {}
    per = shore.get("per_class") or {}
    usv_cls = per.get("usv") or {}
    return {
        "shore_mAP_50_95": shore.get("mAP_50_95"),
        "shore_mAP_50": shore.get("mAP_50"),
        "usv_source_mAP_50_95": usv_src.get("mAP_50_95"),
        "usv_class_AP_50_95": usv_cls.get("AP_50_95"),
        "usv_class_AP_50": usv_cls.get("AP_50"),
        "n_images_shore": shore.get("n_images"),
        "n_gt_shore": shore.get("n_gt"),
    }


def smoke_check(families: dict[str, Any]) -> None:
    from counterusv.models import DetectorBaseline

    print("[freeze] smoke: loading wrappers…")
    for fam, e in families.items():
        det = DetectorBaseline.from_freeze(fam)
        assert det.family == fam
        assert det.transfer_role == e["transfer_role"]
        assert det.attack_crafting_allowed == e["attack_crafting_allowed"]
        if e["attack_crafting_allowed"]:
            det.assert_attack_crafting_allowed()  # must not raise
            print(f"  {fam}: surrogate OK (crafting allowed)")
        else:
            try:
                det.assert_attack_crafting_allowed()
                raise AssertionError(f"{fam} should refuse crafting")
            except PermissionError:
                print(f"  {fam}: held-out OK (crafting refused)")
    print("[freeze] smoke: passed")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-root", type=Path, default=DEFAULT_WEIGHTS_ROOT)
    ap.add_argument("--clean-map", type=Path, default=DEFAULT_CLEAN_MAP)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--skip-smoke", action="store_true",
                    help="skip wrapper load / sequestration smoke check")
    args = ap.parse_args()

    usv_cap = _load_json(args.clean_map / "usv_capability.json") or {}
    usv_by_fam = (usv_cap.get("families") or {})

    families_out: dict[str, Any] = {}
    for entry in roster_families():
        fam = entry["name"]
        cfg_path = (REPO_ROOT / entry["config"]).resolve()
        cfg = load_family_config(cfg_path)
        det = cfg.get("detector") or {}
        role = str(entry.get("role") or det.get("transfer_role") or "surrogate")
        weights = resolve_weights(args.weights_root, fam)
        clean = _load_json(args.clean_map / f"{fam}_s{args.imgsz}.json")
        metrics = headline_from_clean(clean)
        # Prefer USV capability AP when available (same slice, clearer).
        uc = usv_by_fam.get(fam) or {}
        if uc.get("ap_50") is not None:
            metrics["usv_class_AP_50"] = uc["ap_50"]
            metrics["usv_class_AP_50_95"] = uc.get("ap_50_95")
            op = uc.get("operating_point") or {}
            metrics["usv_presence_recall"] = op.get("presence_recall_any_class")
            metrics["usv_recognition_recall"] = op.get("recognition_recall_usv_class")

        w_sha = _sha256(weights)
        c_sha = _sha256(cfg_path)
        print(f"[freeze] {fam}: weights={weights.name} sha={w_sha[:12]}… "
              f"role={role}")

        families_out[fam] = {
            "arch": str(det.get("arch", "yolo11")),
            "scale": det.get("scale"),
            "transfer_role": role,
            "attack_crafting_allowed": role != "held_out_target",
            "config": str(cfg_path.relative_to(REPO_ROOT)),
            "config_sha256": c_sha,
            "weights": str(weights),
            "weights_rel": str(weights.relative_to(REPO_ROOT)),
            "weights_sha256": w_sha,
            "weights_bytes": weights.stat().st_size,
            "imgsz": int((cfg.get("train") or {}).get("imgsz", args.imgsz)),
            "metrics": metrics,
            "clean_map_json": (
                str((args.clean_map / f"{fam}_s{args.imgsz}.json")
                    .relative_to(REPO_ROOT))
                if clean else None
            ),
        }

    version = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    payload = {
        "version": version,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "imgsz": args.imgsz,
        "class_space": {
            "n_classes": 11,
            "data_yaml": "data/eo_views/yolo/data.yaml",
            "policy": "configs/base.yaml detector (exclude working_service; "
                      "non_target keep)",
            "interface": [
                "box_xyxy", "score", "class_name", "class_id", "role",
            ],
        },
        "transfer_protocol": {
            "surrogates": [
                n for n, e in families_out.items()
                if e["transfer_role"] == "surrogate"
            ],
            "held_out_target": next(
                (n for n, e in families_out.items()
                 if e["transfer_role"] == "held_out_target"),
                None,
            ),
            "doc": "docs/TRANSFER_PROTOCOL.md",
        },
        "access_assumptions": access_assumptions(),
        "families": families_out,
        "artifacts": {
            "clean_map_report": "results/detector_baselines/report.md",
            "usv_capability": "results/detector_baselines/usv_capability.md",
            "train_summary": "results/detector_baselines/train_summary.json",
        },
    }

    out_json = args.weights_root / "FROZEN.json"
    args.weights_root.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[freeze] wrote {out_json}")
    print("[freeze] human detail: report.md + usv_capability.md (not duplicated here)")

    if not args.skip_smoke:
        smoke_check(families_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
