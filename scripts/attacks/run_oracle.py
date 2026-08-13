#!/usr/bin/env python3
"""Perfect-disguise oracle — emit no-patch benign class assertions.

Models an ideal patch / zero-tech visual disguise: the defense is *told*
the contact is ``fishing`` or ``recreational``. No pixels are modified.
Defense evaluation consumes the assertion records for the oracle DDR
condition (``docs/METRICS.md``, ``docs/THREAT_MODEL.md``).

Usage
-----
    python scripts/attacks/run_oracle.py --dry-run
    python scripts/attacks/run_oracle.py --benign-class fishing
    python scripts/attacks/run_oracle.py --all-benigns --max-images 5

Writes ``results/attacks/oracle/<benign>/assertions.json`` + ``report.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.attacks.oracle import (  # noqa: E402
    PerfectDisguiseOracle,
    assertions_to_records,
    build_assertions_for_contacts,
    load_oracle_config,
)

DEFAULT_OUT = REPO_ROOT / "results" / "attacks" / "oracle"
DEFAULT_DATA = REPO_ROOT / "data"


def _usv_contacts(
    data_dir: Path,
    *,
    max_images: int | None,
    split: str = "test",
    source: str = "usv",
) -> list[dict[str, Any]]:
    """Build contact stubs from the curated ``usv`` EO split (largest box)."""
    master_path = data_dir / "annotations" / "coco_master.json"
    if not master_path.is_file():
        raise FileNotFoundError(f"missing annotations: {master_path}")
    master = json.loads(master_path.read_text())
    cats = {c["name"]: int(c["id"]) for c in master.get("categories") or []}
    if "usv" not in cats:
        raise KeyError("usv category missing from coco_master.json")
    usv_id = cats["usv"]

    ann_by_img: dict[int, list[list[float]]] = defaultdict(list)
    for a in master.get("annotations") or []:
        if int(a["category_id"]) == usv_id:
            x, y, w, h = [float(v) for v in a["bbox"]]
            ann_by_img[int(a["image_id"])].append([x, y, x + w, y + h])

    splits_path = data_dir / "splits" / "eo_image_splits.csv"
    if not splits_path.is_file():
        raise FileNotFoundError(f"missing splits: {splits_path}")
    splits = pd.read_csv(splits_path)
    sel = splits[(splits["split"] == split) & (splits["source"] == source)]
    ids = sorted(int(x) for x in sel["image_id"])

    contacts: list[dict[str, Any]] = []
    for iid in ids:
        boxes = ann_by_img.get(iid) or []
        if not boxes:
            continue
        box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        contacts.append(
            {
                "image_id": iid,
                "contact_id": iid,
                "true_class": "usv",
                "box_xyxy": box,
            }
        )
        if max_images is not None and len(contacts) >= max_images:
            break
    return contacts


def _write_report(
    out_dir: Path,
    *,
    benign: str,
    cfg: Any,
    n: int,
) -> None:
    lines = [
        f"# Perfect-disguise oracle — `{benign}`",
        "",
        f"- Generated: `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')}`",
        f"- Config: `{cfg.path}` (v{cfg.version}, frozen {cfg.frozen})",
        f"- Condition: `{cfg.condition}`",
        f"- True class: `{cfg.true_class}` → asserted `{benign}`",
        f"- Assertion score: **{cfg.assertion_score}**",
        f"- Contacts: **{n}**",
        "",
        "No pixels modified. Records are class assertions for the oracle DDR",
        "condition; pair with kinematic features at defense evaluation.",
        "",
        "Artifacts: `assertions.json`.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--benign-class",
        default="fishing",
        help="Target benign class to assert (default: fishing)",
    )
    p.add_argument("--config", type=Path, default=None, help="oracle.yaml path")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--source", default="usv")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config + list contacts; write nothing",
    )
    p.add_argument(
        "--all-benigns",
        action="store_true",
        help="Emit assertions for every target benign in the config",
    )
    args = p.parse_args()

    cfg = load_oracle_config(args.config)
    oracle = PerfectDisguiseOracle(cfg)
    benigns = (
        list(cfg.target_benign_classes)
        if args.all_benigns
        else [args.benign_class]
    )
    for b in benigns:
        oracle.validate_benign(b)

    contacts = _usv_contacts(
        args.data_dir,
        max_images=args.max_images,
        split=args.split,
        source=args.source,
    )
    print(f"oracle: {len(contacts)} {args.source} contacts (split={args.split})")
    if args.dry_run:
        print(f"dry-run OK; would assert {benigns} under {cfg.condition}")
        return 0

    for benign in benigns:
        assertions = build_assertions_for_contacts(oracle, contacts, benign)
        out = args.out_dir / benign
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "condition": cfg.condition,
            "true_class": cfg.true_class,
            "asserted_class": benign,
            "assertion_score": cfg.assertion_score,
            "config": {
                "version": cfg.version,
                "frozen": cfg.frozen,
                "path": str(cfg.path) if cfg.path else None,
            },
            "n_contacts": len(assertions),
            "assertions": assertions_to_records(assertions),
        }
        (out / "assertions.json").write_text(json.dumps(payload, indent=2) + "\n")
        _write_report(out, benign=benign, cfg=cfg, n=len(assertions))
        print(f"wrote {out / 'assertions.json'} ({len(assertions)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
