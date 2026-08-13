#!/usr/bin/env python3
"""Fig. 2 — RQ1 feasibility: ESR vs TMSR at L0 by access × family.

Reads digested attack freeze headlines + transfer severity JSONs.
Writes ``results/paper/fig2_rq1_feasibility.{png,pdf}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "report"))

from _common import (  # noqa: E402
    DETECTOR_LABELS,
    PAPER_DIR,
    apply_style,
    load_json,
    rel,
    save_fig,
    verify_freezes,
)


def _l0(path: str, rate_key: str) -> float:
    data = load_json(path)
    cell = (data.get(rate_key) or {}).get("scale:L0") or {}
    return float(cell.get(rate_key, 0.0))


def collect_rates() -> list[dict]:
    """Rows: family, access, metric, rate."""
    freeze = load_json("results/attacks/FROZEN.json")
    hl = freeze["headlines"]
    rows: list[dict] = []

    for fam, block in (hl.get("white_box_esr_L0") or {}).items():
        rows.append(
            {
                "family": fam,
                "access": "white",
                "metric": "ESR",
                "rate": float(block["rate"]),
                "source": block["source"],
            }
        )
    for key, block in (hl.get("white_box_tmsr_L0") or {}).items():
        # key like yolo11s_fishing
        fam, benign = key.rsplit("_", 1)
        rows.append(
            {
                "family": fam,
                "access": "white",
                "metric": f"TMSR→{benign}",
                "rate": float(block["rate"]),
                "source": block["source"],
            }
        )

    # Grey / black ESR from freeze headlines (path-keyed labels).
    for key, block in (hl.get("transfer_esr_L0") or {}).items():
        # yolo11s_to_yolo11l_grey
        access = "grey" if key.endswith("_grey") else "black"
        fam = key.split("_to_")[0]
        rows.append(
            {
                "family": fam,
                "access": access,
                "metric": "ESR",
                "rate": float(block["rate"]),
                "source": block["source"],
            }
        )

    # Transfer TMSR (fishing only in freeze) — read digested paths.
    for path in freeze.get("results") or {}:
        if "transfer/disguise" not in path or not path.endswith(
            "tmsr_by_severity.json"
        ):
            continue
        data = load_json(path)
        access = str(data.get("access_level") or "transfer")
        fam = str(data.get("surrogate") or path.split("/")[-3].split("_to_")[0])
        rows.append(
            {
                "family": fam,
                "access": access,
                "metric": "TMSR→fishing",
                "rate": _l0(path, "tmsr"),
                "source": path,
            }
        )
    return rows


def build_figure(out_dir: Path) -> dict:
    import matplotlib.pyplot as plt

    apply_style()
    rows = collect_rates()

    families = ["yolo11s", "yolo11l"]
    # Panel layout: ESR (white/grey/black) | TMSR fishing (w/g/b) | TMSR rec (white)
    metrics = [
        ("ESR", "white"),
        ("ESR", "grey"),
        ("ESR", "black"),
        ("TMSR→fishing", "white"),
        ("TMSR→fishing", "grey"),
        ("TMSR→fishing", "black"),
        ("TMSR→recreational", "white"),
    ]
    labels = [
        "ESR\nwhite-box",
        "ESR\ngrey-box",
        "ESR\nblack-box",
        "TMSR fishing\nwhite-box",
        "TMSR fishing\ngrey-box",
        "TMSR fishing\nblack-box",
        "TMSR recreational\nwhite-box",
    ]
    lookup = {
        (r["family"], r["metric"], r["access"]): r["rate"] for r in rows
    }

    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    colors = {"yolo11s": "#4c78a8", "yolo11l": "#f58518"}
    for i, fam in enumerate(families):
        vals = [lookup.get((fam, m, a), 0.0) * 100.0 for m, a in metrics]
        offset = -width / 2 if i == 0 else width / 2
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=DETECTOR_LABELS.get(fam, fam),
            color=colors[fam],
            edgecolor="#333333",
            linewidth=0.6,
        )
        for bar, v in zip(bars, vals):
            if v < 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    1.5,
                    "0",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#555555",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 34)
    ax.legend(frameon=False, loc="upper right")
    ax.axhline(0, color="#333333", linewidth=0.8)
    fig.tight_layout()
    return save_fig(fig, "fig2_rq1_feasibility", out_dir=out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PAPER_DIR)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    verify_freezes(skip=args.no_verify)
    meta = build_figure(args.out)
    print(f"[fig2] wrote {meta['png']} / {meta['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
