#!/usr/bin/env python3
"""Fig. S1 — patch-attributable ESR vs photometric severity (supplementary).

Photometric axes only: glare, spray, sea_state. Transform axes that destroy
the patch denominator (scale/rotation/blur/grazing) are omitted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "report"))

from _common import (  # noqa: E402
    DETECTOR_LABELS,
    PAPER_DIR,
    apply_style,
    legend_below,
    load_json,
    panel_label,
    save_fig,
    verify_freezes,
)

PHOTOMETRIC = ("glare", "spray", "sea_state")
LEVELS = [0, 1, 2, 3, 4]


def _series_from_path(path: str, *, access: str, family: str) -> dict:
    data = load_json(path)
    series = {}
    for axis in PHOTOMETRIC:
        ys = []
        for lv in LEVELS:
            cell = (data.get("esr") or {}).get(f"{axis}:L{lv}") or {}
            val = cell.get("esr_patch_attributable")
            if val is None:
                val = cell.get("esr", 0.0)
            ys.append(float(val) * 100.0)
        series[axis] = ys
    return {
        "access": access,
        "family": family,
        "label": f"{DETECTOR_LABELS.get(family, family)} · {access}-box",
        "series": series,
        "source": path,
    }


def collect_series() -> list[dict]:
    freeze = load_json("results/attacks/FROZEN.json")
    out: list[dict] = []
    # White-box
    for fam in ("yolo11s", "yolo11l"):
        path = f"results/attacks/evasion/{fam}/esr_by_severity.json"
        if path in (freeze.get("results") or {}):
            out.append(_series_from_path(path, access="white", family=fam))
    # Grey / black transfer ESR
    for path in freeze.get("results") or {}:
        if "transfer/evasion" not in path or not path.endswith(
            "esr_by_severity.json"
        ):
            continue
        data = load_json(path)
        out.append(
            _series_from_path(
                path,
                access=str(data.get("access_level") or "transfer"),
                family=str(data.get("surrogate") or "surrogate"),
            )
        )
    return out


def build_figure(out_dir: Path) -> dict:
    import matplotlib.pyplot as plt

    apply_style()
    series_list = collect_series()
    # Order so the legend columns group as white | grey | black.
    access_order = {"white": 0, "grey": 1, "black": 2}
    family_order = {"yolo11s": 0, "yolo11l": 1}
    series_list.sort(
        key=lambda r: (
            access_order.get(r["access"], 9),
            family_order.get(r["family"], 9),
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2), sharey=True)
    # Distinct line styles by access; color by family
    style = {
        ("yolo11s", "white"): ("#4c78a8", "-"),
        ("yolo11l", "white"): ("#f58518", "-"),
        ("yolo11s", "grey"): ("#4c78a8", "--"),
        ("yolo11l", "grey"): ("#f58518", "--"),
        ("yolo11s", "black"): ("#4c78a8", ":"),
        ("yolo11l", "black"): ("#f58518", ":"),
    }
    for ax, axis, letter in zip(axes, PHOTOMETRIC, "abc"):
        for row in series_list:
            key = (row["family"], row["access"])
            color, ls = style.get(key, ("#666666", "-."))
            ax.plot(
                LEVELS,
                row["series"][axis],
                color=color,
                linestyle=ls,
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                label=row["label"] if axis == PHOTOMETRIC[0] else None,
            )
        panel_label(ax, letter, suffix=axis.replace("_", " "), dx=0.0)
        ax.set_xlabel("Severity level")
        ax.set_xticks(LEVELS)
        ax.set_xticklabels([f"L{lv}" for lv in LEVELS])
        ax.set_xlim(-0.2, 4.2)
        ax.set_ylim(-2, 48)
    axes[0].set_ylabel("Patch-attributable ESR (%)")
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    legend_below(fig, handles, labels, ncol=3, y=0.0, handlelength=3.0)
    return save_fig(fig, "figS1_severity", out_dir=out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PAPER_DIR)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    verify_freezes(skip=args.no_verify)
    meta = build_figure(args.out)
    print(f"[figS1] wrote {meta['png']} / {meta['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
