#!/usr/bin/env python3
"""Fig. 6 — FAR: kinematics calibrated targets + geometry placement FAR.

Panel (a): kinematics test FAR at calibrated operating points.
Panel (b): geometry placement FAR (operating vs sensitivity).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "report"))

from _common import (  # noqa: E402
    PAPER_DIR,
    apply_style,
    load_json,
    panel_label,
    save_fig,
    verify_freezes,
)

KIN_FAR = "results/behavior_model/far_summary.json"
GEO_FAR = "results/behavior_model_geometry/far_placement_summary.json"


def build_figure(out_dir: Path) -> dict:
    import matplotlib.pyplot as plt

    apply_style()
    kin = load_json(KIN_FAR)
    geo = load_json(GEO_FAR)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.0, 3.8))

    # Panel a: overall + a few envelopes at FAR@1/5/10%
    targets = [("far_0.01", 0.01), ("far_0.05", 0.05), ("far_0.1", 0.10)]
    test = (kin.get("by_split") or {}).get("test") or {}
    overall = (test.get("overall") or {}).get("at_calibrated") or {}
    envelopes = test.get("envelopes") or {}
    highlight = [
        "fishing",
        "recreational",
        "sailing",
        "passenger_ferry",
        "cargo_merchant",
    ]

    x = np.arange(len(targets))
    width = 0.12
    ax_a.bar(
        x - 2.5 * width,
        [float(overall.get(k, np.nan)) * 100 for k, _ in targets],
        width,
        label="overall",
        color="#333333",
        edgecolor="#111111",
        linewidth=0.5,
    )
    palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"]
    for i, env in enumerate(highlight):
        block = (envelopes.get(env) or {}).get("at_calibrated") or {}
        vals = [float(block.get(k, np.nan)) * 100 for k, _ in targets]
        ax_a.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            label=env.replace("_", " "),
            color=palette[i],
            edgecolor="#333333",
            linewidth=0.4,
        )
    # Target line markers
    for j, (_, tgt) in enumerate(targets):
        ax_a.hlines(
            tgt * 100,
            j - 0.45,
            j + 0.45,
            colors="#888888",
            linestyles=":",
            linewidth=1.0,
        )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["@1%", "@5%", "@10%"])
    ax_a.set_ylabel("Realized FAR (%)")
    panel_label(ax_a, "a")
    ax_a.set_ylim(0, 14)
    ax_a.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")

    # Panel b: placement FAR
    placements = geo.get("by_placement_class_test") or []
    operating = set(geo.get("operating_placements") or [])
    names = [p["placement_class"] for p in placements]
    fars = [float(p["far"]) * 100 for p in placements]
    colors = [
        "#54a24b" if n in operating else "#bbbbbb" for n in names
    ]
    bars = ax_b.bar(
        range(len(names)),
        fars,
        color=colors,
        edgecolor="#333333",
        linewidth=0.6,
    )
    for bar, v, p in zip(bars, fars, placements):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.3,
            f"{v:.1f}\nn={p['n_scored']}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax_b.axhline(5.0, color="#888888", linestyle=":", linewidth=1.0)
    ax_b.set_xticks(range(len(names)))
    ax_b.set_xticklabels(
        [n.replace("_", "\n") for n in names], fontsize=8
    )
    ax_b.set_ylabel("FAR (%)")
    panel_label(ax_b, "b")
    ax_b.set_ylim(0, max(fars + [5.0]) * 1.40)
    # Legend proxies
    from matplotlib.patches import Patch

    ax_b.legend(
        handles=[
            Patch(facecolor="#54a24b", edgecolor="#333333", label="operating"),
            Patch(facecolor="#bbbbbb", edgecolor="#333333", label="sensitivity"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper right",
    )

    fig.tight_layout()
    return save_fig(fig, "fig6_far", out_dir=out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PAPER_DIR)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    verify_freezes(skip=args.no_verify)
    meta = build_figure(args.out)
    print(f"[fig6] wrote {meta['png']} / {meta['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
