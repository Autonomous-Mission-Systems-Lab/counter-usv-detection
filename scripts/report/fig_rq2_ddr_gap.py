#!/usr/bin/env python3
"""Fig. 3 — RQ2 oracle DDR + presence gap (both consistency arms).

Source: ``results/oracle_ddr/oracle_ddr_summary.json`` (defense-freeze pinned).
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
    legend_below,
    load_json,
    panel_label,
    save_fig,
    verify_freezes,
)


def build_figure(out_dir: Path) -> dict:
    import matplotlib.pyplot as plt

    apply_style()
    oracle = load_json("results/oracle_ddr/oracle_ddr_summary.json")
    classes = ["fishing", "recreational", "sailing"]
    arms = ["kinematics_only", "kinematics_geometry"]
    arm_labels = {
        "kinematics_only": "kinematics only",
        "kinematics_geometry": "kinematics + geometry",
    }

    cons = {
        (r["mimicked_class"], r["arm"]): float(r["ddr"]) * 100.0
        for r in oracle.get("consistency_ddr") or []
    }
    presence = {
        r["mimicked_class"]: float(r["ddr"]) * 100.0
        for r in oracle.get("presence_ddr") or []
    }
    unc = {
        (r["mimicked_class"], r["arm"]): float(r["ddr"]) * 100.0
        for r in oracle.get("unconstrained_ddr") or []
    }
    x = np.arange(len(classes))
    width = 0.25
    fig, (ax, ax_ins) = plt.subplots(
        1,
        2,
        figsize=(8.4, 3.6),
        gridspec_kw={"width_ratios": [2.0, 1.25]},
    )

    # Main: presence + two arms (presence is 0 on disguise — show as gap)
    colors = {
        "presence": "#bbbbbb",
        "kinematics_only": "#4c78a8",
        "kinematics_geometry": "#54a24b",
    }
    pres_vals = [presence.get(c, 0.0) for c in classes]
    bars_p = ax.bar(
        x - width,
        # Draw a stub so the 0% presence bar is visible as a gap marker.
        [max(v, 1.5) for v in pres_vals],
        width,
        label="presence check",
        color=colors["presence"],
        edgecolor="#333333",
        linewidth=0.6,
    )
    for bar, v in zip(bars_p, pres_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            3.0,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555555",
        )
    for i, arm in enumerate(arms):
        offset = 0 if i == 0 else width
        vals = [cons.get((c, arm), 0.0) for c in classes]
        ax.bar(
            x + offset,
            vals,
            width,
            label=arm_labels[arm],
            color=colors[arm],
            edgecolor="#333333",
            linewidth=0.6,
        )
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("DDR (%)")
    ax.set_ylim(0, 112)
    panel_label(ax, "a")

    # Inset: unconstrained recreational
    unc_classes = ["fishing", "recreational", "sailing"]
    x2 = np.arange(len(unc_classes))
    w2 = 0.35
    for i, arm in enumerate(arms):
        offset = -w2 / 2 if i == 0 else w2 / 2
        vals = [unc.get((c, arm), 0.0) for c in unc_classes]
        bars = ax_ins.bar(
            x2 + offset,
            vals,
            w2,
            label=arm_labels[arm],
            color=colors[arm],
            edgecolor="#333333",
            linewidth=0.6,
        )
        for bar, v in zip(bars, vals):
            ax_ins.text(
                bar.get_x() + bar.get_width() / 2,
                max(v, 0) + 2,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax_ins.set_xticks(x2)
    ax_ins.set_xticklabels(unc_classes, fontsize=8)
    ax_ins.set_ylim(0, 112)
    panel_label(ax_ins, "b")

    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    legend_below(fig, handles, labels, ncol=3, y=0.0)
    return save_fig(fig, "fig3_rq2_ddr_gap", out_dir=out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PAPER_DIR)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    verify_freezes(skip=args.no_verify)
    meta = build_figure(args.out)
    print(f"[fig3] wrote {meta['png']} / {meta['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
