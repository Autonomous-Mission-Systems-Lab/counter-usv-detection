#!/usr/bin/env python3
"""Fig. 4 — Real-track label-swap flag-rate heatmap + matched control.

Source: ``results/label_swap/label_swap_summary.json`` (defense-freeze pinned).
Headline arm: kinematics_only. Thin-n cells (n < 20) are hatched.
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
    save_fig,
    verify_freezes,
)

ARM = "kinematics_only"
THIN_N = 20


def _display(name: str) -> str:
    """Canonical class name for a tick label; compound names wrap to two lines."""
    return {
        "cargo_merchant": "cargo\nmerchant",
        "passenger_ferry": "passenger\nferry",
        "working_service": "working\nservice",
        "small_craft": "small\ncraft",
    }.get(name, name)


def build_figure(out_dir: Path) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    apply_style()
    data = load_json("results/label_swap/label_swap_summary.json")
    sources = list((data.get("kinematics") or {}).get("sources") or [])
    asserted = list((data.get("kinematics") or {}).get("asserted") or [])
    if not sources or not asserted:
        # Fall back from cells
        cells = [c for c in data.get("swap_cells") or [] if c["arm"] == ARM]
        sources = sorted({c["source_class"] for c in cells})
        asserted = sorted({c["asserted_class"] for c in cells})

    swap = {
        (c["source_class"], c["asserted_class"]): c
        for c in data.get("swap_cells") or []
        if c["arm"] == ARM
    }
    matched = {
        c["source_class"]: c
        for c in data.get("matched_controls") or []
        if c["arm"] == ARM
    }

    n_src = len(sources)
    n_as = len(asserted)
    # Extra column for matched control
    mat = np.full((n_src, n_as + 1), np.nan)
    thin = np.zeros((n_src, n_as + 1), dtype=bool)
    annot = [[""] * (n_as + 1) for _ in range(n_src)]

    for i, src in enumerate(sources):
        for j, asn in enumerate(asserted):
            if src == asn:
                annot[i][j] = "—"
                continue
            cell = swap.get((src, asn))
            if not cell:
                continue
            rate = cell.get("flag_rate")
            n = int(cell.get("n_scored") or 0)
            is_thin = bool(cell.get("thin_n")) or n < THIN_N
            thin[i, j] = is_thin
            if rate is None:
                continue
            mat[i, j] = float(rate) * 100.0
            mark = "†" if is_thin else ""
            annot[i][j] = f"{mat[i, j]:.0f}{mark}"

        ctrl = matched.get(src)
        if ctrl and ctrl.get("flag_rate") is not None:
            n = int(ctrl.get("n_scored") or 0)
            is_thin = bool(ctrl.get("thin_n")) or n < THIN_N
            thin[i, n_as] = is_thin
            mat[i, n_as] = float(ctrl["flag_rate"]) * 100.0
            mark = "†" if is_thin else ""
            annot[i][n_as] = f"{mat[i, n_as]:.0f}{mark}"
        elif ctrl is not None:
            annot[i][n_as] = "—"
            thin[i, n_as] = True

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    cmap = LinearSegmentedColormap.from_list(
        "flag", ["#f7fbff", "#6baed6", "#08306b"]
    )
    cmap.set_bad(color="#eeeeee")
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(n_src):
        for j in range(n_as + 1):
            if thin[i, j] and not np.isnan(mat[i, j]):
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="////",
                        edgecolor="#666666",
                        linewidth=0.4,
                    )
                )
            if annot[i][j]:
                color = (
                    "#111111"
                    if np.isnan(mat[i, j]) or mat[i, j] < 55
                    else "#ffffff"
                )
                ax.text(
                    j,
                    i,
                    annot[i][j],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )

    # Separator before matched column
    ax.axvline(n_as - 0.5, color="#111111", linewidth=1.2)
    col_labels = [_display(a) for a in asserted] + ["matched\n(true class)"]
    ax.set_xticks(range(n_as + 1))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(n_src))
    ax.set_yticklabels([_display(s) for s in sources])
    ax.set_xlabel("Asserted class")
    ax.set_ylabel("Source class (real AIS window)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Flag rate (%)")
    fig.tight_layout()
    return save_fig(fig, "fig4_label_swap", out_dir=out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PAPER_DIR)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    verify_freezes(skip=args.no_verify)
    meta = build_figure(args.out)
    print(f"[fig4] wrote {meta['png']} / {meta['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
