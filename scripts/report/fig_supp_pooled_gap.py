#!/usr/bin/env python3
"""Fig. S2 — Pooled vs class-conditional discriminability.

Panel (a): label-swap gap (swap − matched, pp) for headline cells under
conditional envelopes vs a single ``pooled_benign`` kinematics envelope.
Panel (b): oracle DDR for fishing / recreational / sailing, conditional vs
pooled, both consistency arms.

Sources (defense-freeze pinned):
  results/label_swap/label_swap_summary.json
  results/label_swap_pooled/label_swap_summary.json
  results/oracle_ddr/oracle_ddr_summary.json
  results/oracle_ddr_pooled/oracle_ddr_summary.json
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

ARM = "kinematics_only"
THIN_N = 20

# Headline cells from the writeup (§9.9).
HEADLINE_CELLS = [
    ("recreational", "fishing", "recreational\nas fishing"),
    ("recreational", "sailing", "recreational\nas sailing"),
    ("passenger_ferry", "fishing", "passenger ferry\nas fishing"),
]


def _gap_pp(swap: dict, matched: dict, src: str, asn: str) -> tuple[float | None, int, int, bool]:
    """Return (gap_pp, n_swap, n_matched, thin)."""
    s = swap.get((src, asn))
    m = matched.get(src)
    if not s or not m:
        return None, 0, 0, True
    if s.get("flag_rate") is None or m.get("flag_rate") is None:
        return None, int(s.get("n_scored") or 0), int(m.get("n_scored") or 0), True
    n_s = int(s.get("n_scored") or 0)
    n_m = int(m.get("n_scored") or 0)
    thin = (
        bool(s.get("thin_n"))
        or bool(m.get("thin_n"))
        or n_s < THIN_N
        or n_m < THIN_N
    )
    gap = (float(s["flag_rate"]) - float(m["flag_rate"])) * 100.0
    return gap, n_s, n_m, thin


def _index_swap(data: dict) -> tuple[dict, dict]:
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
    return swap, matched


def _ddr_map(summary: dict) -> dict[tuple[str, str], float]:
    return {
        (r["mimicked_class"], r["arm"]): float(r["ddr"]) * 100.0
        for r in summary.get("consistency_ddr") or []
    }


def build_figure(out_dir: Path) -> dict:
    import matplotlib.pyplot as plt

    apply_style()
    cond_swap = load_json("results/label_swap/label_swap_summary.json")
    pool_swap = load_json("results/label_swap_pooled/label_swap_summary.json")
    cond_ddr = load_json("results/oracle_ddr/oracle_ddr_summary.json")
    pool_ddr = load_json("results/oracle_ddr_pooled/oracle_ddr_summary.json")

    c_swap, c_matched = _index_swap(cond_swap)
    p_swap, p_matched = _index_swap(pool_swap)

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(8.6, 3.7),
        gridspec_kw={"width_ratios": [1.15, 1.35]},
    )

    # ---- Panel (a): swap−matched gaps ----
    labels = [lab for _, _, lab in HEADLINE_CELLS]
    x = np.arange(len(HEADLINE_CELLS))
    width = 0.36
    cond_gaps: list[float] = []
    pool_gaps: list[float] = []
    thin_flags: list[bool] = []
    for src, asn, _ in HEADLINE_CELLS:
        g_c, _, _, t_c = _gap_pp(c_swap, c_matched, src, asn)
        g_p, _, _, t_p = _gap_pp(p_swap, p_matched, src, asn)
        cond_gaps.append(0.0 if g_c is None else g_c)
        pool_gaps.append(0.0 if g_p is None else g_p)
        thin_flags.append(t_c or t_p)

    bars_c = ax_a.bar(
        x - width / 2,
        cond_gaps,
        width,
        label="class-conditional",
        color="#4c78a8",
        edgecolor="#333333",
        linewidth=0.6,
    )
    bars_p = ax_a.bar(
        x + width / 2,
        pool_gaps,
        width,
        label="pooled benign",
        color="#e45756",
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax_a.axhline(0.0, color="#666666", linewidth=0.7, zorder=0)
    for bars, vals in ((bars_c, cond_gaps), (bars_p, pool_gaps)):
        for bar, v, thin in zip(bars, vals, thin_flags):
            if thin:
                bar.set_hatch("///")
                bar.set_alpha(0.85)
            y = v + (1.8 if v >= 0 else -1.8)
            va = "bottom" if v >= 0 else "top"
            ax_a.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{v:+.1f}",
                ha="center",
                va=va,
                fontsize=7,
            )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel("swap − matched gap (pp)")
    ymin = min(0.0, min(cond_gaps + pool_gaps) - 8)
    ymax = max(cond_gaps + pool_gaps) + 12
    ax_a.set_ylim(ymin, ymax)
    panel_label(ax_a, "a")

    # ---- Panel (b): oracle DDR ----
    classes = ["fishing", "recreational", "sailing"]
    arms = [
        ("kinematics_only", "kin", "#4c78a8", "#9ecae1"),
        ("kinematics_geometry", "geo", "#54a24b", "#a1d99b"),
    ]
    c_map = _ddr_map(cond_ddr)
    p_map = _ddr_map(pool_ddr)
    x2 = np.arange(len(classes))
    w = 0.18
    # Offsets: kin-cond, kin-pool, geo-cond, geo-pool
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    series = [
        ("kinematics only · conditional",
         [c_map.get((c, "kinematics_only"), 0.0) for c in classes], arms[0][2], ""),
        ("kinematics only · pooled",
         [p_map.get((c, "kinematics_only"), 0.0) for c in classes], arms[0][3], "///"),
        ("kinematics + geometry · conditional",
         [c_map.get((c, "kinematics_geometry"), 0.0) for c in classes], arms[1][2], ""),
        ("kinematics + geometry · pooled",
         [p_map.get((c, "kinematics_geometry"), 0.0) for c in classes], arms[1][3], "///"),
    ]
    for off, (lab, vals, color, hatch) in zip(offsets, series):
        bars = ax_b.bar(
            x2 + off,
            vals,
            w,
            label=lab,
            color=color,
            edgecolor="#333333",
            linewidth=0.6,
            hatch=hatch or None,
        )
        for bar, v in zip(bars, vals):
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                v + 1.2,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    ax_b.set_xticks(x2)
    ax_b.set_xticklabels(classes)
    ax_b.set_ylabel("DDR (%)")
    ax_b.set_ylim(0, 118)
    panel_label(ax_b, "b")

    fig.tight_layout()
    handles_a, labels_a = ax_a.get_legend_handles_labels()
    handles_b, labels_b = ax_b.get_legend_handles_labels()
    # Combined legend under both panels
    legend_below(
        fig,
        handles_a + handles_b,
        labels_a + labels_b,
        ncol=3,
        y=-0.02,
    )
    return save_fig(fig, "figS2_pooled_gap", out_dir=out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PAPER_DIR)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    if not args.no_verify:
        verify_freezes()
    meta = build_figure(args.out)
    print(f"[figS2] wrote {meta['png']} / {meta['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
