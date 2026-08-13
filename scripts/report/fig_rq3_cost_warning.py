#!/usr/bin/env python3
"""Fig. 5 — RQ3 adaptive cost: DDR vs commit range + warning/standoff.

Panel (a) is end-of-track DDR vs ``commit_range_nm`` — the range at which the
adversary breaks off mimicry and runs in, so short = held the disguise longest
and paid the most added approach time. The axis therefore runs costly → cheap,
which the panel annotates. Fishing/sailing stay flat at 100%.

The largest commit range is the *unconstrained* end-member (burst speed,
radial, no mimicry, Δt≈0), not another sweep cell, so it is drawn detached
with an open marker. ``assert_headlines`` is what guarantees that reading: it
fails the build unless that point equals the pinned Δt≈0 headline.

Both panels colour by defense arm (``ARM_COLORS``) so the figure decodes the
same way throughout; panel (a) adds line style as a redundant arm cue for
greyscale, and greys out fishing/sailing, which are flat at 100% either way.

Panel (b) is first-flag warning medians.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

CURVES = REPO_ROOT / "results" / "adaptive_cost" / "adaptive_cost_curves.parquet"
SUMMARY = "results/adaptive_cost/adaptive_cost_summary.json"

# Shared by both panels so "arm" decodes the same way across the figure.
ARM_COLORS = {"kinematics_only": "#4c78a8", "kinematics_geometry": "#54a24b"}
ARM_LABELS = {
    "kinematics_only": "kinematics only",
    "kinematics_geometry": "kinematics + geometry",
}
FLAT_COLOR = "#8c8c8c"


def assert_headlines(df: pd.DataFrame, summary: dict) -> None:
    """Guardrail: parquet Δt≈0 / unconstrained commit must match pinned summary."""
    pinned = summary.get("recreational_delta_t_zero") or {}
    sub_dt = df[
        (df["axis"] == "delta_t_add_min")
        & (df["mimicked_class"] == "recreational")
        & (df["delta_t_add_min"].abs() < 1e-9)
    ]
    sub_c = df[
        (df["axis"] == "commit_range_nm")
        & (df["mimicked_class"] == "recreational")
        & (df["commit_range_nm"] >= 5.5)
    ]
    mismatches: list[str] = []
    for arm, block in pinned.items():
        exp = float(block["ddr"])
        row_dt = sub_dt[sub_dt["arm"] == arm]
        if row_dt.empty:
            mismatches.append(f"{arm}: no Δt≈0 row in curves parquet")
        else:
            got = float(row_dt.iloc[0]["ddr"])
            if abs(got - exp) > 1e-9:
                mismatches.append(f"{arm}: Δt≈0 curves ddr={got} vs summary={exp}")
        row_c = sub_c[sub_c["arm"] == arm]
        if not row_c.empty:
            got_c = float(row_c.iloc[0]["ddr"])
            if abs(got_c - exp) > 1e-9:
                mismatches.append(
                    f"{arm}: commit≈6 nm ddr={got_c} vs summary Δt≈0={exp}"
                )
    if mismatches:
        raise SystemExit(
            "RQ3 headline drift (curves vs pinned summary):\n  "
            + "\n  ".join(mismatches)
        )


def build_figure(out_dir: Path) -> dict:
    import matplotlib.pyplot as plt

    apply_style()
    if not CURVES.is_file():
        raise FileNotFoundError(CURVES)
    summary = load_json(SUMMARY)
    df = pd.read_parquet(CURVES)
    assert_headlines(df, summary)

    curves = df[df["axis"] == "commit_range_nm"].copy()
    warning = summary.get("warning_summary") or []

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    styles = {"kinematics_only": "-", "kinematics_geometry": "--"}

    # Largest commit range is the unconstrained end-member, not a sweep cell —
    # guaranteed by assert_headlines above. Draw it detached so the curve does
    # not imply the sweep continues through a regime change.
    commit_vals = sorted(float(v) for v in curves["commit_range_nm"].unique())
    unc_x = commit_vals[-1]

    # fishing/sailing sit flat at 100% for both arms — collapse them into one
    # muted legend entry so the recreational curves stay legible.
    flat_drawn = False
    for cls in ("fishing", "sailing", "recreational"):
        for arm in ("kinematics_only", "kinematics_geometry"):
            sub = curves[
                (curves["mimicked_class"] == cls) & (curves["arm"] == arm)
            ].sort_values("commit_range_nm")
            if sub.empty:
                continue
            if cls == "recreational":
                label = f"recreational / {ARM_LABELS[arm]}"
                color, lw, alpha, ms, z = ARM_COLORS[arm], 2.2, 1.0, 4, 3
            else:
                label = "_nolegend_" if flat_drawn else "fishing, sailing (both arms)"
                flat_drawn = True
                color, lw, alpha, ms, z = FLAT_COLOR, 1.2, 0.7, 3, 2

            swp = sub[sub["commit_range_nm"] < unc_x]
            ax_a.plot(
                swp["commit_range_nm"],
                swp["ddr"] * 100.0,
                linestyle=styles[arm],
                marker="o",
                color=color,
                linewidth=lw,
                alpha=alpha,
                markersize=ms,
                label=label,
                zorder=z,
            )
            unc = sub[sub["commit_range_nm"] >= unc_x]
            ax_a.plot(
                unc["commit_range_nm"],
                unc["ddr"] * 100.0,
                linestyle="none",
                marker="o",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.5,
                markersize=ms + 2,
                alpha=alpha,
                label="_nolegend_",
                zorder=z,
            )

    split_x = (commit_vals[-2] + unc_x) / 2.0
    ax_a.axvline(split_x, color="#bbbbbb", linewidth=0.9, linestyle=(0, (2, 3)))
    ax_a.annotate(
        "unconstrained\n(no mimicry)",
        xy=(unc_x, 50),
        xytext=(unc_x, 42),
        ha="center",
        va="center",
        fontsize=7,
        color="#555555",
    )

    ax_a.set_xlabel("Commit range (nm)")
    ax_a.set_ylabel("End-of-track DDR (%)")
    # No interior gap fits a 3-entry legend: the flat 100% lines hold the top,
    # the recreational descent cuts the middle, and the 0 rule holds the floor.
    # Reserve headroom above 100 instead and keep the ticks on the real range.
    ax_a.set_ylim(-5, 143)
    ax_a.set_yticks(range(0, 101, 20))
    panel_label(ax_a, "a")
    ax_a.legend(frameon=False, fontsize=8, loc="upper left", borderaxespad=0.2)
    ax_a.axhline(0, color="#333333", linewidth=0.6)

    # Commit range runs opposite to cost: added approach time accrues over the
    # mimicked leg, so breaking off early is the cheap attack.
    ax_a.annotate(
        "",
        xy=(1.0, -0.25),
        xytext=(0.0, -0.25),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#888888", linewidth=0.9),
    )
    ax_a.text(
        0.0,
        -0.23,
        "more mimicry · costlier",
        transform=ax_a.transAxes,
        fontsize=7,
        color="#555555",
        ha="left",
        va="bottom",
    )
    ax_a.text(
        1.0,
        -0.23,
        "less mimicry · cheaper",
        transform=ax_a.transAxes,
        fontsize=7,
        color="#555555",
        ha="right",
        va="bottom",
    )

    classes = ["fishing", "recreational", "sailing"]
    arms = ["kinematics_only", "kinematics_geometry"]
    x = np.arange(len(classes))
    width = 0.35
    lookup = {(w["mimicked_class"], w["arm"]): w for w in warning}

    r_vals_kin = [
        float((lookup.get((c, "kinematics_only")) or {}).get("R_flag_nm_median") or np.nan)
        for c in classes
    ]
    r_vals_geo = [
        float(
            (lookup.get((c, "kinematics_geometry")) or {}).get("R_flag_nm_median")
            or np.nan
        )
        for c in classes
    ]
    bars1 = ax_b.bar(
        x - width / 2,
        r_vals_kin,
        width,
        label=ARM_LABELS["kinematics_only"],
        color=ARM_COLORS["kinematics_only"],
        edgecolor="#333333",
        linewidth=0.6,
    )
    bars2 = ax_b.bar(
        x + width / 2,
        r_vals_geo,
        width,
        label=ARM_LABELS["kinematics_geometry"],
        color=ARM_COLORS["kinematics_geometry"],
        edgecolor="#333333",
        linewidth=0.6,
    )

    for i, c in enumerate(classes):
        for arm in arms:
            w = lookup.get((c, arm)) or {}
            t = w.get("t_flag_min_median")
            if t is None:
                continue
            bar = bars1[i] if arm == "kinematics_only" else bars2[i]
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{round(float(t), 1):g} min",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(classes)
    ax_b.set_ylabel("Median range at first flag (nm)")
    panel_label(ax_b, "b")
    ax_b.legend(frameon=False, fontsize=8, loc="upper left")
    ymax = max([v for v in r_vals_kin + r_vals_geo if np.isfinite(v)] + [1.0])
    ax_b.set_ylim(0, ymax * 1.30)

    fig.tight_layout()
    return save_fig(fig, "fig5_rq3_cost_warning", out_dir=out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PAPER_DIR)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    verify_freezes(skip=args.no_verify)
    meta = build_figure(args.out)
    print(f"[fig5] wrote {meta['png']} / {meta['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
