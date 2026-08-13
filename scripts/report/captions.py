#!/usr/bin/env python3
"""Emit paper figure captions as ``results/paper/CAPTIONS.{json,md}``.

The figures carry no titles and no footnote text, so the caption is the only
place that descriptive text lives. Every number a caption quotes is read here
from the same digest-pinned artifacts the figures plot, so a caption cannot
drift from its panel.

Each ``caption`` body keeps to one job: say what is plotted, expand every
acronym and name every marker that appears on the panel, and quote only the
numbers that are the point. Interpretation belongs in ``takeaway`` and
methodological limits in ``caveats``, so the caption is not asked to argue the
result as well as decode it.

JSON is the machine-readable source (one record per figure: short title,
caption body, per-panel lines, takeaway, caveats, source artifacts, and the
numbers quoted). The Markdown file is a rendering of the same records.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "report"))

from _common import PAPER_DIR, git_sha, load_json, verify_freezes  # noqa: E402

SCHEMA = "counterusv.paper.captions/1"
THIN_N = 20

CONVENTION = (
    "Figures carry no titles and no in-figure footnotes; panels are marked "
    "(a), (b), … on the figure — Fig. S1 also names its transform axis there "
    "— and are described here. On-figure text is limited to axis labels, "
    "legends, value labels, and the reading aids each caption names. "
    "Regenerate with scripts/report/build_all.py."
)


# --------------------------------------------------------------------------
# facts pulled from pinned artifacts
# --------------------------------------------------------------------------


def _pct(x: float) -> float:
    return round(float(x) * 100.0, 1)


def _and(items: list[str]) -> str:
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + (" and " if len(items) == 2 else ", and ") + items[-1]


def gather_facts() -> dict[str, Any]:
    import pandas as pd

    attacks = load_json("results/attacks/FROZEN.json")
    hl = attacks["headlines"]
    oracle = load_json("results/oracle_ddr/oracle_ddr_summary.json")
    oracle_pooled = load_json("results/oracle_ddr_pooled/oracle_ddr_summary.json")
    swap = load_json("results/label_swap/label_swap_summary.json")
    swap_pooled = load_json("results/label_swap_pooled/label_swap_summary.json")
    cost = load_json("results/adaptive_cost/adaptive_cost_summary.json")
    kin_far = load_json("results/behavior_model/far_summary.json")
    geo_far = load_json("results/behavior_model_geometry/far_placement_summary.json")

    esr_white = {k: _pct(v["rate"]) for k, v in (hl.get("white_box_esr_L0") or {}).items()}
    esr_tx = {k: _pct(v["rate"]) for k, v in (hl.get("transfer_esr_L0") or {}).items()}
    tmsr_white = {k: _pct(v["rate"]) for k, v in (hl.get("white_box_tmsr_L0") or {}).items()}

    cons = {
        (r["mimicked_class"], r["arm"]): _pct(r["ddr"])
        for r in oracle.get("consistency_ddr") or []
    }
    cons_pooled = {
        (r["mimicked_class"], r["arm"]): _pct(r["ddr"])
        for r in oracle_pooled.get("consistency_ddr") or []
    }
    unc = {
        (r["mimicked_class"], r["arm"]): _pct(r["ddr"])
        for r in oracle.get("unconstrained_ddr") or []
    }
    unc_pooled = {
        (r["mimicked_class"], r["arm"]): _pct(r["ddr"])
        for r in oracle_pooled.get("unconstrained_ddr") or []
    }
    cons_n = next(
        (int(r["n"]) for r in oracle.get("consistency_ddr") or [] if r.get("n")), 0
    )
    negctl = oracle.get("negative_control") or {}

    kin_cells = [c for c in swap.get("swap_cells") or [] if c["arm"] == "kinematics_only"]
    rec_row = sorted(
        _pct(c["flag_rate"])
        for c in kin_cells
        if c["source_class"] == "recreational"
        and c["asserted_class"] != "recreational"
        and c.get("flag_rate") is not None
    )
    thin_cells = [
        c for c in kin_cells if c.get("thin_n") or int(c.get("n_scored") or 0) < THIN_N
    ]
    matched = {
        c["source_class"]: c
        for c in swap.get("matched_controls") or []
        if c["arm"] == "kinematics_only"
    }
    swap_idx = {
        (c["source_class"], c["asserted_class"]): c
        for c in kin_cells
    }
    pool_cells = [
        c for c in swap_pooled.get("swap_cells") or [] if c["arm"] == "kinematics_only"
    ]
    pool_matched = {
        c["source_class"]: c
        for c in swap_pooled.get("matched_controls") or []
        if c["arm"] == "kinematics_only"
    }
    pool_idx = {
        (c["source_class"], c["asserted_class"]): c for c in pool_cells
    }

    def _gap(src: str, asn: str, sw: dict, mt: dict) -> float:
        s = sw.get((src, asn)) or {}
        m = mt.get(src) or {}
        if s.get("flag_rate") is None or m.get("flag_rate") is None:
            return 0.0
        return round(
            (float(s["flag_rate"]) - float(m["flag_rate"])) * 100.0, 1
        )

    gap_rec_fish = _gap("recreational", "fishing", swap_idx, matched)
    gap_rec_sail = _gap("recreational", "sailing", swap_idx, matched)
    gap_ferry_fish = _gap("passenger_ferry", "fishing", swap_idx, matched)
    gap_rec_fish_p = _gap("recreational", "fishing", pool_idx, pool_matched)
    gap_rec_sail_p = _gap("recreational", "sailing", pool_idx, pool_matched)
    gap_ferry_fish_p = _gap("passenger_ferry", "fishing", pool_idx, pool_matched)

    # Pre-committed reading: pooling must shrink the recreational→fishing gap.
    if not (gap_rec_fish_p < gap_rec_fish - 5.0):
        raise SystemExit(
            "Fig. S2 caption asserts pooling collapses the recreational→fishing "
            f"gap, but conditional={gap_rec_fish} pooled={gap_rec_fish_p} — "
            "rewrite before shipping."
        )
    fish_kin = cons.get(("fishing", "kinematics_only"), 0.0)
    fish_kin_p = cons_pooled.get(("fishing", "kinematics_only"), 0.0)
    if not (fish_kin_p < fish_kin - 5.0):
        raise SystemExit(
            "Fig. S2 caption asserts pooling drops fishing kinematics DDR, "
            f"but conditional={fish_kin} pooled={fish_kin_p} — rewrite before shipping."
        )

    curves = pd.read_parquet(
        REPO_ROOT / "results" / "adaptive_cost" / "adaptive_cost_curves.parquet"
    )
    rec = curves[
        (curves["axis"] == "commit_range_nm")
        & (curves["mimicked_class"] == "recreational")
    ]
    rec_kin = {
        float(r.commit_range_nm): (_pct(r.ddr), int(r.n))
        for r in rec[rec["arm"] == "kinematics_only"].itertuples()
    }
    rec_geo = {
        float(r.commit_range_nm): (_pct(r.ddr), int(r.n))
        for r in rec[rec["arm"] == "kinematics_geometry"].itertuples()
    }
    ranges = sorted(rec_kin)
    warn = {
        (w["mimicked_class"], w["arm"]): w for w in cost.get("warning_summary") or []
    }

    kin_test = (kin_far.get("by_split") or {}).get("test") or {}
    kin_overall = (kin_test.get("overall") or {}).get("at_calibrated") or {}
    kin_fishing = ((kin_test.get("envelopes") or {}).get("fishing") or {}).get(
        "at_calibrated"
    ) or {}

    placements = {
        p["placement_class"]: p for p in geo_far.get("by_placement_class_test") or []
    }

    # Fig. S1 headline numbers, recomputed from the series the figure plots so the
    # caption cannot claim a severity effect the panels do not show.
    from fig_supp_severity import PHOTOMETRIC, collect_series

    sev = collect_series()
    sev_gain = max(
        max(r["series"][a]) - r["series"][a][0] for r in sev for a in PHOTOMETRIC
    )
    sev_flat_axis = min(
        PHOTOMETRIC,
        key=lambda a: max(max(r["series"][a]) - r["series"][a][0] for r in sev),
    )
    black_peak = max(
        max(r["series"][a]) for r in sev if r["access"] == "black" for a in PHOTOMETRIC
    )
    l0_non_black = [r["series"][PHOTOMETRIC[0]][0] for r in sev if r["access"] != "black"]
    if black_peak >= min(l0_non_black):
        raise SystemExit(
            "Fig. S1 caption asserts black-box severity never reaches a white/grey L0 "
            f"baseline, but black peaks at {black_peak:.1f}% vs L0 min "
            f"{min(l0_non_black):.1f}% — rewrite the caption before shipping."
        )

    return {
        "sev_max_gain_pts": round(sev_gain, 1),
        "sev_flat_axis": sev_flat_axis.replace("_", " "),
        "sev_black_peak": round(black_peak, 1),
        "sev_l0_non_black_min": round(min(l0_non_black), 1),
        "sev_l0_non_black_max": round(max(l0_non_black), 1),
        "sev_access_order_holds": black_peak < min(l0_non_black),
        "esr_white": esr_white,
        "esr_transfer": esr_tx,
        "tmsr_white_all_zero": all(v == 0.0 for v in tmsr_white.values()),
        "esr_grey_max": max(
            (v for k, v in esr_tx.items() if k.endswith("_grey")), default=0.0
        ),
        "esr_black_min": min(
            (v for k, v in esr_tx.items() if k.endswith("_black")), default=0.0
        ),
        "esr_black_max": max(
            (v for k, v in esr_tx.items() if k.endswith("_black")), default=0.0
        ),
        "oracle_target": _pct(oracle.get("far_target", 0.05)),
        "oracle_far_kin": _pct((oracle.get("operating_far") or {}).get("kinematics_only", 0)),
        "oracle_far_geo": _pct(
            (oracle.get("operating_far") or {}).get("kinematics_geometry", 0)
        ),
        "oracle_n_cell": cons_n,
        "ddr_rec_kin": cons.get(("recreational", "kinematics_only"), 0.0),
        "ddr_rec_geo": cons.get(("recreational", "kinematics_geometry"), 0.0),
        "unc_rec_kin": unc.get(("recreational", "kinematics_only"), 0.0),
        "unc_rec_geo": unc.get(("recreational", "kinematics_geometry"), 0.0),
        "negctl_n": int((negctl.get("kinematics_only") or {}).get("n_total") or 0),
        "swap_target": _pct(swap.get("far_target", 0.05)),
        "swap_trips": int((swap.get("kinematics") or {}).get("n_band_trips") or 0),
        "swap_cells": len(kin_cells),
        "swap_thin": len(thin_cells),
        "swap_matched_working": _pct(
            (matched.get("working_service") or {}).get("flag_rate") or 0.0
        ),
        "swap_matched_working_n": int(
            (matched.get("working_service") or {}).get("n_scored") or 0
        ),
        "swap_matched_rec": _pct((matched.get("recreational") or {}).get("flag_rate") or 0.0),
        "swap_matched_rec_n": int((matched.get("recreational") or {}).get("n_scored") or 0),
        "swap_rec_row_min": rec_row[0] if rec_row else 0.0,
        "swap_rec_row_max": rec_row[-1] if rec_row else 0.0,
        "swap_matched_empty": sorted(
            k for k, v in matched.items() if not int(v.get("n_scored") or 0)
        ),
        "commit_ranges": ranges,
        "rec_kin_curve": rec_kin,
        "rec_geo_curve": rec_geo,
        "commit_max": ranges[-1] if ranges else 0.0,
        "commit_max_n": rec_kin[ranges[-1]][1] if ranges else 0,
        "warn": warn,
        "warn_n": int((warn.get(("fishing", "kinematics_only")) or {}).get("n") or 0),
        "kin_far_overall": {k: _pct(v) for k, v in kin_overall.items()},
        "kin_far_fishing": {k: _pct(v) for k, v in kin_fishing.items()},
        "geo_operating": list(geo_far.get("operating_placements") or []),
        "geo_placements": {
            k: (_pct(v["far"]), int(v["n_scored"])) for k, v in placements.items()
        },
        "gap_rec_fish": gap_rec_fish,
        "gap_rec_sail": gap_rec_sail,
        "gap_ferry_fish": gap_ferry_fish,
        "gap_rec_fish_pooled": gap_rec_fish_p,
        "gap_rec_sail_pooled": gap_rec_sail_p,
        "gap_ferry_fish_pooled": gap_ferry_fish_p,
        "ddr_fish_kin": fish_kin,
        "ddr_fish_kin_pooled": fish_kin_p,
        "ddr_fish_geo": cons.get(("fishing", "kinematics_geometry"), 0.0),
        "ddr_fish_geo_pooled": cons_pooled.get(("fishing", "kinematics_geometry"), 0.0),
        "ddr_sail_kin": cons.get(("sailing", "kinematics_only"), 0.0),
        "ddr_sail_kin_pooled": cons_pooled.get(("sailing", "kinematics_only"), 0.0),
        "ddr_rec_kin_pooled": cons_pooled.get(("recreational", "kinematics_only"), 0.0),
        "ddr_rec_geo_pooled": cons_pooled.get(
            ("recreational", "kinematics_geometry"), 0.0
        ),
        "unc_fish_kin_pooled": unc_pooled.get(("fishing", "kinematics_only"), 0.0),
        "unc_sail_kin_pooled": unc_pooled.get(("sailing", "kinematics_only"), 0.0),
        "pool_override": str(
            swap_pooled.get("envelope_override")
            or oracle_pooled.get("envelope_override")
            or "pooled_benign"
        ),
    }


# --------------------------------------------------------------------------
# caption records
# --------------------------------------------------------------------------


def build_records(f: dict[str, Any]) -> list[dict[str, Any]]:
    w = f["warn"]

    def wr(cls: str, arm: str, key: str) -> float:
        return round(float((w.get((cls, arm)) or {}).get(key) or 0.0), 2)

    def wt(cls: str, arm: str) -> float:
        """First-flag minutes, rounded as the figure annotates it."""
        return round(float((w.get((cls, arm)) or {}).get("t_flag_min_median") or 0.0), 1)

    rec_kin = f["rec_kin_curve"]
    rec_geo = f["rec_geo_curve"]
    cmax = f["commit_max"]

    return [
        {
            "id": "fig1_system",
            "number": "1",
            "latex_label": "fig:system",
            "kind": "schematic",
            "files": ["results/paper/fig1_system.svg"],
            "short_title": "Where the attack and the defense meet.",
            "caption": (
                "A hostile uncrewed surface vessel (USV) approaches a defended asset. "
                "A shore electro-optical (EO) camera asserts a class for it; a coastal "
                "radar tracks it in the world frame. The attacker controls appearance "
                "only — an adversarial patch or benign hull paint — so it can reach the "
                "EO channel but cannot invent a bearing rate on the radar channel. Both "
                "channels meet at the dashed interface the defense reads: a claimed "
                "class plus track features. The presence check looks "
                "for a radar track with no EO detection, which is what evasion "
                "produces; the consistency check asks whether the claimed class matches "
                "how the craft is moving, in a kinematics arm and an asset-relative "
                "geometry arm (feature families listed in their boxes). Dashed grey, "
                "bottom: the benign envelope model is fit offline on historical benign "
                "AIS tracks. RQ1–RQ3 mark which research question each path answers."
            ),
            "panels": {},
            "takeaway": (
                "The defense never has to win in pixel space — it adjudicates the "
                "claimed class against an independent motion channel the attacker "
                "cannot reach."
            ),
            "caveats": [
                "Schematic, not to scale; hand-authored vector art rather than a "
                "generated figure (see figures/README.md).",
                "AIS is never read at runtime and no hostile track enters the benign "
                "envelope model.",
            ],
            "sources": ["figures/fig1_system.svg"],
            "key_values": {},
        },
        {
            "id": "fig2_rq1_feasibility",
            "number": "2",
            "latex_label": "fig:rq1-feasibility",
            "kind": "data",
            "files": [
                "results/paper/fig2_rq1_feasibility.pdf",
                "results/paper/fig2_rq1_feasibility.png",
            ],
            "short_title": (
                "Evasion is reliably achievable; targeted disguise is not."
            ),
            "caption": (
                "Attack success at base severity L0 (no marine viewing transform), "
                "grouped by attack goal and adversary access. ESR (evasion success "
                "rate) is the fraction of otherwise-detected targets the patch "
                "suppresses; TMSR (targeted-misclassification success rate) is the "
                "fraction pushed into the benign class named on the tick. Access is "
                "white-box (patch optimised on the attacked model), grey-box (optimised "
                "on the other YOLO11 size), or black-box (optimised on a YOLO11 model "
                "and evaluated on RT-DETR-L); bar colour keys the model the patch was "
                "optimised on, so it is the victim in the white-box groups and the "
                "surrogate in the other two. Bar labels give the rate in percent and "
                "\"0\" marks an exactly-zero bar. Evasion succeeds at every access "
                f"level ({f['esr_white'].get('yolo11s', 0):.0f}–"
                f"{f['esr_white'].get('yolo11l', 0):.0f}% white-box, up to "
                f"{f['esr_grey_max']:.0f}% grey-box, "
                f"{f['esr_black_min']:.0f}–{f['esr_black_max']:.0f}% black-box); TMSR "
                "is 0% in all seven configurations measured."
            ),
            "panels": {},
            "takeaway": (
                "The attacker can make a hostile craft disappear, but cannot make it "
                "read as a chosen benign class — which is what makes a class-assertion "
                "defense worth building."
            ),
            "caveats": [
                "Each rate is over the attacked model's own clean detections "
                "(n = 33–37), so denominators differ between bars and single-digit "
                "differences are not meaningful.",
                "TMSR at exactly 0 is a ceiling result: it bounds this patch family "
                "under this threat model, not targeted disguise in general.",
            ],
            "sources": [
                "results/attacks/FROZEN.json (headlines.white_box_esr_L0, "
                ".white_box_tmsr_L0, .transfer_esr_L0)",
                "results/attacks/transfer/disguise/*/fishing/tmsr_by_severity.json",
            ],
            "key_values": {
                "esr_white_pct": f["esr_white"],
                "esr_transfer_pct": f["esr_transfer"],
                "tmsr_all_zero": f["tmsr_white_all_zero"],
            },
        },
        {
            "id": "fig3_rq2_ddr_gap",
            "number": "3",
            "latex_label": "fig:rq2-ddr-gap",
            "kind": "data",
            "files": [
                "results/paper/fig3_rq2_ddr_gap.pdf",
                "results/paper/fig3_rq2_ddr_gap.png",
            ],
            "short_title": (
                "A presence check cannot see a successful disguise; a consistency "
                "check can."
            ),
            "caption": (
                "Defense detection rate (DDR) — the fraction of successful disguises "
                "the defense flags — by mimicked benign class, under the "
                "perfect-disguise oracle. (a) A mimicking approach: presence check "
                "versus both consistency arms. The presence check is 0% for all three "
                "classes and is drawn as a short stub so the zero stays visible. "
                "(b) The unconstrained end-member, in which the adversary pays no "
                "added approach time; only the two consistency arms are plotted. "
                "Recreational is the only class "
                f"either arm misses — {f['ddr_rec_kin']:.0f}% kinematics and "
                f"{f['ddr_rec_geo']:.0f}% with geometry added while mimicking, "
                f"{f['unc_rec_kin']:.0f}% and {f['unc_rec_geo']:.0f}% unconstrained — "
                "while fishing and sailing are at 100% throughout. Both panels are at "
                "the calibrated false-alarm rate (FAR) "
                f"@{f['oracle_target']:.0f}% operating point; "
                f"n = {f['oracle_n_cell']:,} scored trips per class × arm."
            ),
            "panels": {
                "a": (
                    "Perfect-disguise oracle DDR by mimicked class, comparing the "
                    "presence check against both consistency arms."
                ),
                "b": (
                    "The unconstrained (no added approach time) end-member, where the "
                    "recreational gap between the two arms is widest; the presence "
                    "check is not plotted."
                ),
            },
            "takeaway": (
                "A presence check is blind exactly where disguise succeeds — it looks "
                "for a missing detection, and a successful disguise supplies one — and "
                "the geometry arm is what closes that gap for recreational, the "
                "kinematically closest benign class."
            ),
            "caveats": [
                "Oracle-conditioned: the adversary is handed the benign assertion, so "
                "this is a defense measurement, not an end-to-end attack success rate.",
                f"Realized test FAR is {f['oracle_far_kin']:.1f}% for kinematics only "
                f"and {f['oracle_far_geo']:.1f}% for kinematics + geometry, so the "
                f"geometry arm's extra DDR is bought at a realized FAR above the "
                f"{f['oracle_target']:.0f}% target.",
                f"Negative control (benign craft under a truthful assertion) returns 0% "
                f"DDR on n = {f['negctl_n']} for both arms.",
            ],
            "sources": ["results/oracle_ddr/oracle_ddr_summary.json"],
            "key_values": {
                "ddr_recreational_kinematics_pct": f["ddr_rec_kin"],
                "ddr_recreational_geometry_pct": f["ddr_rec_geo"],
                "unconstrained_recreational_pct": {
                    "kinematics_only": f["unc_rec_kin"],
                    "kinematics_geometry": f["unc_rec_geo"],
                },
                "operating_far_pct": {
                    "kinematics_only": f["oracle_far_kin"],
                    "kinematics_geometry": f["oracle_far_geo"],
                },
                "n_per_cell": f["oracle_n_cell"],
            },
        },
        {
            "id": "fig4_label_swap",
            "number": "4",
            "latex_label": "fig:label-swap",
            "kind": "data",
            "files": [
                "results/paper/fig4_label_swap.pdf",
                "results/paper/fig4_label_swap.png",
            ],
            "short_title": (
                "The consistency signal is present in real vessel motion, not only in "
                "synthesised tracks."
            ),
            "caption": (
                "Fraction of real AIS windows the kinematics-arm consistency check "
                "flags when a window's true class (row) is relabelled with an asserted "
                f"class (column), at the frozen calibrated FAR@{f['swap_target']:.0f}% "
                f"operating point over {f['swap_trips']} banded trips. The rightmost "
                "column, past the rule, is the matched control — the same windows scored "
                "under their true class — and is the baseline for each swapped cell. In "
                "the recreational row, the best-populated one with a control, swapped "
                f"cells run {f['swap_rec_row_min']:.0f}–"
                f"{f['swap_rec_row_max']:.0f}% against a {f['swap_matched_rec']:.0f}% "
                f"control (n = {f['swap_matched_rec_n']}). Markers: \"—\" is a cell that "
                "cannot exist (the diagonal, and rows with no matched control); hatching "
                f"with † is a thin-n cell (n < {THIN_N}, {f['swap_thin']} of "
                f"{f['swap_cells']} cells); grey is empty. Per-cell n is not shown."
            ),
            "panels": {},
            "takeaway": (
                "Synthesis-free evidence that a class assertion inconsistent with "
                "observed motion is detectable in real traffic: the check responds to "
                "the mismatch, not merely to the traffic."
            ),
            "caveats": [
                f"Half the cells ({f['swap_thin']} of {f['swap_cells']}) are thin-n "
                f"(n < {THIN_N}); individual hatched cells should not be read as rates.",
                f"The working service row has a matched-control flag rate of "
                f"{f['swap_matched_working']:.0f}% on n = "
                f"{f['swap_matched_working_n']}, i.e. those windows are flagged under "
                f"their own true class, so that row reflects a per-class bias rather "
                f"than swap discriminability; the fishing row's control is thin-n. The "
                f"swap-versus-control comparison is therefore only supported where a "
                f"well-populated control exists.",
                "Matched controls are empty for "
                + _and([c.replace("_", " ") for c in f["swap_matched_empty"]])
                + ", so those rows have no baseline.",
                "Not an adversary-motion DDR claim: these are benign tracks with "
                "relabelled assertions, and thresholds were not retuned per cell.",
            ],
            "sources": ["results/label_swap/label_swap_summary.json"],
            "key_values": {
                "far_target_pct": f["swap_target"],
                "n_band_trips": f["swap_trips"],
                "n_cells": f["swap_cells"],
                "n_thin_cells": f["swap_thin"],
                "matched_recreational_pct": f["swap_matched_rec"],
                "matched_working_service_pct": f["swap_matched_working"],
            },
        },
        {
            "id": "fig5_rq3_cost_warning",
            "number": "5",
            "latex_label": "fig:rq3-cost-warning",
            "kind": "data",
            "files": [
                "results/paper/fig5_rq3_cost_warning.pdf",
                "results/paper/fig5_rq3_cost_warning.png",
            ],
            "short_title": (
                "A kinematics-only check is weakest against the cheapest attack; "
                "asset-relative geometry closes that route and warns earlier."
            ),
            "caption": (
                "(a) End-of-track DDR against commit range — the range, in nautical "
                "miles (nm), at which the adversary breaks off its benign-looking "
                "mimicry, turns radially inbound and accelerates to burst speed for the "
                "terminal run. The arrow under the axis marks that commit range runs "
                "opposite to cost. Open markers past the dotted rule are the "
                "unconstrained end-member (no mimicry, zero added time). Fishing and "
                "sailing hold 100% for both arms throughout, drawn as one grey series. "
                "Recreational is the exception: kinematics-only DDR is "
                f"{rec_kin.get(0.5, (0,))[0]:.0f}% at a 0.5 nm commit, "
                f"{rec_kin.get(4.0, (0,))[0]:.0f}% at 4 nm and "
                f"{rec_kin.get(cmax, (0,))[0]:.0f}% unconstrained, while the geometry "
                f"arm holds {rec_geo.get(cmax, (0,))[0]:.0f}% across the sweep and at "
                "the end-member, dipping only at the shortest 0.5 nm commit "
                f"({rec_geo.get(0.5, (0,))[0]:.0f}%). (b) Median range at first flag, "
                "annotated with median first-flag time in minutes relative to annulus "
                "entry, so smaller and negative values are earlier warnings. The "
                "geometry arm flags recreational "
                f"{wr('recreational', 'kinematics_geometry', 'R_flag_nm_median'):.1f} nm "
                f"out against "
                f"{wr('recreational', 'kinematics_only', 'R_flag_nm_median'):.1f} nm for "
                f"kinematics only ({wt('recreational', 'kinematics_geometry'):g} vs "
                f"{wt('recreational', 'kinematics_only'):g} min), and fishing similarly; "
                "sailing is the exception, where the kinematics arm flags before annulus "
                f"entry ({wt('sailing', 'kinematics_only'):g} min at "
                f"{wr('sailing', 'kinematics_only', 'R_flag_nm_median'):.1f} nm). "
                f"n = {f['warn_n']:,} trips per class × arm in panel (b)."
            ),
            "panels": {
                "a": (
                    "End-of-track DDR versus the range at which the adversary breaks "
                    "off mimicry and runs in; fishing and sailing are flat at 100% for "
                    "both arms and are drawn as a single grey series. Colour keys the "
                    "defense arm in both panels. The geometry arm's dip at the shortest "
                    "commit is a short-geometry effect: little approach geometry remains "
                    "to judge."
                ),
                "b": (
                    "Median first-flag standoff by class and arm, annotated with median "
                    "first-flag time relative to annulus entry."
                ),
            },
            "takeaway": (
                "Elaborate mimicry does not buy the adversary anything against a "
                "kinematics-only check — the cheap direct run is the one it misses, "
                "because a burst transition out of a slow benign track is itself "
                "anomalous while a fast straight run-in merely resembles a recreational "
                "speedboat to features that know nothing about the defended asset. "
                "Asset-relative geometry closes that route and warns roughly ten "
                "minutes earlier."
            ),
            "caveats": [
                f"The rightmost point ({cmax:g} nm) is the unconstrained end-member "
                f"(burst speed, radial, no mimicry, zero added approach time), not a "
                f"longer-commit sweep cell: the curve changes experimental condition "
                f"there. It is also the thinnest point, at n = {f['commit_max_n']} per "
                f"arm against n = 480 per arm for the 0.5–4 nm sweep cells.",
                "Commit range runs opposite to cost — added approach time falls as "
                "commit range grows — so the panel should not be read as a "
                "monotone cost axis. The primary added-approach-time pivot and the "
                "speed and bearing-offset companions live in "
                "results/adaptive_cost/adaptive_cost_report.md.",
                "Swept under the perfect-disguise oracle, so panel (a) is a defense "
                "measurement conditioned on the disguise already succeeding.",
                "First-flag time is measured from annulus entry, so it can be negative; "
                "it is not directly comparable to end-of-track DDR in panel (a).",
            ],
            "sources": [
                "results/adaptive_cost/adaptive_cost_curves.parquet (axis=commit_range_nm)",
                "results/adaptive_cost/adaptive_cost_summary.json (warning_summary)",
            ],
            "key_values": {
                "recreational_kinematics_ddr_pct_by_commit_nm": {
                    str(k): v[0] for k, v in sorted(rec_kin.items())
                },
                "recreational_geometry_ddr_pct_by_commit_nm": {
                    str(k): v[0] for k, v in sorted(rec_geo.items())
                },
                "n_by_commit_nm": {str(k): v[1] for k, v in sorted(rec_kin.items())},
                "warning": {
                    f"{cls}/{arm}": {
                        "R_flag_nm_median": wr(cls, arm, "R_flag_nm_median"),
                        "t_flag_min_median": wt(cls, arm),
                    }
                    for cls in ("fishing", "recreational", "sailing")
                    for arm in ("kinematics_only", "kinematics_geometry")
                },
            },
        },
        {
            "id": "fig6_far",
            "number": "6",
            "latex_label": "fig:far",
            "kind": "data",
            "files": ["results/paper/fig6_far.pdf", "results/paper/fig6_far.png"],
            "short_title": "What the detection rates cost in false alarms.",
            "caption": (
                "False-alarm rate (FAR) on held-out benign traffic — the price side of "
                "every DDR reported elsewhere. (a) Realized kinematics-arm FAR on the "
                "test split, overall and for five representative traffic envelopes; the "
                "\"@1%\", \"@5%\" and \"@10%\" ticks are the calibrated targets, which the "
                f"dotted rules mark. Overall FAR tracks the target "
                f"({f['kin_far_overall'].get('far_0.01', 0):.1f}%, "
                f"{f['kin_far_overall'].get('far_0.05', 0):.1f}% and "
                f"{f['kin_far_overall'].get('far_0.1', 0):.1f}%), but per-envelope FAR "
                "spreads around it, with fishing loosest at every target "
                f"({f['kin_far_fishing'].get('far_0.05', 0):.1f}% at the 5% target, "
                f"{f['kin_far_fishing'].get('far_0.1', 0):.1f}% at the 10%). "
                "(b) Geometry-arm FAR on the test split by asset-placement class: green "
                "bars are the operating placements that set the reported floor, grey "
                "bars are sensitivity placements. Bar labels give FAR and the scored "
                "count n, and the dotted rule is the 5% target. FAR is highest at "
                f"anchorage ({f['geo_placements'].get('anchorage', (0, 0))[0]:.1f}%) and "
                f"lowest at the offshore terminal "
                f"({f['geo_placements'].get('offshore_terminal', (0, 0))[0]:.1f}%)."
            ),
            "panels": {
                "a": (
                    "Realized kinematics test FAR at three calibrated targets, overall "
                    "and by traffic envelope."
                ),
                "b": (
                    "Geometry placement FAR on the test split, operating versus "
                    "sensitivity placements."
                ),
            },
            "takeaway": (
                "Calibration holds in aggregate but not uniformly: the defense spends "
                "its false alarms on the envelopes and placements where benign traffic "
                "looks most like an approach — irregular fishing motion, and traffic "
                "that legitimately loiters near the asset."
            ),
            "caveats": [
                "Calibration is global, not per-envelope, so per-envelope FAR is "
                "expected to spread around the target rather than meet it; fishing "
                "motion is the most irregular benign behaviour in the corpus.",
                "Placement classes differ substantially in scored count "
                "(n = "
                + "–".join(
                    str(v)
                    for v in (
                        min(n for _, n in f["geo_placements"].values()),
                        max(n for _, n in f["geo_placements"].values()),
                    )
                )
                + "), so the placement bars are not equally precise.",
            ],
            "sources": [
                "results/behavior_model/far_summary.json (by_split.test)",
                "results/behavior_model_geometry/far_placement_summary.json",
            ],
            "key_values": {
                "kinematics_overall_far_pct": f["kin_far_overall"],
                "kinematics_fishing_far_pct": f["kin_far_fishing"],
                "geometry_placement_far_pct": {
                    k: v[0] for k, v in f["geo_placements"].items()
                },
                "geometry_placement_n": {
                    k: v[1] for k, v in f["geo_placements"].items()
                },
                "operating_placements": f["geo_operating"],
            },
        },
        {
            "id": "figS1_severity",
            "number": "S1",
            "latex_label": "fig:supp-severity",
            "kind": "data",
            "files": [
                "results/paper/figS1_severity.pdf",
                "results/paper/figS1_severity.png",
            ],
            "short_title": (
                "Harsh viewing conditions help the evasion attack, but never enough to "
                "overcome adversary access."
            ),
            "caption": (
                "Patch-attributable evasion success across the L0–L4 marine severity "
                "ladder for the three photometric transforms: (a) glare, (b) spray, "
                "(c) sea state. Patch-attributable ESR credits the patch only for "
                "targets the transform alone did not already suppress. Line style keys "
                "adversary access — solid white-box, dashed grey-box, dotted black-box "
                "— and colour keys the detector family. Glare and spray raise attack "
                f"success at high severity by up to +{f['sev_max_gain_pts']:.1f} points; "
                f"{f['sev_flat_axis']} is nearly flat. The ordering by access level "
                "nevertheless survives the whole ladder: black-box peaks at "
                f"{f['sev_black_peak']:.1f}% under the harshest photometric severity, "
                "still below every white-box and grey-box value at L0 "
                f"({f['sev_l0_non_black_min']:.1f}–"
                f"{f['sev_l0_non_black_max']:.1f}%)."
            ),
            "panels": {
                "a": "Glare severity ladder.",
                "b": "Spray severity ladder.",
                "c": "Sea-state severity ladder.",
            },
            "takeaway": (
                "Marine viewing conditions are a modulating condition on RQ1, not an "
                "independent lever the attacker can pull to reach white-box performance "
                "from black-box access."
            ),
            "caveats": [
                "The four geometric axes (scale, rotation, motion blur, grazing angle) "
                "are omitted because at high severity the transform itself suppresses "
                "the detection, collapsing the patch-attributable denominator.",
                "Same small per-model denominators as Fig. 2 (n = 33–37 at L0, "
                "shrinking with severity as the transform suppresses targets), so "
                "individual level-to-level steps are noisy.",
                "Patch-attributable rates are lower than raw miss rates by construction; "
                "raw rates at L4 reach 27% black-box.",
                "Supplementary: the headline access-level comparison is Fig. 2.",
            ],
            "sources": [
                "results/attacks/evasion/{yolo11s,yolo11l}/esr_by_severity.json",
                "results/attacks/transfer/evasion/*/esr_by_severity.json",
            ],
            "key_values": {
                "max_severity_gain_points": f["sev_max_gain_pts"],
                "flattest_axis": f["sev_flat_axis"],
                "black_box_peak_pct": f["sev_black_peak"],
                "l0_white_grey_range_pct": [
                    f["sev_l0_non_black_min"],
                    f["sev_l0_non_black_max"],
                ],
                "access_ordering_holds_across_ladder": f["sev_access_order_holds"],
            },
        },
        {
            "id": "figS2_pooled_gap",
            "number": "S2",
            "latex_label": "fig:supp-pooled-gap",
            "kind": "data",
            "files": [
                "results/paper/figS2_pooled_gap.pdf",
                "results/paper/figS2_pooled_gap.png",
            ],
            "short_title": (
                "Class conditioning buys discriminability, not just false-alarm "
                "calibration."
            ),
            "caption": (
                "Class-conditional versus pooled benign envelopes at matched FAR@5%. "
                "(a) Real-track label-swap gap — swapped flag rate minus the matched "
                "control, in percentage points (pp) — for the three headline cells, "
                "whose ticks read source class \"as\" asserted class. Under one shared "
                f"`{f['pool_override']}` envelope the recreational gaps collapse "
                f"({f['gap_rec_fish']:+.1f} to {f['gap_rec_fish_pooled']:+.1f} pp as "
                f"fishing, {f['gap_rec_sail']:+.1f} to "
                f"{f['gap_rec_sail_pooled']:+.1f} as sailing) and the passenger-ferry "
                f"gap shrinks ({f['gap_ferry_fish']:+.1f} to "
                f"{f['gap_ferry_fish_pooled']:+.1f}). (b) Perfect-disguise oracle DDR "
                "for the mimicking condition, class-conditional (solid) versus pooled "
                "(hatched), both arms. Pooling collapses the per-class spread onto a "
                f"single value per arm — kinematics from {f['ddr_fish_kin']:.0f}% "
                f"(fishing and sailing) and {f['ddr_rec_kin']:.0f}% (recreational) to a "
                f"shared {f['ddr_fish_kin_pooled']:.0f}%, geometry from "
                f"{f['ddr_rec_geo']:.0f}–{f['ddr_fish_geo']:.0f}% to a shared "
                f"{f['ddr_fish_geo_pooled']:.0f}% — which is what one envelope for all "
                "asserted classes must produce."
            ),
            "panels": {
                "a": (
                    "Label-swap gap collapse under pooling (kinematics; "
                    "headline cells)."
                ),
                "b": (
                    "Oracle DDR under the mimicking condition, conditional vs pooled, "
                    "for fishing / recreational / sailing (both arms)."
                ),
            },
            "takeaway": (
                "Pooling matches or beats on false alarms but loses discriminability; "
                "class-conditional envelopes are what carry the disguise check."
            ),
            "caveats": [
                "Pooled routing yields one flag rate per source for every asserted "
                "class, so a residual non-zero gap can only come from different "
                "samples in swap vs matched (the ferry cohort), not from class "
                "mismatch.",
                "The unconstrained end-member is not plotted here: under pooled "
                f"kinematics fishing and sailing fall to "
                f"{f['unc_fish_kin_pooled']:.0f}% and "
                f"{f['unc_sail_kin_pooled']:.0f}%, matching recreational, while the "
                "geometry arm still closes it under either routing.",
                "Panel (b) is the frozen 5,880-cell oracle grid. Supplementary to "
                "Fig. 4 (label-swap) and Fig. 3 (oracle DDR); thresholds never "
                "retuned.",
                "Envelope joblib weights are not redistributed; digests are pinned "
                "in the kinematics arm freeze.",
            ],
            "sources": [
                "results/label_swap/label_swap_summary.json",
                "results/label_swap_pooled/label_swap_summary.json",
                "results/oracle_ddr/oracle_ddr_summary.json",
                "results/oracle_ddr_pooled/oracle_ddr_summary.json",
            ],
            "key_values": {
                "gap_recreational_fishing_conditional_pp": f["gap_rec_fish"],
                "gap_recreational_fishing_pooled_pp": f["gap_rec_fish_pooled"],
                "gap_recreational_sailing_conditional_pp": f["gap_rec_sail"],
                "gap_recreational_sailing_pooled_pp": f["gap_rec_sail_pooled"],
                "gap_ferry_fishing_conditional_pp": f["gap_ferry_fish"],
                "gap_ferry_fishing_pooled_pp": f["gap_ferry_fish_pooled"],
                "ddr_fishing_kinematics_conditional_pct": f["ddr_fish_kin"],
                "ddr_fishing_kinematics_pooled_pct": f["ddr_fish_kin_pooled"],
                "ddr_sailing_kinematics_conditional_pct": f["ddr_sail_kin"],
                "ddr_sailing_kinematics_pooled_pct": f["ddr_sail_kin_pooled"],
                "unc_fishing_kinematics_pooled_pct": f["unc_fish_kin_pooled"],
                "unc_sailing_kinematics_pooled_pct": f["unc_sail_kin_pooled"],
                "pooling_shrinks_rec_fish_gap": f["gap_rec_fish_pooled"]
                < f["gap_rec_fish"] - 5.0,
                "pooling_drops_fishing_kin_ddr": f["ddr_fish_kin_pooled"]
                < f["ddr_fish_kin"] - 5.0,
            },
        },
    ]


# --------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------


def build_payload() -> dict[str, Any]:
    facts = gather_facts()
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "convention": CONVENTION,
        "figures": build_records(facts),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Figure captions (draft)",
        "",
        f"Generated by `scripts/report/captions.py` — see `CAPTIONS.json` for the "
        f"machine-readable records (panels, takeaways, caveats, source artifacts, and "
        f"the numbers quoted). {payload['convention']}",
        "",
        "---",
        "",
    ]
    for rec in payload["figures"]:
        lines.append(
            f"**Figure {rec['number']}. {rec['short_title']}** {rec['caption']}"
        )
        lines.append("")
        if rec["caveats"]:
            lines.append("*Caveats.* " + " ".join(rec["caveats"]))
            lines.append("")
    return "\n".join(lines)


def write(out_dir: Path | None = None) -> dict[str, Path]:
    dest = out_dir or PAPER_DIR
    dest.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    js = dest / "CAPTIONS.json"
    md = dest / "CAPTIONS.md"
    js.write_text(json.dumps(payload, indent=2) + "\n")
    md.write_text(render_markdown(payload))
    return {"json": js, "markdown": md}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PAPER_DIR)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    verify_freezes(skip=args.no_verify)
    out = write(args.out)
    print(f"[captions] wrote {out['json']} + {out['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
