#!/usr/bin/env python3
"""Build the security-lite paper figure set into ``results/paper/``.

Copies the hand-authored Fig. 1 SVG, regenerates Figs 2–6 + S1–S2 from digest-
verified attack/defense artifacts, and writes ``PROVENANCE.json``.

Usage
-----
    python scripts/report/build_all.py
    python scripts/report/build_all.py --no-verify   # draft only
    python scripts/report/build_all.py --smoke       # temp out dir + no verify
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "report"))

from _common import (  # noqa: E402
    PAPER_DIR,
    copy_fig1,
    rel,
    verify_freezes,
    write_provenance,
)
from captions import write as write_captions  # noqa: E402
from fig_far import build_figure as build_far  # noqa: E402
from fig_label_swap import build_figure as build_label_swap  # noqa: E402
from fig_rq1_feasibility import build_figure as build_rq1  # noqa: E402
from fig_rq2_ddr_gap import build_figure as build_rq2  # noqa: E402
from fig_rq3_cost_warning import build_figure as build_rq3  # noqa: E402
from fig_supp_severity import build_figure as build_s1  # noqa: E402
from fig_supp_pooled_gap import build_figure as build_s2  # noqa: E402


def run(out_dir: Path, *, skip_verify: bool) -> dict:
    verification = verify_freezes(skip=skip_verify)
    figures: dict = {}

    print("[build] Fig. 1 (static SVG) …")
    figures["fig1_system"] = copy_fig1(out_dir=out_dir)
    if figures["fig1_system"].get("cairosvg_note"):
        print(f"  note: {figures['fig1_system']['cairosvg_note']}")

    print("[build] Fig. 2 RQ1 feasibility …")
    figures["fig2_rq1_feasibility"] = build_rq1(out_dir)

    print("[build] Fig. 3 RQ2 DDR gap …")
    figures["fig3_rq2_ddr_gap"] = build_rq2(out_dir)

    print("[build] Fig. 4 label-swap …")
    figures["fig4_label_swap"] = build_label_swap(out_dir)

    print("[build] Fig. 5 RQ3 cost + warning …")
    figures["fig5_rq3_cost_warning"] = build_rq3(out_dir)

    print("[build] Fig. 6 FAR …")
    figures["fig6_far"] = build_far(out_dir)

    print("[build] Fig. S1 severity …")
    figures["figS1_severity"] = build_s1(out_dir)

    print("[build] Fig. S2 pooled-vs-conditional …")
    figures["figS2_pooled_gap"] = build_s2(out_dir)

    print("[build] captions …")
    captions = write_captions(out_dir)
    figures["captions"] = {k: rel(v) for k, v in captions.items()}

    prov = write_provenance(figures, verification, out_dir=out_dir)
    print(f"[build] wrote {prov}")
    return {"figures": figures, "provenance": str(prov), "verification": verification}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path, default=PAPER_DIR)
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip freeze digest verification (local drafting only)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="write to a temp dir and skip digest verification",
    )
    args = ap.parse_args()

    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="paper_figs_") as tmp:
            out = Path(tmp)
            run(out, skip_verify=True)
            expected = [
                "fig1_system.svg",
                "fig2_rq1_feasibility.png",
                "fig3_rq2_ddr_gap.png",
                "fig4_label_swap.png",
                "fig5_rq3_cost_warning.png",
                "fig6_far.png",
                "figS1_severity.png",
                "figS2_pooled_gap.png",
                "CAPTIONS.json",
                "CAPTIONS.md",
                "PROVENANCE.json",
            ]
            missing = [n for n in expected if not (out / n).is_file()]
            if missing:
                raise SystemExit(f"smoke missing outputs: {missing}")
            print(f"[build] smoke OK ({len(expected)} artifacts in {out})")
        return 0

    run(args.out, skip_verify=args.no_verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
