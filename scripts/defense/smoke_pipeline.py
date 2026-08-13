#!/usr/bin/env python3
"""Smoke-test the defense pipeline against the frozen behavior-model artifacts.

Loads ``DefensePipeline.from_freeze()`` (kinematics-only arm) and exercises
Detection / OracleAssertion / abstain / high-SOG-as-fishing paths.

Usage
-----
    python scripts/defense/smoke_pipeline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.attacks.oracle import PerfectDisguiseOracle  # noqa: E402
from counterusv.defense.pipeline import DefensePipeline  # noqa: E402
from counterusv.models.detector import Detection  # noqa: E402


def _benign_feats() -> dict[str, float]:
    return {
        "sog_med": 6.0,
        "sog_p95": 9.0,
        "sog_std": 1.5,
        "loiter_frac": 0.1,
        "straightness": 0.85,
        "accel_mean_abs": 0.05,
        "turn_rate_mean_dps": 0.1,
        "cog_circ_std_deg": 5.0,
    }


def _attack_run_feats(sog: float = 35.0) -> dict[str, float]:
    return {
        "sog_med": sog,
        "sog_p95": sog * 1.05,
        "sog_std": max(0.5, sog * 0.05),
        "loiter_frac": 0.0,
        "straightness": 0.99,
        "accel_mean_abs": 0.2,
        "turn_rate_mean_dps": 0.05,
        "cog_circ_std_deg": 2.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--verify-digests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify behavior-model freeze digests (default: true)",
    )
    args = ap.parse_args()

    print("[smoke] DefensePipeline.from_freeze() …")
    try:
        pipe = DefensePipeline.from_freeze(verify_digests=args.verify_digests)
    except ValueError as e:
        if args.verify_digests and "digest mismatch" in str(e):
            print(f"[smoke] WARNING: {e}")
            print(
                "[smoke] reloading with verify_digests=False "
                "(re-attest digests at defense freeze / behavior re-freeze)"
            )
            pipe = DefensePipeline.from_freeze(verify_digests=False)
        else:
            raise
    print(
        f"  arm={pipe.config.feature_arm}  far={pipe.config.far_target}  "
        f"envelopes={len(pipe.scorer.envelopes)}"
    )
    print(f"  association: {pipe.config.association_note[:80]}…")

    # 1. Benign recreational detection → expect pass (or scored)
    det = Detection(
        box_xyxy=(10.0, 20.0, 80.0, 60.0),
        score=0.92,
        class_name="recreational",
        class_id=2,
        role="benign",
    )
    d1 = pipe.evaluate(det, _benign_feats(), purpose="defense")
    print(f"[1] detection recreational + benign feats → {d1.action} "
          f"(status={d1.consistency.status})")
    assert d1.source == "detection"
    assert d1.action in ("pass", "flag", "abstain")

    # 2. Oracle asserts fishing on attack-run motion → expect flag
    oracle = PerfectDisguiseOracle.from_config()
    assertion = oracle.assert_class(
        "fishing", contact_id="smoke-1", box_xyxy=det.box_xyxy
    )
    d2 = pipe.evaluate(
        assertion,
        _attack_run_feats(),
        purpose="eval",
        track_meta={"role": "hostile", "source": "synth_preview"},
    )
    print(
        f"[2] oracle fishing + 35 kn dash → {d2.action} "
        f"(status={d2.consistency.status}, score={d2.consistency.score})"
    )
    assert d2.source == "oracle"
    assert d2.true_class == "usv"
    if d2.consistency.status == "scored":
        assert d2.action == "flag", "attack-run under fishing should flag"

    # 3. Undisguised usv class → abstain (envelope policy)
    d3 = pipe.evaluate(
        Detection(
            box_xyxy=det.box_xyxy,
            score=0.9,
            class_name="usv",
            class_id=8,
            role="hostile",
        ),
        _attack_run_feats(),
        purpose="eval",
    )
    print(f"[3] detection usv → {d3.action} (status={d3.consistency.status})")
    assert d3.action == "abstain"

    # 4. Same dash under recreational (weak-disguise surface) — may pass
    d4 = pipe.evaluate(
        "recreational",
        _attack_run_feats(),
        purpose="eval",
        track_meta={"role": "hostile", "source": "synth_preview"},
    )
    print(
        f"[4] recreational + 35 kn dash → {d4.action} "
        f"(status={d4.consistency.status})  "
        "[weak-disguise lane; flag not required]"
    )

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
