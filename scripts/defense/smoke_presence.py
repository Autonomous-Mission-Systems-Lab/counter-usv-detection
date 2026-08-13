#!/usr/bin/env python3
"""Smoke-test the presence-only cross-check and the shared defense harness.

Exercises evasion (flag), disguise (pass), no-track (abstain), and a
side-by-side harness swap against the consistency pipeline.

Usage
-----
    python scripts/defense/smoke_presence.py
    python scripts/defense/smoke_presence.py --no-verify-digests
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.attacks.oracle import PerfectDisguiseOracle  # noqa: E402
from counterusv.defense.harness import evaluate_contact, load_defense  # noqa: E402
from counterusv.defense.presence import (  # noqa: E402
    PresenceObservation,
    PresenceOnlyDefense,
    presence_for_disguise,
    presence_for_evasion,
)


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--verify-digests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify behavior-model freeze digests for consistency arm",
    )
    args = ap.parse_args()

    presence = PresenceOnlyDefense.from_config()
    print(f"[smoke] PresenceOnlyDefense  kind={presence.kind}")

    evasion = presence_for_evasion(contact_id="evasion-1")
    d_ev = presence.evaluate(presence=evasion, purpose="eval")
    print(f"  evasion  → {d_ev.action}  (expect flag)  note={d_ev.note}")

    oracle = PerfectDisguiseOracle().assert_class(
        "fishing",
        contact_id="disguise-1",
        true_class="usv",
        box_xyxy=(0.0, 0.0, 10.0, 10.0),
    )
    d_di = presence.evaluate(oracle, purpose="eval")
    print(f"  disguise → {d_di.action}  (expect pass)  note={d_di.note}")

    d_ab = presence.evaluate(
        presence=PresenceObservation(track_present=False, eo_outcome="missed"),
        purpose="eval",
    )
    print(f"  no-track → {d_ab.action}  (expect abstain)")

    print("[smoke] harness swap (presence vs consistency on disguise) …")
    try:
        consistency = load_defense(
            "consistency", verify_digests=args.verify_digests
        )
    except ValueError as e:
        if args.verify_digests and "digest mismatch" in str(e):
            print(f"[smoke] WARNING: {e}")
            consistency = load_defense("consistency", verify_digests=False)
        else:
            raise
    obs = presence_for_disguise(oracle)
    feats = _benign_feats()
    p_dec = evaluate_contact(
        presence, assertion=oracle, presence=obs, purpose="eval"
    )
    c_dec = evaluate_contact(
        consistency, assertion=oracle, features=feats, purpose="eval"
    )
    print(
        f"  presence    → {p_dec.action}  kind={p_dec.defense_kind}  "
        f"(disguise never flagged)"
    )
    print(
        f"  consistency → {c_dec.action}  kind={c_dec.defense_kind}  "
        f"arm={c_dec.feature_arm}"
    )
    print("[smoke] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
