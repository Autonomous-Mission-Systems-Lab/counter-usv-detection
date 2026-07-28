"""Class-kinematics consistency defense and baseline defenses.

Implements the class–kinematics consistency check, plus presence-only,
APRICOT-style, and PercepGuard-style baselines.
"""

from counterusv.defense.consistency import (
    ConsistencyResult,
    ConsistencyScorer,
    FirewallError,
    assert_benign_train_allowed,
    filter_benign_training,
)

__all__ = [
    "ConsistencyResult",
    "ConsistencyScorer",
    "FirewallError",
    "assert_benign_train_allowed",
    "filter_benign_training",
]
