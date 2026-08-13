"""Class-kinematics consistency defense and presence-only cross-check.

Implements the class–kinematics consistency check, the wired defense
pipeline (detector / oracle → decision), the presence-only RQ2 comparator,
and the shared harness that swaps defenses behind one evaluate surface.
APRICOT-style and PercepGuard-style baselines are not implemented.
"""

from counterusv.defense.consistency import (
    ConsistencyResult,
    ConsistencyScorer,
    FirewallError,
    assert_benign_train_allowed,
    filter_benign_training,
)
from counterusv.defense.engagement import (
    EngagementGeometryConfig,
    PlacementClass,
    PortRegion,
    load_engagement_geometry,
)
from counterusv.defense.geometry_features import (
    GEOMETRY_FEATURE_KEYS,
    geometry_features_from_points,
    range_bearing_nm,
)
from counterusv.defense.harness import (
    DefenseBackend,
    evaluate_contact,
    load_defense,
)
from counterusv.defense.pipeline import (
    ASSOCIATION_ASSUMPTION,
    DefenseDecision,
    DefenseKind,
    DefensePipeline,
    PipelineConfig,
    decide_action,
    load_pipeline_config,
)
from counterusv.defense.placements import (
    materialize_placements,
    offset_nm,
)
from counterusv.defense.presence import (
    PresenceConfig,
    PresenceObservation,
    PresenceOnlyDefense,
    PresenceResult,
    decide_presence,
    load_presence_config,
    presence_for_disguise,
    presence_for_evasion,
)

__all__ = [
    "ASSOCIATION_ASSUMPTION",
    "ConsistencyResult",
    "ConsistencyScorer",
    "DefenseBackend",
    "DefenseDecision",
    "DefenseKind",
    "DefensePipeline",
    "EngagementGeometryConfig",
    "FirewallError",
    "GEOMETRY_FEATURE_KEYS",
    "PipelineConfig",
    "PlacementClass",
    "PortRegion",
    "PresenceConfig",
    "PresenceObservation",
    "PresenceOnlyDefense",
    "PresenceResult",
    "assert_benign_train_allowed",
    "decide_action",
    "decide_presence",
    "evaluate_contact",
    "filter_benign_training",
    "geometry_features_from_points",
    "load_defense",
    "load_engagement_geometry",
    "load_pipeline_config",
    "load_presence_config",
    "materialize_placements",
    "offset_nm",
    "presence_for_disguise",
    "presence_for_evasion",
    "range_bearing_nm",
]
