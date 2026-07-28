"""Track-kinematics and benign-behavior modeling.

Ingests real vessel trajectories (AIS archives + video-derived tracks), derives
class-conditional kinematic features, and fits the one-class benign-behavior model
the class-kinematics consistency defense scores against. The benign model is trained
on real benign tracks only; hostile/adaptive trajectories are synthesized at
evaluation time and never enter training, which keeps the defense free of
by-construction circularity.
"""

from counterusv.kinematics.features import (
    extract_last_windows,
    features_from_points,
    last_window_mask,
)
from counterusv.kinematics.behavior_model import (
    EnvelopeModel,
    MultiHorizonEnvelope,
    fit_envelope,
    load_envelope,
    save_envelope,
)

__all__ = [
    "extract_last_windows",
    "features_from_points",
    "last_window_mask",
    "EnvelopeModel",
    "MultiHorizonEnvelope",
    "fit_envelope",
    "load_envelope",
    "save_envelope",
]
