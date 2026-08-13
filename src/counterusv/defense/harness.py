"""Shared defense interface — swap consistency / presence / (later) baselines.

Every defense kind emits :class:`~counterusv.defense.pipeline.DefenseDecision`
with the same ``flag`` / ``pass`` / ``abstain`` vocabulary so evaluation
steps can score the same contacts under multiple defenses in one harness.

Config entry points
-------------------
- consistency: ``configs/defense/pipeline.yaml`` (via :class:`DefensePipeline`)
- presence_only: ``configs/defense/presence.yaml`` (via :class:`PresenceOnlyDefense`)
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from counterusv.attacks.oracle import OracleAssertion
from counterusv.defense.pipeline import (
    DefenseDecision,
    DefenseKind,
    DefensePipeline,
    Purpose,
)
from counterusv.defense.presence import (
    PresenceObservation,
    PresenceOnlyDefense,
    load_presence_config,
)
from counterusv.models.detector import Detection

AssertionLike = Detection | OracleAssertion | str


@runtime_checkable
class DefenseBackend(Protocol):
    """Common evaluate surface for every defense kind."""

    kind: DefenseKind

    def evaluate(
        self,
        assertion: AssertionLike | None = None,
        features: Mapping[str, Any] | None = None,
        *,
        presence: PresenceObservation | None = None,
        features_by_window: Mapping[int, Mapping[str, Any]] | None = None,
        complete_windows: set[int] | Sequence[int] | None = None,
        purpose: Purpose = "defense",
        track_meta: Mapping[str, Any] | None = None,
        far_target: float | None = None,
        contact_id: str | int | None = None,
        **kwargs: Any,
    ) -> DefenseDecision: ...


def load_defense(
    kind: DefenseKind,
    *,
    pipeline_config: str | None = None,
    presence_config: str | None = None,
    verify_digests: bool = True,
) -> DefenseBackend:
    """Construct a defense backend by kind.

    ``consistency`` loads :class:`DefensePipeline` from its freeze.
    ``presence_only`` loads :class:`PresenceOnlyDefense` (no freeze).
    """
    if kind == "consistency":
        from counterusv.defense.pipeline import load_pipeline_config

        cfg = load_pipeline_config(pipeline_config)
        return DefensePipeline.from_freeze(
            cfg.freeze_path,
            far_target=cfg.far_target,
            verify_digests=verify_digests,
            pipeline_config=pipeline_config,
        )
    if kind == "presence_only":
        cfg = load_presence_config(presence_config)
        return PresenceOnlyDefense(cfg)
    raise ValueError(f"unknown defense kind {kind!r}")


def evaluate_contact(
    defense: DefenseBackend,
    *,
    assertion: AssertionLike | None = None,
    features: Mapping[str, Any] | None = None,
    presence: PresenceObservation | None = None,
    purpose: Purpose = "eval",
    contact_id: str | int | None = None,
    track_meta: Mapping[str, Any] | None = None,
    far_target: float | None = None,
    **kwargs: Any,
) -> DefenseDecision:
    """Score one contact under ``defense`` — the parity entry point for 6.x."""
    return defense.evaluate(
        assertion,
        features,
        presence=presence,
        purpose=purpose,
        contact_id=contact_id,
        track_meta=track_meta,
        far_target=far_target,
        **kwargs,
    )
