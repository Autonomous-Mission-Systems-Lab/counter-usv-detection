"""Defense pipeline — Detection / OracleAssertion + track → decision.

Wires the frozen :class:`~counterusv.defense.consistency.ConsistencyScorer`
to EO detections and perfect-disguise oracle assertions at a named FAR
operating point. Implements the shared :class:`~counterusv.defense.harness.DefenseBackend`
surface (``kind="consistency"``) so evaluation can swap this defense with
presence-only (and later baselines) in one harness.

**Association assumption.** The world-frame track is *given* (EO tracker
and/or coastal radar fusion). Contact↔track linking is out of scope
(``docs/THREAT_MODEL.md``); this module does not infer it.

Config: ``configs/defense/pipeline.yaml``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import yaml

from counterusv.attacks.oracle import OracleAssertion
from counterusv.defense.consistency import ConsistencyResult, ConsistencyScorer
from counterusv.models.detector import Detection

if TYPE_CHECKING:
    from counterusv.defense.presence import PresenceObservation, PresenceResult

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "defense" / "pipeline.yaml"

Purpose = Literal["defense", "eval", "train"]
DecisionAction = Literal["flag", "pass", "abstain"]
AssertionSource = Literal["detection", "oracle", "class_name"]
FeatureArm = Literal["kinematics_only", "kinematics_geometry"]
DefenseKind = Literal["consistency", "presence_only"]

ASSOCIATION_ASSUMPTION = (
    "World-frame track is given (EO tracker and/or coastal radar). "
    "Contact↔track association is out of scope."
)


@dataclass(frozen=True)
class PipelineConfig:
    version: int
    frozen: str
    far_target: float
    feature_arm: FeatureArm
    freeze_path: Path
    verify_digests: bool
    primary_model: str
    association_note: str
    path: Path | None = None


def load_pipeline_config(path: str | Path | None = None) -> PipelineConfig:
    p = Path(path) if path is not None else DEFAULT_CONFIG
    if not p.is_file():
        alt = Path(path) if path is not None else Path("configs/defense/pipeline.yaml")
        if alt.is_file():
            p = alt
        else:
            raise FileNotFoundError(f"pipeline config not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    scorer = raw.get("scorer") or {}
    assoc = raw.get("association") or {}
    arm = str(raw.get("feature_arm") or "kinematics_only")
    if arm not in ("kinematics_only", "kinematics_geometry"):
        raise ValueError(f"unknown feature_arm {arm!r}")
    arms = raw.get("arms") or {}
    arm_block = arms.get(arm) or {}
    freeze_rel = (
        arm_block.get("freeze_path")
        or scorer.get("freeze_path")
        or "results/behavior_model/FROZEN.json"
    )
    freeze_path = Path(freeze_rel)
    if not freeze_path.is_absolute():
        freeze_path = REPO_ROOT / freeze_path
    note = str(assoc.get("note") or ASSOCIATION_ASSUMPTION).strip()
    return PipelineConfig(
        version=int(raw.get("version", 1)),
        frozen=str(raw.get("frozen", "")),
        far_target=float(raw.get("far_target", 0.05)),
        feature_arm=arm,  # type: ignore[arg-type]
        freeze_path=freeze_path,
        verify_digests=bool(scorer.get("verify_digests", True)),
        primary_model=str(scorer.get("primary_model") or "gmm"),
        association_note=note,
        path=p.resolve(),
    )


@dataclass(frozen=True)
class DefenseDecision:
    """Operating-point decision for one contact.

    ``action``
        ``flag`` — scored inconsistent at ``far_target`` (consistency) or
        track-present ∧ EO miss/mislabel (presence);
        ``pass`` — scored consistent / presence satisfied;
        ``abstain`` — unscored (policy abstain, missing window, no track, …).

    ``defense_kind``
        ``consistency`` or ``presence_only``.
    """

    action: DecisionAction
    asserted_class: str
    far_target: float
    defense_kind: DefenseKind
    feature_arm: FeatureArm | None
    consistency: ConsistencyResult | None
    source: AssertionSource
    presence: PresenceResult | None = None
    contact_id: str | int | None = None
    true_class: str | None = None
    oracle_condition: str | None = None
    association: str = ASSOCIATION_ASSUMPTION
    note: str | None = None

    @property
    def is_flagged(self) -> bool:
        return self.action == "flag"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["consistency"] = (
            self.consistency.as_dict() if self.consistency is not None else None
        )
        d["presence"] = (
            self.presence.as_dict() if self.presence is not None else None
        )
        return d


def decide_action(result: ConsistencyResult) -> DecisionAction:
    """Map a :class:`ConsistencyResult` to ``flag`` / ``pass`` / ``abstain``."""
    if result.status != "scored" or result.is_inconsistent is None:
        return "abstain"
    return "flag" if result.is_inconsistent else "pass"


def _resolve_assertion(
    assertion: Detection | OracleAssertion | str,
) -> tuple[str, AssertionSource, Detection | None, str | int | None, str | None, str | None]:
    """Return asserted_class, source, detection-or-None, contact_id, true_class, oracle_condition."""
    if isinstance(assertion, OracleAssertion):
        return (
            assertion.asserted_class,
            "oracle",
            assertion.as_detection(),
            assertion.contact_id,
            assertion.true_class,
            assertion.condition,
        )
    if isinstance(assertion, Detection):
        return (
            assertion.class_name,
            "detection",
            assertion,
            None,
            None,
            None,
        )
    if isinstance(assertion, str):
        return (assertion, "class_name", None, None, None, None)
    raise TypeError(
        f"assertion must be Detection, OracleAssertion, or str; got {type(assertion)!r}"
    )


class DefensePipeline:
    """Detector / oracle → consistency scorer → operating-point decision.

    Implements the shared defense-backend surface (``kind="consistency"``).

    Parameters
    ----------
    scorer
        Loaded :class:`ConsistencyScorer` (typically ``from_freeze()``).
    config
        Pipeline operating-point + association note.
    """

    kind: Literal["consistency"] = "consistency"

    def __init__(
        self,
        scorer: ConsistencyScorer,
        config: PipelineConfig | None = None,
    ) -> None:
        self.scorer = scorer
        self.config = config or load_pipeline_config()
        if self.config.feature_arm == "kinematics_geometry":
            # Geometry arm requires geometry columns on the feature row;
            # missing geometry → scorer returns nan_features / abstain.
            if not self.config.freeze_path.is_file():
                raise FileNotFoundError(
                    f"kinematics_geometry freeze missing: {self.config.freeze_path}. "
                    f"Run scripts/behavior/fit_geometry_model.py first."
                )

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "DefensePipeline":
        cfg = load_pipeline_config(path)
        scorer = ConsistencyScorer.from_freeze(
            cfg.freeze_path,
            verify_digests=cfg.verify_digests,
            far_target=cfg.far_target,
            primary_model=cfg.primary_model,  # type: ignore[arg-type]
        )
        return cls(scorer, cfg)

    @classmethod
    def from_freeze(
        cls,
        freeze_path: str | Path | None = None,
        *,
        far_target: float | None = None,
        verify_digests: bool = True,
        pipeline_config: str | Path | None = None,
    ) -> "DefensePipeline":
        """Load scorer from the behavior-model freeze + pipeline config."""
        base = load_pipeline_config(pipeline_config)
        fp = Path(freeze_path) if freeze_path is not None else base.freeze_path
        if not fp.is_absolute():
            fp = REPO_ROOT / fp
        ft = float(base.far_target if far_target is None else far_target)
        cfg = PipelineConfig(
            version=base.version,
            frozen=base.frozen,
            far_target=ft,
            feature_arm=base.feature_arm,
            freeze_path=fp,
            verify_digests=verify_digests,
            primary_model=base.primary_model,
            association_note=base.association_note,
            path=base.path,
        )
        scorer = ConsistencyScorer.from_freeze(
            cfg.freeze_path,
            verify_digests=cfg.verify_digests,
            far_target=cfg.far_target,
            primary_model=cfg.primary_model,  # type: ignore[arg-type]
        )
        return cls(scorer, cfg)

    def evaluate(
        self,
        assertion: Detection | OracleAssertion | str | None = None,
        features: Mapping[str, Any] | None = None,
        *,
        presence: PresenceObservation | None = None,
        features_by_window: Mapping[int, Mapping[str, Any]] | None = None,
        complete_windows: set[int] | Sequence[int] | None = None,
        purpose: Purpose = "defense",
        track_meta: Mapping[str, Any] | None = None,
        far_target: float | None = None,
        contact_id: str | int | None = None,
        **score_kwargs: Any,
    ) -> DefenseDecision:
        """Score ``assertion`` against track features and decide at FAR target.

        Parameters
        ----------
        assertion
            :class:`Detection`, :class:`OracleAssertion`, or asserted class
            name string.
        features / features_by_window
            Track feature row(s) in the scorer contract. Association to the
            contact is assumed given — not computed here.
        presence
            Accepted for harness parity with presence-only; ignored here
            (consistency scores kinematics, not EO presence).
        purpose
            ``defense`` (runtime) or ``eval`` (hostile / adaptive allowed).
        """
        del presence  # harness parity only
        if assertion is None:
            raise ValueError(
                "DefensePipeline.evaluate requires assertion "
                "(Detection, OracleAssertion, or class name)"
            )
        (
            asserted_class,
            source,
            detection,
            cid_from_assertion,
            true_class,
            oracle_condition,
        ) = _resolve_assertion(assertion)
        far = float(
            self.config.far_target if far_target is None else far_target
        )
        if detection is not None:
            result = self.scorer.score_detection(
                detection,
                features,
                features_by_window=features_by_window,
                complete_windows=complete_windows,
                far_target=far,
                purpose=purpose,
                track_meta=track_meta,
                **score_kwargs,
            )
        else:
            result = self.scorer.score(
                asserted_class,
                features,
                features_by_window=features_by_window,
                complete_windows=complete_windows,
                far_target=far,
                purpose=purpose,
                track_meta=track_meta,
                **score_kwargs,
            )
        action = decide_action(result)
        note = None
        if action == "abstain" and result.note:
            note = result.note
        return DefenseDecision(
            action=action,
            asserted_class=asserted_class,
            far_target=far,
            defense_kind="consistency",
            feature_arm=self.config.feature_arm,
            consistency=result,
            presence=None,
            source=source,
            contact_id=contact_id if contact_id is not None else cid_from_assertion,
            true_class=true_class,
            oracle_condition=oracle_condition,
            association=self.config.association_note,
            note=note,
        )
