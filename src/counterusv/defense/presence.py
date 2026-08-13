"""Presence-only cross-check — the RQ2 comparator.

An independent world-frame track exists, but EO missed or mislabeled the
contact. This is the existing-fusion-defense equivalent: it catches
**evasion** (track paint, no EO detection) and, by construction, cannot
reach **disguise** (EO detected the contact, just as a benign class).

Decision rules
--------------
- ``track_present`` and EO ``missed`` / ``mislabeled`` → ``flag``
- ``track_present`` and EO ``detected`` → ``pass`` (presence satisfied)
- no independent track → ``abstain``

``mislabeled`` means the EO output failed presence association (treated
like a miss for fusion purposes). A successful hostile→benign disguise
is ``detected`` — EO saw a vessel — so this layer never flags it.

Config: ``configs/defense/presence.yaml``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import yaml

from counterusv.attacks.oracle import OracleAssertion
from counterusv.defense.pipeline import (
    ASSOCIATION_ASSUMPTION,
    AssertionSource,
    DecisionAction,
    DefenseDecision,
    Purpose,
)
from counterusv.models.detector import Detection

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "defense" / "presence.yaml"

EoOutcome = Literal["missed", "detected", "mislabeled"]
_FLAG_OUTCOMES = frozenset({"missed", "mislabeled"})
_PASS_OUTCOMES = frozenset({"detected"})


@dataclass(frozen=True)
class PresenceConfig:
    version: int
    frozen: str
    flag_outcomes: tuple[str, ...]
    pass_outcomes: tuple[str, ...]
    association_note: str
    path: Path | None = None


def load_presence_config(path: str | Path | None = None) -> PresenceConfig:
    p = Path(path) if path is not None else DEFAULT_CONFIG
    if not p.is_file():
        alt = Path(path) if path is not None else Path("configs/defense/presence.yaml")
        if alt.is_file():
            p = alt
        else:
            raise FileNotFoundError(f"presence config not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    outcomes = raw.get("eo_outcomes") or {}
    flag = tuple(outcomes.get("flag") or ["missed", "mislabeled"])
    pas = tuple(outcomes.get("pass") or ["detected"])
    assoc = raw.get("association") or {}
    note = str(assoc.get("note") or ASSOCIATION_ASSUMPTION).strip()
    return PresenceConfig(
        version=int(raw.get("version", 1)),
        frozen=str(raw.get("frozen", "")),
        flag_outcomes=flag,
        pass_outcomes=pas,
        association_note=note,
        path=p.resolve(),
    )


@dataclass(frozen=True)
class PresenceObservation:
    """Inputs for the presence-only layer on one contact.

    Parameters
    ----------
    track_present
        Independent world-frame track exists (coastal radar / EO tracker).
    eo_outcome
        ``missed`` — no EO detection for this contact;
        ``detected`` — EO produced a detection (any vessel class, including
        a successful disguise);
        ``mislabeled`` — EO output that fails presence association (flagged
        like a miss).
    asserted_class
        Optional EO-asserted / oracle class for harness parity; ignored by
        the decision rule itself.
    """

    track_present: bool
    eo_outcome: EoOutcome
    asserted_class: str = ""
    contact_id: str | int | None = None
    true_class: str | None = None
    source: AssertionSource = "class_name"
    oracle_condition: str | None = None


@dataclass(frozen=True)
class PresenceResult:
    """Outcome of one presence-only check."""

    track_present: bool
    eo_outcome: EoOutcome
    status: str  # scored | abstain_no_track
    is_inconsistent: bool | None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_presence(
    obs: PresenceObservation,
    *,
    flag_outcomes: Sequence[str] | None = None,
    pass_outcomes: Sequence[str] | None = None,
) -> tuple[DecisionAction, PresenceResult]:
    """Map a presence observation to ``flag`` / ``pass`` / ``abstain``."""
    flag_set = frozenset(flag_outcomes) if flag_outcomes is not None else _FLAG_OUTCOMES
    pass_set = frozenset(pass_outcomes) if pass_outcomes is not None else _PASS_OUTCOMES
    outcome = obs.eo_outcome
    if outcome not in flag_set | pass_set:
        raise ValueError(f"unknown eo_outcome {outcome!r}")

    if not obs.track_present:
        result = PresenceResult(
            track_present=False,
            eo_outcome=outcome,
            status="abstain_no_track",
            is_inconsistent=None,
            note="no independent track — presence layer abstains",
        )
        return "abstain", result

    if outcome in flag_set:
        result = PresenceResult(
            track_present=True,
            eo_outcome=outcome,
            status="scored",
            is_inconsistent=True,
            note=f"track present, EO {outcome} — presence inconsistency",
        )
        return "flag", result

    result = PresenceResult(
        track_present=True,
        eo_outcome=outcome,
        status="scored",
        is_inconsistent=False,
        note="track present, EO detected — presence satisfied (cannot catch disguise)",
    )
    return "pass", result


def presence_for_evasion(
    *,
    contact_id: str | int | None = None,
    true_class: str | None = "usv",
    eo_outcome: EoOutcome = "missed",
) -> PresenceObservation:
    """Observation for a successful EO evasion: track paint, no (usable) EO."""
    if eo_outcome == "detected":
        raise ValueError("evasion observation cannot use eo_outcome='detected'")
    return PresenceObservation(
        track_present=True,
        eo_outcome=eo_outcome,
        asserted_class="",
        contact_id=contact_id,
        true_class=true_class,
        source="class_name",
    )


def presence_for_disguise(
    assertion: Detection | OracleAssertion | str,
    *,
    contact_id: str | int | None = None,
) -> PresenceObservation:
    """Observation for a disguise condition: EO detected the contact.

    By construction the presence layer **passes** — disguise is EO success
    with a wrong (benign) label, not an EO miss.
    """
    if isinstance(assertion, OracleAssertion):
        return PresenceObservation(
            track_present=True,
            eo_outcome="detected",
            asserted_class=assertion.asserted_class,
            contact_id=contact_id if contact_id is not None else assertion.contact_id,
            true_class=assertion.true_class,
            source="oracle",
            oracle_condition=assertion.condition,
        )
    if isinstance(assertion, Detection):
        return PresenceObservation(
            track_present=True,
            eo_outcome="detected",
            asserted_class=assertion.class_name,
            contact_id=contact_id,
            source="detection",
        )
    if isinstance(assertion, str):
        return PresenceObservation(
            track_present=True,
            eo_outcome="detected",
            asserted_class=assertion,
            contact_id=contact_id,
            source="class_name",
        )
    raise TypeError(
        f"assertion must be Detection, OracleAssertion, or str; got {type(assertion)!r}"
    )


class PresenceOnlyDefense:
    """Independent-track presence check behind the shared defense interface."""

    kind: Literal["presence_only"] = "presence_only"

    def __init__(self, config: PresenceConfig | None = None) -> None:
        self.config = config or load_presence_config()

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "PresenceOnlyDefense":
        return cls(load_presence_config(path))

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
        **kwargs: Any,
    ) -> DefenseDecision:
        """Decide at the presence layer.

        Prefer an explicit ``presence`` observation. If omitted and
        ``assertion`` is given, treat the contact as a **disguise**
        condition (EO detected) via :func:`presence_for_disguise` — the
        default that makes oracle/disguise cells score ``pass``.

        ``features`` / window kwargs are accepted for harness parity and
        ignored — presence does not score kinematics.
        """
        del features, features_by_window, complete_windows, purpose, track_meta, kwargs
        obs = presence
        if obs is None:
            if assertion is None:
                raise ValueError(
                    "PresenceOnlyDefense.evaluate requires presence=... "
                    "or an assertion (disguise → EO detected)"
                )
            obs = presence_for_disguise(assertion, contact_id=contact_id)
        elif contact_id is not None and obs.contact_id is None:
            obs = PresenceObservation(
                track_present=obs.track_present,
                eo_outcome=obs.eo_outcome,
                asserted_class=obs.asserted_class,
                contact_id=contact_id,
                true_class=obs.true_class,
                source=obs.source,
                oracle_condition=obs.oracle_condition,
            )

        action, result = decide_presence(
            obs,
            flag_outcomes=self.config.flag_outcomes,
            pass_outcomes=self.config.pass_outcomes,
        )
        return DefenseDecision(
            action=action,
            asserted_class=obs.asserted_class,
            far_target=float(far_target if far_target is not None else 0.0),
            defense_kind="presence_only",
            feature_arm=None,
            consistency=None,
            presence=result,
            source=obs.source,
            contact_id=obs.contact_id,
            true_class=obs.true_class,
            oracle_condition=obs.oracle_condition,
            association=self.config.association_note,
            note=result.note,
        )
