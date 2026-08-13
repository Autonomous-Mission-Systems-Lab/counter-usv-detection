"""Perfect-disguise oracle — no-patch benign class assertion (RQ2/RQ3).

The defense is *told* a hostile contact carries a chosen benign class
(``fishing`` / ``recreational``). No adversarial patch and no attack imagery —
models an ideal patch and zero-tech visual disguise
(``docs/THREAT_MODEL.md``, ``docs/METRICS.md``).

Defense evaluation consumes :class:`OracleAssertion` / oracle-emitted
:class:`Detection` records so DDR under the oracle condition is not
confounded by patch fragility. Patch-condition results decompose as
oracle performance × patch reliability.

Config: ``configs/attacks/oracle.yaml``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from counterusv.models.detector import Detection, load_class_maps

DEFAULT_CONFIG = Path("configs/attacks/oracle.yaml")
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class OracleConfig:
    version: int
    frozen: str
    true_class: str
    target_benign_classes: tuple[str, ...]
    assertion_score: float
    condition: str
    notes: str
    path: Path | None = None


def load_oracle_config(path: str | Path | None = None) -> OracleConfig:
    p = Path(path) if path is not None else _REPO_ROOT / DEFAULT_CONFIG
    if not p.is_file():
        alt = Path(path) if path is not None else Path(DEFAULT_CONFIG)
        if alt.is_file():
            p = alt
        else:
            raise FileNotFoundError(f"oracle config not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    benign = tuple(raw.get("target_benign_classes") or ["fishing", "recreational"])
    return OracleConfig(
        version=int(raw.get("version", 1)),
        frozen=str(raw.get("frozen", "")),
        true_class=str(raw.get("true_class", "usv")),
        target_benign_classes=benign,
        assertion_score=float(raw.get("assertion_score", 1.0)),
        condition=str(raw.get("condition", "perfect_disguise_oracle")),
        notes=str(raw.get("notes") or "").strip(),
        path=p.resolve(),
    )


@dataclass(frozen=True)
class OracleAssertion:
    """One perfect-disguise class assertion for the consistency scorer.

    ``asserted_class`` is what the defense sees. ``true_class`` is ground truth
    (hostile) — never passed to the scorer as the asserted label under this
    condition.
    """

    contact_id: str | int
    true_class: str
    asserted_class: str
    condition: str
    score: float
    role: str = "benign"
    box_xyxy: tuple[float, float, float, float] | None = None
    class_id: int | None = None
    source_detection: dict[str, Any] | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.box_xyxy is not None:
            d["box_xyxy"] = list(self.box_xyxy)
        return d

    def as_detection(self) -> Detection:
        """Materialize a canonical :class:`Detection` for ``score_detection``."""
        if self.box_xyxy is None:
            box = (0.0, 0.0, 0.0, 0.0)
        else:
            box = self.box_xyxy
        return Detection(
            box_xyxy=box,
            score=float(self.score),
            class_name=self.asserted_class,
            class_id=int(self.class_id if self.class_id is not None else -1),
            role=self.role,
        )


def _role_for_class(class_name: str) -> str:
    """Best-effort role from taxonomy / EO view maps."""
    try:
        _names, roles, _ = load_class_maps()
        name_to_idx = {v: k for k, v in _names.items()}
        if class_name in name_to_idx:
            return str(roles.get(name_to_idx[class_name], "benign"))
    except Exception:
        pass
    tax_path = _REPO_ROOT / "data" / "taxonomy.yaml"
    if tax_path.is_file():
        tax = yaml.safe_load(tax_path.read_text()) or {}
        for c in tax.get("canonical_classes") or []:
            if c.get("name") == class_name:
                return str(c.get("role", "benign"))
    return "benign"


def _class_id_for_name(class_name: str) -> int | None:
    try:
        names, _, _ = load_class_maps()
        for i, n in names.items():
            if n == class_name:
                return int(i)
    except Exception:
        return None
    return None


class PerfectDisguiseOracle:
    """No-patch label assertion: hostile contact → chosen benign class.

    Parameters
    ----------
    config
        Loaded :class:`OracleConfig` (defaults from ``oracle.yaml``).
    """

    def __init__(self, config: OracleConfig | None = None):
        self.config = config or load_oracle_config()

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "PerfectDisguiseOracle":
        return cls(load_oracle_config(path))

    def validate_benign(self, benign_class: str) -> None:
        if benign_class not in self.config.target_benign_classes:
            raise ValueError(
                f"benign_class {benign_class!r} not in "
                f"{self.config.target_benign_classes}"
            )

    def assert_class(
        self,
        benign_class: str,
        *,
        contact_id: str | int = -1,
        true_class: str | None = None,
        box_xyxy: Sequence[float] | None = None,
        detection: Detection | None = None,
        note: str | None = None,
    ) -> OracleAssertion:
        """Assert ``benign_class`` for a hostile contact (perfect disguise).

        Pass either ``box_xyxy`` or a source ``detection`` (box/score preserved
        geometrically; class/role replaced). ``true_class`` defaults to the
        config hostile class (``usv``).
        """
        self.validate_benign(benign_class)
        true = true_class or self.config.true_class
        src: dict[str, Any] | None = None
        box: tuple[float, float, float, float] | None = None
        if detection is not None:
            src = detection.as_dict()
            box = tuple(float(v) for v in detection.box_xyxy)  # type: ignore[assignment]
        if box_xyxy is not None:
            box = (
                float(box_xyxy[0]),
                float(box_xyxy[1]),
                float(box_xyxy[2]),
                float(box_xyxy[3]),
            )
        return OracleAssertion(
            contact_id=contact_id,
            true_class=true,
            asserted_class=benign_class,
            condition=self.config.condition,
            score=float(self.config.assertion_score),
            role=_role_for_class(benign_class),
            box_xyxy=box,
            class_id=_class_id_for_name(benign_class),
            source_detection=src,
            note=note,
        )

    def apply_to_detection(
        self,
        detection: Detection,
        benign_class: str,
        *,
        contact_id: str | int | None = None,
    ) -> OracleAssertion:
        """Replace a detector's class with the oracle benign assertion."""
        return self.assert_class(
            benign_class,
            contact_id=contact_id if contact_id is not None else -1,
            detection=detection,
            true_class=self.config.true_class,
            note="applied_to_detection",
        )

    def asserted_class_name(self, benign_class: str) -> str:
        """String form for ``ConsistencyScorer.score(asserted_class, ...)``."""
        self.validate_benign(benign_class)
        return benign_class

    def score_with(
        self,
        scorer: Any,
        benign_class: str,
        features: Mapping[str, Any] | None = None,
        *,
        contact_id: str | int = -1,
        detection: Detection | None = None,
        purpose: str = "eval",
        **score_kwargs: Any,
    ) -> tuple[OracleAssertion, Any]:
        """Assert then score via ``ConsistencyScorer`` (``purpose='eval'``).

        Returns ``(assertion, ConsistencyResult)``. Hostile / adaptive tracks
        are allowed under ``purpose='eval'``.
        """
        assertion = self.assert_class(
            benign_class, contact_id=contact_id, detection=detection
        )
        # Prefer score_detection when a box-bearing assertion exists.
        if detection is not None or assertion.box_xyxy is not None:
            result = scorer.score_detection(
                assertion.as_detection(), features, purpose=purpose, **score_kwargs
            )
        else:
            result = scorer.score(
                assertion.asserted_class, features, purpose=purpose, **score_kwargs
            )
        return assertion, result


def build_assertions_for_contacts(
    oracle: PerfectDisguiseOracle,
    contacts: Sequence[Mapping[str, Any]],
    benign_class: str,
) -> list[OracleAssertion]:
    """Build oracle assertions for a list of contact dicts.

    Each contact may include ``contact_id`` / ``image_id``, ``true_class``,
    ``box_xyxy``, and optional nested ``detection`` fields.
    """
    oracle.validate_benign(benign_class)
    out: list[OracleAssertion] = []
    for c in contacts:
        cid = c.get("contact_id", c.get("image_id", -1))
        box = c.get("box_xyxy") or c.get("target_xyxy")
        true = c.get("true_class")
        det = None
        if "detection" in c and isinstance(c["detection"], Detection):
            det = c["detection"]
        out.append(
            oracle.assert_class(
                benign_class,
                contact_id=cid,
                true_class=true,
                box_xyxy=box,
                detection=det,
            )
        )
    return out


def assertions_to_records(
    assertions: Sequence[OracleAssertion],
) -> list[dict[str, Any]]:
    """JSON-serializable records for ``results/attacks/oracle/``."""
    return [a.as_dict() for a in assertions]
