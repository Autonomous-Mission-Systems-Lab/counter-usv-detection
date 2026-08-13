"""Engagement-geometry spec — defended asset, annulus, placement policy.

Loads ``configs/defense/engagement_geometry.yaml``. Feature extraction and
FAR sweeps consume this contract; see ``docs/ENGAGEMENT_GEOMETRY.md`` for
the geography-vs-behavior rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "defense" / "engagement_geometry.yaml"


@dataclass(frozen=True)
class PortRegion:
    id: str
    approx_lat: float
    approx_lon: float
    note: str = ""


@dataclass(frozen=True)
class PlacementClass:
    """A defended-asset archetype (what is being protected), not an offset.

    ``role`` is ``fit`` for realistic assets that supply the benign envelope
    training population, or ``far_only`` for placements scored at false-alarm
    validation but never fit.
    """

    name: str
    role: str
    represents: str
    derivation: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def is_fit(self) -> bool:
        return self.role == "fit"


@dataclass(frozen=True)
class EngagementGeometryConfig:
    version: int
    frozen: str
    frame: dict[str, Any]
    asset: dict[str, Any]
    annulus: dict[str, Any]
    inbound_leg: dict[str, Any]
    placement_policy: dict[str, Any]
    consumers: dict[str, Any]
    path: Path | None = None

    @property
    def max_range_nm(self) -> float:
        return float(self.annulus["max_range_nm"])

    @property
    def min_range_nm(self) -> float:
        return float(self.annulus["min_range_nm"])

    @property
    def expected_cadence_s(self) -> float:
        return float(self.inbound_leg.get("expected_point_cadence_s", 60))

    def port_regions(self) -> list[PortRegion]:
        rows = (self.placement_policy.get("seed_port_regions") or [])
        return [
            PortRegion(
                id=str(r["id"]),
                approx_lat=float(r["approx_lat"]),
                approx_lon=float(r["approx_lon"]),
                note=str(r.get("note") or ""),
            )
            for r in rows
        ]

    def placement_classes(self) -> list[PlacementClass]:
        raw = self.placement_policy.get("placement_classes") or {}
        out: list[PlacementClass] = []
        for name, body in raw.items():
            body = dict(body or {})
            role = str(body.pop("role", "far_only")).strip()
            represents = str(body.pop("represents", "")).strip()
            derivation = str(body.pop("derivation", "")).strip()
            out.append(
                PlacementClass(
                    name=str(name),
                    role=role,
                    represents=represents,
                    derivation=derivation,
                    params=body,
                )
            )
        return out

    def fit_population(self) -> list[str]:
        """Placement classes whose encounters train the geometry envelopes.

        Declared explicitly in the config; cross-checked against per-class
        ``role`` so the two cannot drift apart silently.
        """
        declared = [str(n) for n in (self.placement_policy.get("fit_population") or [])]
        by_role = [c.name for c in self.placement_classes() if c.is_fit]
        if declared and set(declared) != set(by_role):
            raise ValueError(
                "engagement geometry: fit_population "
                f"{sorted(declared)} disagrees with role=fit classes {sorted(by_role)}"
            )
        return declared or by_role

    def in_annulus(self, range_nm: float) -> bool:
        r = float(range_nm)
        return self.min_range_nm <= r <= self.max_range_nm

    def geometry_scoreable(self, range_nm: float) -> bool:
        """True if geometry features should be computed (inside max range)."""
        return float(range_nm) <= self.max_range_nm


def load_engagement_geometry(
    path: str | Path | None = None,
) -> EngagementGeometryConfig:
    p = Path(path) if path is not None else DEFAULT_CONFIG
    if not p.is_file():
        alt = Path(path) if path is not None else Path(
            "configs/defense/engagement_geometry.yaml"
        )
        if alt.is_file():
            p = alt
        else:
            raise FileNotFoundError(f"engagement geometry config not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    required = ("annulus", "inbound_leg", "placement_policy", "asset", "frame")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"engagement geometry config missing keys: {missing}")
    return EngagementGeometryConfig(
        version=int(raw.get("version", 1)),
        frozen=str(raw.get("frozen", "")),
        frame=dict(raw["frame"] or {}),
        asset=dict(raw["asset"] or {}),
        annulus=dict(raw["annulus"] or {}),
        inbound_leg=dict(raw["inbound_leg"] or {}),
        placement_policy=dict(raw["placement_policy"] or {}),
        consumers=dict(raw.get("consumers") or {}),
        path=p.resolve(),
    )
