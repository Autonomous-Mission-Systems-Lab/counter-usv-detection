"""Class–kinematics consistency scorer (defense interface).

Given an EO-asserted class and a contact's kinematic features, returns an
anomaly score against the matching benign envelope (or abstains). Thresholds
are FAR-target percentiles from held-out benign calibration — exposed as a
knob for DDR/FAR curves.

Firewall
--------
Hostile / ``usv`` / ``non_target`` tracks must never enter training or
calibration. Call ``assert_benign_train_allowed`` (or pass
``purpose="train"``) before using track metadata as fit material. Runtime
defense scoring and the explicit ``purpose="eval"`` path accept any contact —
including hostile/adaptive kinematics synthesized only at evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd
import yaml

from counterusv.kinematics.behavior_model import (
    EnvelopeModel,
    ModelName,
    MultiHorizonEnvelope,
    load_envelope,
)
from counterusv.models.detector import Detection

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENVELOPE_DIR = REPO_ROOT / "results" / "behavior_model" / "envelopes"
DEFAULT_MAP = REPO_ROOT / "configs" / "defense" / "class_envelope_map.yaml"
DEFAULT_MODEL_CFG = REPO_ROOT / "configs" / "defense" / "behavior_model.yaml"
DEFAULT_FREEZE = REPO_ROOT / "results" / "behavior_model" / "FROZEN.json"

Purpose = Literal["defense", "eval", "train"]

# Roles / sources that must never enter benign-envelope training or calibration.
_BLOCKED_ROLES = frozenset({"hostile", "non_target"})
_BLOCKED_SOURCES = frozenset({"usv"})
_BLOCKED_CANONICAL = frozenset({"usv", "military"})


class FirewallError(PermissionError):
    """Raised when hostile / usv / non_target material is offered for training."""


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _far_key(far_target: float) -> str:
    return f"far_{float(far_target):g}"


def assert_benign_train_allowed(
    meta: Mapping[str, Any] | pd.Series | None,
) -> None:
    """Refuse hostile / ``usv`` / ``non_target`` tracks as training material.

    Runtime defense scoring does **not** call this — the defense must score
    contacts of unknown (possibly hostile) identity. Use this (or
    ``purpose="train"``) only when a track would enter fit / calibration.
    """
    if meta is None:
        raise FirewallError(
            "track metadata required when purpose='train' "
            "(cannot attest benign-only without role/source)."
        )
    get = meta.get if hasattr(meta, "get") else lambda k, d=None: meta[k] if k in meta else d  # type: ignore[index]

    role = str(get("role") or "").strip().lower()
    source = str(get("source") or "").strip().lower()
    canon = str(
        get("canonical_class") or get("class_name") or ""
    ).strip().lower()

    reasons: list[str] = []
    if role in _BLOCKED_ROLES:
        reasons.append(f"role={role!r}")
    if source in _BLOCKED_SOURCES:
        reasons.append(f"source={source!r}")
    if canon in _BLOCKED_CANONICAL:
        reasons.append(f"canonical_class={canon!r}")
    if reasons:
        raise FirewallError(
            "firewall: refusing non-benign track as training/calibration "
            f"material ({', '.join(reasons)}). Hostile/adaptive tracks are "
            "accepted only through purpose='eval'."
        )


def filter_benign_training(
    df: pd.DataFrame,
    *,
    raise_on_blocked: bool = False,
) -> pd.DataFrame:
    """Drop rows that the firewall would refuse for training.

    If ``raise_on_blocked`` and any blocked row is present, raises
    :class:`FirewallError` instead of silently filtering.
    """
    if df.empty:
        return df
    blocked = pd.Series(False, index=df.index)
    if "role" in df.columns:
        role = df["role"].astype(str).str.lower()
        blocked |= role.isin(_BLOCKED_ROLES)
        blocked |= role.notna() & role.ne("") & role.ne("benign") & role.ne("nan")
    if "source" in df.columns:
        blocked |= df["source"].astype(str).str.lower().isin(_BLOCKED_SOURCES)
    if "canonical_class" in df.columns:
        blocked |= df["canonical_class"].astype(str).str.lower().isin(
            _BLOCKED_CANONICAL
        )
    n_blocked = int(blocked.sum())
    if n_blocked and raise_on_blocked:
        raise FirewallError(
            f"firewall: {n_blocked} non-benign row(s) in candidate training "
            "table; hostile/usv/non_target must not enter fit/calibration."
        )
    return df.loc[~blocked].copy()


@dataclass(frozen=True)
class ConsistencyResult:
    """Outcome of one consistency check.

    ``score`` is the envelope **anomaly** score (higher = more inconsistent
    with the asserted class's benign kinematics). ``is_inconsistent`` is
    ``score > threshold`` at the requested FAR target; ``None`` when abstained
    or unscored.
    """

    asserted_class: str
    status: str  # scored | abstain | unknown_class | missing_envelope | no_window | nan_features
    envelope_used: str | None
    score: float | None
    is_inconsistent: bool | None
    far_target: float
    threshold: float | None
    subspace: str | None = None
    window_s: int | None = None
    model: str | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConsistencyScorer:
    """EO-asserted class × track features → consistency (anomaly) score.

    Parameters
    ----------
    envelopes
        Name → fitted :class:`MultiHorizonEnvelope` (or single-horizon
        :class:`EnvelopeModel` wrapped at load).
    eo_class_map
        From ``configs/defense/class_envelope_map.yaml`` ``eo_class_map``.
    far_target
        Default FAR operating point (e.g. 0.05). Overridable per call.
    primary_model
        Model family used for scoring (default ``gmm``).
    """

    def __init__(
        self,
        envelopes: Mapping[str, MultiHorizonEnvelope | EnvelopeModel],
        eo_class_map: Mapping[str, Mapping[str, Any]],
        *,
        far_target: float = 0.05,
        primary_model: ModelName = "gmm",
        far_targets: Sequence[float] | None = None,
    ) -> None:
        self.envelopes: dict[str, MultiHorizonEnvelope | EnvelopeModel] = dict(
            envelopes
        )
        self.eo_class_map = {str(k): dict(v) for k, v in eo_class_map.items()}
        self.far_target = float(far_target)
        self.primary_model: ModelName = primary_model  # type: ignore[assignment]
        self.far_targets = list(
            far_targets if far_targets is not None else [0.01, 0.05, 0.10]
        )

    @classmethod
    def from_artifacts(
        cls,
        *,
        envelope_dir: Path = DEFAULT_ENVELOPE_DIR,
        map_path: Path = DEFAULT_MAP,
        model_cfg: Path = DEFAULT_MODEL_CFG,
        far_target: float | None = None,
        primary_model: ModelName | None = None,
        extra_envelopes: Sequence[str] | None = None,
    ) -> "ConsistencyScorer":
        """Load envelope map + fitted bundles from disk (pre- / post-freeze).

        Map-listed envelopes are always loaded. Additional names from
        ``extra_envelopes`` and ``pooled_ablation`` in the model config are
        loaded when the joblib exists — without being added to
        ``class_envelope_map.yaml`` (runtime override / ablation path).
        """
        emap = _load_yaml(map_path)
        cfg = _load_yaml(model_cfg) if model_cfg.is_file() else {}
        cal = cfg.get("calibration") or {}
        models_cfg = cfg.get("models") or {}
        ft = float(
            far_target
            if far_target is not None
            else cal.get("default_far", 0.05)
        )
        pm = str(
            primary_model
            if primary_model is not None
            else models_cfg.get("primary", "gmm")
        )
        names: list[str] = list(emap.get("envelopes") or {})
        pooled_cfg = cfg.get("pooled_ablation") or {}
        if bool(pooled_cfg.get("enabled", False)):
            pname = str(pooled_cfg.get("name") or "pooled_benign")
            if pname not in names:
                names.append(pname)
        for name in extra_envelopes or ():
            if name and name not in names:
                names.append(str(name))
        envelopes: dict[str, MultiHorizonEnvelope | EnvelopeModel] = {}
        for name in names:
            path = Path(envelope_dir) / f"{name}.joblib"
            if not path.is_file():
                continue
            envelopes[name] = load_envelope(path)
        if not envelopes:
            raise FileNotFoundError(
                f"No envelope artifacts under {envelope_dir}. "
                f"Run scripts/behavior/fit_behavior_model.py first."
            )
        return cls(
            envelopes,
            emap.get("eo_class_map") or {},
            far_target=ft,
            primary_model=pm,  # type: ignore[arg-type]
            far_targets=list(cal.get("far_targets") or [0.01, 0.05, 0.10]),
        )

    @classmethod
    def from_freeze(
        cls,
        freeze_path: Path = DEFAULT_FREEZE,
        *,
        verify_digests: bool = True,
        far_target: float | None = None,
        primary_model: ModelName | None = None,
    ) -> "ConsistencyScorer":
        """Load the pinned freeze manifest (digests optional but default on).

        Prefer this for freeze-pinned consumers so they pin the attested artifact
        set. Config YAML digests are not checked (git owns configs); envelope
        and non-git data-pin digests are.
        """
        if not freeze_path.is_file():
            raise FileNotFoundError(
                f"Missing freeze manifest {freeze_path}. "
                f"Run scripts/behavior/freeze_behavior_model.py first."
            )
        data = json.loads(freeze_path.read_text())
        cfg_block = (data.get("configs") or {}).get("class_envelope_map") or {}
        map_rel = cfg_block.get("path") or "configs/defense/class_envelope_map.yaml"
        map_path = REPO_ROOT / map_rel
        model_rel = (
            (data.get("configs") or {}).get("behavior_model") or {}
        ).get("path") or "configs/defense/behavior_model.yaml"
        model_cfg = REPO_ROOT / model_rel

        if verify_digests:
            # Config YAMLs are path-only (git owns them). Digests cover envelopes
            # and any non-git data pins nested under configs (e.g. placements).
            for label, block in (data.get("configs") or {}).items():
                rel = block.get("path")
                if not rel or not block.get("sha256"):
                    continue
                if rel.startswith("configs/"):
                    continue
                p = REPO_ROOT / rel
                got = _file_sha256(p)
                exp = block["sha256"]
                if got != exp:
                    raise ValueError(
                        f"freeze digest mismatch for data pin {label}: "
                        f"{p} sha256={got[:12]}… expected {exp[:12]}…"
                    )
            for name, block in (data.get("envelopes") or {}).items():
                p = REPO_ROOT / block["path"]
                got = _file_sha256(p)
                exp = block.get("sha256")
                if exp and got != exp:
                    raise ValueError(
                        f"freeze digest mismatch for envelope {name}: "
                        f"{p} sha256={got[:12]}… expected {exp[:12]}…"
                    )

        # Envelope dir = parent of any listed envelope path
        env_entries = data.get("envelopes") or {}
        if not env_entries:
            raise FileNotFoundError("Freeze manifest lists no envelopes.")
        first = next(iter(env_entries.values()))
        envelope_dir = (REPO_ROOT / first["path"]).parent

        ft = far_target
        if ft is None:
            ft = (data.get("far_floor") or {}).get("default_far")
        pm = primary_model or data.get("primary_model") or "gmm"
        # Freeze roster may include ablation envelopes (e.g. pooled_benign)
        # that are not in class_envelope_map — pass them as extras so they load.
        freeze_extras = list(env_entries.keys())
        pooled = data.get("pooled_ablation")
        if pooled and str(pooled) not in freeze_extras:
            freeze_extras.append(str(pooled))
        return cls.from_artifacts(
            envelope_dir=envelope_dir,
            map_path=map_path,
            model_cfg=model_cfg,
            far_target=ft,
            primary_model=pm,  # type: ignore[arg-type]
            extra_envelopes=freeze_extras,
        )

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------

    def attach_envelope(
        self,
        name: str,
        path: Path | str,
    ) -> None:
        """Load an extra envelope joblib into ``self.envelopes`` (ablation).

        Does not modify ``eo_class_map``. Prefer ``from_freeze`` when the
        kinematics freeze already lists the ablation (``pooled_ablation`` /
        roster entry); this helper remains for ad-hoc overrides.
        """
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.is_file():
            raise FileNotFoundError(f"envelope artifact not found: {p}")
        self.envelopes[str(name)] = load_envelope(p)

    def resolve_envelope(
        self,
        asserted_class: str,
        *,
        envelope_override: str | None = None,
    ) -> tuple[str, str | None, str | None]:
        """Return ``(policy, envelope_name, reason)`` for an EO class name.

        When ``envelope_override`` is set and the class is scoreable, the
        override name replaces the map envelope. Abstain / unknown_class are
        unchanged (override ignored) so routing policy stays map-driven.
        """
        entry = self.eo_class_map.get(asserted_class)
        if entry is None:
            return "unknown_class", None, f"class {asserted_class!r} not in eo_class_map"
        policy = str(entry.get("policy") or "abstain")
        if policy == "abstain":
            return "abstain", None, entry.get("reason")
        env = entry.get("envelope")
        if not env:
            return "abstain", None, "score policy but no envelope name"
        if envelope_override:
            return "score", str(envelope_override), "envelope_override"
        return "score", str(env), entry.get("reason")

    def scoreable_classes(self) -> list[str]:
        return sorted(
            c for c, e in self.eo_class_map.items()
            if e.get("policy") == "score"
        )

    def abstain_classes(self) -> list[str]:
        return sorted(
            c for c, e in self.eo_class_map.items()
            if e.get("policy") == "abstain"
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        asserted_class: str,
        features: Mapping[str, Any] | pd.Series | None = None,
        *,
        features_by_window: Mapping[int, Mapping[str, Any] | pd.Series] | None = None,
        complete_windows: set[int] | Sequence[int] | None = None,
        far_target: float | None = None,
        model_name: ModelName | None = None,
        purpose: Purpose = "defense",
        track_meta: Mapping[str, Any] | pd.Series | None = None,
        envelope_override: str | None = None,
    ) -> ConsistencyResult:
        """Score ``(asserted_class, track features)`` against the benign envelope.

        Parameters
        ----------
        asserted_class
            EO class name (e.g. from ``Detection.class_name``).
        features
            Single-horizon feature row (used when ``features_by_window`` is
            omitted). Treated as the primary / only available window.
        features_by_window
            Multi-horizon feature rows keyed by window length in seconds.
        complete_windows
            Horizons whose window is complete for this contact. Defaults to
            the keys of ``features_by_window``, or the envelope primary window
            when only ``features`` is given.
        far_target
            Operating-point FAR (default: scorer's ``far_target``).
        purpose
            ``defense`` — runtime scoring (any contact).
            ``eval`` — evaluation path; hostile/adaptive tracks allowed.
            ``train`` — training/calibration; firewall refuses non-benign meta.
        track_meta
            Optional role/source/canonical_class tags. Required when
            ``purpose="train"``.
        envelope_override
            When set and the asserted class is scoreable, score against this
            envelope name instead of the class map (pooled ablation). Abstain
            / unknown classes ignore the override.
        """
        far = float(self.far_target if far_target is None else far_target)
        model = model_name or self.primary_model

        if purpose == "train":
            assert_benign_train_allowed(track_meta)
        elif purpose == "eval":
            pass  # hostile / adaptive explicitly allowed
        elif purpose != "defense":
            raise ValueError(f"unknown purpose {purpose!r}")

        policy, env_name, reason = self.resolve_envelope(
            asserted_class, envelope_override=envelope_override
        )
        if policy == "unknown_class":
            return ConsistencyResult(
                asserted_class=asserted_class,
                status="unknown_class",
                envelope_used=None,
                score=None,
                is_inconsistent=None,
                far_target=far,
                threshold=None,
                note=reason,
            )
        if policy == "abstain":
            return ConsistencyResult(
                asserted_class=asserted_class,
                status="abstain",
                envelope_used=None,
                score=None,
                is_inconsistent=None,
                far_target=far,
                threshold=None,
                note=str(reason) if reason else "unscoreable EO class",
            )
        assert env_name is not None
        bundle = self.envelopes.get(env_name)
        if bundle is None:
            return ConsistencyResult(
                asserted_class=asserted_class,
                status="missing_envelope",
                envelope_used=env_name,
                score=None,
                is_inconsistent=None,
                far_target=far,
                threshold=None,
                note=f"envelope artifact {env_name!r} not loaded",
            )

        rows_by_w, complete = self._resolve_windows(
            bundle, features, features_by_window, complete_windows
        )
        raw = self._score_bundle(bundle, rows_by_w, complete, model_name=model)
        score = raw.get("score")
        note = raw.get("note")
        window_s = raw.get("window_s")
        subspace = raw.get("subspace")
        if score is None:
            status = "no_window" if note == "no_complete_window" else (
                "nan_features" if note == "nan_features" else "no_window"
            )
            return ConsistencyResult(
                asserted_class=asserted_class,
                status=status,
                envelope_used=env_name,
                score=None,
                is_inconsistent=None,
                far_target=far,
                threshold=None,
                subspace=subspace,
                window_s=window_s,
                model=model,
                note=note,
            )

        thr = self._threshold(bundle, subspace or "core", model, far, window_s)
        inconsistent = bool(score > thr) if thr is not None else None
        return ConsistencyResult(
            asserted_class=asserted_class,
            status="scored",
            envelope_used=env_name,
            score=float(score),
            is_inconsistent=inconsistent,
            far_target=far,
            threshold=thr,
            subspace=subspace,
            window_s=int(window_s) if window_s is not None else None,
            model=model,
            note=None,
        )

    def score_detection(
        self,
        detection: Detection,
        features: Mapping[str, Any] | pd.Series | None = None,
        **kwargs: Any,
    ) -> ConsistencyResult:
        """Score using ``detection.class_name`` (DetectorBaseline wiring)."""
        return self.score(detection.class_name, features, **kwargs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_windows(
        self,
        bundle: MultiHorizonEnvelope | EnvelopeModel,
        features: Mapping[str, Any] | pd.Series | None,
        features_by_window: Mapping[int, Mapping[str, Any] | pd.Series] | None,
        complete_windows: set[int] | Sequence[int] | None,
    ) -> tuple[dict[int, Mapping[str, Any] | pd.Series], set[int]]:
        if features_by_window:
            rows = {int(w): v for w, v in features_by_window.items()}
            if complete_windows is None:
                complete = set(rows)
            else:
                complete = {int(w) for w in complete_windows}
            return rows, complete

        if features is None:
            return {}, set()

        if isinstance(bundle, MultiHorizonEnvelope):
            w = int(bundle.primary_window_s)
            avail = bundle.available_windows()
            if complete_windows is not None:
                complete = {int(x) for x in complete_windows}
                # Single feature row: attach to longest requested complete window
                # that we have a model for; else primary.
                cand = [c for c in complete if c in avail]
                w = max(cand) if cand else w
            return {w: features}, {w}

        w = int(bundle.window_s)
        return {w: features}, {w}

    def _score_bundle(
        self,
        bundle: MultiHorizonEnvelope | EnvelopeModel,
        rows_by_w: dict[int, Mapping[str, Any] | pd.Series],
        complete: set[int],
        *,
        model_name: ModelName,
    ) -> dict[str, Any]:
        if isinstance(bundle, MultiHorizonEnvelope):
            return bundle.score_row(rows_by_w, complete, model_name=model_name)
        # Single-horizon EnvelopeModel
        w = int(bundle.window_s)
        if w not in complete or w not in rows_by_w:
            if not rows_by_w:
                return {"score": None, "window_s": None, "note": "no_complete_window"}
            # use whatever row we have
            w_use = next(iter(rows_by_w))
            res = bundle.score_row(rows_by_w[w_use], model_name=model_name)
            res["window_s"] = w_use
            return res
        res = bundle.score_row(rows_by_w[w], model_name=model_name)
        res["window_s"] = w
        return res

    def _threshold(
        self,
        bundle: MultiHorizonEnvelope | EnvelopeModel,
        subspace: str,
        model_name: ModelName,
        far_target: float,
        window_s: int | None,
    ) -> float | None:
        env: EnvelopeModel
        if isinstance(bundle, MultiHorizonEnvelope):
            w = window_s if window_s in bundle.horizons else bundle.primary_window_s
            if w not in bundle.horizons:
                return None
            env = bundle.horizons[w]
        else:
            env = bundle
        fams = env.subspaces.get(subspace) or env.subspaces.get("core") or {}
        fit = fams.get(model_name)
        if fit is None:
            return None
        return fit.thresholds.get(_far_key(far_target))
