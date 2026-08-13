"""Real-track label-swap helpers (synthesis-free discriminability).

Filters attack-run-like benign windows and decides which splits are admissible
when scoring under a swapped asserted class. Scoring orchestration lives in
``scripts/behavior/run_label_swap.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CFG = REPO_ROOT / "configs" / "defense" / "label_swap.yaml"
DEFAULT_MAP = REPO_ROOT / "configs" / "defense" / "class_envelope_map.yaml"

PROXY_SOURCE_DEFAULT = frozenset({"recreational", "fishing", "sailing"})


def load_label_swap_config(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_CFG
    if not p.is_absolute():
        p = REPO_ROOT / p
    return yaml.safe_load(p.read_text()) or {}


def load_envelope_map(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_MAP
    if not p.is_absolute():
        p = REPO_ROOT / p
    return yaml.safe_load(p.read_text()) or {}


def attack_run_mask(
    df: pd.DataFrame,
    *,
    sog_p95_min: float = 30.0,
    straightness_min: float = 0.95,
) -> pd.Series:
    """Boolean mask: complete-window kinematics in the attack-run band."""
    if df.empty:
        return pd.Series(dtype=bool)
    sog = pd.to_numeric(df["sog_p95"], errors="coerce")
    straight = pd.to_numeric(df["straightness"], errors="coerce")
    mask = (sog >= float(sog_p95_min)) & (straight >= float(straightness_min))
    if "window_complete" in df.columns:
        mask = mask & df["window_complete"].astype(bool)
    if "role" in df.columns:
        mask = mask & (df["role"].astype(str) == "benign")
    return mask.fillna(False)


def scoreable_eo_classes(emap: Mapping[str, Any] | None = None) -> list[str]:
    emap = dict(emap or load_envelope_map())
    listed = list(emap.get("scoreable_eo_classes") or [])
    if listed:
        return [str(c) for c in listed]
    return sorted(
        c for c, e in (emap.get("eo_class_map") or {}).items()
        if (e or {}).get("policy") == "score"
    )


def resolve_envelope_name(
    asserted_class: str,
    emap: Mapping[str, Any] | None = None,
) -> str | None:
    emap = dict(emap or load_envelope_map())
    entry = (emap.get("eo_class_map") or {}).get(asserted_class) or {}
    if str(entry.get("policy") or "") != "score":
        return None
    env = entry.get("envelope")
    return str(env) if env else None


def envelope_member_classes(
    envelope_name: str,
    emap: Mapping[str, Any] | None = None,
) -> set[str]:
    emap = dict(emap or load_envelope_map())
    members = ((emap.get("envelopes") or {}).get(envelope_name) or {}).get("members") or {}
    return {str(c) for c in (members.get("canonical_class") or [])}


def touches_proxy_pool(
    source_class: str,
    asserted_class: str,
    *,
    cfg: Mapping[str, Any] | None = None,
    emap: Mapping[str, Any] | None = None,
) -> bool:
    """True when the pairing risks train leakage into the Class-B proxy."""
    cfg = dict(cfg or {})
    emap = dict(emap or load_envelope_map())
    proxy_env = str(cfg.get("proxy_envelope") or "small_craft_class_b_proxy")
    proxy_sources = {
        str(c) for c in (cfg.get("proxy_source_classes") or PROXY_SOURCE_DEFAULT)
    }
    proxy_asserted = {
        str(c) for c in (cfg.get("proxy_asserted_eo_classes") or ["small_craft"])
    }
    env_name = resolve_envelope_name(asserted_class, emap)
    if env_name == proxy_env or asserted_class in proxy_asserted:
        return source_class in proxy_sources
    return False


def admissible_splits(
    source_class: str,
    asserted_class: str,
    *,
    cfg: Mapping[str, Any] | None = None,
    emap: Mapping[str, Any] | None = None,
) -> set[str]:
    """Splits allowed when scoring ``source_class`` under ``asserted_class``.

    When the asserted envelope's train members exclude the source class, train
    rows are admissible (they never trained that envelope). Pairings that touch
    the Class-B small-craft proxy are held-out only.
    """
    cfg = dict(cfg or load_label_swap_config())
    emap = dict(emap or load_envelope_map())
    all_splits = {str(s) for s in (cfg.get("all_splits") or ["train", "val", "test"])}
    held_out = {str(s) for s in (cfg.get("held_out_splits") or ["val", "test"])}

    if source_class == asserted_class:
        # Matched-class control: held-out only (standard FAR hygiene).
        return set(held_out)

    if touches_proxy_pool(source_class, asserted_class, cfg=cfg, emap=emap):
        return set(held_out)

    env_name = resolve_envelope_name(asserted_class, emap)
    if env_name is None:
        return set()
    members = envelope_member_classes(env_name, emap)
    if source_class in members:
        # Source class helped train this envelope → held-out only.
        return set(held_out)
    return set(all_splits)


def swap_pairs(
    source_classes: Sequence[str],
    asserted_classes: Sequence[str],
) -> list[tuple[str, str]]:
    """Off-diagonal (source, asserted) pairs."""
    out: list[tuple[str, str]] = []
    for src in source_classes:
        for asserted in asserted_classes:
            if str(src) == str(asserted):
                continue
            out.append((str(src), str(asserted)))
    return out


def matrix_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    rate_key: str = "flag_rate",
    n_key: str = "n_scored",
    source_key: str = "source_class",
    asserted_key: str = "asserted_class",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot cell records into (rate_matrix, n_matrix)."""
    rows = list(records)
    if not rows:
        empty = pd.DataFrame()
        return empty, empty
    df = pd.DataFrame(rows)
    rates = df.pivot(index=source_key, columns=asserted_key, values=rate_key)
    ns = df.pivot(index=source_key, columns=asserted_key, values=n_key)
    return rates, ns


def thin_n_label(n: int | None, threshold: int = 20) -> str | None:
    if n is None:
        return None
    if int(n) < int(threshold):
        return "thin-n"
    return None
