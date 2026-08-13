"""Access-level transfer eval (grey-box + black-box) for crafted patches.

White-box craft+score lives in the evasion/disguise runners. This module:

* classifies surrogate→target pairs as ``white`` / ``grey`` / ``black``
* saves / loads a **patch bank** (attacked letterbox canvas + patch tensor +
  placement) so transfer does not require re-optimization
* hard-evaluates saved attacks on a target family (including the held-out
  ``rtdetr_l``) with the same ESR/TMSR marine-EOT scoring as white-box

Config: ``configs/attacks/access_levels.yaml``. Protocol:
``docs/TRANSFER_PROTOCOL.md``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import yaml
from PIL import Image

from counterusv.attacks.disguise import aggregate_tmsr, score_instance_tmsr
from counterusv.attacks.evasion import (
    aggregate_esr,
    make_predict_fn,
    score_instance_esr,
    to_ultralytics_device,
)
from counterusv.attacks.patch import Placement
from counterusv.models.detector import DetectorBaseline

DEFAULT_CONFIG = Path("configs/attacks/access_levels.yaml")
_REPO_ROOT = Path(__file__).resolve().parents[3]

AccessLevel = Literal["white", "grey", "black"]


@dataclass(frozen=True)
class AccessLevelsConfig:
    version: int
    frozen: str
    surrogates: tuple[str, ...]
    held_out_target: str
    grey_box_peers: dict[str, str]
    patch_bank_dirname: str
    evaluation: dict[str, Any]
    path: Path | None = None

    @property
    def conf_threshold(self) -> float:
        return float(self.evaluation.get("conf_threshold", 0.25))

    @property
    def iou_match(self) -> float:
        return float(self.evaluation.get("iou_match", 0.5))

    @property
    def nms_iou(self) -> float:
        return float(self.evaluation.get("nms_iou", 0.7))

    @property
    def severity_axes(self) -> list[str]:
        return list(self.evaluation.get("severity_axes") or [])

    @property
    def severity_levels(self) -> list[str]:
        return list(self.evaluation.get("severity_levels") or [])


def load_access_levels_config(path: str | Path | None = None) -> AccessLevelsConfig:
    p = Path(path) if path is not None else _REPO_ROOT / DEFAULT_CONFIG
    if not p.is_file():
        alt = Path(path) if path is not None else Path(DEFAULT_CONFIG)
        if alt.is_file():
            p = alt
        else:
            raise FileNotFoundError(f"access_levels config not found: {p}")
    raw = yaml.safe_load(p.read_text())
    peers = {str(k): str(v) for k, v in (raw.get("grey_box_peers") or {}).items()}
    return AccessLevelsConfig(
        version=int(raw["version"]),
        frozen=str(raw["frozen"]),
        surrogates=tuple(raw.get("surrogates") or ["yolo11s", "yolo11l"]),
        held_out_target=str(raw.get("held_out_target") or "rtdetr_l"),
        grey_box_peers=peers,
        patch_bank_dirname=str(raw.get("patch_bank_dirname") or "patch_bank"),
        evaluation=dict(raw.get("evaluation") or {}),
        path=p.resolve(),
    )


def access_level(
    surrogate: str,
    target: str,
    cfg: AccessLevelsConfig | None = None,
) -> AccessLevel:
    """Classify a surrogate→target pair for reporting."""
    cfg = cfg or load_access_levels_config()
    if surrogate == target:
        return "white"
    if target == cfg.held_out_target:
        return "black"
    peer = cfg.grey_box_peers.get(surrogate)
    if peer is not None and target == peer:
        return "grey"
    if surrogate in cfg.surrogates and target in cfg.surrogates:
        return "grey"
    return "black"


def default_transfer_targets(
    surrogate: str,
    cfg: AccessLevelsConfig | None = None,
    *,
    include_white: bool = False,
) -> list[str]:
    """Default eval families for a surrogate (grey peer + held-out)."""
    cfg = cfg or load_access_levels_config()
    out: list[str] = []
    if include_white:
        out.append(surrogate)
    peer = cfg.grey_box_peers.get(surrogate)
    if peer and peer not in out:
        out.append(peer)
    if cfg.held_out_target not in out:
        out.append(cfg.held_out_target)
    return out


def load_eval_baseline(
    family: str,
    *,
    device: str = "auto",
) -> DetectorBaseline:
    """Load a detector for hard-eval only (held-out target allowed)."""
    ultra = to_ultralytics_device(device)
    try:
        return DetectorBaseline.from_freeze(family, device=ultra)
    except FileNotFoundError:
        return DetectorBaseline.from_roster(family, device=ultra)


@dataclass
class PatchBankEntry:
    image_id: int
    target_xyxy: tuple[float, float, float, float]
    placement: Placement
    attacked_rgb: np.ndarray  # HxWx3 uint8
    patch_chw: np.ndarray  # (3,H,W) float32 in [0,1]


def patch_bank_dir(craft_out_dir: Path, dirname: str = "patch_bank") -> Path:
    return Path(craft_out_dir) / dirname


def save_patch_bank_entry(
    bank_dir: Path,
    *,
    image_id: int,
    target_xyxy: Sequence[float],
    placement: Placement,
    attacked_rgb: np.ndarray,
    patch_chw: np.ndarray,
) -> dict[str, Any]:
    """Write one instance into an on-disk patch bank; return manifest row."""
    bank_dir = Path(bank_dir)
    (bank_dir / "attacked").mkdir(parents=True, exist_ok=True)
    (bank_dir / "patches").mkdir(parents=True, exist_ok=True)
    Image.fromarray(attacked_rgb).save(bank_dir / "attacked" / f"{image_id}.png")
    np.save(bank_dir / "patches" / f"{image_id}.npy", patch_chw.astype(np.float32))
    meta = {
        "image_id": int(image_id),
        "target_xyxy": [float(v) for v in target_xyxy],
        "placement": asdict(placement),
    }
    (bank_dir / "patches" / f"{image_id}_placement.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    return meta


def write_patch_bank_manifest(
    bank_dir: Path,
    *,
    attack: str,
    surrogate: str,
    instances: Sequence[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> None:
    bank_dir = Path(bank_dir)
    bank_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "attack": attack,
        "surrogate": surrogate,
        "n_instances": len(instances),
        "instances": list(instances),
        **(extra or {}),
    }
    (bank_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def load_patch_bank(bank_dir: Path) -> tuple[dict[str, Any], list[PatchBankEntry]]:
    """Load manifest + attacked canvases / patches from a craft output bank."""
    bank_dir = Path(bank_dir)
    man_path = bank_dir / "manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(
            f"patch bank manifest missing: {man_path} "
            "(re-run craft with --save-patches)"
        )
    manifest = json.loads(man_path.read_text())
    entries: list[PatchBankEntry] = []
    for row in manifest.get("instances") or []:
        iid = int(row["image_id"])
        atk_path = bank_dir / "attacked" / f"{iid}.png"
        patch_path = bank_dir / "patches" / f"{iid}.npy"
        if not atk_path.is_file():
            raise FileNotFoundError(f"missing attacked canvas: {atk_path}")
        attacked = np.asarray(Image.open(atk_path).convert("RGB"), dtype=np.uint8)
        patch = (
            np.load(patch_path).astype(np.float32)
            if patch_path.is_file()
            else np.zeros((3, 1, 1), dtype=np.float32)
        )
        pl = row.get("placement") or {}
        placement = Placement(
            x0=int(pl["x0"]),
            y0=int(pl["y0"]),
            side=int(pl["side"]),
            size_frac=float(pl.get("size_frac", 0.0)),
        )
        xyxy = (float(row["target_xyxy"][0]), float(row["target_xyxy"][1]),
                float(row["target_xyxy"][2]), float(row["target_xyxy"][3]))
        entries.append(
            PatchBankEntry(
                image_id=iid,
                target_xyxy=xyxy,
                placement=placement,
                attacked_rgb=attacked,
                patch_chw=patch,
            )
        )
    return manifest, entries


def transfer_gap(
    surrogate_rate: float | None,
    target_rate: float | None,
) -> float | None:
    """Target success − surrogate (white-box) success on the same attacks."""
    if surrogate_rate is None or target_rate is None:
        return None
    return float(target_rate) - float(surrogate_rate)


def base_rate_from_agg(
    agg: dict[str, dict[str, Any]],
    *,
    rate_key: str,
    axes: Sequence[str],
) -> dict[str, Any] | None:
    """Pick scale:L0 (or first axis L0) as the headline base rate cell."""
    if not agg:
        return None
    preferred = f"{axes[0]}:L0" if axes else None
    if preferred and preferred in agg:
        return agg[preferred]
    for k, v in agg.items():
        if k.endswith(":L0"):
            return v
    return next(iter(agg.values()), None)
