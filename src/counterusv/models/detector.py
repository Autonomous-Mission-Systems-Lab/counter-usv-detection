"""Frozen detector baselines — canonical inference interface.

Downstream EO attacks and the class–kinematics consistency defense consume
detections as ``(box, score, canonical_class)``. This module wraps Ultralytics
YOLO11 / RT-DETR so every family emits that interface with taxonomy names and
roles (hostile / benign / non_target), not framework-local class indices.

The held-out transfer target (``rtdetr_l``) is still loadable for white-box
eval, but ``assert_attack_crafting_allowed()`` refuses it so transfer attacks
cannot accidentally optimize against the sequestered family.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS_ROOT = REPO_ROOT / "results" / "detector_baselines"
DEFAULT_FREEZE = DEFAULT_WEIGHTS_ROOT / "FROZEN.json"
FAMILIES_YAML = REPO_ROOT / "configs" / "detector" / "families.yaml"
DEFAULT_DATA_YAML = REPO_ROOT / "data" / "eo_views" / "yolo" / "data.yaml"


@dataclass(frozen=True)
class Detection:
    """Canonical detection consumed by attacks and the class–kinematics defense.

    Coordinates are in the **native image** frame (Ultralytics xyxy after
    reversing letterbox), not the padded training canvas.
    """

    box_xyxy: tuple[float, float, float, float]
    score: float
    class_name: str
    class_id: int
    role: str  # hostile | benign | non_target

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def box_xywh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.box_xyxy
        return (x1, y1, x2 - x1, y2 - y1)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def load_class_maps(
    data_yaml: Path = DEFAULT_DATA_YAML,
) -> tuple[dict[int, str], dict[int, str], dict[str, int]]:
    """YOLO index → name, YOLO index → role, name → master category_id."""
    d = _load_yaml(data_yaml)
    names = {int(k): str(v) for k, v in (d.get("names") or {}).items()}
    roles = {int(k): str(v) for k, v in (d.get("roles") or {}).items()}
    master_path = REPO_ROOT / "data" / "annotations" / "coco_master.json"
    if master_path.is_file():
        master = json.loads(master_path.read_text())
        name_to_master = {
            c["name"]: int(c["id"]) for c in master.get("categories") or []
        }
    else:
        name_to_master = {n: i + 1 for i, n in sorted(names.items())}
    if not roles:
        tax = _load_yaml(REPO_ROOT / "data" / "taxonomy.yaml")
        role_by_name = {
            c["name"]: str(c.get("role", "benign"))
            for c in (tax.get("canonical_classes") or [])
        }
        roles = {i: role_by_name.get(n, "benign") for i, n in names.items()}
    return names, roles, name_to_master


def resolve_device(requested: str = "auto") -> str:
    import torch

    req = (requested or "auto").lower()
    if req != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_ultralytics(arch: str, weights: Path):
    from ultralytics import RTDETR, YOLO

    arch_l = (arch or "yolo11").lower()
    if arch_l in ("rtdetr", "rt-detr", "transformer"):
        return RTDETR(str(weights))
    return YOLO(str(weights))


class DetectorBaseline:
    """Frozen family loaded for inference with the canonical detection interface.

    Parameters
    ----------
    family
        Roster name (``yolo11s`` / ``yolo11l`` / ``rtdetr_l``).
    weights
        Path to ``best.pt``. Defaults to
        ``results/detector_baselines/<family>/weights/best.pt``.
    transfer_role
        ``surrogate`` or ``held_out_target`` (from the freeze / families roster).
    """

    def __init__(
        self,
        family: str,
        *,
        weights: Path | None = None,
        arch: str = "yolo11",
        transfer_role: str = "surrogate",
        imgsz: int = 640,
        device: str = "auto",
        data_yaml: Path = DEFAULT_DATA_YAML,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
    ) -> None:
        self.family = family
        self.arch = arch
        self.transfer_role = transfer_role
        self.imgsz = imgsz
        self.device = resolve_device(device)
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.weights = Path(
            weights
            or (DEFAULT_WEIGHTS_ROOT / family / "weights" / "best.pt")
        )
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"No weights for {family!r} at {self.weights}. "
                f"Pull from RunPod (docs/RUNPOD.md) or pass weights=."
            )
        self.yolo_names, self.yolo_roles, self.name_to_master = load_class_maps(data_yaml)
        # Master category ids are 1-indexed in the COCO master; YOLO indices are
        # 0-indexed. Downstream defense matches primarily on class_name.
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_ultralytics(self.arch, self.weights)
        return self._model

    @property
    def attack_crafting_allowed(self) -> bool:
        return self.transfer_role != "held_out_target"

    def assert_attack_crafting_allowed(self) -> None:
        """Raise if this family is the sequestered held-out transfer target."""
        if not self.attack_crafting_allowed:
            raise PermissionError(
                f"{self.family!r} is the held-out transfer target "
                f"(role={self.transfer_role}). Craft attacks only on surrogates "
                f"(see configs/detector/families.yaml / docs/TRANSFER_PROTOCOL.md)."
            )

    @classmethod
    def from_freeze(
        cls,
        family: str,
        freeze_path: Path = DEFAULT_FREEZE,
        **kwargs: Any,
    ) -> "DetectorBaseline":
        """Load a family from the frozen baseline manifest."""
        if not freeze_path.is_file():
            raise FileNotFoundError(
                f"Missing freeze manifest {freeze_path}. "
                f"Run scripts/detector/freeze_baselines.py first."
            )
        data = json.loads(freeze_path.read_text())
        fams = data.get("families") or {}
        if family not in fams:
            raise KeyError(
                f"Family {family!r} not in freeze manifest. "
                f"Available: {sorted(fams)}"
            )
        entry = fams[family]
        # Prefer the absolute path when it still resolves (same machine that
        # froze); otherwise fall back to weights_rel under the repo root so a
        # RunPod / other host can load the same freeze after a weights sync.
        weights = Path(entry["weights"])
        if not weights.is_file():
            rel = entry.get("weights_rel")
            if rel:
                alt = REPO_ROOT / rel
                if alt.is_file():
                    weights = alt
        return cls(
            family,
            weights=weights,
            arch=entry.get("arch", "yolo11"),
            transfer_role=entry.get("transfer_role", "surrogate"),
            imgsz=int(entry.get("imgsz", 640)),
            **kwargs,
        )

    @classmethod
    def from_roster(
        cls,
        family: str,
        families_yaml: Path = FAMILIES_YAML,
        **kwargs: Any,
    ) -> "DetectorBaseline":
        """Load a family using the live roster (pre-freeze / development)."""
        roster = _load_yaml(families_yaml)
        match = next(
            (f for f in (roster.get("families") or []) if f.get("name") == family),
            None,
        )
        if match is None:
            raise KeyError(f"Unknown family {family!r} in {families_yaml}")
        cfg_path = REPO_ROOT / match["config"]
        cfg = _load_yaml(cfg_path)
        # shallow merge extends
        if cfg.get("extends"):
            base = _load_yaml((cfg_path.parent / cfg["extends"]).resolve())
            merged = dict(base)
            for k, v in cfg.items():
                if k == "extends":
                    continue
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k] = {**merged[k], **v}
                else:
                    merged[k] = v
            cfg = merged
        det = cfg.get("detector") or {}
        return cls(
            family,
            arch=str(det.get("arch", "yolo11")),
            transfer_role=str(
                match.get("role") or det.get("transfer_role") or "surrogate"
            ),
            imgsz=int((cfg.get("train") or {}).get("imgsz", 640)),
            **kwargs,
        )

    def predict(
        self,
        source: str | Path | Sequence[str | Path],
        *,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        augment: bool = False,
        verbose: bool = False,
    ) -> list[list[Detection]]:
        """Run inference; return one Detection list per input image.

        ``source`` may be a path, a directory, or a list of paths (Ultralytics
        conventions). Boxes are native-image xyxy.
        """
        results = self.model.predict(
            source=source,
            imgsz=imgsz or self.imgsz,
            conf=self.conf if conf is None else conf,
            iou=self.iou if iou is None else iou,
            max_det=self.max_det,
            augment=augment,
            device=self.device,
            verbose=verbose,
        )
        out: list[list[Detection]] = []
        for r in results:
            dets: list[Detection] = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), sc, yi in zip(xyxy, confs, clss):
                    yi = int(yi)
                    name = self.yolo_names.get(yi)
                    if name is None:
                        continue
                    master_id = self.name_to_master.get(name)
                    if master_id is None:
                        continue
                    dets.append(
                        Detection(
                            box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                            score=float(sc),
                            class_name=name,
                            class_id=int(master_id),
                            role=self.yolo_roles.get(yi, "benign"),
                        )
                    )
            out.append(dets)
        return out

    def predict_one(
        self,
        source: str | Path,
        **kwargs: Any,
    ) -> list[Detection]:
        """Convenience: predict a single image → flat Detection list."""
        batches = self.predict(source, **kwargs)
        return batches[0] if batches else []
