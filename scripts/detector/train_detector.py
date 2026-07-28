#!/usr/bin/env python3
"""Train EO detector baselines from ``configs/detector/``.

Reads the family roster in ``configs/detector/families.yaml`` (or a single
``--family`` / ``--config``), trains with Ultralytics (YOLO11 or RT-DETR) on the
harmonized EO view (``data/eo_views/yolo/data.yaml``), and writes under
``results/detector_baselines/<family>/``.

Designed for RunPod / single-GPU CUDA. Also supports ``--smoke`` (1 epoch,
small fraction) and ``--dry-run`` (print plan, no train) for laptop checks.

Usage
-----
    python scripts/detector/train_detector.py --all
    python scripts/detector/train_detector.py --family yolo11s
    python scripts/detector/train_detector.py --family yolo11s --smoke
    python scripts/detector/train_detector.py --all --dry-run
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

FAMILIES_YAML = REPO_ROOT / "configs" / "detector" / "families.yaml"
DEFAULT_OUT = REPO_ROOT / "results" / "detector_baselines"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def load_family_config(config_path: Path) -> dict[str, Any]:
    """Load a per-family config; shallow-merge ``extends`` base if present."""
    cfg = _load_yaml(config_path)
    extends = cfg.get("extends")
    if extends:
        base_path = (config_path.parent / extends).resolve()
        if base_path.is_file():
            base = _load_yaml(base_path)
            # Family wins on overlapping keys; nested dicts merged one level.
            merged = dict(base)
            for k, v in cfg.items():
                if k == "extends":
                    continue
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k] = {**merged[k], **v}
                else:
                    merged[k] = v
            cfg = merged
    cfg["_config_path"] = str(config_path)
    return cfg


def roster_families(families_yaml: Path = FAMILIES_YAML) -> list[dict]:
    data = _load_yaml(families_yaml)
    return list(data.get("families") or [])


def resolve_device(requested: str) -> str:
    """Map ``auto`` → cuda:0 / mps / cpu; otherwise return the user string."""
    import torch

    req = (requested or "auto").lower()
    if req != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_info(device: str) -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "device": device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    return info


def build_model(det: dict[str, Any]):
    """Instantiate Ultralytics YOLO or RTDETR from the family ``detector`` block."""
    from ultralytics import RTDETR, YOLO

    arch = str(det.get("arch", "yolo11")).lower()
    weights = det.get("pretrained") or det.get("model")
    if not weights:
        raise ValueError("family config detector.pretrained (or .model) is required")
    if arch in ("rtdetr", "rt-detr", "transformer"):
        return RTDETR(str(weights))
    return YOLO(str(weights))


def resolve_data_yaml(cfg: dict[str, Any], data_dir: Path) -> Path:
    rel = cfg.get("data_view") or "data/eo_views/yolo/data.yaml"
    path = Path(rel)
    if not path.is_absolute():
        path = REPO_ROOT / path
    # Prefer the view under --data-dir if the default relative path is used.
    alt = data_dir / "eo_views" / "yolo" / "data.yaml"
    if alt.is_file() and ("eo_views/yolo/data.yaml" in str(rel).replace("\\", "/")):
        path = alt
    if not path.is_file():
        raise FileNotFoundError(
            f"EO view not found: {path}. On the GPU box run "
            "`python scripts/data/export_eo_views.py` first (see docs/RUNPOD.md)."
        )
    _ensure_absolute_data_path(path)
    return path


def _ensure_absolute_data_path(data_yaml: Path) -> None:
    """Force the YOLO data.yaml ``path`` to the yaml's own absolute directory.

    Ultralytics resolves a relative ``path`` against the CWD (or datasets dir),
    NOT the yaml location, so a portable ``path: .`` breaks training from any
    other CWD. We own this file, so rewrite it in place to the absolute view dir.
    """
    import yaml as _yaml

    d = _yaml.safe_load(data_yaml.read_text()) or {}
    want = str(data_yaml.parent.resolve())
    if str(d.get("path", "")) != want:
        d["path"] = want
        data_yaml.write_text(_yaml.safe_dump(d, sort_keys=False))


def train_kwargs_from_cfg(
    cfg: dict[str, Any],
    *,
    imgsz: int | None,
    epochs: int | None,
    batch: int | None,
    smoke: bool,
) -> dict[str, Any]:
    train = dict(cfg.get("train") or {})
    # Ultralytics-facing keys only (drop anything it wouldn't understand).
    drop = {"device", "batch_size", "multiscale"}  # base.yaml leftovers
    for k in list(train):
        if k in drop:
            train.pop(k)
    if imgsz is not None:
        train["imgsz"] = imgsz
    if epochs is not None:
        train["epochs"] = epochs
    if batch is not None:
        train["batch"] = batch
    if smoke:
        train["epochs"] = min(int(train.get("epochs", 1)), 1)
        train["patience"] = 1
        # ~2% of the train set — enough to exercise the pipeline, not to converge.
        train["fraction"] = float(train.get("fraction") or 0.02)
        train["batch"] = min(int(train.get("batch", 8)), 8)
        train["workers"] = min(int(train.get("workers", 2)), 2)
    # Always record plots / CSV for the run folder.
    train.setdefault("plots", True)
    train.setdefault("save", True)
    train.setdefault("exist_ok", True)
    return train


def write_run_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def train_one(
    *,
    family_name: str,
    config_path: Path,
    out_root: Path,
    data_dir: Path,
    device: str,
    imgsz: int | None,
    epochs: int | None,
    batch: int | None,
    smoke: bool,
    dry_run: bool,
    resume: bool,
) -> dict[str, Any]:
    cfg = load_family_config(config_path)
    det = cfg.get("detector") or {}
    data_yaml = resolve_data_yaml(cfg, data_dir)
    train_kw = train_kwargs_from_cfg(
        cfg, imgsz=imgsz, epochs=epochs, batch=batch, smoke=smoke)
    run_name = family_name + ("_smoke" if smoke else "")
    run_dir = out_root / run_name
    info = device_info(device)

    plan = {
        "family": family_name,
        "config": str(config_path),
        "arch": det.get("arch"),
        "pretrained": det.get("pretrained"),
        "transfer_role": det.get("transfer_role"),
        "data_yaml": str(data_yaml),
        "project": str(out_root),
        "name": run_name,
        "device": device,
        "train": train_kw,
        "device_info": info,
        "smoke": smoke,
        "resume": resume,
    }
    print(f"\n[train] === {run_name} ===")
    print(f"[train] arch={det.get('arch')} pretrained={det.get('pretrained')} "
          f"role={det.get('transfer_role')}")
    print(f"[train] data={data_yaml}")
    print(f"[train] device={device} ({info.get('gpu_name') or info.get('platform')})")
    print(f"[train] out={run_dir}")
    print(f"[train] kwargs={json.dumps({k: train_kw[k] for k in sorted(train_kw) if k in ('imgsz','epochs','batch','patience','optimizer','seed','fraction')}, default=str)}")

    if dry_run:
        write_run_meta(run_dir, {**plan, "status": "dry_run",
                                 "timestamp": datetime.now(timezone.utc).isoformat()})
        print("[train] dry-run — not starting Ultralytics")
        return {**plan, "status": "dry_run"}

    model = build_model(det)
    t0 = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    results = model.train(
        data=str(data_yaml),
        project=str(out_root),
        name=run_name,
        device=device,
        resume=resume,
        **train_kw,
    )
    elapsed = time.perf_counter() - t0
    finished = datetime.now(timezone.utc).isoformat()

    # Ultralytics writes weights under run_dir/weights/{best,last}.pt
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    meta = {
        **plan,
        "status": "completed",
        "started_utc": started,
        "finished_utc": finished,
        "wall_clock_s": round(elapsed, 1),
        "wall_clock_h": round(elapsed / 3600.0, 3),
        "best_weights": str(best) if best.is_file() else None,
        "last_weights": str(last) if last.is_file() else None,
        "ultralytics_save_dir": str(getattr(results, "save_dir", run_dir)),
    }
    write_run_meta(run_dir, meta)
    print(f"[train] done {run_name} in {meta['wall_clock_h']} h → {best}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true",
                   help="train every family in configs/detector/families.yaml")
    g.add_argument("--family", type=str,
                   help="family name from families.yaml (e.g. yolo11s)")
    g.add_argument("--config", type=Path,
                   help="path to a single family config YAML")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="results root (default: results/detector_baselines)")
    ap.add_argument("--device", type=str, default="auto",
                    help="auto | 0 | cpu | mps | cuda:0 … (default: auto)")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="override train imgsz (default from family config; 1280 for the variant)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch + 2%% fraction — pipeline sanity check")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the train plan and write run_meta.json; do not train")
    ap.add_argument("--resume", action="store_true",
                    help="resume an interrupted Ultralytics run in the same out dir")
    args = ap.parse_args()

    device = resolve_device(args.device)
    jobs: list[tuple[str, Path]] = []
    if args.config:
        name = args.config.stem
        jobs.append((name, args.config.resolve()))
    elif args.family:
        match = [f for f in roster_families() if f.get("name") == args.family]
        if not match:
            known = [f.get("name") for f in roster_families()]
            ap.error(f"unknown family {args.family!r}; known: {known}")
        cfg_rel = match[0]["config"]
        jobs.append((args.family, (REPO_ROOT / cfg_rel).resolve()))
    else:
        for f in roster_families():
            jobs.append((f["name"], (REPO_ROOT / f["config"]).resolve()))

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, cfg_path in jobs:
        if not cfg_path.is_file():
            raise FileNotFoundError(cfg_path)
        summaries.append(train_one(
            family_name=name,
            config_path=cfg_path,
            out_root=args.out,
            data_dir=args.data_dir,
            device=device,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            smoke=args.smoke,
            dry_run=args.dry_run,
            resume=args.resume,
        ))

    summary_path = args.out / ("train_summary_smoke.json" if args.smoke
                               else "train_summary.json")
    if args.dry_run:
        summary_path = args.out / "train_summary_dry_run.json"
    summary_path.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "jobs": summaries,
    }, indent=2) + "\n")
    print(f"\n[train] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
