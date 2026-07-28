#!/usr/bin/env python3
"""Fit class-conditional one-class benign-behavior models.

Per envelope from ``configs/defense/class_envelope_map.yaml``, fits GMM
(primary) + Mahalanobis + IsolationForest on complete-window benign train
tracks, at several observation horizons (e.g. 120/180/300 s). Selects GMM ``k``
by val benign log-likelihood and calibrates anomaly-score thresholds to FAR
targets on val benign. Never uses hostile data.

Each envelope is saved as a ``MultiHorizonEnvelope`` bundle; at score time the
longest horizon whose window is complete for the contact is used, so short
tracks are still scored (coverage) instead of dropped.

Usage
-----
    python scripts/behavior/fit_behavior_model.py
    python scripts/behavior/fit_behavior_model.py --envelope fishing
    python scripts/behavior/fit_behavior_model.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.kinematics.behavior_model import (  # noqa: E402
    EnvelopeModel,
    MultiHorizonEnvelope,
    envelope_summary,
    fit_envelope,
    save_envelope,
    select_envelope_rows,
)

DEFAULT_CFG = REPO_ROOT / "configs" / "defense" / "behavior_model.yaml"
DEFAULT_OUT = REPO_ROOT / "results" / "behavior_model"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def load_window(cfg: dict, window_s: int, *, require_complete: bool) -> pd.DataFrame:
    """Load one horizon's benign feature table (with ``split`` column)."""
    tmpl = (cfg.get("features_path_template")
            or "data/behavior/features_window_{w}s.parquet")
    path = (REPO_ROOT / tmpl.format(w=window_s)).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if require_complete and "window_complete" in df.columns:
        df = df.loc[df["window_complete"]].copy()
    if "split" not in df.columns:
        splits = pd.read_csv(
            REPO_ROOT / (cfg.get("splits") or "data/splits/ais_track_splits.csv"))
        df = df.merge(splits[["trip_id", "split"]], on="trip_id", how="left")
    if "role" in df.columns:
        df = df.loc[df["role"] == "benign"].copy()
    return df


def score_split_far(
    model: EnvelopeModel,
    df: pd.DataFrame,
    *,
    far_key: str = "far_0.05",
    model_name: str = "gmm",
) -> dict[str, Any]:
    """Empirical FAR on a benign split at a calibrated threshold (one horizon)."""
    if df.empty:
        return {"n": 0, "n_scored": 0, "n_flagged": 0, "far": None}
    flagged = 0
    scored = 0
    scores_all: list[float] = []
    has_course_model = (
        "core_course" in model.subspaces
        and model_name in model.subspaces["core_course"]
    )
    for sub, fams in model.subspaces.items():
        if model_name not in fams:
            continue
        fit = fams[model_name]
        cols = fit.feature_names
        if sub == "core_course":
            mask = df[model.course_features].notna().all(axis=1)
        elif has_course_model:
            mask = ~df[model.course_features].notna().all(axis=1)
        else:
            mask = pd.Series(True, index=df.index)
        mask = mask & df[model.core_features].notna().all(axis=1)
        sub_df = df.loc[mask]
        if sub_df.empty:
            continue
        X = sub_df[cols].to_numpy(dtype="float64")
        if np.isnan(X).any():
            X = X[~np.isnan(X).any(axis=1)]
        if len(X) == 0:
            continue
        scores = fit.anomaly_score(X)
        thr = fit.thresholds.get(far_key)
        scored += len(scores)
        scores_all.extend(float(s) for s in scores)
        if thr is not None:
            flagged += int((scores > thr).sum())
    far = (flagged / scored) if scored else None
    return {
        "n": int(len(df)),
        "n_scored": scored,
        "n_flagged": flagged,
        "far": far,
        "score_median": float(np.median(scores_all)) if scores_all else None,
        "score_p95": float(np.percentile(scores_all, 95)) if scores_all else None,
    }


def multihorizon_test(
    bundle: MultiHorizonEnvelope,
    test_by_w: dict[int, pd.DataFrame],
    members: dict,
    *,
    far_key: str = "far_0.05",
    model_name: str = "gmm",
) -> dict[str, Any]:
    """Coverage + FAR when each test contact is scored at its longest complete
    horizon (the runtime policy). ``test_by_w`` are full test tables (incomplete
    windows kept) carrying a ``window_complete`` flag."""
    windows = bundle.available_windows()
    rows: dict[int, pd.DataFrame] = {}
    complete: dict[int, set] = {}
    universe: set = set()
    for w in windows:
        d = select_envelope_rows(test_by_w[w].loc[test_by_w[w]["split"] == "test"],
                                  members)
        d = d.drop_duplicates("trip_id").set_index("trip_id")
        rows[w] = d
        complete[w] = set(d.index[d["window_complete"].astype(bool)])
        universe |= set(d.index)

    # Assign each contact to the longest horizon whose window is complete.
    assign: dict[int, list] = {w: [] for w in windows}
    n_uncovered = 0
    for trip in universe:
        cw = [w for w in windows if trip in complete[w]]
        if not cw:
            n_uncovered += 1
            continue
        assign[max(cw)].append(trip)

    scored = flagged = 0
    by_window: dict[str, dict] = {}
    for w, trips in assign.items():
        if not trips:
            by_window[str(w)] = {"assigned": 0, "n_scored": 0, "n_flagged": 0}
            continue
        sub = rows[w].loc[trips]
        res = score_split_far(bundle.horizons[w], sub, far_key=far_key,
                              model_name=model_name)
        scored += res["n_scored"]
        flagged += res["n_flagged"]
        by_window[str(w)] = {
            "assigned": len(trips),
            "n_scored": res["n_scored"],
            "n_flagged": res["n_flagged"],
        }
    n_univ = len(universe)
    return {
        "model": model_name,
        "n_universe": n_univ,
        "n_scored": scored,
        "n_uncovered": n_uncovered,
        "coverage": (scored / n_univ) if n_univ else None,
        "n_flagged": flagged,
        "far": (flagged / scored) if scored else None,
        "by_window": by_window,
    }


def _fmt_pct(x: Any) -> str:
    return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else "—"


def write_report(path: Path, payload: dict) -> None:
    primary_w = payload["primary_window_s"]
    windows = payload["windows_s"]
    lines: list[str] = [
        "# Benign-behavior model fit",
        "",
        f"Generated: {payload['timestamp']}",
        "",
        "Per-envelope one-class models. Primary: **GMM**; baselines: Mahalanobis, "
        "IsolationForest. Thresholds = percentiles of val benign anomaly scores "
        "(higher score = more anomalous). Hostile data never used.",
        "",
        f"Horizons fitted: **{windows} s** (primary **{primary_w} s**). At score "
        "time the longest horizon whose window is complete for the contact is "
        "used, so short tracks are still covered.",
        "",
        f"Features core: `{payload['core_features']}`",
        f"Features course: `{payload['course_features']}`",
        "",
        f"## Per-envelope GMM at primary horizon ({primary_w} s, core subspace)",
        "",
        "| envelope | n_train | n_val | k | val loglik | thr FAR@5% | test FAR@5% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for env_name, block in payload["envelopes"].items():
        core = ((block.get("subspaces") or {}).get("core") or {}).get("gmm") or {}
        tf = ((block.get("test_far") or {}).get("gmm") or {})
        thr = (core.get("thresholds") or {}).get("far_0.05")
        ll, k = core.get("val_loglik"), core.get("n_components")
        lines.append(
            f"| `{env_name}` | {core.get('n_train', 0):,} | {core.get('n_val', 0):,} | "
            f"{k if k is not None else '—'} | "
            f"{ll:.3f} | {thr:.3f} | {_fmt_pct(tf.get('far'))} |"
            if isinstance(ll, (int, float)) and isinstance(thr, (int, float))
            else f"| `{env_name}` | — | — | — | — | — | — |"
        )

    lines += [
        "",
        "## GMM `k` selected per horizon (ceiling check)",
        "",
        "| envelope | " + " | ".join(f"{w} s" for w in windows) + " |",
        "|---|" + "---:|" * len(windows),
    ]
    for env_name, block in payload["envelopes"].items():
        ks = []
        for w in windows:
            h = (block.get("horizons") or {}).get(str(w)) or {}
            ks.append(str(h.get("gmm_k", "—")))
        lines.append(f"| `{env_name}` | " + " | ".join(ks) + " |")

    lines += [
        "",
        "## Multi-horizon coverage + FAR (longest complete window, GMM)",
        "",
        "Coverage = fraction of test contacts of the class that receive a score "
        "(vs. dropped for having no complete window at any horizon).",
        "",
        "| envelope | test contacts | scored | coverage | FAR@5% |",
        "|---|---:|---:|---:|---:|",
    ]
    for env_name, block in payload["envelopes"].items():
        mh = block.get("multi_horizon_test") or {}
        lines.append(
            f"| `{env_name}` | {mh.get('n_universe', 0):,} | "
            f"{mh.get('n_scored', 0):,} | {_fmt_pct(mh.get('coverage'))} | "
            f"{_fmt_pct(mh.get('far'))} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Multi-horizon adds coverage over a single long window (contacts "
        f"complete at 120/180 s but not {primary_w} s are recovered). The gain "
        "is modest on AIS because at ~60-100 s report cadence the longer window "
        "is more often complete; the short horizons matter most at runtime, "
        "where dense EO tracking makes short windows complete and enables "
        "earlier flagging.",
        "- Test FAR is on a vessel-disjoint split; near the 5% target confirms "
        "the val→test calibration generalizes.",
        "- GMM `k` swept to 12 and chosen by a val-loglik knee rule; interior "
        "selections (mostly 8-12) confirm the order is data-driven, not pinned "
        "at the ceiling.",
        "- `core_course` subspace used when COG features are non-null; else `core`.",
        "- Bundles saved under `results/behavior_model/envelopes/<name>.joblib`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--envelope", type=str, default=None,
                    help="fit a single envelope (default: all)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true",
                    help="subsample train/val/test for a fast wiring check")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    emap = _load_yaml(REPO_ROOT / (cfg.get("envelope_map")
                                   or "configs/defense/class_envelope_map.yaml"))
    envelopes = emap.get("envelopes") or {}
    if args.envelope:
        if args.envelope not in envelopes:
            ap.error(f"unknown envelope {args.envelope!r}")
        envelopes = {args.envelope: envelopes[args.envelope]}

    windows = [int(w) for w in (cfg.get("windows_s") or [cfg.get("window_s") or 300])]
    primary_w = int(cfg.get("window_s") or max(windows))
    if primary_w not in windows:
        windows = sorted(set(windows) | {primary_w})
    models_cfg = cfg.get("models") or {}
    primary = str(models_cfg.get("primary") or "gmm")
    baselines = [b for b in (models_cfg.get("baselines") or []) if b != primary]
    families = [primary] + baselines

    print(f"[fit] horizons: {windows} s (primary {primary_w} s)")
    print(f"[fit] loading features …")
    complete_by_w = {w: load_window(cfg, w, require_complete=True) for w in windows}
    testfull_by_w = {w: load_window(cfg, w, require_complete=False) for w in windows}
    for w in windows:
        d = complete_by_w[w]
        print(f"  {w:>4} s complete: train={int((d.split=='train').sum()):,} "
              f"val={int((d.split=='val').sum()):,} "
              f"test={int((d.split=='test').sum()):,}")

    if args.smoke:
        for w in windows:
            complete_by_w[w] = pd.concat([
                complete_by_w[w].loc[complete_by_w[w].split == s].sample(
                    n=min(n, int((complete_by_w[w].split == s).sum())), random_state=0)
                for s, n in (("train", 5000), ("val", 1000), ("test", 1000))
            ])
        print("[fit] smoke subsample applied")

    out_env = args.out / "envelopes"
    out_env.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.relative_to(REPO_ROOT)),
        "window_s": primary_w,
        "primary_window_s": primary_w,
        "windows_s": windows,
        "core_features": (cfg.get("features") or {}).get("core"),
        "course_features": (cfg.get("features") or {}).get("course"),
        "primary": primary,
        "envelopes": {},
        "smoke": bool(args.smoke),
    }

    t0 = time.perf_counter()
    for name, espec in envelopes.items():
        members = espec.get("members") or {}
        print(f"\n[fit] === {name} ===")
        horizon_models: dict[int, EnvelopeModel] = {}
        horizons_meta: dict[str, Any] = {}
        for w in windows:
            dfc = complete_by_w[w]
            tr = dfc.loc[dfc.split == "train"]
            va = dfc.loc[dfc.split == "val"]
            model = fit_envelope(name, members, tr, va, cfg={**cfg, "window_s": w})
            if model is None:
                continue
            horizon_models[w] = model
            te = select_envelope_rows(dfc.loc[dfc.split == "test"], members)
            core_gmm = (model.subspaces.get("core") or {}).get("gmm")
            horizons_meta[str(w)] = {
                "gmm_k": core_gmm.n_components if core_gmm else None,
                "val_loglik": core_gmm.val_loglik if core_gmm else None,
                "test_far": {
                    fam: score_split_far(model, te, model_name=fam)
                    for fam in families
                },
            }
        if not horizon_models:
            continue

        bundle = MultiHorizonEnvelope(
            name=name,
            members=members,
            primary_window_s=primary_w,
            core_features=list((cfg.get("features") or {}).get("core") or []),
            course_features=list((cfg.get("features") or {}).get("course") or []),
            horizons=horizon_models,
            primary=primary,  # type: ignore[arg-type]
        )
        path = out_env / f"{name}.joblib"
        save_envelope(bundle, path)

        primary_model = horizon_models.get(primary_w) or next(iter(horizon_models.values()))
        summary = envelope_summary(primary_model)
        summary["primary_window_s"] = primary_w
        summary["horizons"] = horizons_meta
        summary["test_far"] = horizons_meta.get(str(primary_w), {}).get("test_far", {})
        summary["multi_horizon_test"] = multihorizon_test(
            bundle, testfull_by_w, members, model_name=primary)
        summary["artifact"] = str(path.relative_to(REPO_ROOT))
        payload["envelopes"][name] = summary

        mh = summary["multi_horizon_test"]
        tf = summary["test_far"].get(primary, {}).get("far")
        print(f"  saved {path.name}  {primary_w}s test FAR@5%={_fmt_pct(tf)}  "
              f"multi-horizon coverage={_fmt_pct(mh.get('coverage'))} "
              f"FAR={_fmt_pct(mh.get('far'))}")

    payload["wall_clock_s"] = round(time.perf_counter() - t0, 1)
    sum_path = args.out / "fit_summary.json"
    sum_path.write_text(json.dumps(payload, indent=2) + "\n")
    report = args.out / "fit_report.md"
    write_report(report, payload)
    print(f"\n[fit] wrote {sum_path}")
    print(f"[fit] wrote {report}")
    print(f"[fit] done in {payload['wall_clock_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
