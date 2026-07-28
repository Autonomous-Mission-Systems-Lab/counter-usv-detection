#!/usr/bin/env python3
"""Video-derived non-cooperative tracks from SMD on-shore .

The **runtime-representative** trajectory check: shore-camera tracks of *non-
transmitting* craft, the realistic counter-USV sensing situation (unlike AIS, which
only sees cooperative transmitters). Used to expose AIS carriage bias — but only as
far as the camera geometry defensibly allows.

Design (from the plan)
----------------------
* **Identity source = SMD on-shore `TrackGT`.** Each MATLAB ``Track`` element is one
  persistent object with a full per-frame ``BB`` sequence — we do NOT reconstruct
  identities. Padding slots (``ObjectType == '0'``) are skipped.
* **Labels transferred from SMD-Plus `ObjectGT`** by framewise box IoU (SMD-Plus is
  the cleaner, corrected 7-class labeling). Fallback to the track's own (original
  SMD) ``ObjectType`` when no SMD-Plus box matches. Match/retention rates reported.
* **World-frame calibration gate.** Before comparing to AIS we check whether SMD
  supports a defensible water-plane homography + metric scale. It does **not**
  (`HorizonGT` gives a per-frame horizon line, but there are no camera intrinsics,
  no camera height, and no ground-control points), so the gate **fails** and we keep
  only **scale-normalized image-plane** motion/shape features (also the PercepGuard-
  style baseline input). We explicitly do **not** claim these validate AIS speed or
  metric turn-rate envelopes. A horizon-relative vertical position is emitted as an
  *ordinal* pseudo-range, flagged non-metric.

Why image-plane features are still comparable-ish: normalizing image speed by the
object's own bbox diagonal ("body-lengths per second") removes the distance/zoom
scale factor, giving a scale-robust motion proxy for the non-cooperative sensitivity
check — without pretending it is metric.

Outputs
-------
* ``data/tracks/tracks_video.parquet`` — one row per video track (image-plane,
  scale-normalized features + transferred class + calibration flag + provenance).
* ``data/tracks/tracks_video_points.parquet`` — per-frame image-plane centroid tracks.
* ``results/smd_tracks/report.md`` + ``figures/*.png`` + ``summary.json``.

Usage
-----
    python scripts/data/ingest_smd_tracks.py
    python scripts/data/ingest_smd_tracks.py --iou 0.3 --fps 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_RESULTS = REPO_ROOT / "results" / "smd_tracks"
SMD = "raw/smd"

IOU_MATCH = 0.3 # min IoU to transfer an SMD-Plus label onto a TrackGT box
LOITER_BL_S = 0.10 # image-plane loiter: < 0.1 body-lengths/sec
MIN_TRACK_FRAMES = 15 # ~0.5 s at 30 fps; below this features are unreliable
DEFAULT_FPS = 30.0

# Fallback: original SMD TrackGT ObjectType strings -> canonical (used only when no
# SMD-Plus box matches). SMD-Plus (via taxonomy `smd_plus`) is the primary source.
ORIG_SMD_MAP = {
    "Ferry": "passenger_ferry", "Buoy": "static_aid", "Vessel/ship": "cargo_merchant",
    "Speed boat": "small_craft", "Boat": "small_craft", "Kayak": "small_craft",
    "Sail boat": "sailing", "Swimming person": "unknown_other",
    "Flying bird/plane": "unknown_other", "Other": "unknown_other",
}


# ---------------------------------------------------------------------------
# Taxonomy (SMD-Plus native -> canonical) + roles
# ---------------------------------------------------------------------------
def load_maps(tax_path: Path):
    import yaml
    tax = yaml.safe_load(tax_path.read_text())
    smd_plus = tax["eo_sources"]["smd_plus"]["native"]
    roles = {n: m.get("role", "benign") for n, m in tax["canonical_classes"].items()}
    return smd_plus, roles


# ---------------------------------------------------------------------------
# .mat loading helpers
# ---------------------------------------------------------------------------
def _load(mat_path):
    import scipy.io as sio
    return sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)


def frame_boxes_labels(structxml, nmap):
    """SMD-Plus ObjectGT -> list per frame of (canonical_label, [x,y,w,h])."""
    frames = np.atleast_1d(structxml)
    out = []
    for fr in frames:
        ot = getattr(fr, "ObjectType", None)
        bb = getattr(fr, "BB", None)
        items = []
        if ot is not None and bb is not None:
            ot = np.atleast_1d(ot)
            bb = np.atleast_2d(bb)
            for i in range(min(len(ot), bb.shape[0])):
                if bb.shape[1] < 4:
                    continue
                lab = str(ot[i]).strip()
                canon = nmap.get(lab)
                if canon is not None:
                    items.append((canon, [float(v) for v in bb[i][:4]]))
        out.append(items)
    return out


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------------------
# Calibration gate
# ---------------------------------------------------------------------------
def calibration_gate(has_horizon: bool) -> dict:
    """World-frame metric calibration requires more than a horizon line. SMD on-shore
    ships no intrinsics, no camera height, no ground-control points, so a defensible
    water-plane homography cannot be built. Gate FAILS -> image-plane only."""
    available = {"horizon_line": has_horizon, "camera_intrinsics": False,
                 "camera_height": False, "ground_control_points": False,
                 "ais_matched_world_points": False}
    calibratable = (available["camera_intrinsics"] and available["camera_height"]
                    and (available["ground_control_points"]
                         or available["ais_matched_world_points"]))
    reasons = []
    if not available["camera_intrinsics"]:
        reasons.append("no camera intrinsics (focal length / principal point)")
    if not available["camera_height"]:
        reasons.append("no camera height above waterline")
    if not (available["ground_control_points"] or available["ais_matched_world_points"]):
        reasons.append("no ground-control points or AIS-matched world points to fit a homography")
    return {
        "world_frame_calibratable": bool(calibratable),
        "available": available,
        "reasons": reasons,
        "decision": ("scale-normalized image-plane features only; do NOT claim these "
                     "validate AIS metric speed / turn-rate envelopes. A horizon-"
                     "relative vertical position is provided as an ORDINAL pseudo-"
                     "range (non-metric)."),
    }


# ---------------------------------------------------------------------------
# Per-clip processing
# ---------------------------------------------------------------------------
def get_fps(video_path: Path, default: float) -> float:
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps) if fps and fps > 1 else default
    except Exception:
        return default


def horizon_y_per_frame(horizon_structxml):
    """Return an array of horizon Y (image row) per frame, or None."""
    if horizon_structxml is None:
        return None
    frames = np.atleast_1d(horizon_structxml)
    ys = []
    for fr in frames:
        y = getattr(fr, "Y", np.nan)
        ys.append(float(y) if y is not None else np.nan)
    return np.array(ys, dtype="float64")


def process_clip(clip, data_dir, nmap, roles, fps, iou_thr):
    trackgt = data_dir / SMD / "VIS_Onshore" / "TrackGT" / f"{clip}_TrackGT.mat"
    objgt = data_dir / SMD / "SMD-Plus" / "ObjectGT" / f"{clip}_ObjectGT.mat"
    horgt = data_dir / SMD / "VIS_Onshore" / "HorizonGT" / f"{clip}_HorizonGT.mat"
    if not trackgt.exists() or not objgt.exists():
        return [], [], {"clip": clip, "skipped": "missing TrackGT or SMD-Plus ObjectGT"}

    tr = np.atleast_1d(_load(trackgt)["Track"])
    smdp_frames = frame_boxes_labels(_load(objgt)["structXML"], nmap)
    nframes_lbl = len(smdp_frames)
    hy = horizon_y_per_frame(_load(horgt)["structXML"]) if horgt.exists() else None

    track_rows, point_rows = [], []
    n_obj_total = n_obj_real = 0
    for oid, t in enumerate(tr):
        n_obj_total += 1
        bb = np.atleast_2d(t.BB).astype("float64")
        nframes = bb.shape[0]
        ot = np.atleast_1d(getattr(t, "ObjectType", np.array(["0"] * nframes)))
        mt = np.atleast_1d(getattr(t, "MotionType", np.array([""] * nframes)))
        dt = np.atleast_1d(getattr(t, "DistanceType", np.array([""] * nframes)))
        present = ~np.isnan(bb).any(axis=1)
        # padding slot: no real ObjectType among present frames
        types_present = [str(ot[f]).strip() for f in np.where(present)[0]
                         if str(ot[f]).strip() not in ("0", "", "nan")]
        if present.sum() < MIN_TRACK_FRAMES or not types_present:
            continue
        n_obj_real += 1

        # --- class transfer: framewise IoU to SMD-Plus ---
        matched, votes = 0, {}
        for f in np.where(present)[0]:
            if f >= nframes_lbl:
                break
            box = list(bb[f])
            best_iou, best_lab = 0.0, None
            for lab, sbox in smdp_frames[f]:
                i = iou_xywh(box, sbox)
                if i > best_iou:
                    best_iou, best_lab = i, lab
            if best_lab is not None and best_iou >= iou_thr:
                matched += 1
                votes[best_lab] = votes.get(best_lab, 0) + 1
        match_frac = matched / max(int(present.sum()), 1)
        if votes:
            canon = max(votes, key=votes.get)
            label_source = "smdplus_iou"
        else: # fallback to the track's own (original SMD) label
            fb = max(set(types_present), key=types_present.count)
            canon = ORIG_SMD_MAP.get(fb, "unknown_other")
            label_source = "trackgt_fallback"

        # --- image-plane trajectory + scale-normalized features ---
        idx = np.where(present)[0]
        cx = bb[idx, 0] + bb[idx, 2] / 2.0
        cy = bb[idx, 1] + bb[idx, 3] / 2.0
        w, h = bb[idx, 2], bb[idx, 3]
        diag = np.sqrt(w ** 2 + h ** 2)
        tsec = idx / fps
        dframe = np.diff(idx).astype("float64")
        dsec = dframe / fps
        dcx, dcy = np.diff(cx), np.diff(cy)
        step_px = np.sqrt(dcx ** 2 + dcy ** 2)
        diag_mid = (diag[1:] + diag[:-1]) / 2.0
        with np.errstate(invalid="ignore", divide="ignore"):
            speed_px_s = np.where(dsec > 0, step_px / dsec, np.nan)
            norm_speed = np.where(diag_mid > 0, speed_px_s / diag_mid, np.nan) # body-len/s
            vel_ang = np.degrees(np.arctan2(dcy, dcx))
            dang = np.abs((np.diff(vel_ang) + 180) % 360 - 180)
            turn_dps = np.where(dsec[1:] > 0, dang / dsec[1:], np.nan)
        net_px = float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0]))
        path_px = float(np.nansum(step_px))
        straightness = net_px / path_px if path_px > 0 else np.nan
        # bbox size trend (approach/recede): normalized slope of diag over time
        if len(tsec) >= 2 and np.ptp(tsec) > 0:
            slope = np.polyfit(tsec, diag, 1)[0]
            size_trend = slope / (np.nanmean(diag) + 1e-9) # rel. size change per s
        else:
            size_trend = np.nan
        aspect = w / np.where(h > 0, h, np.nan)
        # ordinal horizon-relative vertical position (NON-METRIC pseudo-range)
        if hy is not None:
            hf = hy[idx[idx < len(hy)]]
            hz_rel = float(np.nanmean(cy[:len(hf)] - hf)) if len(hf) else np.nan
        else:
            hz_rel = np.nan
        moving_frac = float(np.mean([str(mt[f]).strip() == "Moving"
                                     for f in idx if f < len(mt)]))
        dvals = [str(dt[f]).strip() for f in idx if f < len(dt) and str(dt[f]).strip()]
        dist_modal = max(set(dvals), key=dvals.count) if dvals else ""

        track_id = f"{clip}#{oid}"
        track_rows.append({
            "track_id": track_id, "clip": clip, "obj_index": oid,
            "canonical_class": canon, "role": roles.get(canon, "non_target"),
            "label_source": label_source, "label_match_frac": round(match_frac, 3),
            "n_frames": int(present.sum()), "duration_s": round(float(tsec[-1] - tsec[0]), 3),
            "fps": round(fps, 2),
            "norm_speed_bl_s_mean": _r(np.nanmean(norm_speed)),
            "norm_speed_bl_s_p95": _r(np.nanquantile(norm_speed, 0.95) if np.isfinite(norm_speed).any() else np.nan),
            "norm_speed_bl_s_max": _r(np.nanmax(norm_speed) if np.isfinite(norm_speed).any() else np.nan),
            "loiter_frac": _r(np.nanmean(norm_speed < LOITER_BL_S)),
            "turn_rate_mean_dps": _r(np.nanmean(turn_dps)),
            "turn_rate_p95_dps": _r(np.nanquantile(turn_dps, 0.95) if np.isfinite(turn_dps).any() else np.nan),
            "straightness": _r(straightness),
            "speed_px_s_mean": _r(np.nanmean(speed_px_s)),
            "diag_px_mean": _r(np.nanmean(diag)), "aspect_mean": _r(np.nanmean(aspect)),
            "aspect_std": _r(np.nanstd(aspect)), "size_trend_per_s": _r(size_trend),
            "moving_frac_annotated": round(moving_frac, 3), "distance_modal": dist_modal,
            "horizon_rel_y_px": _r(hz_rel), # ORDINAL pseudo-range, non-metric
            "source": "smd_video",
        })
        for k in range(len(idx)):
            point_rows.append({
                "track_id": track_id, "clip": clip, "frame": int(idx[k]),
                "t_s": round(float(tsec[k]), 3), "cx": round(float(cx[k]), 2),
                "cy": round(float(cy[k]), 2), "w": round(float(w[k]), 2),
                "h": round(float(h[k]), 2),
                "norm_speed_bl_s": _r(norm_speed[k - 1]) if k > 0 else np.nan,
                "canonical_class": canon,
            })

    stats = {"clip": clip, "nframes": nframes_lbl, "fps": round(fps, 2),
             "objects_total": n_obj_total, "objects_kept": n_obj_real,
             "has_horizon": hy is not None}
    return track_rows, point_rows, stats


def _r(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return np.nan
        return round(float(x), nd)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Report + figures
# ---------------------------------------------------------------------------
def make_figures(tracks, fig_dir):
    import os
    fig_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(fig_dir.parent / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = tracks["canonical_class"].value_counts().index.tolist()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    tracks["canonical_class"].value_counts().plot(kind="bar", ax=ax)
    ax.set_ylabel("video tracks"); ax.set_title("SMD video tracks per canonical class")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(fig_dir / "tracks_per_class.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    data = [tracks.loc[tracks.canonical_class == c, "norm_speed_bl_s_mean"].dropna().values
            for c in classes]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(classes) + 1)); ax.set_xticklabels(classes, rotation=30)
    ax.set_ylabel("mean image speed (body-lengths / s)")
    ax.set_title("Scale-normalized speed by class (image-plane; NON-metric)")
    fig.tight_layout(); fig.savefig(fig_dir / "norm_speed_by_class.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for c in classes:
        sub = tracks[tracks.canonical_class == c]
        ax.scatter(sub["loiter_frac"].median(), sub["straightness"].median(), s=60, label=c)
    ax.set_xlabel("median loiter fraction"); ax.set_ylabel("median straightness (image-plane)")
    ax.set_title("Image-plane kinematic separation"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "straightness_vs_loiter.png", dpi=120); plt.close(fig)


def build_summary(tracks, clip_stats, gate, params):
    per_class = tracks.groupby("canonical_class").agg(
        role=("role", "first"), tracks=("track_id", "count"),
        clips=("clip", "nunique"),
        norm_speed_med=("norm_speed_bl_s_mean", "median"),
        straightness_med=("straightness", "median"),
        loiter_med=("loiter_frac", "median")).round(3)
    per_class = per_class.sort_values("tracks", ascending=False)
    src = tracks["label_source"].value_counts().to_dict()
    return {
        "params": params,
        "coverage": {
            "clips_processed": int(len(clip_stats)),
            "clips_with_horizon": int(sum(c.get("has_horizon", False) for c in clip_stats)),
            "tracks": int(len(tracks)),
            "total_track_seconds": round(float(tracks["duration_s"].sum()), 1),
            "duration_s": {
                "min": round(float(tracks["duration_s"].min()), 1),
                "median": round(float(tracks["duration_s"].median()), 1),
                "max": round(float(tracks["duration_s"].max()), 1)},
        },
        "class_transfer": {
            "by_source": src,
            "label_match_frac_median": round(float(tracks["label_match_frac"].median()), 3),
            "smdplus_transfer_rate": round(src.get("smdplus_iou", 0) / max(len(tracks), 1), 3),
        },
        "calibration_gate": gate,
        "per_class": json.loads(per_class.reset_index().to_json(orient="records")),
        "ais_comparability_note": (
            "Time-horizon + sampling mismatch vs the AIS corpus : SMD tracks are "
            "~7-20 s at ~30 fps; AIS tracks are minutes-to-hours at ~60 s cadence. There "
            "is no observation window where both are well-sampled (AIS needs minutes for "
            ">1 point; SMD offers seconds). So SMD is NOT pooled with AIS: it is a "
            "separate held-out evaluation pool , compared to the AIS-learned "
            "benign envelope at the level of short-horizon / instantaneous feature "
            "DISTRIBUTIONS per class, and only for calibration-gated features. "
            "config `observation_window_s`=300 exceeds every SMD clip, so SMD informs "
            "only the short end of the time-to-flag curve (6.6)."),
    }


def write_report(summary, path):
    s = summary; cov = s["coverage"]; ct = s["class_transfer"]; g = s["calibration_gate"]
    L = ["# SMD video-derived non-cooperative tracks \n",
         "Auto-generated by `scripts/data/ingest_smd_tracks.py`. The runtime-representative "
         "check: shore-camera tracks of **non-transmitting** craft, to expose AIS "
         "carriage bias as far as the camera geometry defensibly allows.\n",
         "## Coverage\n",
         f"- Clips processed: **{cov['clips_processed']}** "
         f"({cov['clips_with_horizon']} with HorizonGT)\n",
         f"- Video tracks: **{cov['tracks']}** · total track time "
         f"**{cov['total_track_seconds']:.0f} s** (~{cov['total_track_seconds']/60:.1f} min)\n",
         f"- Track duration s: min={cov['duration_s']['min']}, "
         f"median={cov['duration_s']['median']}, max={cov['duration_s']['max']}\n",
         "## Class transfer (SMD-Plus labels onto TrackGT identities)\n",
         f"- Method: framewise IoU>={s['params']['iou_match']} match of each TrackGT box "
         "to SMD-Plus `ObjectGT`; per-track class = modal matched label; fallback to the "
         "track's own (original SMD) `ObjectType` when unmatched.\n",
         f"- By label source: {ct['by_source']} · SMD-Plus transfer rate: "
         f"**{100*ct['smdplus_transfer_rate']:.0f}%** · median per-track frame-match "
         f"fraction: **{ct['label_match_frac_median']}**\n",
         "## World-frame calibration gate\n",
         f"- **world_frame_calibratable = {g['world_frame_calibratable']}**\n",
         f"- Available: {g['available']}\n",
         f"- Reasons: {'; '.join(g['reasons']) if g['reasons'] else 'n/a'}\n",
         f"- Decision: {g['decision']}\n",
         "## Per canonical class (image-plane, scale-normalized — NON-metric)\n",
         "| class | role | tracks | clips | med norm-speed (bl/s) | med straightness | med loiter |",
         "|---|---|---|---|---|---|---|"]
    for r in s["per_class"]:
        L.append(f"| {r['canonical_class']} | {r.get('role','?')} | {r['tracks']} | "
                 f"{r['clips']} | {r.get('norm_speed_med')} | {r.get('straightness_med')} "
                 f"| {r.get('loiter_med')} |")
    L += ["\n## Limitations (reported honestly)\n",
          f"- {s['ais_comparability_note']}\n",
          "- Image-plane features are **scale-normalized but NON-metric**; "
          "`horizon_rel_y_px` is an **ordinal** pseudo-range only. Do not use these to "
          "claim validation of AIS metric speed / turn-rate envelopes (calibration gate "
          "failed).\n",
          "- Perspective foreshortening biases image-plane straightness/turn-rate; "
          "treat cross-class comparisons qualitatively.\n",
          "- Small corpus (tens of tracks, minutes of motion): a sensitivity check and "
          "the PercepGuard-style baseline input, not a second training corpus.\n",
          "\n## Figures\n",
          "- `figures/tracks_per_class.png` · `figures/norm_speed_by_class.png` · "
          "`figures/straightness_vs_loiter.png`\n"]
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SMD video track ingestion .")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--iou", type=float, default=IOU_MATCH)
    p.add_argument("--fps", type=float, default=None, help="override fps (else read per video)")
    p.add_argument("--no-points", action="store_true")
    args = p.parse_args(argv)

    nmap, roles = load_maps(args.data_dir / "taxonomy.yaml")
    tg_dir = args.data_dir / SMD / "VIS_Onshore" / "TrackGT"
    clips = sorted(f.name[:-len("_TrackGT.mat")] for f in tg_dir.glob("*_TrackGT.mat"))
    if not clips:
        print(f"[error] no TrackGT under {tg_dir}")
        return 1
    print(f"[smd] {len(clips)} TrackGT clips")

    all_tracks, all_points, clip_stats = [], [], []
    for clip in clips:
        vid = args.data_dir / SMD / "VIS_Onshore" / "Videos" / f"{clip}.avi"
        fps = args.fps or get_fps(vid, DEFAULT_FPS)
        trs, pts, st = process_clip(clip, args.data_dir, nmap, roles, fps, args.iou)
        all_tracks += trs
        all_points += pts
        clip_stats.append(st)
        print(f" {clip:28} tracks={len(trs):2d} "
              f"objs={st.get('objects_kept','?')}/{st.get('objects_total','?')} "
              f"fps={st.get('fps','?')}")

    if not all_tracks:
        print("[error] no tracks produced")
        return 1
    tracks = pd.DataFrame(all_tracks)
    points = pd.DataFrame(all_points)

    gate = calibration_gate(has_horizon=any(c.get("has_horizon") for c in clip_stats))
    params = {"iou_match": args.iou, "loiter_bl_s": LOITER_BL_S,
              "min_track_frames": MIN_TRACK_FRAMES}

    tracks_dir = args.data_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    tracks.to_parquet(tracks_dir / "tracks_video.parquet", index=False)
    if not args.no_points:
        points.to_parquet(tracks_dir / "tracks_video_points.parquet", index=False)

    summary = build_summary(tracks, clip_stats, gate, params)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    try:
        make_figures(tracks, args.results_dir / "figures")
    except Exception as e:
        print(f"[warn] figures failed: {e}")
    write_report(summary, args.results_dir / "report.md")

    cov = summary["coverage"]; ct = summary["class_transfer"]
    print(f"\n[smd] tracks={cov['tracks']} over {cov['clips_processed']} clips "
          f"({cov['total_track_seconds']:.0f}s total)")
    print(f" class transfer: {ct['by_source']} (SMD-Plus rate {100*ct['smdplus_transfer_rate']:.0f}%)")
    print(f" calibration gate: world_frame_calibratable={gate['world_frame_calibratable']} "
          f"-> image-plane only")
    print(f" per-class: {[(r['canonical_class'], r['tracks']) for r in summary['per_class']]}")
    print(f"\nWrote:\n {tracks_dir}/tracks_video.parquet"
          + ("" if args.no_points else f"\n {tracks_dir}/tracks_video_points.parquet")
          + f"\n {args.results_dir}/report.md + summary.json + figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
