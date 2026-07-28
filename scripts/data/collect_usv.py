#!/usr/bin/env python3
"""Provenance-first curation kit for the ``usv`` EO imagery set .

This tool is a **manifest-first curation kit**, not a downloader:

  * you obtain each image/clip from an authoritative, license-legible source
    (DVIDS / navy.mil public-domain, Wikimedia Commons per-file license,
    manufacturer/OSINT press pages — link only), respecting its terms;
  * you register it here with full provenance in one append-only manifest;
  * video clips are frame-extracted (near-duplicate frames skipped) with each frame
    inheriting the clip's provenance;
  * ``build_coco_master.py``'s ``usv`` adapter merges the annotated set into the COCO
    master with ``source="usv"``, ``channel="eo_only"`` and the ``synthetic`` flag, so
    the scorer pipeline can hard-exclude it and the EO audit picks it up unchanged.

Layout produced under ``data/raw/usv/`` (all gitignored; released via the data card):
    manifest.csv append-only per-image provenance (THE deliverable)
    sources.yaml per-platform authoritative seed source catalog
    images/ registered stills
    frames/<clip_id>/ extracted video frames
    synthetic/ flagged generated frames (synthetic=true)
    annotations.coco.json your CVAT/LabelImg export (you create this) [or]
    Annotations/*.xml LabelImg Pascal-VOC per-image boxes

Subcommands
-----------
    seeds write/refresh the per-platform authoritative seed source catalog
    scrape web-image-search + download per platform (records source URL/image);
                 QC loop: you delete bad files, then re-run scrape/sync to auto-update
    videos search + download YouTube videos per platform (yt-dlp; QC-reconcilable)
    video-frames split QC'd videos into deduped frames (once/video) with provenance
    sync reconcile after QC (drop deleted stills/frames, tombstone deleted
                  videos; rebuild manifest to survivors)
    add-image register one already-obtained still (min: --platform + --source-url)
    frames extract frames from a LOCAL video file (+ per-frame provenance)
    synth register a generated frame (synthetic=true, excludable)
    backfill fill in provenance recorded later (license/date/…) for existing rows
    verify sha256 + manifest<->files integrity + firewall + dedup report
    status coverage summary (by platform / role / viewpoint / real-vs-synthetic)

Video (YouTube) loop — the volume booster:
    1) python collect_usv.py videos --all --max 8 # search+download per platform
    2) browse data/raw/usv/videos/<platform>/ and DELETE clips that aren't useful
    3) python collect_usv.py video-frames --fps 1 # split survivors into frames
    4) browse data/raw/usv/frames/<clip>/ and DELETE bad frames
    5) python collect_usv.py sync # manifest auto-updates to survivors
Videos are frame-extracted once (marked processed), so deleted frames don't regenerate;
deleted videos are remembered and never re-downloaded. Frames inherit the video's URL +
uploader + date as provenance (license left blank to backfill).

Scrape+QC loop (the fast path to volume):
    1) python collect_usv.py scrape --all --max 60 # or --platform magura_v5 sea_baby
    2) browse data/raw/usv/scraped/<platform>/ and DELETE the images that aren't useful
    3) python collect_usv.py scrape ... (or `sync`) # manifest auto-updates to
       survivors; the URLs you rejected are remembered and never re-downloaded.
The scrape ledger (data/raw/usv/scrape_index.csv) keeps image_url + source_url for every
candidate, so provenance survives your deletions. Default engine is DuckDuckGo image
search (keyless, returns a source page per image); Google has no stable keyless scrape.

Simplified capture: only ``--platform`` and ``--source-url`` are required per image;
``--date``/``--license``/``--viewpoint`` are optional and can be added later with
``backfill``. Role is auto-derived from the platform via the seed catalog.

Every command only *appends* provenance; nothing here deletes source rows.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
USV_SUBDIR = ("raw", "usv")

# Manifest schema (order = CSV column order). Kept flat + text so it diffs cleanly
# and is trivially re-releasable as the "links + annotations only" artifact.
MANIFEST_FIELDS = [
    "image_id", # stable id, e.g. usv_000123 or <clip_id>_f000042
    "file_name", # path RELATIVE to the data dir (portable, matches COCO master)
    "kind", # image | frame | synthetic
    "platform", # magura_v5, sea_baby, mantas_t12, saildrone_explorer, ...
    "role", # hostile | platform | unknown (hostile-only vs all-in slicing)
    "viewpoint", # near_waterline | shore | oblique | aerial | onboard | render | unknown
    "source_url", # the STABLE source page (not a CDN/thumbnail url)
    "source_title", # human label for the source page
    "date_accessed", # YYYY-MM-DD
    "license", # e.g. PD-USGov | CC0 | CC-BY-4.0 | press-link-only | synthetic
    "attribution", # credit string to reproduce on release
    "clip_id", # for frames; blank for stills
    "frame_index", # for frames; blank for stills
    "synthetic", # true | false
    "channel", # ALWAYS eo_only — the non-negotiable firewall marker
    "sha256", # content hash (integrity + dedup)
    "width",
    "height",
    "added_utc",
    "notes",
]

# License strings we recognize (free-form allowed, but warn on unknowns).
KNOWN_LICENSES = {
    "PD-USGov", "PD", "CC0", "CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-3.0",
    "CC-BY-SA-3.0", "press-link-only", "fair-use-research", "synthetic",
}
VIEWPOINTS = {"near_waterline", "shore", "oblique", "aerial", "onboard",
              "render", "unknown"}

# Scrape ledger (URL provenance survives QC deletions). One row per candidate image
# ever seen; `status` flips present->deleted when you QC away the file on disk, and
# deleted URLs are never re-downloaded on a later scrape (your QC "sticks").
SCRAPE_INDEX_FIELDS = [
    "scrape_id", "platform", "role", "query", "engine",
    "image_url", "source_url", "source_title", "file_name",
    "sha256", "width", "height", "first_seen", "status", # status: present | deleted
]

# Video ledger (YouTube search+download). status flips present->deleted when you QC a
# video away; `processed` marks that frames were already extracted (so frame-QC sticks
# and re-runs don't regenerate deleted frames).
VIDEO_INDEX_FIELDS = [
    "video_id", "platform", "role", "query", "url", "title", "uploader",
    "upload_date", "duration_s", "file_name", "first_seen",
    "status", "processed", # status: present|deleted ; processed: ""|yes
]

# ---------------------------------------------------------------------------
# Per-platform authoritative seed source catalog.
# These are DISCOVERY starting points, not auto-downloaders. License notes are the
# *typical* status per host and MUST be verified per image at registration time.
# ---------------------------------------------------------------------------
SEED_CATALOG = {
    "_source_hosts": {
        "dvids": {
            "url": "https://www.dvidshub.net/search/?q={q}&filter[type]=image",
            "typical_license": "PD-USGov",
            "note": "US DoD imagery; most is public domain. Best for Task Force 59 "
                    "MANTAS/Saildrone/GARC and USN unmanned-systems releases. Stable "
                    "asset pages; near-waterline/pier framing common. VERIFY the "
                    "per-asset 'public domain' line — a few are restricted.",
        },
        "navy_mil": {
            "url": "https://www.navy.mil/Resources/Photo-Gallery/",
            "typical_license": "PD-USGov",
            "note": "US Navy photo gallery; public domain. Cite photographer + unit.",
        },
        "wikimedia_commons": {
            "url": "https://commons.wikimedia.org/w/index.php?search={q}&title=Special:MediaSearch&type=image",
            "typical_license": "per-file (CC0/CC-BY/CC-BY-SA/PD)",
            "note": "Per-file machine-readable license — easiest clean provenance. "
                    "Copy the exact license + author from the file page.",
        },
        "wikipedia": {
            "url": "https://en.wikipedia.org/wiki/{q}",
            "typical_license": "per-file (via Commons)",
            "note": "Platform pages surface Commons-hosted images; follow through to "
                    "the Commons file page for the real license/attribution.",
        },
        "covert_shores": {
            "url": "https://www.hisutton.com/",
            "typical_license": "press-link-only",
            "note": "H.I. Sutton OSINT USV catalog — the best single index of these "
                    "platforms. Copyrighted analysis: LINK ONLY, do not re-host; use "
                    "for identification + to find the original press source.",
        },
        "naval_news": {
            "url": "https://www.navalnews.com/?s={q}",
            "typical_license": "press-link-only",
            "note": "Press imagery; link only. Use to locate the primary source.",
        },
        "manufacturer": {
            "url": "",
            "typical_license": "press-link-only",
            "note": "Official platform pages (MARTAC, Saildrone, Saronic, Textron, "
                    "Havoc AI, Maritime Robotics, Ocius, Elbit, Meteksan/ARES, etc.). "
                    "Press-kit stills; link only. Best clean IDs, but skew to oblique "
                    "marketing angles — prefer near-waterline where offered.",
        },
    },
    "platforms": {
        # Ukrainian combat USVs — the archetypal small hostile platform. Real
        # imagery is dominated by press + OSINT combat footage (near-waterline!).
        "magura_v5": {"role": "hostile", "operator": "UA HUR",
                      "queries": ["Magura V5 USV", "Magura V5 sea drone"],
                      "primary_hosts": ["wikipedia", "covert_shores", "naval_news"]},
        "magura_v7": {"role": "hostile", "operator": "UA HUR",
                      "queries": ["Magura V7 USV"],
                      "primary_hosts": ["covert_shores", "naval_news"]},
        "sea_baby": {"role": "hostile", "operator": "UA SBU",
                     "queries": ["Sea Baby USV", "Sea Baby naval drone"],
                     "primary_hosts": ["covert_shores", "naval_news", "wikipedia"]},
        "mamai": {"role": "hostile", "operator": "UA",
                  "queries": ["Mamai USV Ukraine"],
                  "primary_hosts": ["covert_shores"]},
        # Western/allied unmanned surface craft — right size class, clean PD imagery
        # via US Navy Task Force 59. Useful as undisguised-small-USV appearance even
        # though not "hostile"; label role via annotation, keep canonical class usv.
        "mantas_t12": {"role": "platform", "operator": "MARTAC / USN TF59",
                       "queries": ["MANTAS T12 USV Task Force 59",
                                   "MARTAC MANTAS unmanned surface vessel"],
                       "primary_hosts": ["dvids", "navy_mil", "manufacturer"]},
        "devil_ray_t38": {"role": "platform", "operator": "MARTAC / USN",
                          "queries": ["Devil Ray T38 USV", "MARTAC Devil Ray"],
                          "primary_hosts": ["dvids", "manufacturer"]},
        "garc": {"role": "platform", "operator": "USN",
                 "queries": ["Global Autonomous Reconnaissance Craft GARC",
                             "USV GARC Navy"],
                 "primary_hosts": ["dvids", "navy_mil"]},
        "seagull": {"role": "platform", "operator": "Elbit Systems",
                    "queries": ["Elbit Seagull USV"],
                    "primary_hosts": ["manufacturer", "naval_news"]},
        "ulaq": {"role": "platform", "operator": "ARES/Meteksan (TR)",
                 "queries": ["ULAQ armed USV Turkey"],
                 "primary_hosts": ["manufacturer", "naval_news"]},
        "marlin": {"role": "platform", "operator": "Aselsan/Sefine (TR)",
                   "queries": ["MARLIN USV Turkey"],
                   "primary_hosts": ["manufacturer", "naval_news"]},
        # Saronic (US) — add only the SMALL tactical hulls (right size class for the
        # `usv` appearance baseline). Mirage (52') / Marauder (180') are large MUSVs
        # → the large `military` pole, deliberately EXCLUDED here.
        "saronic_spyglass": {"role": "platform", "operator": "Saronic (US)",
                             "size_class": "small", "queries": ["Saronic Spyglass ASV"],
                             "primary_hosts": ["manufacturer", "naval_news"]},
        "saronic_cutlass": {"role": "platform", "operator": "Saronic (US)",
                            "size_class": "small", "queries": ["Saronic Cutlass ASV"],
                            "primary_hosts": ["manufacturer", "naval_news"]},
        "saronic_corsair": {"role": "platform", "operator": "Saronic (US)",
                            "size_class": "small-medium",
                            "queries": ["Saronic Corsair USV", "Saronic Corsair Navy drone boat"],
                            "primary_hosts": ["manufacturer", "naval_news", "dvids"]},
        # Other US/allied small tactical ASVs — appearance-diversity proxies.
        "textron_tsunami": {"role": "platform", "operator": "Textron (US)",
                            "size_class": "small",
                            "queries": ["Textron Tsunami USV", "Textron CUSV unmanned surface"],
                            "primary_hosts": ["dvids", "manufacturer", "naval_news"]},
        "havoc_rampage": {"role": "platform", "operator": "Havoc AI (US)",
                          "size_class": "small",
                          "queries": ["Havoc AI Rampage USV"],
                          "primary_hosts": ["manufacturer", "naval_news"]},
        "maritime_robotics_otter": {"role": "platform", "operator": "Maritime Robotics (NO)",
                                    "size_class": "small",
                                    "queries": ["Maritime Robotics Otter USV",
                                                "Maritime Robotics Mariner USV"],
                                    "primary_hosts": ["manufacturer"]},
        # Hostile small USVs — HIGHEST priority for the `usv` threat class (current
        # hostile set is UA-only; broaden the threat appearance).
        "houthi_usv": {"role": "hostile", "operator": "Houthi (Yemen)",
                       "size_class": "small",
                       "queries": ["Houthi USV Tufan-1", "Houthi drone boat WBIED",
                                   "Houthi explosive boat Red Sea"],
                       "primary_hosts": ["covert_shores", "naval_news"]},
        "toloka_tlk150": {"role": "hostile", "operator": "UA",
                          "size_class": "small", "queries": ["Toloka TLK-150 USV"],
                          "primary_hosts": ["covert_shores", "naval_news"]},
    },
    "_guidance": [
        "Add only SMALL tactical hulls (~<=8 m). Large MUSVs (Saronic Mirage/Marauder, "
        "Textron large variants) are the large `military` pole, not the small `usv` class.",
        "Hostile-platform imagery (Magura/Sea Baby/Houthi/Toloka) is the priority; "
        "`platform`-role craft (MANTAS/Saildrone/Saronic/...) are size-class appearance "
        "proxies — tag role at annotation time; canonical class stays `usv`.",
        "Prefer near_waterline / shore viewpoints (fielded EO match); log viewpoint honestly.",
        "One STABLE source-page URL per image (not a CDN/thumbnail link).",
        "Real imagery first; use `synth` only as clearly-flagged augmentation.",
        "Everything registered here is EO/appearance-channel only (channel=eo_only).",
        "This set is press/OSINT-biased by nature — state it in the coverage note; the "
        "threat model brackets it with the perfect-disguise oracle.",
    ],
}


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def usv_dir(data_dir: Path) -> Path:
    return data_dir.joinpath(*USV_SUBDIR)


def manifest_path(data_dir: Path) -> Path:
    return usv_dir(data_dir) / "manifest.csv"


def _now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s or "src"


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) from the file header. PNG/JPEG/GIF/WEBP; falls back to cv2."""
    with open(path, "rb") as fh:
        head = fh.read(32)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", head[16:24])
        return int(w), int(h)
    if head[:6] in (b"GIF87a", b"GIF89a"):
        w, h = struct.unpack("<HH", head[6:10])
        return int(w), int(h)
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        # try cv2 for the several WEBP sub-formats rather than parse each chunk
        return _cv2_size(path)
    if head[:2] == b"\xff\xd8": # JPEG: scan for SOF
        with open(path, "rb") as fh:
            fh.seek(2)
            while True:
                b = fh.read(1)
                if not b:
                    break
                if b != b"\xff":
                    continue
                marker = fh.read(1)
                while marker == b"\xff":
                    marker = fh.read(1)
                if marker in (b"\xc0", b"\xc1", b"\xc2", b"\xc3"):
                    fh.read(3)
                    hh, ww = struct.unpack(">HH", fh.read(4))
                    return int(ww), int(hh)
                seg = fh.read(2)
                if len(seg) < 2:
                    break
                fh.seek(struct.unpack(">H", seg)[0] - 2, 1)
    return _cv2_size(path)


def _cv2_size(path: Path) -> tuple[int, int]:
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h)
    except Exception:
        pass
    raise ValueError(f"could not read image size: {path}")


def dhash64(path: Path) -> int | None:
    """64-bit difference hash (row gradient). Complements the 256-bit dHash in the
    EO audit; used here only to skip near-identical frames at extraction time."""
    try:
        import cv2
        import numpy as np
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
        diff = small[:, 1:] > small[:, :-1]
        bits = 0
        for b in diff.flatten():
            bits = (bits << 1) | int(b)
        return bits
    except Exception:
        return None


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# manifest read / write (append-only)
# ---------------------------------------------------------------------------
def load_manifest(data_dir: Path) -> list[dict]:
    p = manifest_path(data_dir)
    if not p.exists():
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _write_manifest(data_dir: Path, rows: list[dict]) -> None:
    p = manifest_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})


def append_rows(data_dir: Path, new_rows: list[dict]) -> None:
    rows = load_manifest(data_dir)
    rows.extend(new_rows)
    _write_manifest(data_dir, rows)


def next_still_id(rows: list[dict]) -> int:
    n = 0
    for r in rows:
        m = re.fullmatch(r"usv_(\d+)", r.get("image_id", ""))
        if m:
            n = max(n, int(m.group(1)))
    return n + 1


def _warn_license(lic: str) -> None:
    if lic and lic not in KNOWN_LICENSES:
        print(f"[warn] license '{lic}' not in the known set {sorted(KNOWN_LICENSES)}; "
              f"kept as-is — make sure it is accurate for release.")


def _base_row(**kw) -> dict:
    row = {k: "" for k in MANIFEST_FIELDS}
    row.update(kw)
    row["channel"] = "eo_only" # firewall: never overridable here
    row["added_utc"] = _now_utc()
    return row


def resolve_role(platform: str, override: str | None) -> str:
    """Per-image threat role for hostile-only vs all-in reporting. Uses an explicit
    override, else the seed-catalog role for the platform, else 'unknown' (+warn)."""
    if override:
        return override
    role = SEED_CATALOG["platforms"].get(platform, {}).get("role")
    if role:
        return role
    print(f"[warn] role for platform '{platform}' not in the seed catalog; set to "
          f"'unknown' — pass --role hostile|platform to tag it explicitly.")
    return "unknown"


# ---------------------------------------------------------------------------
# subcommand: seeds
# ---------------------------------------------------------------------------
def cmd_seeds(args) -> int:
    out = usv_dir(args.data_dir) / "sources.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        text = yaml.safe_dump(SEED_CATALOG, sort_keys=False, allow_unicode=True)
    except Exception:
        text = json.dumps(SEED_CATALOG, indent=2) # readable fallback
    out.write_text(text)
    print(f"Wrote authoritative seed source catalog -> {out}")
    print("\nPer-platform starting points (discovery, not auto-download):")
    for name, meta in SEED_CATALOG["platforms"].items():
        hosts = ", ".join(meta["primary_hosts"])
        print(f" {name:20} [{meta['role']:8}] {meta['operator']:22} via {hosts}")
    print("\nHost license posture:")
    for host, meta in SEED_CATALOG["_source_hosts"].items():
        print(f" {host:18} {meta['typical_license']}")
    print("\nWorkflow: find on an authoritative host -> obtain per its terms -> "
          "`add-image`/`frames` with provenance -> annotate -> build_coco_master.")
    return 0


# ---------------------------------------------------------------------------
# subcommand: add-image
# ---------------------------------------------------------------------------
def cmd_add_image(args) -> int:
    src = Path(args.file).expanduser().resolve()
    if not src.is_file():
        print(f"[error] not a file: {src}", file=sys.stderr)
        return 2
    if args.viewpoint not in VIEWPOINTS:
        print(f"[error] --viewpoint must be one of {sorted(VIEWPOINTS)}", file=sys.stderr)
        return 2
    _warn_license(args.license)

    rows = load_manifest(args.data_dir)
    sha = sha256_file(src)
    dup = next((r for r in rows if r.get("sha256") == sha), None)
    if dup and not args.allow_dup:
        print(f"[skip] identical content already registered as {dup['image_id']} "
              f"({dup['file_name']}). Use --allow-dup to force.")
        return 0

    seq = next_still_id(rows)
    image_id = f"usv_{seq:06d}"
    ext = src.suffix.lower() or ".jpg"
    images_dir = usv_dir(args.data_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    dst = images_dir / f"{image_id}{ext}"
    dst.write_bytes(src.read_bytes())
    try:
        w, h = image_size(dst)
    except ValueError:
        w = h = 0
        print(f"[warn] could not read dimensions for {dst.name}; set to 0 "
              f"(build_coco_master re-reads dims at convert time).")

    rel = dst.relative_to(args.data_dir).as_posix()
    role = resolve_role(args.platform, args.role)
    row = _base_row(
        image_id=image_id, file_name=rel, kind="image",
        platform=args.platform, role=role, viewpoint=args.viewpoint,
        source_url=args.source_url, source_title=args.source_title or "",
        date_accessed=args.date, license=args.license,
        attribution=args.attribution or "", synthetic="false",
        sha256=sha, width=w, height=h, notes=args.notes or "",
    )
    append_rows(args.data_dir, [row])
    print(f"[add] {image_id} {args.platform:16} [{role}] {args.viewpoint:14} {w}x{h} {rel}")
    return 0


# ---------------------------------------------------------------------------
# shared frame extraction (used by `frames` and `video-frames`)
# ---------------------------------------------------------------------------
def extract_frames(data_dir: Path, video: Path, *, platform: str, role: str,
                   clip_id: str, source_url: str, source_title: str = "",
                   date: str = "", license: str = "", attribution: str = "",
                   viewpoint: str = "unknown", fps: float = 1.0, stride: int = 0,
                   max_frames: int = 0, dedup_hamming: int = 6,
                   seen_sha: set | None = None, notes: str = "") -> tuple[list[dict], int]:
    """Sample a video to deduped JPEG frames + per-frame manifest rows. Returns
    (new_rows, skipped_dup). Does NOT append — the caller persists the rows."""
    import cv2
    out_dir = usv_dir(data_dir) / "frames" / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native_fps / fps)) if fps and fps > 0 else max(1, stride)
    if seen_sha is None:
        seen_sha = set()
    kept_hashes: list[int] = []
    new_rows: list[dict] = []
    idx = kept = skipped_dup = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            fpath = out_dir / f"{clip_id}_f{idx:06d}.jpg"
            cv2.imwrite(str(fpath), frame)
            hv = dhash64(fpath)
            if hv is not None and any(_hamming(hv, k) <= dedup_hamming for k in kept_hashes):
                fpath.unlink(missing_ok=True); skipped_dup += 1; idx += 1; continue
            sha = sha256_file(fpath)
            if sha in seen_sha:
                fpath.unlink(missing_ok=True); skipped_dup += 1; idx += 1; continue
            if hv is not None:
                kept_hashes.append(hv)
            seen_sha.add(sha)
            h, w = frame.shape[:2]
            new_rows.append(_base_row(
                image_id=f"{clip_id}_f{idx:06d}",
                file_name=fpath.relative_to(data_dir).as_posix(), kind="frame",
                platform=platform, role=role, viewpoint=viewpoint,
                source_url=source_url, source_title=source_title,
                date_accessed=date, license=license, attribution=attribution,
                clip_id=clip_id, frame_index=idx, synthetic="false", sha256=sha,
                width=w, height=h, notes=notes,
            ))
            kept += 1
            if max_frames and kept >= max_frames:
                break
        idx += 1
    cap.release()
    return new_rows, skipped_dup


# ---------------------------------------------------------------------------
# subcommand: frames (local video file -> frames, per-frame provenance)
# ---------------------------------------------------------------------------
def cmd_frames(args) -> int:
    try:
        import cv2 # noqa: F401
    except Exception as e:
        print(f"[error] OpenCV (cv2) required for frame extraction: {e}", file=sys.stderr)
        return 2
    vid = Path(args.file).expanduser().resolve()
    if not vid.is_file():
        print(f"[error] not a file: {vid}\n"
              f" Obtain the clip per its terms first (e.g. `yt-dlp <url>` if you "
              f"have rights), then point --file at the local file.", file=sys.stderr)
        return 2
    if args.viewpoint not in VIEWPOINTS:
        print(f"[error] --viewpoint must be one of {sorted(VIEWPOINTS)}", file=sys.stderr)
        return 2
    _warn_license(args.license)

    rows = load_manifest(args.data_dir)
    clip_id = args.clip_id or f"{slugify(args.platform)}_{slugify(vid.stem)}"
    if any(r.get("clip_id") == clip_id for r in rows) and not args.allow_dup:
        print(f"[skip] clip_id '{clip_id}' already has frames in the manifest. "
              f"Use --clip-id to name a different clip or --allow-dup to force.")
        return 0

    new_rows, skipped = extract_frames(
        args.data_dir, vid, platform=args.platform,
        role=resolve_role(args.platform, args.role), clip_id=clip_id,
                source_url=args.source_url, source_title=args.source_title or "",
        date=args.date, license=args.license, attribution=args.attribution or "",
        viewpoint=args.viewpoint, fps=args.fps, stride=args.stride,
        max_frames=args.max_frames, dedup_hamming=args.dedup_hamming,
        seen_sha={r.get("sha256", "") for r in rows}, notes=args.notes or "")
    append_rows(args.data_dir, new_rows)
    print(f"[frames] kept {len(new_rows)} frame(s), skipped {skipped} near/exact-dup "
          f"-> {usv_dir(args.data_dir) / 'frames' / clip_id}")
    return 0


# ---------------------------------------------------------------------------
# subcommand: synth
# ---------------------------------------------------------------------------
def cmd_synth(args) -> int:
    src = Path(args.file).expanduser().resolve()
    if not src.is_file():
        print(f"[error] not a file: {src}", file=sys.stderr)
        return 2
    rows = load_manifest(args.data_dir)
    seq = next_still_id(rows)
    image_id = f"usv_{seq:06d}"
    ext = src.suffix.lower() or ".png"
    synth_dir = usv_dir(args.data_dir) / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)
    dst = synth_dir / f"{image_id}{ext}"
    dst.write_bytes(src.read_bytes())
    try:
        w, h = image_size(dst)
    except ValueError:
        w = h = 0
    rel = dst.relative_to(args.data_dir).as_posix()
    row = _base_row(
        image_id=image_id, file_name=rel, kind="synthetic",
        platform=args.platform, role=resolve_role(args.platform, args.role),
        viewpoint=args.viewpoint if args.viewpoint in VIEWPOINTS else "render",
        source_url=args.generator or "synthetic", source_title="generated",
        date_accessed=args.date, license="synthetic",
        attribution=args.generator or "", synthetic="true",
        sha256=sha256_file(dst), width=w, height=h,
        notes=(args.notes or "") + " | flagged synthetic; excludable from any reported number",
    )
    append_rows(args.data_dir, [row])
    print(f"[synth] {image_id} synthetic=true {rel}")
    return 0


# ---------------------------------------------------------------------------
# scrape + QC-reconcile: web image search -> download -> you delete bad files ->
# re-run to auto-update the manifest (survivors only; rejects remembered).
# ---------------------------------------------------------------------------
def scrape_index_path(data_dir: Path) -> Path:
    return usv_dir(data_dir) / "scrape_index.csv"


def load_index(data_dir: Path) -> list[dict]:
    p = scrape_index_path(data_dir)
    if not p.exists():
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def save_index(data_dir: Path, rows: list[dict]) -> None:
    p = scrape_index_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SCRAPE_INDEX_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SCRAPE_INDEX_FIELDS})


def resolve_platforms(args) -> list[str]:
    cat = list(SEED_CATALOG["platforms"])
    if getattr(args, "all", False):
        return cat
    if args.platform:
        bad = [p for p in args.platform if p not in cat]
        if bad:
            print(f"[warn] not in seed catalog (scraped anyway): {bad}")
        return args.platform
    return []


def search_images(query: str, max_results: int, engine: str) -> list[dict]:
    """Return [{image_url, source_url, source_title, width, height}, ...]."""
    if engine == "ddg":
        from ddgs import DDGS
        out = []
        for r in DDGS().images(query, max_results=max_results):
            if r.get("image"):
                out.append({"image_url": r["image"], "source_url": r.get("url", ""),
                            "source_title": r.get("title", ""),
                            "width": r.get("width", 0), "height": r.get("height", 0)})
        return out
    raise SystemExit(f"unknown engine '{engine}' (supported: ddg)")


_EXT_BY_CT = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png",
              "image/webp": ".webp", "image/gif": ".gif"}


def download_image(url: str, dest: Path, referer: str = "", timeout: int = 20):
    """Fetch one image to `dest`. Returns (sha256, w, h, final_path) or None. Rejects
    non-images / HTML error pages via a header-size read."""
    import urllib.request
    headers = {"User-Agent": "Mozilla/5.0 (curation-bot; counter-usv research)",
               "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"}
    if referer:
        headers["Referer"] = referer
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ct = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            data = resp.read(25 * 1024 * 1024) # cap 25 MB
    except Exception:
        return None
    if not data:
        return None
    ext = _EXT_BY_CT.get(ct) or (Path(url.split("?")[0]).suffix.lower()
                                 if Path(url.split("?")[0]).suffix.lower() in
                                 {".jpg", ".jpeg", ".png", ".webp", ".gif"} else ".jpg")
    dest = dest.with_suffix(ext)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    try:
        w, h = image_size(dest)
    except (ValueError, FileNotFoundError):
        dest.unlink(missing_ok=True)
        return None
    return sha256_file(dest), w, h, dest


def _reconcile_index(data_dir: Path) -> tuple[list[dict], int]:
    """Flip present->deleted for index rows whose file was QC'd away on disk."""
    index = load_index(data_dir)
    removed = 0
    for r in index:
        if r.get("status") == "present":
            if not (data_dir / r.get("file_name", "")).is_file():
                r["status"] = "deleted"
                removed += 1
    if removed:
        save_index(data_dir, index)
    return index, removed


def _project_manifest(data_dir: Path, index: list[dict]) -> tuple[int, int]:
    """Rebuild the scraped rows of manifest.csv from present index rows, preserving
    manually-added rows and any backfilled fields on surviving scraped rows."""
    rows = load_manifest(data_dir)
    # keep manual (image/frame/synthetic) rows only if the file still exists — this is
    # what makes QC-by-deletion work uniformly for stills AND video-derived frames.
    manual = [r for r in rows if r.get("kind") != "scraped"
              and (data_dir / r.get("file_name", "")).is_file()]
    prior = {r["image_id"]: r for r in rows if r.get("kind") == "scraped"}
    rebuilt = []
    for r in index:
        if r.get("status") != "present":
            continue
        if not (data_dir / r.get("file_name", "")).is_file():
            continue
        image_id = f"{r['platform']}_{r['scrape_id']}"
        base = prior.get(image_id) or _base_row(
            image_id=image_id, kind="scraped", synthetic="false",
            date_accessed="", license="", viewpoint="unknown")
        base.update({
            "image_id": image_id, "file_name": r["file_name"], "kind": "scraped",
            "platform": r["platform"], "role": r.get("role", ""),
            "source_url": r.get("source_url", ""),
            "source_title": r.get("source_title", ""),
            "sha256": r.get("sha256", ""), "width": r.get("width", ""),
            "height": r.get("height", ""), "channel": "eo_only",
        })
        base.setdefault("viewpoint", "unknown")
        base.setdefault("synthetic", "false")
        rebuilt.append(base)
    _write_manifest(data_dir, manual + rebuilt)
    return len(rebuilt), len(manual)


def cmd_scrape(args) -> int:
    platforms = resolve_platforms(args)
    if not platforms:
        print("[error] pass --platform <name...> or --all", file=sys.stderr)
        return 2
    # respect prior QC first: mark files you've deleted as rejected so they don't return
    index, removed = _reconcile_index(args.data_dir)
    if removed:
        print(f"[scrape] noted {removed} file(s) you deleted since last run (won't re-fetch).")
    seen_urls = {r["image_url"] for r in index}
    seen_shas = {r["sha256"] for r in index if r.get("sha256")}

    added = 0
    for platform in platforms:
        role = resolve_role(platform, "")
        queries = list(SEED_CATALOG["platforms"].get(platform, {}).get("queries", []))
        queries = queries or [platform.replace("_", " ") + " USV"]
        got = 0
        for q in queries:
            if got >= args.max:
                break
            try:
                results = search_images(q, args.max * 2, args.engine)
            except Exception as e:
                print(f"[warn] search failed for '{q}': {e}")
                continue
            for res in results:
                if got >= args.max:
                    break
                iu = res["image_url"]
                if iu in seen_urls:
                    continue
                seen_urls.add(iu)
                scrape_id = hashlib.sha1(iu.encode()).hexdigest()[:12]
                dest = usv_dir(args.data_dir) / "scraped" / platform / f"{platform}_{scrape_id}"
                dl = download_image(iu, dest, referer=res.get("source_url", ""),
                                    timeout=args.timeout)
                if dl is None:
                    continue
                sha, w, h, fpath = dl
                if min(w, h) < args.min_px:
                    fpath.unlink(missing_ok=True)
                    continue
                if sha in seen_shas:
                    fpath.unlink(missing_ok=True)
                    continue
                seen_shas.add(sha)
                index.append({
                    "scrape_id": scrape_id, "platform": platform, "role": role,
                    "query": q, "engine": args.engine, "image_url": iu,
                    "source_url": res.get("source_url", ""),
                    "source_title": res.get("source_title", ""),
                    "file_name": fpath.relative_to(args.data_dir).as_posix(),
                    "sha256": sha, "width": w, "height": h,
                    "first_seen": _now_utc(), "status": "present",
                })
                got += 1
                added += 1
        print(f"[scrape] {platform:22} [{role:8}] +{got} new image(s)")

    save_index(args.data_dir, index)
    kept, manual = _project_manifest(args.data_dir, index)
    print(f"\n[scrape] downloaded {added} new; manifest now {kept} scraped + {manual} manual.")
    print(f"[scrape] review + delete bad files under {usv_dir(args.data_dir)/'scraped'}/<platform>/,")
    print( " then re-run `scrape` (or `sync`) to auto-update the manifest.")
    return 0


def cmd_sync(args) -> int:
    index, removed = _reconcile_index(args.data_dir)
    _, vremoved = _reconcile_videos(args.data_dir)
    kept, manual = _project_manifest(args.data_dir, index)
    extra = f", {vremoved} video(s)" if vremoved else ""
    print(f"[sync] removed {removed} QC-deleted scraped image(s){extra}; manifest now "
          f"{kept} scraped + {manual} manual row(s) (QC-deleted frames/stills dropped too).")
    return 0


# ---------------------------------------------------------------------------
# YouTube video pipeline: search+download -> (you QC videos) -> frames -> (QC frames)
# ---------------------------------------------------------------------------
def video_index_path(data_dir: Path) -> Path:
    return usv_dir(data_dir) / "video_index.csv"


def load_video_index(data_dir: Path) -> list[dict]:
    p = video_index_path(data_dir)
    if not p.exists():
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def save_video_index(data_dir: Path, rows: list[dict]) -> None:
    p = video_index_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=VIDEO_INDEX_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in VIDEO_INDEX_FIELDS})


def _reconcile_videos(data_dir: Path) -> tuple[list[dict], int]:
    """Flip present->deleted for videos you QC'd away on disk (won't re-download)."""
    vindex = load_video_index(data_dir)
    removed = 0
    for r in vindex:
        if r.get("status") == "present" and not (data_dir / r.get("file_name", "")).is_file():
            r["status"] = "deleted"
            removed += 1
    if removed:
        save_video_index(data_dir, vindex)
    return vindex, removed


def cmd_videos(args) -> int:
    import yt_dlp
    platforms = resolve_platforms(args)
    if not platforms:
        print("[error] pass --platform <name...> or --all", file=sys.stderr)
        return 2
    vindex, removed = _reconcile_videos(args.data_dir)
    if removed:
        print(f"[videos] noted {removed} video(s) you deleted since last run (won't re-fetch).")
    seen_urls = {r["url"] for r in vindex}

    added = 0
    for platform in platforms:
        role = resolve_role(platform, "")
        queries = list(SEED_CATALOG["platforms"].get(platform, {}).get("queries", []))
        queries = queries or [platform.replace("_", " ") + " USV"]
        out_dir = usv_dir(args.data_dir) / "videos" / platform
        got = 0
        for q in queries:
            if got >= args.max:
                break
            # 1) search (flat, no download) for candidate video urls
            try:
                with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True,
                                       "extract_flat": True, "noplaylist": True}) as ydl:
                    info = ydl.extract_info(f"ytsearch{args.per_query}:{q}", download=False)
                entries = info.get("entries", []) or []
            except Exception as e:
                print(f"[warn] search failed for '{q}': {e}")
                continue
            # 2) download each new candidate as a single-stream mp4 (no ffmpeg needed)
            for e in entries:
                if got >= args.max:
                    break
                url = e.get("url") or e.get("webpage_url") or ""
                if url and not url.startswith("http"):
                    url = f"https://www.youtube.com/watch?v={url}"
                if not url or url in seen_urls:
                    continue
                dur = e.get("duration") or 0
                if args.max_duration and dur and dur > args.max_duration:
                    continue
                seen_urls.add(url)
                ydl_opts = {
                    "quiet": True, "noplaylist": True, "no_warnings": True,
                    "format": (f"bv*[height<={args.height}][ext=mp4]/"
                               f"bv*[ext=mp4]/b[height<={args.height}][ext=mp4]/b[ext=mp4]/b"),
                    "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
                }
                if args.max_filesize_mb:
                    ydl_opts["max_filesize"] = args.max_filesize_mb * 1024 * 1024
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        meta = ydl.extract_info(url, download=True)
                    reqd = (meta.get("requested_downloads") or [{}])[0]
                    fpath = reqd.get("filepath")
                    if not fpath or not Path(fpath).is_file():
                        continue
                    fpath = Path(fpath)
                except Exception:
                    continue
                vindex.append({
                    "video_id": meta.get("id", fpath.stem), "platform": platform,
                    "role": role, "query": q, "url": meta.get("webpage_url", url),
                    "title": meta.get("title", ""), "uploader": meta.get("uploader", ""),
                    "upload_date": meta.get("upload_date", ""),
                    "duration_s": meta.get("duration", dur),
                    "file_name": fpath.relative_to(args.data_dir).as_posix(),
                    "first_seen": _now_utc(), "status": "present", "processed": "",
                })
                got += 1
                added += 1
        print(f"[videos] {platform:22} [{role:8}] +{got} new video(s)")

    save_video_index(args.data_dir, vindex)
    print(f"\n[videos] downloaded {added} new video(s) -> {usv_dir(args.data_dir)/'videos'}/<platform>/")
    print( " QC (delete bad clips), then run `video-frames` to split survivors into frames.")
    return 0


def cmd_video_frames(args) -> int:
    try:
        import cv2 # noqa: F401
    except Exception as e:
        print(f"[error] OpenCV (cv2) required: {e}", file=sys.stderr)
        return 2
    vindex, _ = _reconcile_videos(args.data_dir)
    todo = [r for r in vindex if r.get("status") == "present"
            and (args.force or r.get("processed") != "yes")
            and (not args.platform or r.get("platform") in args.platform)]
    if not todo:
        print("[video-frames] nothing to do (no unprocessed present videos match).")
        return 0

    rows = load_manifest(args.data_dir)
    seen_sha = {r.get("sha256", "") for r in rows}
    total_new = 0
    new_rows_all: list[dict] = []
    for r in todo:
        vpath = args.data_dir / r["file_name"]
        clip_id = f"{r['platform']}_yt_{slugify(r['video_id'])}"
        new_rows, skipped = extract_frames(
            args.data_dir, vpath, platform=r["platform"], role=r.get("role", ""),
            clip_id=clip_id, source_url=r.get("url", ""),
            source_title=r.get("title", ""), date=r.get("upload_date", ""),
            license="", attribution=r.get("uploader", ""), viewpoint="unknown",
            fps=args.fps, stride=args.stride, max_frames=args.max_frames,
            dedup_hamming=args.dedup_hamming, seen_sha=seen_sha,
            notes=f"youtube:{r['video_id']}")
        new_rows_all.extend(new_rows)
        r["processed"] = "yes"
        total_new += len(new_rows)
        print(f"[video-frames] {r['platform']:20} {r['video_id']}: +{len(new_rows)} "
              f"frame(s) (skipped {skipped} dup)")
    append_rows(args.data_dir, new_rows_all)
    save_video_index(args.data_dir, vindex)
    print(f"\n[video-frames] added {total_new} frame(s) from {len(todo)} video(s).")
    print( " QC the frames (delete bad ones), then run `sync` to auto-update the manifest.")
    return 0


# ---------------------------------------------------------------------------
# subcommand: backfill (fill in provenance recorded later)
# ---------------------------------------------------------------------------
def cmd_backfill(args) -> int:
    rows = load_manifest(args.data_dir)
    if not rows:
        print("[backfill] manifest empty.")
        return 0
    updates = {k: v for k, v in {
        "license": args.license, "date_accessed": args.date,
        "attribution": args.attribution, "viewpoint": args.viewpoint,
        "source_url": args.source_url, "source_title": args.source_title,
        "role": args.role, "notes": args.notes,
    }.items() if v}
    if not updates:
        print("[backfill] nothing to set — pass e.g. --license / --date / --attribution.")
        return 2
    if args.viewpoint and args.viewpoint not in VIEWPOINTS:
        print(f"[error] --viewpoint must be one of {sorted(VIEWPOINTS)}", file=sys.stderr)
        return 2

    def _match(r):
        if args.image_id:
            return r.get("image_id") == args.image_id
        if args.platform:
            return r.get("platform") == args.platform
        if args.clip_id:
            return r.get("clip_id") == args.clip_id
        return True # --all

    if not (args.image_id or args.platform or args.clip_id or args.all):
        print("[error] scope the backfill: --image-id / --platform / --clip-id / --all",
              file=sys.stderr)
        return 2

    n = 0
    for r in rows:
        if _match(r):
            r.update(updates)
            n += 1
    _write_manifest(args.data_dir, rows)
    print(f"[backfill] updated {n} record(s): {', '.join(f'{k}={v}' for k, v in updates.items())}")
    return 0


# ---------------------------------------------------------------------------
# subcommand: verify
# ---------------------------------------------------------------------------
def cmd_verify(args) -> int:
    rows = load_manifest(args.data_dir)
    if not rows:
        print("[verify] manifest is empty (nothing registered yet).")
        return 0
    problems = backfill = 0
    ids, shas = set(), {}
    for r in rows:
        iid = r.get("image_id", "")
        if iid in ids:
            print(f"[fail] duplicate image_id: {iid}"); problems += 1
        ids.add(iid)
        # firewall assertion
        if r.get("channel") != "eo_only":
            print(f"[fail] {iid}: channel!='eo_only' — firewall violation"); problems += 1
        # hard-required minimum (simplified capture): platform + source_url
        for req in ("platform", "source_url"):
            if not r.get(req):
                print(f"[fail] {iid}: missing {req}"); problems += 1
        # backfill-later provenance: warn only (does not fail verify)
        for opt in ("date_accessed", "license"):
            if not r.get(opt):
                backfill += 1
                break
        fp = args.data_dir / r.get("file_name", "")
        if not fp.is_file():
            print(f"[fail] {iid}: file missing on disk: {r.get('file_name')}"); problems += 1
            continue
        if not args.no_hash:
            got = sha256_file(fp)
            if r.get("sha256") and got != r["sha256"]:
                print(f"[fail] {iid}: sha256 mismatch (file changed)"); problems += 1
            shas.setdefault(got, []).append(iid)
    if not args.no_hash:
        for sha, who in shas.items():
            if len(who) > 1:
                print(f"[warn] exact-duplicate content across ids: {who}")
    real = sum(1 for r in rows if r.get("synthetic") == "false")
    synth = sum(1 for r in rows if r.get("synthetic") == "true")
    print(f"\n[verify] {len(rows)} records ({real} real / {synth} synthetic); "
          f"{problems} problem(s).")
    if backfill:
        print(f"[verify] {backfill} record(s) still need date/license backfill "
              f"(not a failure — capture simplified to platform+url for now).")
    print("[verify] firewall: all records channel=eo_only "
          f"-> {'OK' if all(r.get('channel')=='eo_only' for r in rows) else 'VIOLATION'}")
    return 1 if problems else 0


# ---------------------------------------------------------------------------
# subcommand: status
# ---------------------------------------------------------------------------
def cmd_status(args) -> int:
    rows = load_manifest(args.data_dir)
    if not rows:
        print("[status] manifest empty.")
        return 0
    by_platform, by_view, by_lic, by_role = {}, {}, {}, {}
    real = synth = 0
    for r in rows:
        by_platform[r.get("platform", "?")] = by_platform.get(r.get("platform", "?"), 0) + 1
        by_view[r.get("viewpoint", "?")] = by_view.get(r.get("viewpoint", "?"), 0) + 1
        by_lic[r.get("license", "?")] = by_lic.get(r.get("license", "?"), 0) + 1
        by_role[r.get("role", "?")] = by_role.get(r.get("role", "?"), 0) + 1
        if r.get("synthetic") == "true":
            synth += 1
        else:
            real += 1
    hostile = by_role.get("hostile", 0)

    def _fmt(d):
        return "\n".join(f" {k:22} {v}" for k, v in sorted(d.items(), key=lambda x: -x[1]))

    report = [
        f"# usv imagery coverage — {_now_utc()}",
        "",
        f"Total registered: **{len(rows)}** (real {real} / synthetic {synth})",
        f"Reporting slices: **hostile-only {hostile}** / **all-in {len(rows)}** "
        f"(platform-role proxies {len(rows) - hostile}).",
        "",
        "## By role", _fmt(by_role),
        "", "## By platform", _fmt(by_platform),
        "", "## By viewpoint", _fmt(by_view),
        "", "## By license", _fmt(by_lic),
        "",
        "## Known limitations (carry into the data card, the data card)",
        "- Press/OSINT/manufacturer-biased sample; viewpoint skews oblique vs. a fielded",
        " shore-based EO set. Bracketed by the perfect-disguise oracle (docs/THREAT_MODEL.md).",
        "- EO/appearance channel only (channel=eo_only): never enters the kinematic scorer;",
        " hostile trajectories remain synthesized at eval.",
        "- Mixed hostile + platform-role (friendly/commercial) craft as appearance proxies;",
        " role tagged per image so results are reported BOTH hostile-only and all-in.",
        "- Synthetic frames flagged synthetic=true and excludable from any reported number.",
        "- Inherits the EO-audit pixels-on-target floor + dedup/QA on integration (audit_eo.py).",
    ]
    text = "\n".join(report)
    print(text.replace("**", ""))
    if args.write:
        out = REPO_ROOT / "results" / "usv" / "coverage.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"\n[status] wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Provenance-first curation kit for the usv EO set .",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("seeds", help="write/refresh the per-platform seed source catalog")

    def _prov(sp, need_view=True):
        # Simplified capture: only --platform and --source-url are required now;
        # date/license/viewpoint/attribution are optional and can be backfilled later.
        sp.add_argument("--platform", required=True, help="e.g. magura_v5, mantas_t12")
        sp.add_argument("--role", default="", choices=["", "hostile", "platform", "unknown"],
                        help="threat role (default: derived from the seed catalog)")
        sp.add_argument("--source-url", required=True, help="STABLE source page URL")
        if need_view:
            sp.add_argument("--viewpoint", default="unknown",
                            help=f"one of {sorted(VIEWPOINTS)} (default unknown)")
        sp.add_argument("--source-title", default="")
        sp.add_argument("--date", default="", help="date accessed YYYY-MM-DD (backfill later)")
        sp.add_argument("--license", default="",
                        help=f"e.g. {sorted(KNOWN_LICENSES)} (backfill later)")
        sp.add_argument("--attribution", default="", help="credit string for release")
        sp.add_argument("--notes", default="")

    ai = sub.add_parser("add-image", help="register one already-obtained still")
    ai.add_argument("--file", required=True)
    _prov(ai)
    ai.add_argument("--allow-dup", action="store_true")

    fr = sub.add_parser("frames", help="extract frames from a LOCAL video file")
    fr.add_argument("--file", required=True, help="local video file (obtain per its terms)")
    _prov(fr)
    fr.add_argument("--clip-id", default="", help="stable clip id (default: platform_stem)")
    fr.add_argument("--fps", type=float, default=1.0, help="target frames/sec (default 1)")
    fr.add_argument("--stride", type=int, default=0, help="alt: keep every Nth frame")
    fr.add_argument("--max-frames", type=int, default=0, help="cap frames kept (0=all)")
    fr.add_argument("--dedup-hamming", type=int, default=6,
                    help="skip frames within this 64-bit dHash distance (default 6)")
    fr.add_argument("--allow-dup", action="store_true")

    sy = sub.add_parser("synth", help="register a generated frame (synthetic=true)")
    sy.add_argument("--file", required=True)
    sy.add_argument("--platform", required=True)
    sy.add_argument("--role", default="", choices=["", "hostile", "platform", "unknown"],
                    help="threat role (default: derived from the seed catalog)")
    sy.add_argument("--viewpoint", default="render")
    sy.add_argument("--date", default="", help="date generated YYYY-MM-DD (optional)")
    sy.add_argument("--generator", default="", help="generator/tool id for provenance")
    sy.add_argument("--notes", default="")

    sc = sub.add_parser("scrape", help="web-image-search + download per platform "
                                       "(records source URL per image; QC-reconcilable)")
    scg = sc.add_mutually_exclusive_group(required=True)
    scg.add_argument("--platform", nargs="+", help="platform name(s) to scrape")
    scg.add_argument("--all", action="store_true", help="every catalog platform")
    sc.add_argument("--max", type=int, default=40, help="max new images per platform")
    sc.add_argument("--engine", default="ddg", choices=["ddg"],
                    help="image-search backend (ddg = DuckDuckGo, keyless)")
    sc.add_argument("--min-px", type=int, default=128,
                    help="skip images whose shorter side < this (drops icons/thumbs)")
    sc.add_argument("--timeout", type=int, default=20, help="per-image download timeout (s)")

    vd = sub.add_parser("videos", help="search + download YouTube videos per platform "
                                       "(yt-dlp); QC-reconcilable video ledger")
    vdg = vd.add_mutually_exclusive_group(required=True)
    vdg.add_argument("--platform", nargs="+", help="platform name(s)")
    vdg.add_argument("--all", action="store_true", help="every catalog platform")
    vd.add_argument("--per-query", type=int, default=10, help="YouTube results/query to consider")
    vd.add_argument("--max", type=int, default=8, help="max new videos per platform")
    vd.add_argument("--height", type=int, default=720, help="max video height to fetch")
    vd.add_argument("--max-duration", type=int, default=1200,
                    help="skip videos longer than this many seconds (0=no cap)")
    vd.add_argument("--max-filesize-mb", type=int, default=300, help="per-video cap (0=none)")

    vf = sub.add_parser("video-frames", help="split QC'd videos into deduped frames "
                                             "(once per video) with provenance")
    vf.add_argument("--platform", nargs="+", help="limit to platform(s)")
    vf.add_argument("--fps", type=float, default=1.0, help="target frames/sec (default 1)")
    vf.add_argument("--stride", type=int, default=0, help="alt: keep every Nth frame")
    vf.add_argument("--max-frames", type=int, default=0, help="cap frames per video (0=all)")
    vf.add_argument("--dedup-hamming", type=int, default=6,
                    help="skip frames within this 64-bit dHash distance (default 6)")
    vf.add_argument("--force", action="store_true", help="re-extract already-processed videos")

    sub.add_parser("sync", help="reconcile after QC: drop deleted images/frames, "
                                "tombstone deleted videos, auto-update the manifest")

    bf = sub.add_parser("backfill", help="fill in provenance recorded later "
                                         "(license/date/…) for existing rows")
    scope = bf.add_mutually_exclusive_group()
    scope.add_argument("--image-id", help="one record")
    scope.add_argument("--platform", help="all rows for a platform")
    scope.add_argument("--clip-id", help="all frames of a clip")
    scope.add_argument("--all", action="store_true", help="every record")
    bf.add_argument("--license", default="")
    bf.add_argument("--date", default="", help="YYYY-MM-DD")
    bf.add_argument("--attribution", default="")
    bf.add_argument("--viewpoint", default="")
    bf.add_argument("--source-url", default="")
    bf.add_argument("--source-title", default="")
    bf.add_argument("--role", default="")
    bf.add_argument("--notes", default="")

    ve = sub.add_parser("verify", help="integrity + firewall + dedup check")
    ve.add_argument("--no-hash", action="store_true", help="skip sha256 recompute")

    st = sub.add_parser("status", help="coverage summary")
    st.add_argument("--write", action="store_true", help="write results/usv/coverage.md")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "seeds": cmd_seeds, "add-image": cmd_add_image, "frames": cmd_frames,
        "synth": cmd_synth, "scrape": cmd_scrape, "sync": cmd_sync,
        "videos": cmd_videos, "video-frames": cmd_video_frames,
        "backfill": cmd_backfill, "verify": cmd_verify, "status": cmd_status,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
