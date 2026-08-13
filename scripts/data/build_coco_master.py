#!/usr/bin/env python3
"""Convert every EO source to one COCO master.

Reads each source's native annotations, maps native categories to the canonical
classes defined in ``data/taxonomy.yaml``, and writes:

  * ``data/annotations/<source>.coco.json`` — one converted file per source
  * ``data/annotations/coco_master.json`` — the merged master (global ids)
  * ``data/annotations/convert_summary.json`` — per-source / per-class counts

Every image record carries **provenance** (``source``, ``orig_id``, ``orig_label``
kept on annotations, ``orig_split`` where the source defines one) so per-source
evaluation slices and traceability are possible downstream.

Design
------
* Master categories are the canonical classes from the taxonomy (stable ids in
  taxonomy order). Each category keeps its ``role`` (hostile/benign/non_target).
* ``file_name`` is stored RELATIVE to the data dir (e.g. ``raw/seaships/train/x.jpg``)
  so the master is portable; a loader joins it with the data root.
* bbox is COCO ``[x, y, w, h]`` (floats), clamped to the image; degenerate boxes
  (w or h < 1 px after clamping) are dropped and counted.
* Stdlib only — PNG dimensions (needed for ABOShips, whose CSV omits image size)
  are read from the file header; no numpy/Pillow/scipy required.

Adapters implemented: seaships (native Pascal-VOC, or Roboflow COCO fallback),
mcships (Pascal-VOC), aboships (CSV), smd (video-frame sampling + .mat parsing),
usv (the curated usv set curated set: per-image provenance manifest + CVAT/COCO or LabelImg/VOC
boxes; enforces the EO-only channel firewall + synthetic flag on each image record).

Usage
-----
    python scripts/data/build_coco_master.py # all present sources
    python scripts/data/build_coco_master.py --sources seaships mcships
    python scripts/data/build_coco_master.py --drop-non-target # omit static_aid/unknown_other
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Taxonomy loading (minimal YAML read — avoid a PyYAML dependency)
# ---------------------------------------------------------------------------
def load_taxonomy(path: Path) -> dict:
    """Load taxonomy.yaml. Uses PyYAML if available, else a tiny fallback parser
    tailored to this file's structure."""
    try:
        import yaml # type: ignore
        return yaml.safe_load(path.read_text())
    except Exception:
        return _mini_taxonomy(path)


def _mini_taxonomy(path: Path) -> dict:
    """Fallback: extract only what the converter needs — canonical class names +
    roles, and each eo_source's native->canonical map. Not a general YAML parser."""
    canonical: dict[str, dict] = {}
    eo: dict[str, dict] = {}
    section = None # 'canonical' | 'eo'
    cur_class = None
    cur_source = None
    in_native = False
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        s = raw.strip()
        if indent == 0:
            section = {"canonical_classes:": "canonical", "eo_sources:": "eo"}.get(s)
            cur_class = cur_source = None
            in_native = False
            continue
        if section == "canonical":
            if indent == 2 and s.endswith(":"):
                cur_class = s[:-1]
                canonical[cur_class] = {}
            elif indent == 4 and cur_class and s.startswith("role:"):
                canonical[cur_class]["role"] = s.split(":", 1)[1].strip()
        elif section == "eo":
            if indent == 2 and s.endswith(":"):
                cur_source = s[:-1]
                eo[cur_source] = {"native": {}}
                in_native = False
            elif indent == 4 and s.startswith("native:"):
                in_native = True
            elif indent == 4 and not s.startswith("native:"):
                in_native = False
            elif indent == 6 and in_native and cur_source and ":" in s:
                k, v = s.split(":", 1)
                eo[cur_source]["native"][k.strip().strip('"')] = v.strip()
    return {"canonical_classes": canonical, "eo_sources": eo}


# ---------------------------------------------------------------------------
# Image dimensions from file header (stdlib only)
# ---------------------------------------------------------------------------
def image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) reading only the header. Supports PNG and JPEG."""
    with open(path, "rb") as fh:
        head = fh.read(26)
        # PNG: 8-byte sig, IHDR length+type (8), then width,height big-endian
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", head[16:24])
            return int(w), int(h)
        # JPEG: scan for SOF marker
        if head[:2] == b"\xff\xd8":
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
                    fh.read(3) # length(2) + precision(1)
                    hh, ww = struct.unpack(">HH", fh.read(4))
                    return int(ww), int(hh)
                seg = fh.read(2)
                if len(seg) < 2:
                    break
                fh.seek(struct.unpack(">H", seg)[0] - 2, 1)
    raise ValueError(f"could not read image size: {path}")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class CocoBuilder:
    def __init__(self, data_dir: Path, taxonomy: dict, drop_non_target: bool):
        self.data_dir = data_dir
        self.tax = taxonomy
        self.drop_non_target = drop_non_target
        # canonical name -> id (1-indexed, taxonomy order)
        self.cat_id: dict[str, int] = {}
        self.categories: list[dict] = []
        for i, (name, meta) in enumerate(taxonomy["canonical_classes"].items(), start=1):
            self.cat_id[name] = i
            self.categories.append({"id": i, "name": name,
                                    "role": meta.get("role", "benign")})
        self.non_target = {n for n, m in taxonomy["canonical_classes"].items()
                           if m.get("role") == "non_target"}

    def native_map(self, source: str) -> dict[str, str]:
        return self.tax["eo_sources"][source]["native"]

    def _emit(self, images: list, anns: list, source: str) -> dict:
        """Assemble a per-source COCO dict, dropping non-target if requested and
        renumbering annotation category ids. Returns the coco dict + stats."""
        cats = self.categories
        if self.drop_non_target:
            cats = [c for c in self.categories if c["name"] not in self.non_target]
        return {"info": {"source": source},
                "images": images, "annotations": anns, "categories": cats}


# --- per-source adapters: each returns (images, annotations, stats) -----------
# images/annotations use canonical category NAMES in a temp field; the merge step
# assigns final ids. Provenance fields are attached here.

def adapt_seaships(b: CocoBuilder):
    """SeaShips → COCO. Prefers the NATIVE 1920×1080 Pascal-VOC release
    (Annotations/ + JPEGImages/ + ImageSets/Main); falls back to the older
    Roboflow COCO export (train/test/_annotations.coco.json) if VOC is absent."""
    root = b.data_dir / "raw" / "seaships"
    if (root / "Annotations").is_dir() and (root / "JPEGImages").is_dir():
        return _adapt_seaships_voc(b, root)
    return _adapt_seaships_coco(b, root)


def _adapt_seaships_voc(b: CocoBuilder, root):
    nmap = b.native_map("seaships")
    # split membership from ImageSets/Main (train/val/test; trainval is the union)
    split_of = {}
    for sp in ("train", "val", "test"):
        f = root / "ImageSets" / "Main" / f"{sp}.txt"
        if f.exists():
            for line in f.read_text().split():
                split_of[line.strip()] = sp
    images, anns, dropped, per_class = [], [], 0, {}
    for xml in sorted((root / "Annotations").glob("*.xml")):
        try:
            r = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        stem = xml.stem
        size = r.find("size")
        W = int(size.findtext("width")); H = int(size.findtext("height"))
        fn = r.findtext("filename") or f"{stem}.jpg"
        images.append({
            "file_name": f"raw/seaships/JPEGImages/{fn}",
            "width": W, "height": H, "source": "seaships",
            "orig_id": stem, "orig_split": split_of.get(stem), "_key": stem,
        })
        for obj in r.findall("object"):
            native = (obj.findtext("name") or "").strip()
            if native not in nmap:
                continue
            bnd = obj.find("bndbox")
            x1 = float(bnd.findtext("xmin")); y1 = float(bnd.findtext("ymin"))
            x2 = float(bnd.findtext("xmax")); y2 = float(bnd.findtext("ymax"))
            bb = _clamp([x1, y1, x2 - x1, y2 - y1], W, H)
            if bb is None:
                dropped += 1
                continue
            anns.append({"_img": stem, "category": nmap[native], "bbox": bb,
                         "source": "seaships", "orig_label": native})
            per_class[nmap[native]] = per_class.get(nmap[native], 0) + 1
    return images, anns, {"dropped_boxes": dropped, "per_class": per_class}


def _adapt_seaships_coco(b: CocoBuilder, root):
    nmap = b.native_map("seaships")
    images, anns, dropped, per_class = [], [], 0, {}
    for split in ("train", "test"):
        jp = root / split / "_annotations.coco.json"
        if not jp.exists():
            continue
        d = json.loads(jp.read_text())
        cats = {c["id"]: c["name"] for c in d["categories"]}
        for im in d["images"]:
            images.append({
                "file_name": f"raw/seaships/{split}/{im['file_name']}",
                "width": im["width"], "height": im["height"],
                "source": "seaships", "orig_id": f"{split}/{im['id']}",
                "orig_split": split, "_key": (split, im["id"]),
            })
        dims = {(split, im["id"]): (im["width"], im["height"]) for im in d["images"]}
        for a in d["annotations"]:
            native = cats.get(a["category_id"])
            if native not in nmap: # skips Roboflow root supercategory "ships"
                continue
            canon = nmap[native]
            W, H = dims[(split, a["image_id"])]
            bb = _clamp(a["bbox"], W, H)
            if bb is None:
                dropped += 1
                continue
            anns.append({"_img": (split, a["image_id"]), "category": canon,
                         "bbox": bb, "source": "seaships", "orig_label": native})
            per_class[canon] = per_class.get(canon, 0) + 1
    return images, anns, {"dropped_boxes": dropped, "per_class": per_class}


def adapt_mcships(b: CocoBuilder):
    root = b.data_dir / "raw" / "mcships"
    nmap = b.native_map("mcships")
    # split membership from ImageSets/Main
    split_of = {}
    for sp in ("train", "val", "test"):
        f = root / "ImageSets" / "Main" / f"{sp}.txt"
        if f.exists():
            for line in f.read_text().split():
                split_of[line.strip()] = sp
    images, anns, dropped, per_class = [], [], 0, {}
    ann_dir = root / "Annotations"
    for xml in sorted(ann_dir.glob("*.xml")):
        try:
            r = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        stem = xml.stem
        size = r.find("size")
        W = int(size.findtext("width")); H = int(size.findtext("height"))
        fn = r.findtext("filename") or f"{stem}.jpg"
        images.append({
            "file_name": f"raw/mcships/JPEGImages/{fn}",
            "width": W, "height": H, "source": "mcships",
            "orig_id": stem, "orig_split": split_of.get(stem),
            "_key": stem,
        })
        for obj in r.findall("object"):
            native = (obj.findtext("name") or "").strip()
            if native not in nmap:
                continue
            bnd = obj.find("bndbox")
            x1 = float(bnd.findtext("xmin")); y1 = float(bnd.findtext("ymin"))
            x2 = float(bnd.findtext("xmax")); y2 = float(bnd.findtext("ymax"))
            bb = _clamp([x1, y1, x2 - x1, y2 - y1], W, H)
            if bb is None:
                dropped += 1
                continue
            canon = nmap[native]
            anns.append({"_img": stem, "category": canon, "bbox": bb,
                         "source": "mcships", "orig_label": native})
            per_class[canon] = per_class.get(canon, 0) + 1
    return images, anns, {"dropped_boxes": dropped, "per_class": per_class}


def adapt_aboships(b: CocoBuilder):
    root = b.data_dir / "raw" / "aboships"
    nmap = b.native_map("aboships")
    csv_path = root / "Labels" / "Vesibussi_Labels.csv"
    img_root = root / "Seaships"
    rows_by_file: dict[str, list] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows_by_file.setdefault(row["filename"], []).append(row)
    images, anns, dropped, per_class, missing = [], [], 0, {}, 0
    for fname, rows in rows_by_file.items():
        date = fname[:8]
        img_path = img_root / date / f"{fname}.png"
        if not img_path.exists():
            missing += 1
            continue
        try:
            W, H = image_size(img_path)
        except ValueError:
            missing += 1
            continue
        images.append({
            "file_name": f"raw/aboships/Seaships/{date}/{fname}.png",
            "width": W, "height": H, "source": "aboships",
            "orig_id": fname, "orig_split": None, "_key": fname,
        })
        for row in rows:
            native = row["class"].strip()
            if native not in nmap:
                continue
            x1 = float(row["xmin"]); x2 = float(row["xmax"])
            y1 = float(row["ymin"]); y2 = float(row["ymax"])
            bb = _clamp([x1, y1, x2 - x1, y2 - y1], W, H)
            if bb is None:
                dropped += 1
                continue
            canon = nmap[native]
            anns.append({"_img": fname, "category": canon, "bbox": bb,
                         "source": "aboships", "orig_label": native})
            per_class[canon] = per_class.get(canon, 0) + 1
    return images, anns, {"dropped_boxes": dropped, "missing_images": missing,
                          "per_class": per_class}


def _smd_frame_objects(fr, np):
    """Return [(label, [x,y,w,h]), ...] for one SMD structXML frame."""
    ot = getattr(fr, "ObjectType", None)
    bb = getattr(fr, "BB", None)
    if ot is None or bb is None:
        return []
    ot = np.atleast_1d(ot)
    bb = np.atleast_2d(bb)
    out = []
    for i in range(min(len(ot), bb.shape[0])):
        if bb.shape[1] < 4:
            continue
        x, y, w, h = (float(v) for v in bb[i][:4])
        out.append((str(ot[i]).strip(), [x, y, w, h]))
    return out


def adapt_smd(b: CocoBuilder):
    """SMD on-shore video → sampled frames + COCO boxes. Lazy-imports cv2/scipy so
    the other (image) adapters stay stdlib-only."""
    import numpy as np
    import scipy.io as sio
    import cv2

    root = b.data_dir / "raw" / "smd"
    gt_dir = root / "SMD-Plus" / "ObjectGT"
    vid_dir = root / "VIS_Onshore" / "Videos"
    frames_out = root / "frames"
    nmap = b.native_map("smd_plus")
    stride = max(1, int(getattr(b, "smd_stride", 20)))
    images, anns, dropped, per_class, missing_vid = [], [], 0, {}, 0

    for gt in sorted(gt_dir.glob("*.mat")):
        video = gt.stem.replace("_ObjectGT", "") # e.g. MVI_1469_VIS
        vpath = vid_dir / f"{video}.avi"
        if not vpath.exists():
            missing_vid += 1
            continue
        frames = np.atleast_1d(sio.loadmat(
            gt, struct_as_record=False, squeeze_me=True)["structXML"])
        cap = cv2.VideoCapture(str(vpath))
        outdir = frames_out / video
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0 and idx < len(frames):
                H, W = frame.shape[:2]
                mapped = [(lab, bb) for lab, bb in _smd_frame_objects(frames[idx], np)
                          if lab in nmap]
                if mapped:
                    outdir.mkdir(parents=True, exist_ok=True)
                    fname = f"{video}_f{idx:06d}.jpg"
                    cv2.imwrite(str(outdir / fname), frame)
                    key = f"{video}#{idx}"
                    images.append({
                        "file_name": f"raw/smd/frames/{video}/{fname}",
                        "width": W, "height": H, "source": "smd",
                        "orig_id": key, "orig_split": None, "_key": key,
                    })
                    for lab, bb in mapped:
                        cb = _clamp(bb, W, H)
                        if cb is None:
                            dropped += 1
                            continue
                        canon = nmap[lab]
                        anns.append({"_img": key, "category": canon, "bbox": cb,
                                     "source": "smd", "orig_label": lab})
                        per_class[canon] = per_class.get(canon, 0) + 1
            idx += 1
        cap.release()
    return images, anns, {"dropped_boxes": dropped, "missing_videos": missing_vid,
                          "frames": len(images), "per_class": per_class}


def adapt_usv(b: CocoBuilder):
    """Curated ``usv`` set → COCO. Reads the per-image provenance
    manifest (``data/raw/usv/manifest.csv``, written by ``scripts/data/collect_usv.py``)
    and box annotations from either a CVAT/COCO export (``annotations.coco.json``)
    or LabelImg Pascal-VOC (``Annotations/*.xml``). Every box maps to canonical
    ``usv``; the original CVAT/VOC label is preserved as ``orig_label``.

    The channel firewall is enforced here: each image record carries
    ``channel="eo_only"`` and a ``synthetic`` flag (plus per-image provenance), so
    the kinematic scorer pipeline can hard-exclude this set (``source=="usv"``) and
    any reported number can drop synthetic frames on demand. Images with zero
    boxes are dropped and logged (positive-only hostile class — empty = QC reject).
    """
    root = b.data_dir / "raw" / "usv"
    man_path = root / "manifest.csv"
    if not man_path.exists():
        return [], [], {"per_class": {}, "note": "no manifest.csv"}

    # provenance keyed by image basename (the stable join key across annotation formats)
    prov: dict[str, dict] = {}
    with open(man_path, newline="") as fh:
        for r in csv.DictReader(fh):
            base = Path(r.get("file_name", "")).name
            if base:
                prov[base] = r

    images, anns, dropped, per_class, no_prov = [], [], 0, {}, 0

    def _img_record(base: str, W: int, H: int):
        r = prov[base]
        return {
            "file_name": r["file_name"], "width": W, "height": H,
            "source": "usv", "orig_id": r.get("image_id") or base,
            "orig_split": None, "_key": base,
            # --- channel firewall + provenance passthrough (flow into the master) ---
            "channel": "eo_only",
            "synthetic": (r.get("synthetic", "false").lower() == "true"),
            "usv_platform": r.get("platform", ""),
            "usv_viewpoint": r.get("viewpoint", ""),
            "source_url": r.get("source_url", ""),
            "license": r.get("license", ""),
            "sha256": r.get("sha256", ""),
        }

    coco_json = root / "annotations.coco.json"
    voc_dir = root / "Annotations"

    if coco_json.exists():
        d = json.loads(coco_json.read_text())
        cats = {c["id"]: c.get("name", "usv") for c in d.get("categories", [])}
        # map image_id -> basename, keeping only images we have provenance for
        keep_base, dims = {}, {}
        for im in d["images"]:
            base = Path(im["file_name"]).name
            if base not in prov:
                no_prov += 1
                continue
            W = int(im.get("width") or 0); H = int(im.get("height") or 0)
            if not (W and H):
                fp = b.data_dir / prov[base]["file_name"]
                try:
                    W, H = image_size(fp)
                except (ValueError, FileNotFoundError):
                    pass
            keep_base[im["id"]] = base
            dims[im["id"]] = (W, H)
            images.append(_img_record(base, W, H))
        for a in d["annotations"]:
            base = keep_base.get(a["image_id"])
            if base is None:
                continue
            W, H = dims[a["image_id"]]
            bb = _clamp(a["bbox"], W or 10**6, H or 10**6)
            if bb is None:
                dropped += 1
                continue
            native = cats.get(a.get("category_id"), "usv")
            anns.append({"_img": base, "category": "usv", "bbox": bb,
                         "source": "usv", "orig_label": native})
            per_class["usv"] = per_class.get("usv", 0) + 1

    elif voc_dir.is_dir():
        for xml in sorted(voc_dir.glob("*.xml")):
            try:
                r = ET.parse(xml).getroot()
            except ET.ParseError:
                continue
            fn = r.findtext("filename") or f"{xml.stem}.jpg"
            base = Path(fn).name
            if base not in prov:
                no_prov += 1
                continue
            size = r.find("size")
            W = int(size.findtext("width")) if size is not None else 0
            H = int(size.findtext("height")) if size is not None else 0
            if not (W and H):
                try:
                    W, H = image_size(b.data_dir / prov[base]["file_name"])
                except (ValueError, FileNotFoundError):
                    pass
            images.append(_img_record(base, W, H))
            for obj in r.findall("object"):
                native = (obj.findtext("name") or "usv").strip()
                bnd = obj.find("bndbox")
                if bnd is None:
                    continue
                x1 = float(bnd.findtext("xmin")); y1 = float(bnd.findtext("ymin"))
                x2 = float(bnd.findtext("xmax")); y2 = float(bnd.findtext("ymax"))
                bb = _clamp([x1, y1, x2 - x1, y2 - y1], W or 10**6, H or 10**6)
                if bb is None:
                    dropped += 1
                    continue
                anns.append({"_img": base, "category": "usv", "bbox": bb,
                             "source": "usv", "orig_label": native})
                per_class["usv"] = per_class.get("usv", 0) + 1
    else:
        return [], [], {"per_class": {}, "images_in_manifest": len(prov),
                        "note": "manifest present but no annotations "
                                "(annotations.coco.json or Annotations/*.xml) yet"}

    # Positive-only hostile class: an image with zero boxes is a QC reject, not a
    # silent background negative. Drop and log so empty exports never reach the master.
    ann_keys = {a["_img"] for a in anns}
    n_before = len(images)
    images = [im for im in images if im["_key"] in ann_keys]
    empty_dropped = n_before - len(images)
    if empty_dropped:
        print(f"  [usv] dropped {empty_dropped} image(s) with zero annotations "
              f"(QC rejects; not carried into master)")

    return images, anns, {"dropped_boxes": dropped, "per_class": per_class,
                          "images_without_provenance": no_prov,
                          "empty_annotation_images_dropped": empty_dropped,
                          "manifest_images": len(prov)}


ADAPTERS = {
    "seaships": adapt_seaships,
    "mcships": adapt_mcships,
    "aboships": adapt_aboships,
    "smd": adapt_smd,
    "usv": adapt_usv,
}


def _clamp(bbox, W, H):
    """Clamp [x,y,w,h] to the image; return None if degenerate (<1 px)."""
    x, y, w, h = (float(v) for v in bbox)
    x2, y2 = x + w, y + h
    x = max(0.0, min(x, W)); y = max(0.0, min(y, H))
    x2 = max(0.0, min(x2, W)); y2 = max(0.0, min(y2, H))
    w, h = x2 - x, y2 - y
    if w < 1.0 or h < 1.0:
        return None
    return [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def source_present(data_dir: Path, source: str) -> bool:
    checks = {
        "seaships": data_dir / "raw" / "seaships",
        "mcships": data_dir / "raw" / "mcships" / "Annotations",
        "aboships": data_dir / "raw" / "aboships" / "Labels" / "Vesibussi_Labels.csv",
        "smd": data_dir / "raw" / "smd" / "SMD-Plus" / "ObjectGT",
        "usv": data_dir / "raw" / "usv" / "manifest.csv",
    }
    return checks[source].exists()


def write_source_coco(b: CocoBuilder, source: str, images, anns, out_dir: Path):
    """Assign local ids for a single-source file and write it."""
    img_id = {im["_key"]: i for i, im in enumerate(images, start=1)}
    coco_images = [{k: v for k, v in im.items() if k != "_key"} | {"id": img_id[im["_key"]]}
                   for im in images]
    coco_anns = []
    for j, a in enumerate(anns, start=1):
        coco_anns.append({
            "id": j, "image_id": img_id[a["_img"]],
            "category_id": b.cat_id[a["category"]],
            "bbox": a["bbox"], "area": round(a["bbox"][2] * a["bbox"][3], 2),
            "iscrowd": 0, "source": a["source"], "orig_label": a["orig_label"],
        })
    coco = b._emit(coco_images, coco_anns, source)
    (out_dir / f"{source}.coco.json").write_text(json.dumps(coco))
    return coco_images, coco_anns


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the COCO master .",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--sources", nargs="+", choices=list(ADAPTERS),
                   help="default: all present sources")
    p.add_argument("--drop-non-target", action="store_true",
                   help="omit static_aid/unknown_other categories from outputs")
    p.add_argument("--smd-frame-stride", type=int, default=20,
                   help="sample every Nth SMD video frame (default 20 ~= 1 fps)")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (default: <data-dir>/annotations)")
    args = p.parse_args(argv)

    out_dir = args.out or (args.data_dir / "annotations")
    out_dir.mkdir(parents=True, exist_ok=True)
    tax = load_taxonomy(args.data_dir / "taxonomy.yaml")
    b = CocoBuilder(args.data_dir, tax, args.drop_non_target)
    b.smd_stride = args.smd_frame_stride

    requested = args.sources or list(ADAPTERS)
    summary = {"sources": {}, "master": {}}
    master_images, master_anns = [], []
    gid = 0 # global image id offset

    for source in requested:
        if not source_present(args.data_dir, source):
            print(f"[skip] {source}: not found on disk")
            continue
        print(f"[conv] {source} …")
        try:
            images, anns, stats = ADAPTERS[source](b)
        except NotImplementedError as e:
            print(f"[defer] {source}: {e}")
            continue
        write_source_coco(b, source, images, anns, out_dir)
        # merge into master with globally-unique image ids
        local_to_global = {}
        for im in images:
            gid += 1
            local_to_global[im["_key"]] = gid
            rec = {k: v for k, v in im.items() if k != "_key"}
            rec["id"] = gid
            master_images.append(rec)
        for a in anns:
            master_anns.append({
                "image_id": local_to_global[a["_img"]],
                "category_id": b.cat_id[a["category"]],
                "bbox": a["bbox"], "area": round(a["bbox"][2] * a["bbox"][3], 2),
                "iscrowd": 0, "source": a["source"], "orig_label": a["orig_label"],
            })
        summary["sources"][source] = {
            "images": len(images), "annotations": len(anns), **stats}
        print(f" images={len(images)} anns={len(anns)} "
              f"dropped={stats.get('dropped_boxes', 0)} "
              f"per_class={stats.get('per_class', {})}")

    # finalize master ann ids
    for j, a in enumerate(master_anns, start=1):
        a["id"] = j
    cats = b.categories
    if args.drop_non_target:
        cats = [c for c in cats if c["name"] not in b.non_target]
    master = {"info": {"description": "counter-USV EO COCO master "},
              "images": master_images, "annotations": master_anns,
              "categories": cats}
    (out_dir / "coco_master.json").write_text(json.dumps(master))

    # per-class totals across master
    id2name = {c["id"]: c["name"] for c in cats}
    class_totals = {}
    for a in master_anns:
        n = id2name.get(a["category_id"])
        class_totals[n] = class_totals.get(n, 0) + 1
    summary["master"] = {"images": len(master_images),
                         "annotations": len(master_anns),
                         "per_class": class_totals}
    (out_dir / "convert_summary.json").write_text(json.dumps(summary, indent=2))

    # integrity check
    _verify(master)
    print(f"\nMASTER: images={len(master_images)} anns={len(master_anns)}")
    print("per canonical class:")
    for name in [c["name"] for c in cats]:
        print(f" {name:20} {class_totals.get(name, 0)}")
    print(f"\nWrote {out_dir}/coco_master.json (+ per-source + convert_summary.json)")
    return 0


def _verify(coco: dict) -> None:
    img_ids = {im["id"] for im in coco["images"]}
    cat_ids = {c["id"] for c in coco["categories"]}
    assert len(img_ids) == len(coco["images"]), "duplicate image ids"
    ann_ids = set()
    for a in coco["annotations"]:
        assert a["image_id"] in img_ids, f"ann {a['id']} → missing image"
        assert a["category_id"] in cat_ids, f"ann {a['id']} → missing category"
        assert a["id"] not in ann_ids, "duplicate ann id"
        ann_ids.add(a["id"])


if __name__ == "__main__":
    sys.exit(main())
