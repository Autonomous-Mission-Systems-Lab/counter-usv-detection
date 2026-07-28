#!/usr/bin/env python3
"""Visual QA sheet for the harmonized EO loader.

Overlays remapped boxes on letterboxed images for one sample per source
(SeaShips after overlay mask; small-target ABOShips/SMD; McShips; USV) and
writes a contact sheet + per-sample PNGs under ``results/eo_loader_qa/``.

Usage
-----
    python scripts/data/qa_eo_loader.py
    python scripts/data/qa_eo_loader.py --input-size 640 --split train --per-source 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.data.eo_dataset import EODetectionDataset, default_config  # noqa: E402
from counterusv.data.overlay import SEASHIPS_BANDS  # noqa: E402

# Prefer a source order that exercises every QA concern.
SOURCE_ORDER = ["seaships", "smd", "aboships", "mcships", "usv"]
COLORS = [
    (255, 64, 64), (64, 180, 255), (80, 220, 120), (255, 200, 40),
    (200, 100, 255), (255, 140, 0), (0, 200, 200), (255, 100, 180),
]


def draw_sample(item: dict, title: str) -> Image.Image:
    img = Image.fromarray(item["image_uint8"])
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    boxes = item["boxes"]
    names = item["label_names"]
    for i, (box, name) in enumerate(zip(boxes, names)):
        x, y, w, h = box
        color = COLORS[i % len(COLORS)]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        label = f"{name}"
        draw.rectangle([x, max(0, y - 14), x + 7 * len(label), y], fill=color)
        draw.text((x + 2, max(0, y - 13)), label, fill=(0, 0, 0), font=font_sm)

    # SeaShips: show where the masked bands were (in letterboxed coords).
    if item["source"] == "seaships":
        meta = item["meta"]
        for y0, y1 in SEASHIPS_BANDS:
            yy0 = y0 * meta.scale + meta.pad_y
            yy1 = y1 * meta.scale + meta.pad_y
            draw.rectangle(
                [meta.pad_x, yy0, meta.pad_x + meta.new_w, yy1],
                outline=(255, 255, 0), width=1,
            )

    header = (f"{title}  src={item['source']}  id={item['image_id']}  "
              f"boxes={len(boxes)}  drop_n={item['n_dropped_native']}+"
              f"{item['n_dropped_letterbox']}")
    draw.rectangle([0, 0, img.width, 22], fill=(20, 20, 20))
    draw.text((4, 4), header[:110], fill=(240, 240, 240), font=font)
    return img


def contact_sheet(images: list[Image.Image], cols: int = 3, pad: int = 8) -> Image.Image:
    if not images:
        return Image.new("RGB", (64, 64), (0, 0, 0))
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad,
                              rows * h + (rows + 1) * pad), (30, 30, 30))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        sheet.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--input-size", type=int, default=640)
    ap.add_argument("--per-source", type=int, default=2,
                    help="samples to draw per source")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "eo_loader_qa")
    args = ap.parse_args()

    cfg = default_config(args.data_dir, args.split, input_size=args.input_size, augment=False)
    ds = EODetectionDataset(cfg)
    print(f"[qa] split={args.split} images={len(ds)} input={args.input_size}")

    # Pick first N images per source (stable, cheap).
    by_source: dict[str, list[int]] = {s: [] for s in SOURCE_ORDER}
    for i, im in enumerate(ds.images):
        src = im.get("source", "")
        if src in by_source and len(by_source[src]) < args.per_source:
            by_source[src].append(i)
        if all(len(v) >= args.per_source for v in by_source.values()):
            break

    args.out.mkdir(parents=True, exist_ok=True)
    drawn: list[Image.Image] = []
    manifest = []
    for src in SOURCE_ORDER:
        for j, idx in enumerate(by_source.get(src, [])):
            item = ds[idx]
            title = f"{src}[{j}]"
            viz = draw_sample(item, title)
            # Also dump a side-by-side native-vs-letterbox for SeaShips to show the mask.
            out_path = args.out / f"{args.split}_{src}_{j}_s{args.input_size}.png"
            viz.save(out_path)
            drawn.append(viz)
            manifest.append({
                "file": out_path.name,
                "source": src,
                "image_id": item["image_id"],
                "file_name": item["file_name"],
                "n_boxes": len(item["boxes"]),
                "label_names": item["label_names"],
                "n_dropped_native": item["n_dropped_native"],
                "n_dropped_letterbox": item["n_dropped_letterbox"],
                "input_size": args.input_size,
                "scale": item["meta"].scale,
                "pad_x": item["meta"].pad_x,
                "pad_y": item["meta"].pad_y,
            })
            print(f"  wrote {out_path.name}  boxes={len(item['boxes'])} "
                  f"drop={item['n_dropped_native']}+{item['n_dropped_letterbox']}")

    sheet = contact_sheet(drawn, cols=min(3, max(1, len(drawn))))
    sheet_path = args.out / f"sheet_{args.split}_s{args.input_size}.png"
    sheet.save(sheet_path)

    # Eligibility sweep on a capped sample for a quick retained/dropped report.
    stats = ds.eligibility_stats(max_images=min(len(ds), 2000))
    report = {
        "split": args.split,
        "input_size": args.input_size,
        "dataset_len": len(ds),
        "eligibility_sample": stats.as_dict(),
        "seaships_bands": {"top": list(SEASHIPS_BANDS[0]), "bottom": list(SEASHIPS_BANDS[1])},
        "samples": manifest,
        "sheet": sheet_path.name,
    }
    (args.out / f"qa_summary_{args.split}_s{args.input_size}.json").write_text(
        json.dumps(report, indent=2) + "\n")

    md = [
        f"# EO loader QA — `{args.split}` @ {args.input_size}",
        "",
        "Auto-generated by `scripts/data/qa_eo_loader.py`. Remapped boxes overlaid on",
        "letterboxed images (pad 114). Yellow outlines on SeaShips mark the masked",
        "timestamp / camera-ID bands (in letterboxed coords).",
        "",
        f"**Dataset size:** {len(ds)} images in split `{args.split}`.",
        "",
        f"**Eligibility (first {stats.images} images):** "
        f"kept {stats.boxes_kept}/{stats.boxes_total} boxes "
        f"({100*stats.keep_rate:.1f}%); "
        f"dropped native={stats.boxes_dropped_native}, "
        f"post-letterbox={stats.boxes_dropped_letterbox}; "
        f"images left with 0 boxes={stats.images_with_zero_kept}.",
        "",
        f"**SeaShips overlay bands (native 1920×1080):** "
        f"top `{SEASHIPS_BANDS[0]}`, bottom `{SEASHIPS_BANDS[1]}`.",
        "",
        f"![contact sheet]({sheet_path.name})",
        "",
        "## Per-sample",
        "",
        "| source | image_id | boxes | dropped n+lb | file |",
        "|---|---:|---:|---:|---|",
    ]
    for m in manifest:
        md.append(
            f"| {m['source']} | {m['image_id']} | {m['n_boxes']} | "
            f"{m['n_dropped_native']}+{m['n_dropped_letterbox']} | `{m['file']}` |"
        )
    report_path = args.out / f"report_{args.split}_s{args.input_size}.md"
    report_path.write_text("\n".join(md) + "\n")
    print(f"[qa] sheet → {sheet_path}")
    print(f"[qa] report → {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
