#!/usr/bin/env python3
"""Freeze the EO attack library (config paths + measured results + dual-use).

Records attack config *paths* (git is source of truth) and pins SHA-256 digests
of headline result artifacts, writes a sample gallery of *illustrative*
composites (not printable templates), and records the dual-use redaction of
high-fidelity patch tensors (``docs/DUAL_USE.md``).

Scope is **EO-only** — the adversary motion model freezes with the defense.

Usage
-----
    python scripts/attacks/freeze_attacks.py
    python scripts/attacks/freeze_attacks.py --skip-gallery
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_ROOT = REPO_ROOT / "results" / "attacks"
DEFAULT_ARTIFACT = DEFAULT_ROOT / "artifact_v1"

# Configs recorded by path at freeze (git owns the attack contract).
CONFIG_PATHS = [
    "configs/attacks/marine_eot.yaml",
    "configs/attacks/patch.yaml",
    "configs/attacks/evasion.yaml",
    "configs/attacks/disguise.yaml",
    "configs/attacks/access_levels.yaml",
    "configs/attacks/oracle.yaml",
]

# Headline result artifacts digested (reports / rates — not PNG galleries).
RESULT_PATHS = [
    "results/attacks/evasion/yolo11s/esr_by_severity.json",
    "results/attacks/evasion/yolo11s/report.md",
    "results/attacks/evasion/yolo11l/esr_by_severity.json",
    "results/attacks/evasion/yolo11l/report.md",
    "results/attacks/disguise/yolo11s/fishing/tmsr_by_severity.json",
    "results/attacks/disguise/yolo11s/recreational/tmsr_by_severity.json",
    "results/attacks/disguise/yolo11l/fishing/tmsr_by_severity.json",
    "results/attacks/disguise/yolo11l/recreational/tmsr_by_severity.json",
    "results/attacks/disguise/yolo11s/summary.md",
    "results/attacks/disguise/yolo11l/summary.md",
    "results/attacks/transfer/evasion/yolo11s_summary.md",
    "results/attacks/transfer/evasion/yolo11l_summary.md",
    "results/attacks/transfer/evasion/yolo11s_to_yolo11l/esr_by_severity.json",
    "results/attacks/transfer/evasion/yolo11s_to_rtdetr_l/esr_by_severity.json",
    "results/attacks/transfer/evasion/yolo11l_to_yolo11s/esr_by_severity.json",
    "results/attacks/transfer/evasion/yolo11l_to_rtdetr_l/esr_by_severity.json",
    "results/attacks/transfer/disguise/yolo11s_to_yolo11l/fishing/tmsr_by_severity.json",
    "results/attacks/transfer/disguise/yolo11s_to_rtdetr_l/fishing/tmsr_by_severity.json",
    "results/attacks/transfer/disguise/yolo11l_to_yolo11s/fishing/tmsr_by_severity.json",
    "results/attacks/transfer/disguise/yolo11l_to_rtdetr_l/fishing/tmsr_by_severity.json",
    "results/attacks/oracle/fishing/assertions.json",
    "results/attacks/oracle/recreational/assertions.json",
    "results/attacks/marine_eot/report.md",
    "results/attacks/marine_eot/grid_meta.json",
    "results/attacks/patch_core/report.md",
    "results/attacks/patch_core/smoke_meta.json",
]

# Patch banks hold high-fidelity printable templates — redacted from release.
REDACTED_GLOBS = [
    "results/attacks/**/patch_bank/patches/*.npy",
    "results/attacks/**/patch_bank/patches/*_placement.json",
    "results/attacks/patch_core/patch_optimized.png",
    "results/attacks/patch_core/patch_init.png",
]

# Illustrative composites for the public sample gallery (composited scenes,
# not isolated printable patch textures).
GALLERY_SOURCES = [
    ("marine_eot/severity_grid.png", "marine_eot_severity_grid.png"),
    ("marine_eot/joint_samples.png", "marine_eot_joint_samples.png"),
    ("patch_core/00_clean_annotated.png", "patch_core_clean.png"),
    ("patch_core/02_optimized_composite.png", "patch_core_optimized_composite.png"),
    ("evasion/yolo11s/gallery/25029_clean.png", "evasion_yolo11s_25029_clean.png"),
    ("evasion/yolo11s/gallery/25029_patched.png", "evasion_yolo11s_25029_patched.png"),
    ("evasion/yolo11l/gallery/25029_clean.png", "evasion_yolo11l_25029_clean.png"),
    ("evasion/yolo11l/gallery/25029_patched.png", "evasion_yolo11l_25029_patched.png"),
]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "nogit"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _digest_list(rel_paths: list[str], *, required: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rel in rel_paths:
        p = REPO_ROOT / rel
        if not p.is_file():
            if required:
                raise FileNotFoundError(f"freeze requires {rel}")
            out[rel] = {"missing": True}
            continue
        out[rel] = {
            "sha256": _sha256(p),
            "bytes": p.stat().st_size,
        }
    return out


def _path_list(rel_paths: list[str], *, required: bool = True) -> dict[str, Any]:
    """Record config paths without digests (git is source of truth)."""
    out: dict[str, Any] = {}
    for rel in rel_paths:
        p = REPO_ROOT / rel
        if not p.is_file():
            if required:
                raise FileNotFoundError(f"freeze requires {rel}")
            out[rel] = {"missing": True}
            continue
        out[rel] = {"path": rel}
    return out


def _l0_rate(json_rel: str, rate_key: str) -> dict[str, Any] | None:
    p = REPO_ROOT / json_rel
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    block = d.get(rate_key) or {}
    cell = block.get("scale:L0")
    if not isinstance(cell, dict):
        return None
    rate = cell.get(rate_key)
    n_ok = cell.get("n_success")
    n = cell.get("n_attackable", cell.get("n_eligible"))
    return {
        "rate": float(rate) if rate is not None else None,
        "n_success": n_ok,
        "n": n,
        "source": json_rel,
    }


def _collect_headlines() -> dict[str, Any]:
    return {
        "white_box_esr_L0": {
            "yolo11s": _l0_rate(
                "results/attacks/evasion/yolo11s/esr_by_severity.json", "esr"
            ),
            "yolo11l": _l0_rate(
                "results/attacks/evasion/yolo11l/esr_by_severity.json", "esr"
            ),
        },
        "white_box_tmsr_L0": {
            "yolo11s_fishing": _l0_rate(
                "results/attacks/disguise/yolo11s/fishing/tmsr_by_severity.json",
                "tmsr",
            ),
            "yolo11s_recreational": _l0_rate(
                "results/attacks/disguise/yolo11s/recreational/tmsr_by_severity.json",
                "tmsr",
            ),
            "yolo11l_fishing": _l0_rate(
                "results/attacks/disguise/yolo11l/fishing/tmsr_by_severity.json",
                "tmsr",
            ),
            "yolo11l_recreational": _l0_rate(
                "results/attacks/disguise/yolo11l/recreational/tmsr_by_severity.json",
                "tmsr",
            ),
        },
        "transfer_esr_L0": {
            "yolo11s_to_yolo11l_grey": _l0_rate(
                "results/attacks/transfer/evasion/yolo11s_to_yolo11l/"
                "esr_by_severity.json",
                "esr",
            ),
            "yolo11s_to_rtdetr_l_black": _l0_rate(
                "results/attacks/transfer/evasion/yolo11s_to_rtdetr_l/"
                "esr_by_severity.json",
                "esr",
            ),
            "yolo11l_to_yolo11s_grey": _l0_rate(
                "results/attacks/transfer/evasion/yolo11l_to_yolo11s/"
                "esr_by_severity.json",
                "esr",
            ),
            "yolo11l_to_rtdetr_l_black": _l0_rate(
                "results/attacks/transfer/evasion/yolo11l_to_rtdetr_l/"
                "esr_by_severity.json",
                "esr",
            ),
        },
        "oracle": {
            "fishing_contacts": _oracle_n("fishing"),
            "recreational_contacts": _oracle_n("recreational"),
            "condition": "perfect_disguise_oracle",
        },
    }


def _oracle_n(benign: str) -> int | None:
    p = DEFAULT_ROOT / "oracle" / benign / "assertions.json"
    if not p.is_file():
        return None
    return int(json.loads(p.read_text()).get("n_contacts") or 0)


def _count_redacted() -> dict[str, Any]:
    npy = list(DEFAULT_ROOT.glob("**/patch_bank/patches/*.npy"))
    place = list(DEFAULT_ROOT.glob("**/patch_bank/patches/*_placement.json"))
    banks = sorted({p.parents[1] for p in npy})  # .../patch_bank
    return {
        "policy": "docs/DUAL_USE.md",
        "rule": (
            "Highest-fidelity physically optimized patch templates "
            "(raw patch tensors + isolated patch PNGs) are described, "
            "not distributed. Local patch_bank/ remains for transfer "
            "reproducibility; it is excluded from public release."
        ),
        "redacted_globs": REDACTED_GLOBS,
        "n_patch_tensors_local": len(npy),
        "n_placement_json_local": len(place),
        "n_patch_banks_local": len(banks),
        "patch_banks_local": [_rel(b) for b in banks],
        "released_instead": [
            "configs/attacks/*.yaml (paths recorded; digests in git)",
            "attack library source under src/counterusv/attacks/",
            "headline ESR/TMSR/transfer/oracle JSON + report.md",
            "illustrative composited gallery under artifact_v1/gallery/",
            "marine-EOT severity grid / joint samples",
        ],
    }


def build_gallery(artifact_dir: Path) -> list[dict[str, str]]:
    gal = artifact_dir / "gallery"
    gal.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for src_rel, dest_name in GALLERY_SOURCES:
        src = DEFAULT_ROOT / src_rel
        if not src.is_file():
            print(f"[freeze] gallery skip (missing): {src_rel}")
            continue
        dest = gal / dest_name
        shutil.copy2(src, dest)
        rows.append(
            {
                "source": f"results/attacks/{src_rel}",
                "artifact": _rel(dest),
                "sha256": _sha256(dest),
            }
        )
    (gal / "MANIFEST.json").write_text(
        json.dumps({"n": len(rows), "items": rows}, indent=2) + "\n"
    )
    return rows


def write_redaction(path: Path, redaction: dict[str, Any]) -> None:
    lines = [
        "# Dual-use redaction record — EO attack artifact v1",
        "",
        f"Frozen with `{path.parent.name}/` · see `docs/DUAL_USE.md`.",
        "",
        "## Policy",
        "",
        redaction["rule"],
        "",
        "## Redacted from public release",
        "",
        "Globs (local copies may exist for transfer reproducibility):",
        "",
    ]
    for g in redaction["redacted_globs"]:
        lines.append(f"- `{g}`")
    lines += [
        "",
        f"- Local patch banks: **{redaction['n_patch_banks_local']}** "
        f"({redaction['n_patch_tensors_local']} `.npy` tensors).",
        "",
        "## What *is* released",
        "",
    ]
    for item in redaction["released_instead"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Note",
        "",
        "Illustrative gallery images are **composited letterboxed scenes** "
        "(clean vs attacked) for qualitative inspection. They are not "
        "standalone printable patch templates at physical resolution.",
        "",
    ]
    path.write_text("\n".join(lines))


def _fmt_rate(cell: dict[str, Any] | None) -> str:
    if not cell or cell.get("rate") is None:
        return "—"
    r = float(cell["rate"])
    n_ok, n = cell.get("n_success"), cell.get("n")
    if n_ok is not None and n is not None:
        return f"**{100 * r:.0f}%** ({n_ok}/{n})"
    return f"**{100 * r:.0f}%**"


def write_release_notes(path: Path, payload: dict[str, Any]) -> None:
    h = payload["headlines"]
    lines = [
        "# EO attack library — artifact v1 release notes",
        "",
        f"Freeze version: `{payload['version']}`  ·  "
        f"frozen {payload['frozen_utc']}  ·  git `{payload['git_sha']}`",
        "",
        "Machine pin + SHA-256 digests: **`FROZEN.json`**. "
        "Dual-use redaction: **`REDACTION.md`**. "
        "Sample gallery: **`artifact_v1/gallery/`**.",
        "",
        "## Scope",
        "",
        "EO attack surface only: marine-EOT, physically-realizable patch core, "
        "evasion (ESR), disguise (TMSR), access-level transfer "
        "(white / grey / black), and the perfect-disguise oracle. "
        "The adversary motion model is **not** in this freeze "
        "(ships with the defense).",
        "",
        "## Library surface",
        "",
        "| Module | Role |",
        "|---|---|",
        "| `counterusv.attacks.marine_eot` | Marine transform expectation + severity ladder |",
        "| `counterusv.attacks.patch` | Physically-realizable patch core (TV + NPS) |",
        "| `counterusv.attacks.evasion` | Hostile non-detection (ESR) |",
        "| `counterusv.attacks.disguise` | Hostile→benign flip (TMSR) |",
        "| `counterusv.attacks.transfer` | Access-level hard-eval on saved patch banks |",
        "| `counterusv.attacks.oracle` | No-patch benign class assertion |",
        "",
        "## Frozen configs",
        "",
    ]
    for rel, meta in payload["configs"].items():
        if meta.get("missing"):
            lines.append(f"- `{rel}` — MISSING")
        else:
            lines.append(f"- `{rel}`")
    lines += [
        "",
        "## Headline results (L0 / identity marine-EOT)",
        "",
        "### White-box ESR",
        "",
        f"- yolo11s: {_fmt_rate(h['white_box_esr_L0'].get('yolo11s'))}",
        f"- yolo11l: {_fmt_rate(h['white_box_esr_L0'].get('yolo11l'))}",
        "",
        "### White-box TMSR (both benign targets)",
        "",
        f"- yolo11s → fishing: {_fmt_rate(h['white_box_tmsr_L0'].get('yolo11s_fishing'))}",
        f"- yolo11s → recreational: {_fmt_rate(h['white_box_tmsr_L0'].get('yolo11s_recreational'))}",
        f"- yolo11l → fishing: {_fmt_rate(h['white_box_tmsr_L0'].get('yolo11l_fishing'))}",
        f"- yolo11l → recreational: {_fmt_rate(h['white_box_tmsr_L0'].get('yolo11l_recreational'))}",
        "",
        "### Access-level ESR transfer",
        "",
        f"- grey yolo11s→yolo11l: {_fmt_rate(h['transfer_esr_L0'].get('yolo11s_to_yolo11l_grey'))}",
        f"- grey yolo11l→yolo11s: {_fmt_rate(h['transfer_esr_L0'].get('yolo11l_to_yolo11s_grey'))}",
        f"- black yolo11s→rtdetr_l: {_fmt_rate(h['transfer_esr_L0'].get('yolo11s_to_rtdetr_l_black'))}",
        f"- black yolo11l→rtdetr_l: {_fmt_rate(h['transfer_esr_L0'].get('yolo11l_to_rtdetr_l_black'))}",
        "",
        "TMSR transfer is **0%** at white / grey / black for both surrogates × "
        "`{fishing, recreational}` (see transfer reports under "
        "`results/attacks/transfer/disguise/`).",
        "",
        "### Perfect-disguise oracle",
        "",
        f"- Condition: `{h['oracle']['condition']}`",
        f"- Contacts asserted: fishing **{h['oracle']['fishing_contacts']}**, "
        f"recreational **{h['oracle']['recreational_contacts']}** "
        "(test∩`usv`; no pixels modified).",
        "",
        "## Provisional RQ1 reading",
        "",
        "Under this physical patch recipe: **evasion ≫ disguise** on YOLO "
        "white/grey; black-box evasion to RT-DETR is near-infeasible "
        "(single-digit ESR); disguise fails at every access level "
        "(optimization caveat remains — equal loss weights / soft–hard gap).",
        "",
        "## Dual-use",
        "",
        "See `REDACTION.md` and `docs/DUAL_USE.md`. Raw patch tensors are "
        "**not** part of the public artifact; the gallery ships composited "
        "illustrations only.",
        "",
        "## Reproduce (local)",
        "",
        "```bash",
        "python scripts/attacks/run_evasion.py --family yolo11s --device 0",
        "python scripts/attacks/run_disguise.py --family yolo11s --device 0",
        "python scripts/attacks/run_transfer.py --attack evasion --surrogate yolo11s --device 0",
        "python scripts/attacks/run_oracle.py --all-benigns",
        "python scripts/attacks/freeze_attacks.py",
        "```",
        "",
        "Verify digests against `FROZEN.json` after any result change.",
        "",
    ]
    path.write_text("\n".join(lines))


def smoke_check(payload: dict[str, Any]) -> None:
    """Reload configs through the library; digests are result-only."""
    from counterusv.attacks.disguise import load_disguise_config
    from counterusv.attacks.evasion import load_evasion_config
    from counterusv.attacks.marine_eot import load_config as load_eot
    from counterusv.attacks.oracle import load_oracle_config
    from counterusv.attacks.patch import load_patch_config
    from counterusv.attacks.transfer import load_access_levels_config

    print("[freeze] smoke: reload attack configs …")
    load_eot()
    load_patch_config()
    load_evasion_config()
    load_disguise_config()
    load_access_levels_config()
    load_oracle_config()

    for rel, meta in payload["configs"].items():
        if meta.get("missing"):
            raise AssertionError(f"missing config: {rel}")
        if not (REPO_ROOT / rel).is_file():
            raise AssertionError(f"config path missing on disk: {rel}")
    print("[freeze] smoke OK (configs load)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="results/attacks root",
    )
    ap.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT,
        help="artifact_v1 output directory",
    )
    ap.add_argument("--skip-gallery", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    configs = _path_list(CONFIG_PATHS, required=True)
    # Results: require core white-box + oracle; transfer optional with warn.
    required_results = [p for p in RESULT_PATHS if "transfer/" not in p]
    optional_results = [p for p in RESULT_PATHS if "transfer/" in p]
    results = _digest_list(required_results, required=True)
    results.update(_digest_list(optional_results, required=False))
    missing_opt = [r for r, m in results.items() if m.get("missing")]
    if missing_opt:
        print(f"[freeze] warning: {len(missing_opt)} optional result(s) missing")

    redaction = _count_redacted()
    headlines = _collect_headlines()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    gallery_rows: list[dict[str, str]] = []
    if not args.skip_gallery:
        gallery_rows = build_gallery(args.artifact_dir)
        print(f"[freeze] gallery: {len(gallery_rows)} images → {args.artifact_dir / 'gallery'}")

    version = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    payload: dict[str, Any] = {
        "version": version,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "scope": "eo_attack_library_v1",
        "excludes": [
            "adversary_motion_model",
            "defense_wiring",
            "raw_patch_tensors_public_release",
        ],
        "library": {
            "package": "counterusv.attacks",
            "modules": [
                "marine_eot",
                "patch",
                "evasion",
                "disguise",
                "transfer",
                "oracle",
            ],
            "docs": [
                "docs/THREAT_MODEL.md",
                "docs/METRICS.md",
                "docs/TRANSFER_PROTOCOL.md",
                "docs/DUAL_USE.md",
            ],
        },
        "configs": configs,
        "results": {k: v for k, v in results.items() if not v.get("missing")},
        "results_missing": missing_opt,
        "headlines": headlines,
        "dual_use_redaction": redaction,
        "gallery": {
            "dir": _rel(args.artifact_dir / "gallery"),
            "n_images": len(gallery_rows),
            "items": gallery_rows,
        },
        "artifacts": {
            "release_notes": _rel(args.root / "RELEASE_NOTES.md"),
            "redaction": _rel(args.root / "REDACTION.md"),
            "frozen_json": _rel(args.root / "FROZEN.json"),
            "artifact_dir": _rel(args.artifact_dir),
        },
    }

    out_json = args.root / "FROZEN.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    write_release_notes(args.root / "RELEASE_NOTES.md", payload)
    write_redaction(args.root / "REDACTION.md", redaction)
    print(f"[freeze] wrote {out_json}")
    print(f"[freeze] wrote {args.root / 'RELEASE_NOTES.md'}")
    print(f"[freeze] wrote {args.root / 'REDACTION.md'}")
    print(
        f"[freeze] redacted local patch tensors: "
        f"{redaction['n_patch_tensors_local']} across "
        f"{redaction['n_patch_banks_local']} banks"
    )

    if not args.skip_smoke:
        smoke_check(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
