"""Shared helpers for regenerable paper figures.

Reads only digest-pinned artifacts under ``results/`` (verified against
``results/attacks/FROZEN.json`` and ``results/defense/FROZEN.json``). Fig. 1
is a hand-authored static SVG under ``figures/`` — not generated here.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
ATTACKS_FREEZE = REPO_ROOT / "results" / "attacks" / "FROZEN.json"
DEFENSE_FREEZE = REPO_ROOT / "results" / "defense" / "FROZEN.json"
PAPER_DIR = REPO_ROOT / "results" / "paper"
FIG1_SRC = REPO_ROOT / "figures" / "fig1_system.svg"

# The trailing size letter is ambiguous against the digits in a sans-serif
# legend, so spell it out wherever a detector family is drawn.
DETECTOR_LABELS = {"yolo11s": "yolo11s (small)", "yolo11l": "yolo11l (large)"}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_sha() -> str:
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


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def _collect_digest_entries(freeze: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (rel_path, expected_sha256) pairs from a freeze payload.

    YAML under ``configs/`` is path-only (git owns it). Digests cover results,
    envelopes, placements, and other non-git artifacts.
    """
    out: list[tuple[str, str]] = []

    def _maybe_add(path: str | None, digest: str | None) -> None:
        if not path or not digest:
            return
        if path.startswith("configs/"):
            return
        out.append((path, digest))

    # attacks freeze: results keyed by relative path (configs are path-only)
    results_block = freeze.get("results") or {}
    if isinstance(results_block, dict):
        for key, entry in results_block.items():
            if not isinstance(entry, dict) or entry.get("missing"):
                continue
            _maybe_add(entry.get("path") or key, entry.get("sha256"))

    # Top-level configs may still include non-git data pins (placements, …)
    for _label, entry in (freeze.get("configs") or {}).items():
        if isinstance(entry, dict):
            _maybe_add(entry.get("path"), entry.get("sha256"))

    eval_block = freeze.get("evaluation") or {}
    arts = eval_block.get("artifacts") or {}
    for _label, entry in arts.items():
        if isinstance(entry, dict):
            _maybe_add(entry.get("path"), entry.get("sha256"))

    # defense arm freezes (+ nested artifacts listed inside them)
    arms = freeze.get("arms") or {}
    for _arm, entry in arms.items():
        if isinstance(entry, dict) and entry.get("freeze_path") and entry.get("sha256"):
            out.append((entry["freeze_path"], entry["sha256"]))
            arm_path = REPO_ROOT / entry["freeze_path"]
            if arm_path.is_file():
                try:
                    nested = load_json(arm_path)
                except (OSError, json.JSONDecodeError):
                    nested = {}
                for _lab, nest in (nested.get("artifacts") or {}).items():
                    if isinstance(nest, dict):
                        _maybe_add(nest.get("path"), nest.get("sha256"))
                for _lab, nest in (nested.get("configs") or {}).items():
                    if isinstance(nest, dict):
                        _maybe_add(nest.get("path"), nest.get("sha256"))
                # geometry freeze may pin placement FAR under far_floor
                far_floor = nested.get("far_floor") or {}
                for key in ("summary", "report", "placement_summary", "placement_report"):
                    nest = far_floor.get(key)
                    if isinstance(nest, dict):
                        _maybe_add(nest.get("path"), nest.get("sha256"))
                for _name, nest in (nested.get("envelopes") or {}).items():
                    if isinstance(nest, dict):
                        _maybe_add(nest.get("path"), nest.get("sha256"))

    motion = eval_block.get("motion_freeze") or {}
    _maybe_add(motion.get("path"), motion.get("sha256"))

    return out


def verify_freezes(*, skip: bool = False) -> dict[str, Any]:
    """Recompute digests against both freezes; abort on mismatch unless skip."""
    if skip:
        return {
            "verified": False,
            "skipped": True,
            "attacks_freeze": rel(ATTACKS_FREEZE),
            "defense_freeze": rel(DEFENSE_FREEZE),
            "n_checked": 0,
            "mismatches": [],
        }

    mismatches: list[dict[str, str]] = []
    checked: list[dict[str, str]] = []
    seen: set[str] = set()

    for freeze_path in (ATTACKS_FREEZE, DEFENSE_FREEZE):
        if not freeze_path.is_file():
            raise FileNotFoundError(f"missing freeze: {rel(freeze_path)}")
        payload = load_json(freeze_path)
        for path_rel, expected in _collect_digest_entries(payload):
            if path_rel in seen:
                continue
            seen.add(path_rel)
            path = REPO_ROOT / path_rel
            if not path.is_file():
                mismatches.append(
                    {
                        "path": path_rel,
                        "expected": expected,
                        "got": "MISSING",
                    }
                )
                continue
            got = sha256(path)
            checked.append({"path": path_rel, "sha256": got})
            if got != expected:
                mismatches.append(
                    {
                        "path": path_rel,
                        "expected": expected,
                        "got": got,
                    }
                )

    if mismatches:
        lines = [
            f"  {m['path']}: expected {m['expected'][:12]}… got {m['got'][:12]}…"
            for m in mismatches
        ]
        raise SystemExit(
            "digest verification FAILED "
            f"({len(mismatches)} mismatch(es)):\n" + "\n".join(lines)
        )

    return {
        "verified": True,
        "skipped": False,
        "attacks_freeze": rel(ATTACKS_FREEZE),
        "defense_freeze": rel(DEFENSE_FREEZE),
        "n_checked": len(checked),
        "checked": checked,
        "mismatches": [],
    }


def apply_style() -> None:
    """Shared matplotlib style for paper figures (Agg backend)."""
    mpl.use("Agg")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#111111",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, letter: str, *, suffix: str = "", dx: float = -0.10) -> None:
    """Bold ``(a)`` marker above-left of an axes.

    Figures carry no titles — descriptive text lives in the LaTeX caption (see
    ``results/paper/CAPTIONS.md``). ``suffix`` is only for panel identifiers
    that name a data dimension (e.g. the severity axis in Fig. S1).
    """
    text = f"({letter}) {suffix}".rstrip()
    ax.text(
        dx,
        1.03,
        text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def legend_below(
    fig: plt.Figure,
    handles: list,
    labels: list[str],
    *,
    ncol: int,
    y: float = 0.0,
    handlelength: float = 1.6,
) -> None:
    """Figure-level legend under the axes, so it can never cover data.

    Bump ``handlelength`` for line-style legends — a short handle cannot show
    a dash pattern.
    """
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        fontsize=8,
        handlelength=handlelength,
        columnspacing=1.4,
    )


def save_fig(fig: plt.Figure, stem: str, *, out_dir: Path | None = None) -> dict[str, str]:
    """Write PNG (150 dpi) + PDF vector; return {png, pdf, sha256_png}."""
    dest = out_dir or PAPER_DIR
    dest.mkdir(parents=True, exist_ok=True)
    png = dest / f"{stem}.png"
    pdf = dest / f"{stem}.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "png": rel(png),
        "pdf": rel(pdf),
        "sha256_png": sha256(png),
        "sha256_pdf": sha256(pdf),
        "bytes_png": png.stat().st_size,
        "bytes_pdf": pdf.stat().st_size,
    }


def copy_fig1(*, out_dir: Path | None = None) -> dict[str, Any]:
    """Assert Fig. 1 exists, digest it, copy into results/paper/."""
    if not FIG1_SRC.is_file():
        raise SystemExit(
            f"static Fig. 1 missing: {rel(FIG1_SRC)} "
            "(hand-authored under figures/; not generated by this pipeline)"
        )
    dest = out_dir or PAPER_DIR
    dest.mkdir(parents=True, exist_ok=True)
    out_svg = dest / "fig1_system.svg"
    shutil.copy2(FIG1_SRC, out_svg)
    digest = sha256(FIG1_SRC)
    result: dict[str, Any] = {
        "source": rel(FIG1_SRC),
        "svg": rel(out_svg),
        "sha256": digest,
        "bytes": FIG1_SRC.stat().st_size,
        "static": True,
    }

    # Optional PNG/PDF if cairosvg is importable.
    try:
        import cairosvg  # type: ignore

        png = dest / "fig1_system.png"
        pdf = dest / "fig1_system.pdf"
        cairosvg.svg2png(url=str(FIG1_SRC), write_to=str(png), dpi=150)
        cairosvg.svg2pdf(url=str(FIG1_SRC), write_to=str(pdf))
        result["png"] = rel(png)
        result["pdf"] = rel(pdf)
        result["sha256_png"] = sha256(png)
        result["sha256_pdf"] = sha256(pdf)
        result["cairosvg"] = True
    except Exception as exc:  # noqa: BLE001 — optional dep
        result["cairosvg"] = False
        result["cairosvg_note"] = (
            f"PNG/PDF export skipped ({type(exc).__name__}: {exc}). "
            "SVG is the source of truth; see figures/README.md."
        )
    return result


def write_provenance(
    figures: dict[str, Any],
    verification: dict[str, Any],
    *,
    out_dir: Path | None = None,
) -> Path:
    dest = out_dir or PAPER_DIR
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "verification": {
            k: verification[k]
            for k in (
                "verified",
                "skipped",
                "attacks_freeze",
                "defense_freeze",
                "n_checked",
                "mismatches",
            )
            if k in verification
        },
        "static_assets": {
            "fig1_system.svg": figures.get("fig1_system"),
        },
        "figures": figures,
    }
    # Keep checked digests if verified (useful for audit).
    if verification.get("verified") and verification.get("checked"):
        payload["verification"]["input_digests"] = {
            row["path"]: row["sha256"] for row in verification["checked"]
        }
    out = dest / "PROVENANCE.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out
