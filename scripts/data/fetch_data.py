#!/usr/bin/env python3
"""Fetch the counter-USV datasets from their ORIGINAL providers into ``data/raw/``.

This script automates the sources that expose stable direct-download URLs and
prints exact, copy-pasteable instructions (with the target path) for the ones
that are access-gated (Baidu, Google Drive, or account/ToS walls).

It downloads from each provider directly and does NOT re-host anything, which is
consistent with the repository's redistribution policy (see
``docs/DATA_LICENSES.md``): we ship annotations / derived features + code, and
every user obtains the underlying data from its original source.

Layout produced::

    data/raw/
      seaships/ # SeaShips imagery + VOC annotations
      mcships/ # McShips 9k "lite" (no stated license; citation requested; no re-host)
      smd/ # Singapore Maritime Dataset (video) + SMD-Plus labels
      aboships/ # ABOShips (Zenodo rec. 4736931)
      ais/marinecadastre/ # US AIS daily zips (NOAA OCM)
      CHECKSUMS.sha256 # appended as files land, for the data card

Examples
--------
List every source and whether it is automated::

    python scripts/data/fetch_data.py --list

Fetch the automatable EO sources::

    python scripts/data/fetch_data.py --source seaships aboships

Fetch a week of US AIS (large — hundreds of MB/day)::

    python scripts/data/fetch_data.py --source marinecadastre_ais \
        --ais-start 2023-06-01 --ais-end 2023-06-07

Print the manual steps for a gated source::

    python scripts/data/fetch_data.py --source mcships # prints instructions, no download

Notes
-----
* Core downloading uses only the Python standard library. ``tqdm`` (already in
  ``requirements.txt``) is used for a progress bar if present. ``gdown`` is used
  for Google-Drive sources if installed (``pip install gdown``); otherwise the
  manual link is printed.
* Nothing here re-distributes data. It fetches from providers for your own local
  use. Record exact counts / Class-B share in ``data/INVENTORY.md`` after pulling
  (the source inventory open action items).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

# Repo-root/data by default (this file lives in <repo>/scripts/).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

USER_AGENT = "counterUSV-fetch/1.0 (+research; contact repo maintainer)"
CHUNK = 1 << 20 # 1 MiB

# Set to an unverified ssl.SSLContext by --insecure (last resort for providers
# with broken certs). None = verify.
SSL_CONTEXT: ssl.SSLContext | None = None


class NotABinaryFile(Exception):
    """Raised when a 'binary' source hands back an HTML page (stale/dead link)."""


def _urlopen(req: urllib.request.Request, timeout: int = 60):
    return urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) # noqa: S310

try: # optional progress bar
    from tqdm import tqdm # type: ignore
except Exception: # pragma: no cover - tqdm is optional
    tqdm = None


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
# Each source: kind, target subdir (under data/raw), license reminder, and either
# direct URL(s) or manual instructions. `auto` marks sources this script can pull
# without human interaction.
SOURCES: dict[str, dict] = {
    # ---- EO detection sources -------------------------------------------
    "seaships": {
        "category": "eo",
        "kind": "http",
        "auto": True,
        "subdir": "seaships",
        "license": "Academic (original); CC BY 4.0 (Roboflow mirror). Annotations "
        "over original imagery only — do not re-host.",
        # 7,000-image public subset from the authors' page (WHU LIESMARS).
        "urls": [
            "http://www.lmars.whu.edu.cn/prof_web/shaozhenfeng/datasets/SeaShips(7000).zip"
        ],
        "fallback": "The WHU direct link is frequently down / serves an HTML page. "
        "Use a mirror: Kaggle 'SeaShips' dataset (needs `kaggle` API + token) or the "
        "Roboflow SeaShips-7000 export (CC BY 4.0, needs API key). Unzip into "
        "data/raw/seaships/. The full 31,455-image set is Baidu-only (manual).",
    },
    "aboships": {
        "category": "eo",
        "kind": "zenodo",
        "auto": True,
        "subdir": "aboships",
        "license": "CC BY 4.0 (Zenodo rec. 4736931) — re-host permitted with "
        "attribution; we default to annotations anyway.",
        "record_id": "4736931",
        "note": "Single file ABOshipsDataset.zip is ~8.2 GB — ensure disk/time.",
    },
    "mcships": {
        "category": "eo",
        "kind": "gdrive",
        "auto": False, # Baidu/GDrive gated; no stated license
        "subdir": "mcships",
        "license": "No stated license from the authors' distribution "
        "(Zheng & Zhang, ICME 2020); citation requested. Do not re-host imagery. "
        "McShips-derived annotations are omitted from the permissive release slice "
        "(see docs/DATA_LICENSES.md).",
        "gdrive_id": "1udewXbHCS9WKM-MPpWqouUUGs6Vx5iWf", # 9k "lite" subset
        "manual": [
            "McShips 9k 'lite' (Pascal-VOC). Two provider options:",
            " - Google Drive: https://drive.google.com/file/d/1udewXbHCS9WKM-MPpWqouUUGs6Vx5iWf/view",
            " (auto-download works if `pip install gdown`; then re-run this source)",
            " - Baidu Yun: https://pan.baidu.com/s/1rDeiCPX4EdRUvBl5jnWqDQ password: dqwu",
            "Repo/citation: https://github.com/ZhengYitong2333/Mcships",
            "License: none stated by the authors (citation of Zheng & Zhang, ICME 2020).",
            "Train-only in our splits; do not redistribute imagery. Derived "
            "annotations are omitted from the permissive release (see docs/DATA_LICENSES.md).",
        ],
    },
    "smd": {
        "category": "eo",
        "kind": "manual",
        "auto": False, # original videos on Google Drive; SMD-Plus labels on GitHub
        "subdir": "smd",
        "license": "Academic/research (Prasad et al.). Annotations over original "
        "imagery only.",
        "manual": [
            "Singapore Maritime Dataset (SMD) — on-shore subset is what we use.",
            " 1) Original videos + GT (Prasad et al.), Google Drive:",
            " https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset",
            " (each video is a separate Drive file; `gdown` can pull by id/folder)",
            " 2) SMD-Plus cleaned labels (7 classes; what taxonomy.yaml maps):",
            " https://github.com/kjunhwa/SMD-Plus (git clone into smd/SMD-Plus)",
            "Place videos under data/raw/smd/videos and SMD-Plus GT under "
            "data/raw/smd/SMD-Plus. On-shore subset only is required "
            "(detection frames for the EO master / eval slice).",
        ],
    },
    # ---- Trajectory (AIS) sources ---------------------------------------
    "marinecadastre_ais": {
        "category": "ais",
        "kind": "ais_daily",
        "auto": True, # needs --ais-start/--ais-end
        "subdir": "ais/marinecadastre",
        "license": "US public domain / CC0 (NOAA OCM). Underlying USCG NAIS "
        "'Level C'. Raw AIS: NO retransmit / NO fee — release derived features only.",
        # Daily national zips: AIS_YYYY_MM_DD.zip
        "url_template": "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{y}/AIS_{y}_{m:02d}_{d:02d}.zip",
        "note": "Daily national files are large (~hundreds of MB/day). Pick a short "
        "window; region-filter during cleaning if needed.",
    },
}

AUTO_ORDER = ["seaships", "aboships", "marinecadastre_ais"]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _human(n: int | None) -> str:
    if n is None:
        return "?"
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}TB"


def download_file(url: str, dest: Path, force: bool = False,
                  expect_binary: bool = True) -> Path:
    """Stream ``url`` to ``dest`` (atomic via a .part file). Returns ``dest``.

    Raises ``NotABinaryFile`` if a source expected to be a binary archive returns
    an HTML page (a dead/redirected link that would otherwise silently save junk).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f" [skip] {dest.name} already exists ({_human(dest.stat().st_size)}). "
              f"Use --force to re-download.")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f" [get ] {url}")
    with _urlopen(req) as resp:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if expect_binary and ctype.startswith("text/html"):
            raise NotABinaryFile(
                f"got HTML (Content-Type: {ctype}) — the direct link is likely "
                f"stale/redirected, not the archive.")
        total = resp.length or int(resp.headers.get("Content-Length") or 0) or None
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=f" {dest.name}") if tqdm else None
        got = 0
        with open(part, "wb") as fh:
            while True:
                buf = resp.read(CHUNK)
                if not buf:
                    break
                fh.write(buf)
                got += len(buf)
                if bar:
                    bar.update(len(buf))
                elif total:
                    pct = 100 * got / total
                    print(f"\r {dest.name}: {pct:5.1f}% ({_human(got)}/{_human(total)})",
                          end="", flush=True)
        if bar:
            bar.close()
        elif total:
            print()
    part.replace(dest)
    print(f" [ok ] {dest.name} ({_human(dest.stat().st_size)})")
    return dest


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(CHUNK), b""):
            h.update(buf)
    return h.hexdigest()


def record_checksum(data_dir: Path, path: Path) -> None:
    """Append a sha256 line to data/raw/CHECKSUMS.sha256 (for the data card)."""
    checks = data_dir / "raw" / "CHECKSUMS.sha256"
    checks.parent.mkdir(parents=True, exist_ok=True)
    rel = path.relative_to(data_dir)
    digest = sha256sum(path)
    with open(checks, "a") as fh:
        fh.write(f"{digest} {rel}\n")
    print(f" [sha ] {digest[:16]}… -> {checks.name}")


def print_license(name: str) -> None:
    print(f" [lic ] {SOURCES[name]['license']}")


# ---------------------------------------------------------------------------
# Per-kind fetchers
# ---------------------------------------------------------------------------
def fetch_http(name: str, spec: dict, data_dir: Path, force: bool) -> None:
    out = data_dir / "raw" / spec["subdir"]
    for url in spec["urls"]:
        # Encode characters like parentheses in the SeaShips filename.
        safe_url = quote(url, safe=":/?#[]@!$&'()*+,;=%")
        fname = Path(url.split("?")[0]).name
        try:
            dest = download_file(safe_url, out / fname, force=force)
            record_checksum(data_dir, dest)
        except (urllib.error.URLError, urllib.error.HTTPError, NotABinaryFile) as e:
            print(f" [FAIL] {url}\n {e}")
            if spec.get("fallback"):
                print(f" [note] {spec['fallback']}")


def fetch_zenodo(name: str, spec: dict, data_dir: Path, force: bool) -> None:
    out = data_dir / "raw" / spec["subdir"]
    api = f"https://zenodo.org/api/records/{spec['record_id']}"
    req = urllib.request.Request(api, headers={"User-Agent": USER_AGENT})
    try:
        with _urlopen(req) as resp:
            meta = json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f" [FAIL] Zenodo API {api}\n {e}")
        return
    files = meta.get("files", [])
    print(f" [info] Zenodo record {spec['record_id']}: {len(files)} file(s)")
    for f in files:
        url = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        fname = f.get("key") or Path(url).name
        size = f.get("size")
        print(f" - {fname} ({_human(size)})")
        if not url:
            continue
        try:
            dest = download_file(url, out / fname, force=force)
            # Prefer Zenodo's own checksum when present (md5:...).
            record_checksum(data_dir, dest)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f" [FAIL] {fname}\n {e}")


def _date_range(start: str, end: str):
    y1, m1, d1 = map(int, start.split("-"))
    y2, m2, d2 = map(int, end.split("-"))
    cur, last = date(y1, m1, d1), date(y2, m2, d2)
    if cur > last:
        raise ValueError("--ais-start must be <= --ais-end")
    while cur <= last:
        yield cur
        cur += timedelta(days=1)


def fetch_ais_daily(name: str, spec: dict, data_dir: Path, force: bool,
                    start: str | None, end: str | None) -> None:
    if not (start and end):
        print(f" [need] {name} requires --ais-start YYYY-MM-DD --ais-end YYYY-MM-DD")
        print(f" [note] {spec.get('note', '')}")
        return
    out = data_dir / "raw" / spec["subdir"]
    days = list(_date_range(start, end))
    print(f" [info] {name}: {len(days)} day(s) {start}..{end}")
    print(f" [note] {spec.get('note', '')}")
    for dt in days:
        url = spec["url_template"].format(y=dt.year, m=dt.month, d=dt.day)
        fname = Path(url).name
        try:
            dest = download_file(url, out / fname, force=force)
            record_checksum(data_dir, dest)
        except urllib.error.HTTPError as e:
            print(f" [miss] {fname}: HTTP {e.code} (file may not exist for that date)")
        except urllib.error.URLError as e:
            print(f" [FAIL] {fname}: {e}")


def fetch_gdrive(name: str, spec: dict, data_dir: Path, force: bool) -> None:
    out = data_dir / "raw" / spec["subdir"]
    out.mkdir(parents=True, exist_ok=True)
    gid = spec.get("gdrive_id")
    try:
        import gdown # type: ignore
    except Exception:
        print(" [need] Google-Drive source. Install the optional helper: "
              "`pip install gdown`, then re-run — or follow the manual steps:")
        print_manual(name, spec)
        return
    dest = out / f"{name}.zip"
    if dest.exists() and not force:
        print(f" [skip] {dest.name} exists. Use --force to re-download.")
        return
    print(f" [get ] gdrive id {gid} -> {dest}")
    gdown.download(id=gid, output=str(dest), quiet=False)
    if dest.exists():
        record_checksum(data_dir, dest)


def print_manual(name: str, spec: dict) -> None:
    out = DEFAULT_DATA_DIR / "raw" / spec["subdir"]
    print(f" [manual] target dir: {out}")
    for line in spec.get("manual", ["(no instructions recorded)"]):
        print(f" {line}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_list() -> None:
    print(f"{'source':22} {'cat':4} {'auto':5} license / access")
    print("-" * 100)
    for name, spec in SOURCES.items():
        auto = "yes" if spec.get("auto") else "MAN"
        print(f"{name:22} {spec['category']:4} {auto:5} {spec['license']}")
    print("\nAutomated (no interaction): " + ", ".join(AUTO_ORDER))
    print("Manual (gated): " + ", ".join(n for n, s in SOURCES.items() if not s.get("auto")))
    print("\nAIS sources need --ais-start / --ais-end (YYYY-MM-DD).")


def fetch_one(name: str, data_dir: Path, force: bool,
              ais_start: str | None, ais_end: str | None) -> None:
    spec = SOURCES[name]
    print(f"\n=== {name} [{spec['category']}] ===")
    print_license(name)
    if spec.get("note") and spec["kind"] not in ("ais_daily",):
        print(f" [note] {spec['note']}")
    kind = spec["kind"]
    if kind == "http":
        fetch_http(name, spec, data_dir, force)
    elif kind == "zenodo":
        fetch_zenodo(name, spec, data_dir, force)
    elif kind == "ais_daily":
        fetch_ais_daily(name, spec, data_dir, force, ais_start, ais_end)
    elif kind == "gdrive":
        fetch_gdrive(name, spec, data_dir, force)
    elif kind == "manual":
        print_manual(name, spec)
    else:
        print(f" [FAIL] unknown kind: {kind}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Fetch counter-USV datasets from their original providers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--list", action="store_true", help="list sources and exit")
    p.add_argument("--source", nargs="+", metavar="NAME",
                   help=f"source(s) to fetch (choices: {', '.join(SOURCES)})")
    p.add_argument("--all-auto", action="store_true",
                   help="fetch all non-gated sources (AIS skipped unless dates given)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help=f"root data dir (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--ais-start", metavar="YYYY-MM-DD", help="AIS window start")
    p.add_argument("--ais-end", metavar="YYYY-MM-DD", help="AIS window end")
    p.add_argument("--force", action="store_true", help="re-download existing files")
    p.add_argument("--insecure", action="store_true",
                   help="disable TLS verification (last resort for providers with "
                        "broken certs)")
    args = p.parse_args(argv)

    if args.insecure:
        global SSL_CONTEXT
        SSL_CONTEXT = ssl._create_unverified_context()
        print("[warn] TLS verification DISABLED (--insecure).")

    if args.list or (not args.source and not args.all_auto):
        cmd_list()
        return 0

    if args.all_auto:
        targets = list(AUTO_ORDER)
    else:
        targets = args.source
        unknown = [t for t in targets if t not in SOURCES]
        if unknown:
            p.error(f"unknown source(s): {', '.join(unknown)}. "
                    f"choices: {', '.join(SOURCES)}")

    print(f"data dir: {args.data_dir}")
    for name in targets:
        fetch_one(name, args.data_dir, args.force, args.ais_start, args.ais_end)

    print("\nDone. Record exact counts / Class-B share in data/INVENTORY.md "
          "(the source inventory open action items), and keep data/raw/CHECKSUMS.sha256 for "
          "the data card (data-card packaging).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
