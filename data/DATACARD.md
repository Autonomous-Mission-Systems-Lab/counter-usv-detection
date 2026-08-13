# Data cards

This repository produces two independently-usable derived datasets, each with its own
self-contained data card:

- **[`DATACARD_EO.md`](DATACARD_EO.md)** — harmonized maritime EO **object-detection**
  dataset (SeaShips, McShips, ABOShips, SMD frames + a curated `usv` class), COCO format.
- **[`DATACARD_TRACKS.md`](DATACARD_TRACKS.md)** — derived **vessel trajectory** features
  (MarineCadastre AIS voyage tracks), Parquet.

The two may be released separately. Shared conventions:

- Raw imagery and bulk AIS feeds are **never re-hosted**; obtain sources from their
  providers (`scripts/data/fetch_data.py`; licenses in `docs/DATA_LICENSES.md`) and regenerate
  derived products with the scripts in `scripts/`.
- Both are versioned together by `RELEASE.json` + `CHECKSUMS.derived.sha256`
  (`scripts/data/package_derived.py`); split each card's artifacts if releasing independently.
