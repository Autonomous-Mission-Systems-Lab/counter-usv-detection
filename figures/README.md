# Hand-authored figures

Figures in this directory are authored by hand and committed as source. They are **not** generated
by `scripts/report/`, which builds only the data figures that read digest-verified artifacts under
`results/`. The build step copies these assets into `results/paper/` and records their SHA-256 in
`results/paper/PROVENANCE.json` so the published figure set is fully accounted for.

## `fig1_system.svg`

The attack/defense interface for shore-based counter-USV detection.

Two sensing channels run left to right. The optical channel (camera to detector to asserted class)
is the one an attacker can reach, by adversarial patch or by simply looking like a benign vessel.
The radar channel supplies world-frame range and bearing, which a patch cannot alter. Both meet the
defense at a deliberately narrow interface — the claimed class plus track features — rather than
through joint optimization over pixels and motion.

On the defense side, the presence check catches evasion by finding a radar track with no EO
detection; it reads detection status only and never the claimed class. The consistency check catches
disguise by asking whether the claimed class matches how the craft is actually moving, and reports
its two arms separately: kinematics, and geometry relative to a defended asset.

The dashed path entering from below is offline model fitting. Real benign AIS tracks produce one
envelope per benign class. AIS is never read at runtime and no hostile tracks enter the model, which
is why that path is drawn in a different line style rather than folded into the runtime flow.

### Editing

Plain SVG with no external font or embedded raster, so it opens in Inkscape, Figma, Illustrator, or
a text editor, and diffs readably in git. Conventions to preserve when editing:

- Solid strokes are runtime data flow; dashed strokes are offline training. Do not distinguish the
  two by color alone.
- The figure must stay legible in greyscale and at two-column width. Body text is 9.5–11px against a
  1180x640 viewBox; going smaller will not survive print.
- Fonts are a generic sans-serif stack. Do not reference a font that is not universally installed,
  and do not convert text to paths — the text should stay selectable and searchable.

To preview while editing, serve the repo and open the file in a browser:

```bash
python -m http.server 8971 --bind 127.0.0.1
# then open http://127.0.0.1:8971/figures/fig1_system.svg
```

### Export

SVG is the source of truth and renders natively in GitHub markdown, so no export is required for the
repository. For a manuscript that needs PDF or EPS, convert with whichever tool is available:

```bash
cairosvg figures/fig1_system.svg -o fig1_system.pdf     # pip install cairosvg
rsvg-convert -f pdf -o fig1_system.pdf figures/fig1_system.svg   # brew install librsvg
inkscape figures/fig1_system.svg --export-filename=fig1_system.pdf
```

None of these are currently installed or listed in `requirements.txt`. `scripts/report/build_all.py`
emits PNG and PDF next to the SVG only when `cairosvg` happens to be importable, and otherwise skips
that step with a message rather than failing the build.
