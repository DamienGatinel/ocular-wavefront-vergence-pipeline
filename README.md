# Code — reproduction of the derived wavefront/vergence coefficients

All scripts are plain Python 3.

## Requirements & installation

```
pip install -r requirements.txt
```
Dependencies (pinned in `requirements.txt`): numpy, pandas, sympy, mpmath, scipy, matplotlib, openpyxl (Python 3.11 tested).

## Canonical exact reproducer (from the private aggregate)

- **`build_release.py`** — regenerates the release **exactly** from the private engine
  aggregate. It keeps the last (corrected) duplicate `calc_*` column renamed to its base
  name, keeps `CLASSIFIksi` in the output but excludes it from the topography argmax,
  applies the PTVaxis correction and the 20 D QC rule, writes the 9-field dictionary and
  the FIVE tables (the four descriptive tables plus `qc_sensitivity.csv`), and writes the
  private mapping into `PRIVATE_OUTPUT_DIR`, a directory OUTSIDE the public output tree
  (never inside it). It asserts the output is `1892 x 1235`. Run:
  `OPD_AGGREGATE_XLSX=... OPD_SALT_FILE=... python build_release.py`
- **`dictionary.py`** — the shared 9-field data-dictionary generator.


## Reviewer-runnable on the PUBLIC data (no private data needed)

- **`zernike_to_vergence.py`** — self-contained converter. Given a 45-vector of
  OSA/ANSI Zernike coefficients (order 8), it computes: the Gatinel–Malet (LD/HD)
  coefficients; the vergence coefficients in the fully orthogonal basis Ṽ and the
  partially orthogonal basis VL–VH (high-order projection on the **full pupil**);
  the peak-to-valley normalised coefficients (npvV, npvVT); the oriented-mode
  magnitude and axis; the spherocylindrical refraction (cornea and 12 mm vertex);
  and the analytical sub-pupil rescaling (Lundström & Unsbo 2007).

- **`reproduce_from_release.py`** — recomputes the derived coefficients directly
  from the `OPDc00..OPDc44` columns of `opd_wavefront_release.csv`, for all four
  pupil diameters, and compares them to the released `calc_*` columns.
  `python reproduce_from_release.py --csv ../opd_wavefront_release.csv --n 50`
  Expected: agreement to machine precision (max |difference| < 1e-6).

- **`test_pipeline.py`** — unit tests over the four pupil diameters, both vergence
  bases, the peak-to-valley normalisation, the axis scale-invariance
  (PTVaxis == axis) and the oriented-mode magnitude. `python test_pipeline.py`.

- **`scidata_ptv_descriptive_stats_public_v1.py`** — regenerates the four
  `scidata_tables/` files from the released CSV.
  `python scidata_ptv_descriptive_stats_public_v1.py --csv ../opd_wavefront_release.csv --out ../scidata_tables`

## Provenance scripts (require the private clinical export; shown for transparency)

- **`prepare_release_build.py`** — selection + de-identification (inclusion
  criteria, one-exam-per-eye, salted-hash SubjectID). The private random salt and
  the identifier mapping are **withheld** to prevent re-identification; supply your
  own salt to re-run on your own source export.
- **`rebuild_release_corrected.py`** — DEPRECATED (superseded by `build_release.py`);
  kept only as a stub that exits with a pointer to `build_release.py`.

## Notes on the corrections in this version
- The VL–VH high-order projection is computed on the **full pupil** (not a central
  sub-region); this removes the artificial primary/secondary-coma proportionality
  present in the earlier version.
- The peak-to-valley normalised **axis** columns equal the scale-invariant axis
  (peak-to-valley normalisation multiplies the sine/cosine pair by the same
  positive factor, so the orientation is unchanged).
