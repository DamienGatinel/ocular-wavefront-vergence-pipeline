#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_release.py -- CANONICAL builder: reproduces the FULL public package exactly.

Inputs (PRIVATE, not distributed):
  OPD_AGGREGATE_XLSX : engine aggregate .xlsx (OPDc00..OPDc44 + appended calc_* blocks)
  OPD_SALT_FILE      : private random salt for the stable SubjectID hash
Optional:
  OUTPUT_DIR         : public output dir (default: <repo>/_rebuilt, a sibling of code/)
  PRIVATE_OUTPUT_DIR : where the re-identification mapping is written; MUST be OUTSIDE
                       OUTPUT_DIR (default: <repo>/_rebuilt_PRIVATE). Never inside the
                       public tree, so a recursive zip of OUTPUT_DIR cannot leak it.

Regenerates and asserts the complete public package:
  opd_wavefront_release.csv            1892 x 1235
  data_dictionary_release.csv          9-field rich dictionary, column order (dictionary.py)
  scidata_tables/*.csv                 FIVE tables (4 descriptive + qc_sensitivity.csv)
  README_dataset.md, CHANGELOG.md, LICENSE_data.txt, LICENSE_code.txt   (copied)
  code/*                               (copied)
  manifest_public.json                 SHA-256 of every public file (whole package)
and, OUTSIDE the public tree:
  <PRIVATE_OUTPUT_DIR>/PRIVATE_subject_mapping.csv

Guarantees (asserted): last (corrected) duplicate calc_* kept & renamed to base name
(no ".1"/".2"); CLASSIFIksi kept in output but EXCLUDED from the argmax; PTVaxis == axis;
post-hoc QC (max |scalar npvV component| over all modes/pupils > 20 D) removes the eyes;
five tables present; dictionary == dictionary.py; no PRIVATE_* inside OUTPUT_DIR.

Run:  OPD_AGGREGATE_XLSX=... OPD_SALT_FILE=... python build_release.py
"""
import os, re, sys, csv, json, math, shutil, hashlib, subprocess, importlib.util
import numpy as np, pandas as pd, openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                     # release root (parent of code/)
QC_THRESH_D = 20.0
SENS_THRESHOLDS = (15.0, 20.0, 25.0)
PROB_COLS_FOR_ARGMAX = ["CLASSIFInrm", "CLASSIFIast", "CLASSIFIkcs", "CLASSIFIkc",
                        "CLASSIFIpmd", "CLASSIFIpkp", "CLASSIFImrs", "CLASSIFIhrs", "CLASSIFIoth"]
ALL_CLASSIFI = PROB_COLS_FOR_ARGMAX + ["CLASSIFIksi"]   # ksi kept in output, excluded from argmax
META_KEEP = ["PatID", "Name", "Sex", "DOB", "DOExam", "Exam", "Eye",
             "OPDsph", "OPDcyl", "OPDaxis", "OPDzone", "OPDWFvalidZone", "OPDfitError", "HO_OPDwfError"]
STATIC_FILES = ["README_dataset.md", "CHANGELOG.md", "LICENSE_data.txt", "LICENSE_code.txt"]

b_spec = importlib.util.spec_from_file_location("blt", os.path.join(HERE, "prepare_release_build.py"))
b = importlib.util.module_from_spec(b_spec); b_spec.loader.exec_module(b)
d_spec = importlib.util.spec_from_file_location("dic", os.path.join(HERE, "dictionary.py"))
D = importlib.util.module_from_spec(d_spec); d_spec.loader.exec_module(D)


def read_aggregate(path, sheet="Combined_AllObs"):
    wb = openpyxl.load_workbook(path, read_only=True); ws = wb[sheet]
    hdr = [c.value for c in next(ws.iter_rows(1, 1))]; wb.close()
    keep = set(META_KEEP) | set(ALL_CLASSIFI)
    first, calc_last = {}, {}
    for i, h in enumerate(hdr):
        if not isinstance(h, str):
            continue
        if re.match(r"^OPDc\d{2}$", h) or h in keep:
            first.setdefault(h, i)
        elif re.match(r"^calc_p[3456]mm_", h):
            calc_last[re.sub(r"\.\d+$", "", h)] = i           # last occurrence
    sel = {i: h for h, i in first.items()}
    sel.update({i: base for base, i in calc_last.items()})    # clean base names
    usecols = sorted(sel.keys())
    df = pd.read_excel(path, sheet_name=sheet, usecols=usecols, header=None, skiprows=1)
    df.columns = [sel[i] for i in usecols]
    assert not [c for c in df.columns if re.search(r"\.\d+$", str(c))], "residual .N columns!"
    return df


def write_qc_sensitivity(df_pre_qc, out_csv):
    """Sensitivity of retained-eye count and key |npvV| (VL-VH) medians to the QC threshold.
    All quantities use the VL-VH peak-to-valley (npvV) coefficients at 6 mm, over ALL retained
    eyes, and are absolute values. Column names carry the basis/pupil explicitly."""
    npv = [c for c in df_pre_qc.columns if re.match(r"^calc_p\d+mm_npvV_", c)]
    M = df_pre_qc[npv].apply(pd.to_numeric, errors="coerce").abs().max(axis=1)

    def med_abs(d, col):
        return round(float(pd.to_numeric(d[col], errors="coerce").abs().median()), 4)

    def med_vec(d, a, bcol):
        return round(float(np.hypot(pd.to_numeric(d[a], errors="coerce"),
                                    pd.to_numeric(d[bcol], errors="coerce")).median()), 4)
    rows = []
    for thr in SENS_THRESHOLDS:
        d = df_pre_qc[M <= thr]
        rows.append(dict(qc_threshold_D=thr, subset="all_eyes_retained", basis="VL-VH (npvV)",
                         n_eyes=len(d),
                         median_abs_coma_npvV_6mm_D=med_vec(d, "calc_p6mm_npvV_n3_m1", "calc_p6mm_npvV_n3_m-1"),
                         median_abs_SA_npvV_6mm_D=med_abs(d, "calc_p6mm_npvV_n4_m0"),
                         median_abs_trefoil_npvV_6mm_D=med_vec(d, "calc_p6mm_npvV_n3_m3", "calc_p6mm_npvV_n3_m-3")))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    AGG = os.environ.get("OPD_AGGREGATE_XLSX"); SALT = os.environ.get("OPD_SALT_FILE")
    if not AGG or not SALT:
        sys.exit("Set OPD_AGGREGATE_XLSX and OPD_SALT_FILE (private inputs).")
    pub = os.path.abspath(os.environ.get("OUTPUT_DIR", os.path.join(REPO, "_rebuilt")))
    priv = os.path.abspath(os.environ.get("PRIVATE_OUTPUT_DIR", os.path.join(REPO, "_rebuilt_PRIVATE")))
    assert not (priv == pub or priv.startswith(pub + os.sep)), \
        "PRIVATE_OUTPUT_DIR must be OUTSIDE OUTPUT_DIR (never inside the public tree)"
    # safety: never wipe the repo root, the code dir, or any ancestor of them
    _protected = {os.path.abspath(REPO), os.path.abspath(HERE), os.path.abspath(os.sep)}
    assert pub not in _protected and not REPO.startswith(pub + os.sep) and not HERE.startswith(pub + os.sep), \
        f"refusing to remove OUTPUT_DIR={pub}: it is (or contains) the repo/code/source tree"
    if os.path.isdir(pub):
        shutil.rmtree(pub)
    os.makedirs(os.path.join(pub, "scidata_tables")); os.makedirs(priv, exist_ok=True)
    b.SALT_FILE = SALT

    df = read_aggregate(AGG)
    df_inc = b.apply_inclusion(df)                       # topo argmax excludes ksi (module list)
    df_inc = b.keep_first_exam_per_eye(df_inc)
    write_qc_sensitivity(df_inc, os.path.join(pub, "scidata_tables", "qc_sensitivity.csv"))

    npv = [c for c in df_inc.columns if isinstance(c, str) and re.match(r"^calc_p\d+mm_npvV_", c)]
    M = df_inc[npv].apply(pd.to_numeric, errors="coerce").abs().max(axis=1)
    df_inc = df_inc[M <= QC_THRESH_D].copy()

    rel, pmap = b.anonymize(df_inc, SALT)
    rel = b.order_release_columns(rel)
    for p in [c for c in rel.columns if "PTVaxis" in c]:      # PTVaxis := axis (scale-invariant)
        a = p.replace("PTVaxis", "axis")
        if a in rel.columns:
            rel[p] = rel[a].values

    relcsv = os.path.join(pub, "opd_wavefront_release.csv"); rel.to_csv(relcsv, index=False)
    D.build_rich_dictionary(list(rel.columns)).to_csv(os.path.join(pub, "data_dictionary_release.csv"), index=False)
    pmap.to_csv(os.path.join(priv, "PRIVATE_subject_mapping.csv"), index=False)   # OUTSIDE public tree

    # descriptive tables (fail loudly if the stats script errors)
    stats = os.path.join(HERE, "scidata_ptv_descriptive_stats_public_v1.py")
    subprocess.run([sys.executable, stats, "--csv", relcsv, "--out", os.path.join(pub, "scidata_tables")], check=True)

    # assemble the FULL public package: copy static docs + the code/ folder
    for fn in STATIC_FILES:
        src = os.path.join(REPO, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(pub, fn))
        else:
            print(f"[warn] static file missing, not packaged: {fn}")
    code_dst = os.path.join(pub, "code"); os.makedirs(code_dst, exist_ok=True)
    for fn in os.listdir(HERE):
        if fn.endswith((".py", ".txt", ".md")) and fn != "__pycache__":
            shutil.copy2(os.path.join(HERE, fn), os.path.join(code_dst, fn))

    # manifest of the WHOLE public package (public tree only)
    files = []
    for root, _, fs in os.walk(pub):
        for fn in sorted(fs):
            if fn == "manifest_public.json":
                continue
            p = os.path.join(root, fn)
            e = {"path": os.path.relpath(p, pub).replace("\\", "/"), "bytes": os.path.getsize(p), "sha256": sha(p)}
            if fn.endswith(".csv"):
                with open(p, newline="") as f:
                    r = csv.reader(f); h = next(r); e["rows"] = sum(1 for _ in r); e["cols"] = len(h)
            files.append(e)
    json.dump({"version": "v2 (corrected)", "n_eyes": int(rel.shape[0]), "n_subjects": int(rel["SubjectID"].nunique()),
               "concept_doi": "10.5281/zenodo.18615047",
               "note": "Private re-identification key is written outside this tree and is NOT part of the package.",
               "files": sorted(files, key=lambda x: x["path"])},
              open(os.path.join(pub, "manifest_public.json"), "w"), indent=2)

    # ---- integration assertions ----
    assert rel.shape == (1892, 1235), f"shape {rel.shape} != (1892, 1235)"
    assert not [c for c in rel.columns if re.search(r"\.\d+$", str(c))], "duplicate .N columns present"
    assert "CLASSIFIksi" in rel.columns, "CLASSIFIksi missing from output"
    assert (rel["TopoLabel"].fillna("") == "NRM").all(), "non-NRM eyes present"
    dd = pd.read_csv(os.path.join(pub, "data_dictionary_release.csv"))
    assert list(dd.columns) == ["column", "description", "units", "type", "group", "basis", "n", "m", "pupil_mm"]
    assert list(dd["column"]) == list(rel.columns), "dictionary order != column order"
    tabs = sorted(os.listdir(os.path.join(pub, "scidata_tables")))
    expected_tabs = sorted(["sample_counts_by_pupil.csv", "summary_ptv_key_modes_6mm_subsets.csv",
                            "summary_ptv_key_modes_by_pupil_one_eye_per_subject.csv",
                            "corr_RL_key_modes_6mm.csv", "qc_sensitivity.csv"])
    assert tabs == expected_tabs, f"tables {tabs} != {expected_tabs}"
    leaked = [os.path.join(r, fn) for r, _, fs in os.walk(pub) for fn in fs if fn.startswith("PRIVATE_")]
    assert not leaked, f"PRIVATE_* leaked into public tree: {leaked}"
    print(f"OK: release {rel.shape[0]}x{rel.shape[1]}, {rel['SubjectID'].nunique()} subjects; "
          f"5 tables {tabs}; 9-field dictionary; CLASSIFIksi kept & excluded from argmax; "
          f"full package manifested ({len(files)} public files); PRIVATE mapping in {priv} (outside public tree).")


if __name__ == "__main__":
    main()
