#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_release_build.py  --  PUBLIC template: selection + de-identification helpers.

This module exposes the well-tested helper functions (inclusion criteria,
one-exam-per-eye, salted-hash de-identification, column selection) that the
canonical builder build_release.py uses. It is published as a transparent
template and is NOT meant to be executed directly: build_release.py is the
single entry point that regenerates the release exactly and writes the private
mapping OUTSIDE the public output tree.

What it does
------------
1) Reads your combined OPD-Scan III export (Excel or CSV).
2) Keeps only wavefront-related columns + minimal metadata (no names/DOB/exam date).
3) Applies inclusion criteria:
   - Myopia: sphere <= -0.50 D AND spherical equivalent <= -0.50 D
   - Validated wavefront zone >= 6.0 mm (OPDWFvalidZone)
   - Optional: keep only "NRM" topography label (Klyce/Maeda) using max-probability class.
4) Keeps only ONE examination per eye (earliest by exam date).
5) De-identifies subjects:
   - Creates a stable "SubjectID" using a salted SHA-256 hash of PatID
   - Outputs a PRIVATE mapping file PatID <-> SubjectID (do NOT deposit).
6) Writes a release-ready CSV file for repository deposit.

Entry point
-----------
Use build_release.py (canonical, exact reproducer). It calls the helpers below
and writes the public package plus a private mapping placed OUTSIDE the public
tree (PRIVATE_OUTPUT_DIR).

IMPORTANT
---------
- Keep SALT_FILE and PRIVATE_subject_mapping.csv PRIVATE.
  If you publish the salt, SubjectID could be brute-forced.
- If you create multiple dataset versions over time, REUSE the SAME SALT_FILE
  to preserve stable SubjectID values across releases.

Author: D Gatinel & J Malet
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS (EDIT THESE)
# =============================================================================

# 1) Input file to process (Excel .xlsx/.xls or .csv)
INPUT_FILE = os.environ.get("OPD_AGGREGATE_XLSX", "")   # set via environment; no local paths shipped

# 2) If Excel: sheet to read
INPUT_SHEET = "Combined_AllObs"

# 3) Output directory (will be created if missing)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./_out")

# 4) PRIVATE salt file used to generate stable SubjectID values
#    - Create a new one ONCE, keep it private, and reuse it for future updates.
SALT_FILE = os.environ.get("OPD_SALT_FILE", "")         # private salt; never shipped

# 5) Inclusion criteria
MIN_VALID_ZONE_MM = 6.0
MYOPIA_THRESHOLD_D = -0.50
KEEP_ONLY_NRM_TOPO = True  # uses Klyce/Maeda classifier probabilities if present

# 6) Optional date range filters (set to None to disable)
MIN_EXAM_YEAR: Optional[int] = 2013
MAX_EXAM_YEAR: Optional[int] = 2025

# 7) One exam per eye
KEEP_ONLY_FIRST_EXAM_PER_EYE = True


# =============================================================================
# INTERNALS
# =============================================================================

KLYCE_MAEDA_PROB_COLS = [  # NB: CLASSIFIksi (a severity INDEX, not a probability) is intentionally excluded from the argmax

    "CLASSIFInrm", "CLASSIFIast", "CLASSIFIkcs", "CLASSIFIkc", "CLASSIFIpmd",
    "CLASSIFIpkp", "CLASSIFImrs", "CLASSIFIhrs", "CLASSIFIoth", 
]


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _parse_date_series(s: pd.Series) -> pd.Series:
    """Parse dates in dd/mm/yyyy (day-first)."""
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def stable_subject_id(patient_id: str, salt: str, prefix: str = "SUBJ_", n_hex: int = 12) -> str:
    """Stable pseudonymous ID. Keep the salt PRIVATE."""
    h = _sha256_hex(f"{patient_id}|{salt}")
    return f"{prefix}{h[:n_hex]}"


def compute_spherical_equivalent(sph: pd.Series, cyl: pd.Series) -> pd.Series:
    return sph + (cyl / 2.0)


def pick_usecols(all_cols: List[str]) -> List[str]:
    """
    Compute a minimal 'usecols' list to avoid loading 2000+ columns.

    Keeps:
    - identifiers needed transiently (PatID, Eye, Sex, DOB, DOExam, Exam)
    - myopia/QA: OPDsph/OPDcyl/OPDaxis, OPDWFvalidZone, OPDfitError, HO_OPDwfError, OPDzone
    - Zernike coefficients: OPDc00..OPDc44
    - all derived coefficients: calc_p{3,4,5,6}mm_* (excluding ".1" duplicates when base exists)
    - topography classifier probs (optional; used to derive a label)
    """
    cols = list(all_cols)

    base_meta = [c for c in ["PatID", "Eye", "Sex", "DOB", "DOExam", "Exam"] if c in cols]
    ref_cols = [c for c in ["OPDsph", "OPDcyl", "OPDaxis"] if c in cols]
    qa_cols = [c for c in ["OPDzone", "OPDWFvalidZone", "OPDfitError", "HO_OPDwfError"] if c in cols]

    zern_cols = [c for c in cols if isinstance(c, str) and re.match(r"^OPDc\d{2}$", c)]

    # calc columns: keep the LAST occurrence of each (normalised) calc name.
    # The aggregate may carry several copies of a calc_* column (an older, buggy
    # block at low indices plus the corrected block appended at the end); pandas
    # renames later duplicates with a ".1"/".2" suffix. We therefore group by the
    # base name (suffix stripped) and keep the highest-index column, i.e. the
    # freshly appended, corrected block. (The earlier version kept the FIRST copy,
    # which selected the buggy block -- do NOT revert to that behaviour.)
    calc_last: Dict[str, str] = {}
    for c in cols:
        if not isinstance(c, str) or not re.match(r"^calc_p[3456]mm_", c):
            continue
        base = re.sub(r"\.\d+$", "", c)
        calc_last[base] = c  # later occurrences overwrite earlier ones
    calc_cols = list(calc_last.values())

    topo_cols = [c for c in KLYCE_MAEDA_PROB_COLS if c in cols]

    usecols = set(base_meta + ref_cols + qa_cols + topo_cols + zern_cols + calc_cols)
    # NOTE: pandas usecols selects by name; the exact de-duplication (keeping the
    # LAST/corrected duplicate and RENAMING it to its clean base name) is performed
    # positionally by build_release.py (the canonical exact reproducer). Use that
    # script to regenerate the release; this helper only narrows the columns read.
    return [c for c in cols if c in usecols]


def load_table(input_file: str, sheet_name: str) -> pd.DataFrame:
    p = Path(input_file)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    if p.suffix.lower() in [".xlsx", ".xls"]:
        header = pd.read_excel(p, sheet_name=sheet_name, nrows=0)
        usecols = pick_usecols(list(header.columns))
        df = pd.read_excel(p, sheet_name=sheet_name, usecols=usecols)
        return df

    if p.suffix.lower() == ".csv":
        # For CSV we cannot read "header only" cheaply with pandas in all cases, but it's fine.
        df0 = pd.read_csv(p, nrows=0)
        usecols = pick_usecols(list(df0.columns))
        df = pd.read_csv(p, usecols=usecols)
        return df

    raise ValueError(f"Unsupported input format: {p.suffix}")


def derive_topography_label(df: pd.DataFrame) -> pd.DataFrame:
    """Derive categorical label from Klyce/Maeda classifier probabilities if available."""
    available = [c for c in KLYCE_MAEDA_PROB_COLS if c in df.columns]
    if not available:
        df["TopoLabel"] = np.nan
        return df
    probs = df[available].copy()
    # Force numeric (robust)
    for c in available:
        probs[c] = pd.to_numeric(probs[c], errors="coerce")
    df["TopoLabel"] = probs.idxmax(axis=1).str.replace("CLASSIFI", "", regex=False).str.upper()
    df["TopoNRM"] = pd.to_numeric(df.get("CLASSIFInrm", np.nan), errors="coerce")
    return df


def apply_inclusion(df: pd.DataFrame) -> pd.DataFrame:
    # Date parsing
    df["DOExam_parsed"] = _parse_date_series(df["DOExam"]) if "DOExam" in df.columns else pd.NaT
    df["DOB_parsed"] = _parse_date_series(df["DOB"]) if "DOB" in df.columns else pd.NaT

    # Myopia definition
    if "OPDsph" in df.columns and "OPDcyl" in df.columns:
        df["OPDsph"] = pd.to_numeric(df["OPDsph"], errors="coerce")
        df["OPDcyl"] = pd.to_numeric(df["OPDcyl"], errors="coerce")
        df["OPD_SE"] = compute_spherical_equivalent(df["OPDsph"], df["OPDcyl"])
    else:
        df["OPD_SE"] = np.nan

    myopia_mask = pd.Series(True, index=df.index)
    if "OPDsph" in df.columns:
        myopia_mask &= df["OPDsph"] <= float(MYOPIA_THRESHOLD_D)
    myopia_mask &= df["OPD_SE"] <= float(MYOPIA_THRESHOLD_D)

    # Validated wavefront zone
    zone_mask = pd.Series(True, index=df.index)
    if "OPDWFvalidZone" in df.columns:
        df["OPDWFvalidZone"] = pd.to_numeric(df["OPDWFvalidZone"], errors="coerce")
        zone_mask &= df["OPDWFvalidZone"] >= float(MIN_VALID_ZONE_MM)

    # Topography label (optional)
    df = derive_topography_label(df)
    topo_mask = pd.Series(True, index=df.index)
    if KEEP_ONLY_NRM_TOPO and "TopoLabel" in df.columns:
        topo_mask &= df["TopoLabel"].fillna("") == "NRM"

    # Year filter (optional)
    year_mask = pd.Series(True, index=df.index)
    if MIN_EXAM_YEAR is not None or MAX_EXAM_YEAR is not None:
        y = df["DOExam_parsed"].dt.year
        if MIN_EXAM_YEAR is not None:
            year_mask &= y >= int(MIN_EXAM_YEAR)
        if MAX_EXAM_YEAR is not None:
            year_mask &= y <= int(MAX_EXAM_YEAR)

    out = df[myopia_mask & zone_mask & topo_mask & year_mask].copy()
    return out


def keep_first_exam_per_eye(df: pd.DataFrame) -> pd.DataFrame:
    if "PatID" not in df.columns or "Eye" not in df.columns:
        raise ValueError("Columns PatID and Eye are required to keep one exam per eye.")

    sort_cols: List[str] = []
    if "DOExam_parsed" in df.columns:
        sort_cols.append("DOExam_parsed")
    if "Exam" in df.columns:
        sort_cols.append("Exam")

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=True)

    return df.drop_duplicates(subset=["PatID", "Eye"], keep="first").copy()


def ensure_private_salt(path: Path) -> str:
    if path.exists():
        return _read_text(path)

    # create a new salt only once
    salt = _sha256_hex(f"{datetime.utcnow().isoformat()}|{os.urandom(16).hex()}")
    _safe_mkdir(path.parent)
    _write_text(path, salt)
    return salt


def anonymize(df: pd.DataFrame, salt_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    salt = ensure_private_salt(Path(salt_file))

    df["SubjectID"] = df["PatID"].astype(str).apply(lambda x: stable_subject_id(x, salt))

    # Standardize eye labels
    df["Eye"] = df["Eye"].astype(str).str.upper().str.strip()
    df["Eye"] = df["Eye"].replace({"OD": "R", "OG": "L", "OS": "L"})

    # Sex
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].astype(str).str.upper().str.strip().replace({"MALE": "M", "FEMALE": "F"})

    # Age at exam (in years, rounded)
    dob = pd.to_datetime(df.get("DOB_parsed", pd.NaT), errors="coerce")
    doe = pd.to_datetime(df.get("DOExam_parsed", pd.NaT), errors="coerce")
    df["ExamYear"] = doe.dt.year
    df["AgeYears"] = ((doe - dob).dt.days / 365.25).round(1)

    private_map = df[["PatID", "SubjectID"]].drop_duplicates().sort_values(["PatID"]).reset_index(drop=True)

    # Drop direct identifiers & raw dates
    drop_cols = [c for c in ["PatID", "Name", "DOB", "DOExam", "Diagnosis", "Comment",
                             "DOB_parsed", "DOExam_parsed", "Group", "SourceFile", "SourceSheet"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df, private_map


def order_release_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Place key metadata first, then QA/refraction, then Zernike, then derived calc_* columns.
    """
    cols = list(df.columns)

    first = [c for c in ["SubjectID", "Eye", "Sex", "AgeYears", "ExamYear", "TopoLabel", "TopoNRM", "OPD_SE"] if c in cols]
    qa = [c for c in ["OPDWFvalidZone", "OPDfitError", "HO_OPDwfError", "OPDzone", "OPDsph", "OPDcyl", "OPDaxis"] if c in cols]
    zern = sorted([c for c in cols if isinstance(c, str) and re.match(r"^OPDc\d{2}$", c)])
    calc = sorted([c for c in cols if isinstance(c, str) and re.match(r"^calc_p[3456]mm_", c)])

    used = set(first + qa + zern + calc)
    other = [c for c in cols if c not in used]

    ordered = first + qa + zern + calc + other
    return df[ordered]


def build_data_dictionary(columns):
    """Rich 9-field data dictionary (delegates to dictionary.build_rich_dictionary),
    matching the released data_dictionary_release.csv (column/description/units/type/
    group/basis/n/m/pupil_mm), one row per column in column order."""
    import importlib.util, os
    _h=os.path.dirname(os.path.abspath(__file__))
    _s=importlib.util.spec_from_file_location("dictionary", os.path.join(_h,"dictionary.py"))
    _d=importlib.util.module_from_spec(_s); _s.loader.exec_module(_d)
    return _d.build_rich_dictionary(list(columns))

def run() -> None:  # superseded and disabled
    raise SystemExit(
        "prepare_release_build.run() is superseded and disabled to avoid writing the "
        "private mapping inside a public directory. Use build_release.py instead:\n"
        "  OPD_AGGREGATE_XLSX=... OPD_SALT_FILE=... python build_release.py")


if __name__ == "__main__":
    raise SystemExit("This is a public template of helper functions; run build_release.py instead.")


