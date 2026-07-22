#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scidata_ptv_descriptive_stats_public_v1.py

Public-facing script to reproduce the descriptive statistics tables used in the
Scientific Data (Data Descriptor) manuscript for the OPD-Scan III wavefront/vergence dataset.

Input
-----
- opd_wavefront_release.csv (de-identified release file)

Outputs (CSV)
-------------
- sample_counts_by_pupil.csv
- summary_ptv_key_modes_6mm_subsets.csv
- corr_RL_key_modes_6mm.csv
- summary_ptv_key_modes_by_pupil_one_eye_per_subject.csv

Notes
-----
This script focuses on PTV-normalized metrics (npvV and npvVT).
For oriented modes (m>0), it computes vector magnitudes (primary descriptive metric)
from sine/cosine pairs:
  magnitude = sqrt(cos^2 + sin^2)
  axis      = angle = atan2(sin, cos) / m   (degrees; periodicity depends on m)
Only magnitudes are summarised here (axes are typically handled with circular statistics).

Run
---
python scidata_ptv_descriptive_stats_public_v1.py --csv opd_wavefront_release.csv --out scidata_tables
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


KEY_MODES: Dict[str, Tuple[int, int]] = {
    "Primary coma": (3, 1),
    "Primary trefoil": (3, 3),
    "Primary spherical aberration": (4, 0),
    "Secondary astigmatism": (4, 2),
    "Quadrafoil": (4, 4),
    "Secondary coma": (5, 1),
    "Secondary spherical aberration": (6, 0),
}

PUPILS: List[int] = [3, 4, 5, 6]

BASES = {
    "npvV": "VL–VH (npvV)",
    "npvVT": "Orthogonal V~ (npvVT)",
}


def col_npv(pupil: int, prefix: str, n: int, m: int) -> str:
    """Return the release column name for a peak-to-valley normalised coefficient
    of mode (n, m) in the given basis prefix at the given pupil (mm)."""
    return f"calc_p{pupil}mm_{prefix}_n{n}_m{m}"


def vector_magnitude_and_axis(df: pd.DataFrame, pupil: int, prefix: str, n: int, m_abs: int):
    """
    For m_abs>0:
      cos coefficient uses m=+m_abs
      sin coefficient uses m=-m_abs
    """
    cos_col = col_npv(pupil, prefix, n, m_abs)
    sin_col = col_npv(pupil, prefix, n, -m_abs)
    if cos_col not in df.columns or sin_col not in df.columns:
        raise KeyError(f"Missing columns: {cos_col} and/or {sin_col}")

    c = pd.to_numeric(df[cos_col], errors="coerce")
    s = pd.to_numeric(df[sin_col], errors="coerce")
    mag = np.sqrt(c * c + s * s)
    axis = (np.degrees(np.arctan2(s, c) / m_abs)) % (360.0 / m_abs)
    return mag, axis


def summary_stats(x: pd.Series) -> dict:
    """Return n, mean, sd, median and quartiles of a numeric series (NaNs dropped)."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, median=np.nan, q1=np.nan, q3=np.nan,
                    p5=np.nan, p95=np.nan, min=np.nan, max=np.nan)
    return dict(
        n=int(x.shape[0]),
        mean=float(x.mean()),
        sd=float(x.std(ddof=1)),
        median=float(x.median()),
        q1=float(x.quantile(0.25)),
        q3=float(x.quantile(0.75)),
        p5=float(x.quantile(0.05)),
        p95=float(x.quantile(0.95)),
        min=float(x.min()),
        max=float(x.max()),
    )


def build_subset(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Return the requested eye subset of the release: ``both`` (all eyes),
    ``right_only`` (right eyes), or ``one_eye_per_subject`` (prefers the right eye)."""
    subset = subset.lower()
    df2 = df.copy()
    df2["Eye"] = df2["Eye"].astype(str).str.upper()

    if subset == "both":
        return df2
    if subset == "right_only":
        return df2[df2["Eye"] == "R"].copy()
    if subset == "one_eye_per_subject":
        df2["__eye_rank__"] = df2["Eye"].map({"R": 0, "L": 1}).fillna(2)
        df2 = df2.sort_values(["SubjectID", "__eye_rank__"])
        df2 = df2.drop_duplicates(subset=["SubjectID"], keep="first")
        return df2.drop(columns=["__eye_rank__"])
    raise ValueError("subset must be one of: both, right_only, one_eye_per_subject")


def sample_counts_by_pupil(df: pd.DataFrame) -> pd.DataFrame:
    """Return eye and subject counts at each pupil diameter (sample_counts_by_pupil.csv)."""
    rows = []
    for pupil in PUPILS:
        # use a representative field to define availability
        ref_col = col_npv(pupil, "npvV", 3, 1)
        if ref_col not in df.columns:
            continue
        avail = df[ref_col].notna()
        rows.append({
            "pupil_mm": pupil,
            "n_eyes": int(avail.sum()),
            "n_subjects": int(df.loc[avail, "SubjectID"].nunique()),
        })
    return pd.DataFrame(rows)


def summary_key_modes_6mm(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics of the key peak-to-valley metrics at 6 mm for the
    three subsets (summary_ptv_key_modes_6mm_subsets.csv)."""
    rows = []
    for subset in ["both", "right_only", "one_eye_per_subject"]:
        dsub = build_subset(df, subset)

        for prefix, basis_label in BASES.items():
            for mode, (n, m_abs) in KEY_MODES.items():
                if m_abs == 0:
                    col = col_npv(6, prefix, n, 0)
                    if col not in dsub.columns:
                        continue
                    stats_signed = summary_stats(dsub[col])
                    stats_abs = summary_stats(dsub[col].abs())

                    rows.append({
                        "subset": subset,
                        "basis": basis_label,
                        "metric": f"{mode} (PTV coeff, signed)",
                        **stats_signed
                    })
                    rows.append({
                        "subset": subset,
                        "basis": basis_label,
                        "metric": f"{mode} (PTV coeff, absolute)",
                        **stats_abs
                    })
                else:
                    mag, _ = vector_magnitude_and_axis(dsub, 6, prefix, n, m_abs)
                    stats_mag = summary_stats(mag)
                    rows.append({
                        "subset": subset,
                        "basis": basis_label,
                        "metric": f"{mode} (PTV vector magnitude)",
                        **stats_mag
                    })

    return pd.DataFrame(rows)


def intereye_corr_6mm(df: pd.DataFrame) -> pd.DataFrame:
    """Return right-vs-left (inter-eye) Pearson/Spearman correlations of the key metrics
    at 6 mm, computed on subjects contributing both eyes (corr_RL_key_modes_6mm.csv)."""
    df2 = df.copy()
    df2["Eye"] = df2["Eye"].astype(str).str.upper()

    # subjects with both eyes
    both_ids = df2.groupby("SubjectID")["Eye"].nunique()
    both_ids = both_ids[both_ids >= 2].index
    df_both = df2[df2["SubjectID"].isin(both_ids)].copy()

    rows = []
    for prefix, basis_label in BASES.items():
        for mode, (n, m_abs) in KEY_MODES.items():
            if m_abs == 0:
                col = col_npv(6, prefix, n, 0)
                if col not in df_both.columns:
                    continue
                pivot = df_both.pivot_table(index="SubjectID", columns="Eye", values=col, aggfunc="first")
                if "R" not in pivot.columns or "L" not in pivot.columns:
                    continue
                x, y = pivot["R"], pivot["L"]
                mask = ~(x.isna() | y.isna())
                x, y = x[mask], y[mask]
                if len(x) < 10:
                    continue

                rows.append({
                    "basis": basis_label,
                    "metric": f"{mode} (PTV coeff, signed)",
                    "n_subjects": int(len(x)),
                    "pearson_r": float(x.corr(y, method="pearson")),
                    "spearman_r": float(x.corr(y, method="spearman")),
                })

                x_abs, y_abs = x.abs(), y.abs()
                rows.append({
                    "basis": basis_label,
                    "metric": f"{mode} (PTV coeff, absolute)",
                    "n_subjects": int(len(x_abs)),
                    "pearson_r": float(x_abs.corr(y_abs, method="pearson")),
                    "spearman_r": float(x_abs.corr(y_abs, method="spearman")),
                })

            else:
                mag, _ = vector_magnitude_and_axis(df_both, 6, prefix, n, m_abs)
                tmp = df_both[["SubjectID", "Eye"]].copy()
                tmp["mag"] = mag
                pivot = tmp.pivot_table(index="SubjectID", columns="Eye", values="mag", aggfunc="first")
                if "R" not in pivot.columns or "L" not in pivot.columns:
                    continue
                x, y = pivot["R"], pivot["L"]
                mask = ~(x.isna() | y.isna())
                x, y = x[mask], y[mask]
                if len(x) < 10:
                    continue

                rows.append({
                    "basis": basis_label,
                    "metric": f"{mode} (PTV vector magnitude)",
                    "n_subjects": int(len(x)),
                    "pearson_r": float(x.corr(y, method="pearson")),
                    "spearman_r": float(x.corr(y, method="spearman")),
                })

    return pd.DataFrame(rows)


def summary_key_modes_by_pupil(df: pd.DataFrame, subset: str = "one_eye_per_subject") -> pd.DataFrame:
    """Return descriptive statistics of the key metrics at each pupil diameter for the
    one-eye-per-subject subset (summary_ptv_key_modes_by_pupil_one_eye_per_subject.csv)."""
    dsub = build_subset(df, subset)
    rows = []

    for pupil in PUPILS:
        for prefix, basis_label in BASES.items():
            for mode, (n, m_abs) in KEY_MODES.items():
                if m_abs == 0:
                    col = col_npv(pupil, prefix, n, 0)
                    if col not in dsub.columns:
                        continue
                    stats = summary_stats(dsub[col].abs())
                    rows.append({
                        "subset": subset,
                        "pupil_mm": pupil,
                        "basis": basis_label,
                        "metric": f"{mode} (PTV abs coeff)",
                        **stats
                    })
                else:
                    mag, _ = vector_magnitude_and_axis(dsub, pupil, prefix, n, m_abs)
                    stats = summary_stats(mag)
                    rows.append({
                        "subset": subset,
                        "pupil_mm": pupil,
                        "basis": basis_label,
                        "metric": f"{mode} (PTV magnitude)",
                        **stats
                    })

    return pd.DataFrame(rows)


def main() -> None:
    """Command-line entry point: read the release CSV and write the five descriptive tables."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to opd_wavefront_release.csv")
    ap.add_argument("--out", required=True, help="Output directory for CSV tables")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)

    # basic sanity
    for c in ["SubjectID", "Eye"]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    counts = sample_counts_by_pupil(df)
    counts.to_csv(os.path.join(args.out, "sample_counts_by_pupil.csv"), index=False)

    summary6 = summary_key_modes_6mm(df)
    summary6.to_csv(os.path.join(args.out, "summary_ptv_key_modes_6mm_subsets.csv"), index=False)

    corr6 = intereye_corr_6mm(df)
    for _c in ("pearson_r", "spearman_r"):
        if _c in corr6.columns:
            corr6[_c] = corr6[_c].round(12)  # deterministic export (kill ~1e-16 jitter)
    corr6.to_csv(os.path.join(args.out, "corr_RL_key_modes_6mm.csv"), index=False)

    by_pupil = summary_key_modes_by_pupil(df, subset="one_eye_per_subject")
    by_pupil.to_csv(os.path.join(args.out, "summary_ptv_key_modes_by_pupil_one_eye_per_subject.csv"), index=False)

    print("Wrote tables to:", args.out)


if __name__ == "__main__":
    main()
