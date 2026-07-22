# -*- coding: utf-8 -*-
"""Rich data-dictionary generator (9 fields), shared by the build and audit code.

Produces one row per release column, in the given column order, with fields:
column, description, units, type, group, basis, n, m, pupil_mm.
Every column family of opd_wavefront_release.csv is covered, including Exam,
the Klyce/Maeda CLASSIFI* scores (0-100), TopoNRM, the _v12 spectacle-plane
refraction, the oriented-mode magnitude/axis and their PTV counterparts, the
PTV scalars (m=0) and the per-map distribution statistics.
"""
import re
import pandas as pd

_MODE = {(2, 0): "defocus", (2, 2): "astigmatism (0/90)", (2, -2): "astigmatism (45)",
         (3, 1): "coma (horizontal)", (3, -1): "coma (vertical)", (3, 3): "trefoil", (3, -3): "trefoil",
         (4, 0): "spherical aberration", (4, 2): "secondary astigmatism", (4, -2): "secondary astigmatism",
         (4, 4): "quadrafoil", (4, -4): "quadrafoil", (5, 1): "secondary coma", (5, -1): "secondary coma",
         (5, 3): "secondary trefoil", (5, -3): "secondary trefoil", (5, 5): "pentafoil", (5, -5): "pentafoil",
         (6, 0): "secondary spherical aberration"}
_BASE_FULL = {"GM": "Gatinel–Malet", "VLVH": "vergence VL–VH (partially orthogonal)",
              "TILDE": "vergence Ṽ (fully orthogonal)", "npvV": "peak-to-valley normalised VL–VH",
              "npvVT": "peak-to-valley normalised Ṽ"}
_BASE_UNIT = {"GM": "µm", "VLVH": "D", "TILDE": "D", "npvV": "D", "npvVT": "D"}

_META = {
 "SubjectID": ("Pseudonymous subject identifier (stable salted SHA-256 hash of the clinical PatID).", "", "string", "identifier", "", "", "", ""),
 "Eye": ("Eye laterality (R = right, L = left).", "", "categorical", "metadata", "", "", "", ""),
 "Sex": ("Subject sex as recorded (M/F).", "", "categorical", "metadata", "", "", "", ""),
 "AgeYears": ("Age at examination, in years (rounded to 0.1).", "years", "float", "metadata", "", "", "", ""),
 "ExamYear": ("Calendar year of the examination.", "year", "integer", "metadata", "", "", "", ""),
 "TopoLabel": ("Automated Klyce/Maeda corneal-topography class with the highest score (here always NRM = normal).", "", "categorical", "metadata", "", "", "", ""),
 "TopoNRM": ("Klyce/Maeda classifier score for the normal (NRM) class (0–100 scale).", "0–100", "float", "metadata", "", "", "", ""),
 "OPD_SE": ("Objective spherical equivalent computed from the device sphere and cylinder: SE = sphere + cylinder/2.", "D", "float", "refraction", "", "", "", ""),
 "OPDWFvalidZone": ("Diameter of the validated wavefront-analysis zone reported by the device.", "mm", "float", "quality", "", "", "", ""),
 "OPDfitError": ("Residual error of the device wavefront fit (all orders); larger = poorer fit.", "(device units)", "float", "quality", "", "", "", ""),
 "HO_OPDwfError": ("Higher-order residual error of the device wavefront fit; larger = poorer fit.", "(device units)", "float", "quality", "", "", "", ""),
 "OPDzone": ("Pupil diameter over which the device exported the Zernike expansion (measurement zone).", "mm", "float", "quality", "", "", "", ""),
 "OPDsph": ("Objective sphere reported by the device (negative-cylinder convention).", "D", "float", "refraction", "", "", "", ""),
 "OPDcyl": ("Objective cylinder reported by the device (negative-cylinder convention).", "D", "float", "refraction", "", "", "", ""),
 "OPDaxis": ("Objective cylinder axis reported by the device.", "deg", "float", "refraction", "", "", "", ""),
 "Exam": ("Device examination index for the acquisition (integer; used to order repeat examinations of the same eye).", "", "integer", "metadata", "", "", "", ""),
 "CLASSIFInrm": ("Klyce/Maeda automated classifier score (0–100) for the Normal (NRM) category.", "0–100", "float", "topography_classifier", "", "", "", ""),
 "CLASSIFIast": ("Klyce/Maeda automated classifier score (0–100) for the Regular-astigmatism (AST) category.", "0–100", "float", "topography_classifier", "", "", "", ""),
 "CLASSIFIkcs": ("Klyce/Maeda automated classifier score (0–100) for the Keratoconus-suspect (KCS) category.", "0–100", "float", "topography_classifier", "", "", "", ""),
 "CLASSIFIkc": ("Klyce/Maeda automated classifier score (0–100) for the Keratoconus (KC) category.", "0–100", "float", "topography_classifier", "", "", "", ""),
 "CLASSIFIpmd": ("Klyce/Maeda automated classifier score (0–100) for the Pellucid-marginal-degeneration (PMD) category.", "0–100", "float", "topography_classifier", "", "", "", ""),
 "CLASSIFIpkp": ("Klyce/Maeda automated classifier score (0–100) for the post-penetrating-keratoplasty (PKP) category.", "0–100", "float", "topography_classifier", "", "", "", ""),
 "CLASSIFImrs": ("Klyce/Maeda automated classifier score (0–100) for the post-myopic-refractive-surgery (MRS) category.", "0–100", "float", "topography_classifier", "", "", "", ""),
 "CLASSIFIhrs": ("Klyce/Maeda automated classifier score (0–100) for the post-hyperopic-refractive-surgery (HRS) category.", "0–100", "float", "topography_classifier", "", "", "", ""),
 "CLASSIFIoth": ("Klyce/Maeda automated classifier score (0–100) for the Other/unclassified (OTH) category.", "0–100", "float", "topography_classifier", "", "", "", ""),
 "CLASSIFIksi": ("Klyce/Maeda keratoconus-severity index (KSI); a severity INDEX (not a 0–100 category score), and therefore excluded from the topography argmax.", "index", "float", "topography_classifier", "", "", "", ""),
}


def _zern_nm(k):
    out = []
    for n in range(9):
        for j in range(n + 1):
            out.append((n, 2 * j - n))
    return out[k]


def build_rich_dictionary(columns):
    """Return a DataFrame (one row per column, same order) with the 9 fields."""
    rows = []
    for c in columns:
        if c in _META:
            d, u, t, g, b, n, m, p = _META[c]; rows.append([c, d, u, t, g, b, n, m, p]); continue
        mz = re.match(r"^OPDc(\d\d)$", c)
        if mz:
            k = int(mz.group(1)); n, m = _zern_nm(k); nm = _MODE.get((n, m), "")
            rows.append([c, f"Device-exported OSA/ANSI Zernike coefficient (index {k}, n={n}, m={m}{', ' + nm if nm else ''}), over the validated zone.", "µm", "float", "zernike", "Zernike", n, m, ""]); continue
        p = re.search(r"_p(\d)mm", c); pd_ = p.group(1) if p else ""
        bm = re.search(r"_p\dmm_(GM|VLVH|TILDE|npvVT|npvV)_", c); base = bm.group(1) if bm else ""
        nmm = re.search(r"_n(\d+)_m(-?\d+)$", c); n = nmm.group(1) if nmm else ""; m = nmm.group(2) if nmm else ""
        nm = _MODE.get((int(n), int(m)), "") if n != "" else ""

        def R(desc, unit, typ, grp):
            rows.append([c, desc, unit, typ, grp, _BASE_FULL.get(base, base), n, m, pd_])
        if re.search(r"_(GM|VLVH|TILDE|npvV|npvVT)_n\d+_m-?\d+$", c):
            R(f"{_BASE_FULL.get(base, base)} coefficient (n={n}, m={m}{', ' + nm if nm else ''}) at the {pd_} mm pupil.", _BASE_UNIT.get(base, "D"), "float", "coefficient")
        elif "_PTVaxis_" in c:
            R(f"Orientation (axis) of the peak-to-valley normalised {base} mode (n={n}, m={m}{', ' + nm if nm else ''}) at {pd_} mm; equals the scale-invariant axis, reduced to [0,360/|m|).", "deg", "float", "axis")
        elif "_PTVmag_" in c:
            R(f"Peak-to-valley normalised vector magnitude of the {base} oriented mode (n={n}, m={m}{', ' + nm if nm else ''}) at {pd_} mm.", "D", "float", "magnitude")
        elif "_PTVscal_" in c:
            R(f"Peak-to-valley normalised coefficient of the rotationally symmetric (m=0) {base} mode (n={n}, m={m}) at {pd_} mm, in diopters (identical to the corresponding npv coefficient).", "D", "float", "pv_coefficient")
        elif "_axis_" in c:
            R(f"Orientation (axis) of the {base} oriented mode (n={n}, m={m}{', ' + nm if nm else ''}) at {pd_} mm; (1/m)·atan2(sin,cos), reduced to [0,360/|m|).", "deg", "float", "axis")
        elif "_mag_" in c:
            R(f"Vector magnitude of the {base} oriented mode (n={n}, m={m}{', ' + nm if nm else ''}) at {pd_} mm.", _BASE_UNIT.get(base, "D"), "float", "magnitude")
        else:
            mref = re.search(r"_(VLVH|TILDE)_(S|C|A)(_v12)?$", c)
            if mref:
                base = mref.group(1); comp = mref.group(2); v12 = mref.group(3)
                plane = "spectacle plane (12 mm vertex)" if v12 else "corneal plane"
                name = {"S": "sphere", "C": "cylinder", "A": "axis"}[comp]
                unit = "deg" if comp == "A" else "D"
                rows.append([c, f"Refraction {name} derived from the low-order {base} vergence, at the {plane}, {pd_} mm pupil.", unit, "float", "refraction", _BASE_FULL.get(base, base), "2", "", pd_]); continue
            mstat = re.search(r"_(VLVH|TILDE)_(TOT|VL|VH)_(box_max|box_min|box_median|box_q1|box_q3|max|min|mean|std|rms)$", c)
            if mstat:
                base = mstat.group(1); comp = mstat.group(2); stat = mstat.group(3)
                compname = {"TOT": "total vergence map", "VL": "low-order vergence map", "VH": "high-order vergence map"}[comp]
                rows.append([c, f"Distribution statistic ({stat.replace('box_', '')}) of the {compname} ({base} basis) over the {pd_} mm pupil.", "D", "float", "map_statistic", _BASE_FULL.get(base, base), "", "", pd_]); continue
            rows.append([c, "(derived quantity; see Methods).", "", "float", "derived", _BASE_FULL.get(base, base), n, m, pd_])
    return pd.DataFrame(rows, columns=["column", "description", "units", "type", "group", "basis", "n", "m", "pupil_mm"])
