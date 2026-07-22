#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce_from_release.py  --  INDEPENDENT reproduction check (public data only)

Recomputes EVERY derived column family directly from the public Zernike columns
(OPDc00..OPDc44) of opd_wavefront_release.csv, for every pupil diameter, using
the self-contained converter zernike_to_vergence.py, and compares them to the
released calc_* columns. Requires NO private data.

Families covered: coefficients (GM, VLVH, TILDE, npvV, npvVT); oriented-mode
magnitude and axis (mag/axis) and their peak-to-valley counterparts
(PTVmag/PTVaxis); peak-to-valley scaling factors (PTVscal); spherocylindrical
refraction S/C/A at the corneal and 12 mm spectacle plane (_v12); and the
per-map distribution statistics of the total/low/high vergence maps
(TOT/VL/VH box_* and the VH min/max/mean/std/rms).

Usage:
    python reproduce_from_release.py --csv ../opd_wavefront_release.csv --n 20
Numeric families match to ~1e-9; axis families match circularly within the
1-degree rounding used in the release.
"""
import argparse, importlib.util, os, re, sys, math
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("z2v", os.path.join(HERE, "zernike_to_vergence.py"))
tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
NM = tr.pyramid_indices(8); R, T, MASK = tr.build_polar_grid()
BASE_DIAM = 6.0; DIAMS = (6.0, 5.0, 4.0, 3.0)


def compute_eye(zvec, d):
    z = {NM[k]: zvec[k] for k in range(45)}
    if abs(d - BASE_DIAM) > 1e-9:
        z = tr.apply_subpupil_transform(z, S1_mm=BASE_DIAM, S2_mm=d, n_max=8)
    gm = tr.project_zernike_to_gm(z, n_max=8); gmn = dict(gm)
    for k in [(0, 0), (1, -1), (1, 1)]: gmn[k] = 0.0
    W = tr.wavefront_from_gm(gmn)
    V = tr.compute_vergence(W, d / 2.0, R, T, MASK)
    V = np.nan_to_num(V, nan=0.0); V = np.where(MASK, V, np.nan)
    a, vl_map, vh_map = tr.decompose_VL_VH(np.where(MASK, V, 0.0), d, R, T, MASK)
    at, vl_t, vh_t = tr.decompose_tilde(np.where(MASK, V, 0.0), R, T, MASK)
    npvV = tr.scale_alpha_by_PTV(a, tr.V_PTV); npvVT = tr.scale_alpha_by_PTV(at, tr.VT_PTV)
    coef = {"GM": gm, "VLVH": a, "TILDE": at, "npvV": npvV, "npvVT": npvVT}
    alpha = {"VLVH": a, "TILDE": at, "npvV": npvV, "npvVT": npvVT}
    ptvfac = {"VLVH": tr.V_PTV, "TILDE": tr.VT_PTV}
    S, C, A = tr.extract_sca(a); Sv, Cv, Av = tr.vertex_convert_sca(S, C, A, vertex_m=tr.VERTEX_M)
    VT = tr.build_Vtilde_basis(); tot = np.zeros_like(R)
    for nm in tr.VERGENCE_MODES: tot = tot + at[nm] * VT[nm](R, T)
    tot = np.where(MASK, tot, np.nan)
    c2m2 = tr._projection_lsq(tot, VT[(2, -2)], R, T, MASK, tr.GRID_NR, tr.GRID_NT)
    c20 = tr._projection_lsq(tot, VT[(2, 0)], R, T, MASK, tr.GRID_NR, tr.GRID_NT)
    c22 = tr._projection_lsq(tot, VT[(2, 2)], R, T, MASK, tr.GRID_NR, tr.GRID_NT)
    amp = math.hypot(c2m2, c22); St = c20 + math.sqrt(2) * amp; Ct = -2 * math.sqrt(2) * amp
    At = math.degrees(0.5 * math.atan2(c2m2, c22)) % 180.0
    Stv, Ctv, Atv = tr.vertex_convert_sca(St, Ct, float(round(At)), vertex_m=tr.VERTEX_M)
    refr = {"VLVH": dict(S=S, C=C, A=round(A) % 180, S_v12=Sv, C_v12=Cv, A_v12=round(Av) % 180),
            "TILDE": dict(S=St, C=Ct, A=round(At) % 180, S_v12=Stv, C_v12=Ctv, A_v12=round(Atv) % 180)}
    maps = {"VLVH": {"TOT": tr.map_stats(np.where(MASK, V, np.nan), remove_origin=True),
                     "VL": tr.map_stats(vl_map), "VH": tr.map_stats(vh_map)},
            "TILDE": {"TOT": tr.map_stats(np.where(MASK, V, np.nan), remove_origin=True),
                      "VL": tr.map_stats(vl_t), "VH": tr.map_stats(vh_t)}}
    return dict(coef=coef, alpha=alpha, ptvfac=ptvfac, refr=refr, maps=maps)


def reproduce_column(col, C):
    m = re.match(r"^calc_p(\d)mm_(GM|VLVH|TILDE|npvVT|npvV)_(.+)$", col)
    if not m: return None
    base, rest = m.group(2), m.group(3)
    def nm(s):
        g = re.match(r"^n(\d+)_m(-?\d+)$", s); return (int(g.group(1)), int(g.group(2))) if g else None
    g = nm(rest)
    if g is not None and base in C["coef"]:
        return ("num", C["coef"][base].get(g))
    for kind in ("PTVmag", "PTVaxis", "mag", "axis"):
        if rest.startswith(kind + "_"):
            g = nm(rest[len(kind) + 1:])
            if g is None: return None
            n, mm = g
            src = C["alpha"][base] if kind in ("mag", "axis") else C["alpha"]["npvV" if base == "VLVH" else "npvVT"]
            mag, ax = tr.magnitude_axis(src, n, abs(mm))
            return ("axis" if "axis" in kind else "num", ax if "axis" in kind else mag)
    if rest.startswith("PTVscal_"):
        # PTVscal_n{n}_m0 is the peak-to-valley normalised coefficient of the
        # rotationally symmetric (m=0) mode, i.e. the npvV/npvVT value itself.
        g = nm(rest[len("PTVscal_"):])
        if g is None: return None
        src = C["alpha"]["npvV" if base == "VLVH" else "npvVT"]
        return ("num", src.get(g))
    if rest in ("S", "C", "A", "S_v12", "C_v12", "A_v12"):
        return ("axis" if rest.startswith("A") else "num", C["refr"][base][rest])
    mm = re.match(r"^(TOT|VL|VH)_(box_)?(min|q1|median|q3|max|mean|std|rms)$", rest)
    if mm and base in C["maps"]:
        return ("num", C["maps"][base][mm.group(1)][mm.group(3)])
    return None


def circ_err(a, b, period):
    d = abs((float(a) - float(b)) % period); return min(d, period - d)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--n", type=int, default=20)
    A = ap.parse_args()
    df = pd.read_csv(A.csv)
    calc_cols = [c for c in df.columns if c.startswith("calc_")]
    fam_err, fam_cnt = {}, {}; n_unmapped = 0; unmapped_ex = set()
    for i in df.index[:A.n]:
        zvec = [float(df.loc[i, f"OPDc{k:02d}"]) for k in range(45)]; percache = {}
        for c in calc_cols:
            d = int(re.match(r"^calc_p(\d)mm_", c).group(1))
            if d not in percache: percache[d] = compute_eye(zvec, float(d))
            rep = reproduce_column(c, percache[d])
            if rep is None:
                n_unmapped += 1
                if len(unmapped_ex) < 8: unmapped_ex.add(re.sub(r"p\dmm", "pXmm", c))
                continue
            kind, val = rep
            if val is None or pd.isna(df.loc[i, c]): continue
            rel = float(df.loc[i, c])
            fam = re.sub(r"(VLVH|TILDE|GM|npvVT|npvV)", "BASE", re.sub(r"p\dmm", "pX", re.sub(r"_n\d+_m-?\d+", "_nm", c)))
            if kind == "axis":
                mo = re.search(r"_m(-?\d+)$", c)
                period = 180.0 if (fam.endswith("_A") or fam.endswith("_A_v12")) else (360.0 / abs(int(mo.group(1))) if mo else 180.0)
                e = circ_err(val, rel, period)
            else:
                e = abs(float(val) - rel)
            fam_err[fam] = max(fam_err.get(fam, 0.0), e); fam_cnt[fam] = fam_cnt.get(fam, 0) + 1
    print(f"Reproduction on {min(A.n, len(df))} eyes x {len(DIAMS)} pupils\n{'family':46s}{'n':>7}{'max|diff|':>13}")
    for fam in sorted(fam_err):
        print(f"  {fam:44s}{fam_cnt[fam]:>7}{fam_err[fam]:>13.2e}")
    numeric = [v for f, v in fam_err.items() if not ("axis" in f or f.endswith("_A") or f.endswith("_A_v12"))]
    axisfam = [v for f, v in fam_err.items() if ("axis" in f or f.endswith("_A") or f.endswith("_A_v12"))]
    nw = max(numeric or [0.0]); aw = max(axisfam or [0.0])
    print(f"\nFamilies reproduced: {len(fam_err)}; unmapped instances: {n_unmapped} {sorted(unmapped_ex)}")
    print(f"max |diff| numeric families = {nw:.2e}; max axis error = {aw:.2f} deg")
    ok = nw < 1e-6 and aw <= 1.0 and n_unmapped == 0
    print("RESULT:", "PASS (all calc_* families reproduced from public Zernike)" if ok else
          ("PARTIAL" if n_unmapped else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
