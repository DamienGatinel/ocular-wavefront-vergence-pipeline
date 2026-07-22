#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
                ZERNIKE -> VERGENCE PIPELINE  (v1)
================================================================================

Purpose
-------
Single-case pipeline that converts a measured Zernike wavefront expansion  of
radial order n = 8 (45 OSA/ANSI coefficients, pyramid-ordered) into:

    (1) Gatinel-Malet (GM) coefficients of the same order (45 values).
    (2) Vergence maps (Total / V_L / V_H) in diopters, computed from the
        wavefront via dW/dr / (rho * R^2).
    (3) Vergence-basis coefficients in two complementary decompositions:
            - VL / VH (non-orthogonal): V_L is fitted on a fixed *central*
              physical region; V_H is the remainder.
            - V-tilde (orthogonal):    full-disk orthogonal basis.
    (4) PTV-normalized variants of both bases (npvV and npvVT).
    (5) Spherocylindrical refraction (S, C, A) at the corneal plane and
        at the spectacle plane (12 mm vertex).
    (6) A complete set of publication-quality figures (pyramids,
        half-pyramids, histograms, wavefront triptychs, vergence triptychs).

Input format
------------
A "standard pyramid" Zernike vector (45 coefficients, microns), ordered
top-to-bottom and left-to-right starting from piston:

    n = 0 :  (0,  0)
    n = 1 :  (1, -1), (1,  1)
    n = 2 :  (2, -2), (2,  0), (2,  2)
    n = 3 :  (3, -3), (3, -1), (3,  1), (3,  3)
    n = 4 :  (4, -4), (4, -2), (4,  0), (4,  2), (4,  4)
    n = 5 :  (5, -5), (5, -3), (5, -1), (5,  1), (5,  3), (5,  5)
    n = 6 :  (6, -6), (6, -4), (6, -2), (6,  0), (6,  2), (6,  4), (6,  6)
    n = 7 :  (7, -7), (7, -5), (7, -3), (7, -1), (7,  1), (7,  3), (7,  5), (7,  7)
    n = 8 :  (8, -8), (8, -6), (8, -4), (8, -2), (8,  0), (8,  2), (8,  4), (8,  6), (8,  8)

This is the OSA/ANSI single-index ordering. The pipeline accepts either:

    * a CSV with one coefficient per line (45 lines, in pyramid order); or
    * a 3-column CSV with header `n,m,coeff` (any row order; missing modes
      default to zero).

Usage
-----
    python tracey_zernike_to_vergence_v1.py                       \
        --zernike  zernike_case01.csv                             \
        --diameter-mm 6.0                                         \
        --case-id  Case01                                         \
        --output-dir ./out_case01                                 \
        [--no-figures]

The script produces:

    out_case01/
        Case01_coefficients.csv     # GM, VL/VH, tilde, npv*, refractions
        Case01_summary.txt          # human-readable summary
        *.png                       # all figures (see README inside)

Hooks for future extensions
---------------------------
The code is organized so that two extensions can be plugged in cleanly later
without touching the math:

    * Batch processing of many cases (see `process_batch`, currently a stub).
    * Multi-diameter pupil sweep using the analytic sub-pupil Zernike
      transform (see Section 17 below).

Conventions
-----------
    * Angle convention: theta measured counter-clockwise from the +x axis.
    * Wavefront unit:   microns (um).
    * Vergence unit:    diopters (D).
    * (n, +m) -> coefficient of  cos(m*theta)
    * (n, -m) -> coefficient of  sin(m*theta)
    * Cylinder is reported with negative sign convention (S, C, A).

Author / Maintainer
-------------------
Adapted from my initial OPDscan pipeline by D. Gatinel a
"""

# =============================================================================
# SECTION 2 -- IMPORTS AND GLOBAL CONFIGURATION
# =============================================================================
# All third-party imports are grouped here so that the rest of the file reads
# top-down. Matplotlib is forced to the non-interactive 'Agg' backend so the
# script can run head-less (no X server required) on the iTrace instrument.
# =============================================================================

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
import mpmath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

# -----------------------------------------------------------------------------
# Global physical / numerical configuration
# -----------------------------------------------------------------------------
# Maximum radial order supported by this pipeline.
N_MAX = 8

# Number of OSA Zernike coefficients up to order N_MAX (triangular layout).
# = sum_{n=0}^{N_MAX} (n + 1) = (N_MAX + 1)(N_MAX + 2)/2
N_COEFFS = (N_MAX + 1) * (N_MAX + 2) // 2          # 45 for N_MAX = 8

# Spectacle-plane vertex distance (m). Used to convert corneal-plane refraction
# (S, C, A) to the spectacle plane.
VERTEX_M = 0.012                                    # 12 mm

# VL fit configuration.
#
# V_L (low-degree, n = 2) is fitted on a central region whose *physical* radius
# is constant across pupil diameters.  By default the central radius equals
# 20 % of the 6 mm reference pupil radius (= 0.6 mm), independent of the
# diameter actually being analyzed.
#
# This keeps the local refraction estimate "anchored" near the pupil apex.
VL_REF_DIAMETER_MM     = 6.0
VL_CENTER_FRACTION_REF = 0.20      # -> 0.6 mm fit radius on a 6 mm pupil

# Polar grid resolution used when projecting onto vergence bases and when
# rendering maps.
GRID_NR = 512                       # radial samples
GRID_NT = 1024                      # angular samples

# Pyramid figure grid (used for the schematic Zernike-like layouts).
PYRAMID_NPT = 300

# High-precision context for the exact Zernike -> GM disk-moment projection.
MP_DPS = 80                         # mpmath decimal precision

# Default figure DPI for publication-quality export.
SAVE_DPI = 300
plt.rcParams["savefig.dpi"] = SAVE_DPI
plt.rcParams["figure.dpi"]  = SAVE_DPI

# Sympy symbols used throughout for symbolic wavefront expressions.
R_SYM, T_SYM = sp.symbols("r t", real=True)


# =============================================================================
# SECTION 3 -- ZERNIKE INDEXING (PYRAMID ORDER)
# =============================================================================
# Helpers for converting between the pyramid (n, m) layout and the single
# index k used in the input CSV.  The order is OSA/ANSI:
#
#    k = 0, 1, ..., N_COEFFS - 1 traverses the pyramid row by row,
#    left to right, with m running from -n to +n in steps of 2.
# =============================================================================

def pyramid_indices(n_max: int = N_MAX) -> List[Tuple[int, int]]:
    """Return the list of (n, m) pairs in standard pyramid order.

    Example for n_max = 2:
        [(0,0), (1,-1), (1,1), (2,-2), (2,0), (2,2)]
    """
    out: List[Tuple[int, int]] = []
    for n in range(n_max + 1):
        for j in range(n + 1):
            out.append((n, 2 * j - n))
    return out


def index_to_nm(k: int, n_max: int = N_MAX) -> Tuple[int, int]:
    """Convert a flat pyramid index k -> (n, m)."""
    return pyramid_indices(n_max)[k]


def nm_to_index(n: int, m: int, n_max: int = N_MAX) -> int:
    """Convert (n, m) -> flat pyramid index k."""
    return pyramid_indices(n_max).index((n, m))


def vector_to_dict(coeffs: List[float], n_max: int = N_MAX) -> Dict[Tuple[int, int], float]:
    """Convert a 45-vector (pyramid order) into a {(n, m): value} dict."""
    nm = pyramid_indices(n_max)
    if len(coeffs) != len(nm):
        raise ValueError(
            f"Expected {len(nm)} coefficients (n_max={n_max}); got {len(coeffs)}."
        )
    return {nm[k]: float(coeffs[k]) for k in range(len(nm))}


def dict_to_vector(d: Dict[Tuple[int, int], float], n_max: int = N_MAX) -> List[float]:
    """Convert a {(n, m): value} dict into a 45-vector (pyramid order)."""
    return [float(d.get(nm, 0.0)) for nm in pyramid_indices(n_max)]


# =============================================================================
# SECTION 4 -- SYMBOLIC BASIS: ZERNIKE AND GATINEL-MALET (GM)
# =============================================================================
# Polar definitions used to build the total wavefront expression W(r, theta)
# in microns.  These are the SAME functional forms as the original OPDscan
# pipeline and are kept unchanged for cross-validation.
#
#   * Zernike Z(n, m) uses OSA/ANSI orthonormal normalization.
#   * Azimuthal A(n, m) is the pure r^n * trig(m*theta) form (no lower-order
#     radial terms).
#   * GM(n, m) is defined piecewise (see comments).
# =============================================================================

def zernike_sympy(n: int, m: int):
    """OSA/ANSI orthonormal Zernike polynomial Z(n, m) as a sympy expression."""
    r, t = R_SYM, T_SYM

    def radial(mm: int):
        """Radial polynomial R(n, |m|)(r)."""
        s_var = sp.symbols("s", integer=True, nonnegative=True)
        nm_minus = (n - mm) // 2
        nm_plus  = (n + mm) // 2
        if nm_minus < 0:
            return 0
        return sp.Sum(
            (-1) ** s_var * sp.factorial(n - s_var) / (
                sp.factorial(s_var)
                * sp.factorial(nm_plus  - s_var)
                * sp.factorial(nm_minus - s_var)
            ) * r ** (n - 2 * s_var),
            (s_var, 0, nm_minus),
        ).doit()

    if m > 0:
        z = sp.sqrt(2 * (n + 1)) * radial(m) * sp.cos(m * t)
    elif m < 0:
        z = sp.sqrt(2 * (n + 1)) * radial(-m) * sp.sin(-m * t)
    else:
        z = sp.sqrt(n + 1) * radial(0)
    return sp.expand(z, trig=True)


def azimuthal_sympy(n: int, m: int):
    """Pure azimuthal mode A(n, m) = sqrt(...) * r^n * trig(m * theta)."""
    r, t = R_SYM, T_SYM
    if m > 0:
        return sp.sqrt(2 * (n + 1)) * r ** n * sp.cos(m * t)
    if m < 0:
        return sp.sqrt(2 * (n + 1)) * r ** n * sp.sin(-m * t)
    return sp.sqrt(n + 1) * r ** n


# -----------------------------------------------------------------------------
# Special GM "ladder" polynomials (n = 5, 6, 7, 8).
# These are pre-normalized radial polynomials shared by the cos / sin halves
# of GM at orders 5, 6, 7, 8.  See Gatinel & Malet original publications.
# -----------------------------------------------------------------------------
_GI = 5 * (R_SYM ** 5 - sp.Rational(4, 5) * R_SYM ** 3) * sp.sqrt(6)
_GP = 6 * (R_SYM ** 6 - sp.Rational(5, 6) * R_SYM ** 4) * sp.sqrt(7)
_GJ = (21 * R_SYM ** 7 - 30 * R_SYM ** 5 + 10 * R_SYM ** 3) * sp.sqrt(8)
_GQ = (28 * R_SYM ** 8 - 42 * R_SYM ** 6 + 15 * R_SYM ** 4) * sp.sqrt(9)


def gm_sympy(n: int, m: int):
    """Gatinel-Malet mode GM(n, m) as a sympy expression in (r, theta).

    Definition (consistent with the OPDscan reference pipeline):

        * n in {0, 1, 2}              -> GM(n, m) = Z(n, m)
        * n = 3 and |m| <= 1          -> GM(n, m) = A(n, m)   (azimuthal)
        * n = 4 and |m| <= 2          -> GM(n, m) = A(n, m)
        * n = 5 and |m| = 1           -> sqrt(2) * GI * trig(theta)
        * n = 6 and m in {-2, 0, 2}   -> sqrt(2) * GP * trig(2 theta)  (or GP for m=0)
        * n = 7 and |m| = 1           -> sqrt(2) * GJ * trig(theta)
        * n = 8 and m in {-2, 0, 2}   -> sqrt(2) * GQ * trig(2 theta)  (or GQ for m=0)
        * Otherwise                   -> GM(n, m) = Z(n, m)
    """
    r, t = R_SYM, T_SYM

    if n <= 2:
        return zernike_sympy(n, m)

    if n == 3:
        if abs(m) <= 1:
            return sp.expand(azimuthal_sympy(n, m), trig=True)
        return zernike_sympy(n, m)

    if n == 4:
        if abs(m) <= 2:
            return sp.expand(azimuthal_sympy(n, m), trig=True)
        return zernike_sympy(n, m)

    if n == 5:
        if m == -1:
            return sp.expand(sp.sqrt(2) * _GI * sp.sin(t), trig=True)
        if m ==  1:
            return sp.expand(sp.sqrt(2) * _GI * sp.cos(t), trig=True)
        return zernike_sympy(n, m)

    if n == 6:
        if m == -2:
            return sp.expand(sp.sqrt(2) * _GP * sp.sin(2 * t), trig=True)
        if m ==  2:
            return sp.expand(sp.sqrt(2) * _GP * sp.cos(2 * t), trig=True)
        if m ==  0:
            return sp.expand(_GP, trig=True)
        return zernike_sympy(n, m)

    if n == 7:
        if m == -1:
            return sp.expand(sp.sqrt(2) * _GJ * sp.sin(t), trig=True)
        if m ==  1:
            return sp.expand(sp.sqrt(2) * _GJ * sp.cos(t), trig=True)
        return zernike_sympy(n, m)

    if n == 8:
        if m == -2:
            return sp.expand(sp.sqrt(2) * _GQ * sp.sin(2 * t), trig=True)
        if m ==  2:
            return sp.expand(sp.sqrt(2) * _GQ * sp.cos(2 * t), trig=True)
        if m ==  0:
            return sp.expand(_GQ, trig=True)
        return zernike_sympy(n, m)

    return zernike_sympy(n, m)


def wavefront_from_zernike(z_dict: Dict[Tuple[int, int], float], n_max: int = N_MAX):
    """Build symbolic W(r, theta) = sum c_{n,m} * Z(n, m), in microns."""
    expr = sp.Integer(0)
    for (n, m), c in z_dict.items():
        if n > n_max:
            continue
        expr += sp.Float(c) * zernike_sympy(n, m)
    return sp.expand(expr)


def wavefront_from_gm(gm_dict: Dict[Tuple[int, int], float], n_max: int = N_MAX,
                     n_min: int = 0):
    """Build symbolic W(r, theta) from GM coefficients between n_min and n_max."""
    expr = sp.Integer(0)
    for (n, m), c in gm_dict.items():
        if n < n_min or n > n_max:
            continue
        expr += sp.Float(c) * gm_sympy(n, m)
    return sp.expand(expr)


# =============================================================================
# SECTION 5 -- EXACT ZERNIKE -> GM PROJECTION (DISK MOMENTS, NO QUADRATURE)
# =============================================================================
# The GM basis is NOT orthogonal in the usual L2 sense over the unit disk.
# Direct projection therefore requires solving a Gram-matrix linear system.
#
# To avoid any numerical quadrature, we represent every Z and GM mode as a
# 2D Cartesian polynomial and use the closed-form formula for the unit-disk
# moment of x^px * y^py:
#
#       (1/pi) * integral_{disk} x^px y^py dA
#         = 0                                            if px or py odd
#         = factorial(2i) * factorial(2j)
#           / (4^(i+j) * fact(i) * fact(j) * fact(i+j+1))
#       with i = px / 2, j = py / 2.
#
# The decomposition mirrors the OPDscan pipeline:
#
#       low  = wavefront restricted to total monomial degree <= 2
#       high = remaining terms (degree >= 3)
#
#       low  is projected onto Zernike modes (n <= 2)  (== GM at low order)
#       high is projected onto GM modes      (n >= 3)
# =============================================================================

_DISK_MOMENT_CACHE: Dict[Tuple[int, int], "mpmath.mpf"] = {}
_ZERN_POLY_CACHE:   Dict[int, Dict[Tuple[int, int], Dict[Tuple[int, int], "mpmath.mpf"]]] = {}
_GM_POLY_CACHE:     Dict[int, Dict[Tuple[int, int], Dict[Tuple[int, int], "mpmath.mpf"]]] = {}


def _disk_moment_over_pi(px: int, py: int) -> "mpmath.mpf":
    """Closed-form unit-disk moment of x^px * y^py, divided by pi."""
    key = (px, py)
    if key in _DISK_MOMENT_CACHE:
        return _DISK_MOMENT_CACHE[key]
    if (px % 2) or (py % 2):
        out = mpmath.mpf("0")
    else:
        i = px // 2
        j = py // 2
        num = mpmath.mpf(math.factorial(2 * i) * math.factorial(2 * j))
        den = mpmath.mpf(
            (4 ** (i + j)) * math.factorial(i) * math.factorial(j) * math.factorial(i + j + 1)
        )
        out = num / den
    _DISK_MOMENT_CACHE[key] = out
    return out


def _zernike_cartesian_sympy(n: int, m: int, x_sym, y_sym):
    """Build Z(n, m) as a Cartesian polynomial in x, y (real OSA convention)."""
    m_abs = abs(m)
    if m_abs > n or ((n - m_abs) % 2):
        return sp.Integer(0)

    rho2 = x_sym ** 2 + y_sym ** 2

    if m_abs == 0:
        ang_poly = sp.Integer(1)
    else:
        z = (x_sym + sp.I * y_sym) ** m_abs
        ang_poly = sp.expand(sp.re(z) if m >= 0 else sp.im(z))

    kmax = (n - m_abs) // 2
    expr = sp.Integer(0)
    for k in range(kmax + 1):
        coeff = ((-1) ** k) * sp.factorial(n - k) / (
            sp.factorial(k)
            * sp.factorial((n + m_abs) // 2 - k)
            * sp.factorial((n - m_abs) // 2 - k)
        )
        p = n - 2 * k
        power = (p - m_abs) // 2
        expr += coeff * (rho2 ** power) * ang_poly

    norm = sp.sqrt(n + 1) if m == 0 else sp.sqrt(2 * (n + 1))
    return sp.expand(norm * expr)


def _poly_from_radial_terms(m: int, radial_terms: List[Tuple[int, sp.Expr]], x_sym, y_sym):
    """Build Cartesian polynomial   sum c_k * rho^k * trig(m * theta)."""
    m_abs = abs(m)
    rho2 = x_sym ** 2 + y_sym ** 2

    if m_abs == 0:
        ang_poly = sp.Integer(1)
    else:
        z = (x_sym + sp.I * y_sym) ** m_abs
        ang_poly = sp.expand(sp.re(z) if m >= 0 else sp.im(z))

    expr = sp.Integer(0)
    for power, coeff in radial_terms:
        power = int(power)
        if power < m_abs or ((power - m_abs) % 2):
            raise ValueError(f"Incompatible radial power {power} for m={m}.")
        expr += coeff * (rho2 ** ((power - m_abs) // 2)) * ang_poly
    return sp.expand(expr)


def _gm_cartesian_sympy(n: int, m: int, x_sym, y_sym):
    """Build GM(n, m) as a Cartesian polynomial in x, y (real OSA convention)."""
    if n <= 2:
        return _zernike_cartesian_sympy(n, m, x_sym, y_sym)

    # Pure azimuthal (n^th-degree radial only)
    if n == 3 and abs(m) <= 1:
        norm = sp.sqrt(2 * (n + 1)) if m != 0 else sp.sqrt(n + 1)
        return _poly_from_radial_terms(m, [(n, norm)], x_sym, y_sym)
    if n == 4 and abs(m) <= 2:
        norm = sp.sqrt(2 * (n + 1)) if m != 0 else sp.sqrt(n + 1)
        return _poly_from_radial_terms(m, [(n, norm)], x_sym, y_sym)

    # GI (n=5, |m|=1) : sqrt(2)*GI*trig, GI = sqrt(6)*(5 r^5 - 4 r^3)
    if n == 5 and abs(m) == 1:
        c = sp.sqrt(12)
        return _poly_from_radial_terms(m, [(5,  5 * c), (3, -4 * c)], x_sym, y_sym)

    # GP (n=6, m in {-2,0,2}) : GP = sqrt(7)*(6 r^6 - 5 r^4)
    if n == 6 and (m == 0 or abs(m) == 2):
        c = sp.sqrt(7) if m == 0 else sp.sqrt(14)
        return _poly_from_radial_terms(m, [(6,  6 * c), (4, -5 * c)], x_sym, y_sym)

    # GJ (n=7, |m|=1) : sqrt(2)*GJ*trig, GJ = sqrt(8)*(21 r^7 - 30 r^5 + 10 r^3)
    if n == 7 and abs(m) == 1:
        return _poly_from_radial_terms(m, [(7, 84), (5, -120), (3, 40)], x_sym, y_sym)

    # GQ (n=8, m in {-2,0,2}) : GQ = sqrt(9)*(28 r^8 - 42 r^6 + 15 r^4) = 3 * (...)
    if n == 8 and (m == 0 or abs(m) == 2):
        if m == 0:
            return _poly_from_radial_terms(m, [(8, 84), (6, -126), (4, 45)], x_sym, y_sym)
        c = sp.sqrt(2)
        return _poly_from_radial_terms(m, [(8, 84 * c), (6, -126 * c), (4, 45 * c)], x_sym, y_sym)

    # Default: standard Zernike
    return _zernike_cartesian_sympy(n, m, x_sym, y_sym)


def _poly_terms_dict(expr: sp.Expr, x_sym, y_sym, dps: int) -> Dict[Tuple[int, int], "mpmath.mpf"]:
    """Convert a sympy Cartesian polynomial into {(px, py): mpf coefficient}."""
    poly = sp.Poly(expr, x_sym, y_sym)
    out: Dict[Tuple[int, int], "mpmath.mpf"] = {}
    for (px, py), coeff in poly.terms():
        out[(px, py)] = mpmath.mpf(str(sp.N(coeff, dps)))
    return out


def _get_zernike_poly_terms(n_max: int, dps: int = MP_DPS):
    """Pre-compute Cartesian polynomial terms for all Z(n, m), n <= n_max."""
    if n_max in _ZERN_POLY_CACHE:
        return _ZERN_POLY_CACHE[n_max]
    old_dps = mpmath.mp.dps
    mpmath.mp.dps = dps
    x_sym, y_sym = sp.symbols("x y", real=True)
    out: Dict[Tuple[int, int], Dict[Tuple[int, int], "mpmath.mpf"]] = {}
    for n in range(n_max + 1):
        for m in range(-n, n + 1, 2):
            out[(n, m)] = _poly_terms_dict(
                _zernike_cartesian_sympy(n, m, x_sym, y_sym), x_sym, y_sym, dps
            )
    _ZERN_POLY_CACHE[n_max] = out
    mpmath.mp.dps = old_dps
    return out


def _get_gm_poly_terms(n_max: int, dps: int = MP_DPS):
    """Pre-compute Cartesian polynomial terms for all GM(n, m), n <= n_max."""
    if n_max in _GM_POLY_CACHE:
        return _GM_POLY_CACHE[n_max]
    old_dps = mpmath.mp.dps
    mpmath.mp.dps = dps
    x_sym, y_sym = sp.symbols("x y", real=True)
    out: Dict[Tuple[int, int], Dict[Tuple[int, int], "mpmath.mpf"]] = {}
    for n in range(n_max + 1):
        for m in range(-n, n + 1, 2):
            out[(n, m)] = _poly_terms_dict(
                _gm_cartesian_sympy(n, m, x_sym, y_sym), x_sym, y_sym, dps
            )
    _GM_POLY_CACHE[n_max] = out
    mpmath.mp.dps = old_dps
    return out


def _inner_product_terms(terms_a, terms_b) -> "mpmath.mpf":
    """(1/pi) * integral_{disk} P_a(x, y) * P_b(x, y) dA using disk moments."""
    total = mpmath.mpf("0")
    for (px, py), ca in terms_a.items():
        for (qx, qy), cb in terms_b.items():
            total += ca * cb * _disk_moment_over_pi(px + qx, py + qy)
    return total


def project_zernike_to_gm(z_dict: Dict[Tuple[int, int], float],
                          n_max: int = N_MAX, dps: int = MP_DPS) -> Dict[Tuple[int, int], float]:
    """Exact GM coefficient extraction from a Zernike coefficient dictionary.

    Uses Cartesian polynomial representations + closed-form unit-disk moments.
    Returns a dictionary {(n, m): gm_coeff}, in microns, for all modes up to
    n = n_max (45 entries when n_max = 8).
    """
    old_dps = mpmath.mp.dps
    mpmath.mp.dps = dps

    nm_list = pyramid_indices(n_max)
    z_terms_all = _get_zernike_poly_terms(n_max, dps=dps)
    gm_terms_all = _get_gm_poly_terms(n_max, dps=dps)

    # Total wavefront polynomial W(x, y) = sum c_{n,m} * Z_{n,m}(x, y)
    w_terms: Dict[Tuple[int, int], "mpmath.mpf"] = {}
    for nm in nm_list:
        c = mpmath.mpf(z_dict.get(nm, 0.0))
        if c == 0:
            continue
        for mon, coeff in z_terms_all[nm].items():
            w_terms[mon] = w_terms.get(mon, mpmath.mpf("0")) + c * coeff

    # Split by total monomial degree: low (<=2) vs high (>=3)
    low_mons = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    low_terms  = {mon: w_terms.get(mon, mpmath.mpf("0")) for mon in low_mons}
    high_terms = {mon: v for mon, v in w_terms.items() if mon not in set(low_mons)}

    # Project low onto Z (n<=2) and high onto GM (n>=3).
    gm_out: Dict[Tuple[int, int], float] = {}
    for (n, m) in nm_list:
        if n <= 2:
            val = _inner_product_terms(z_terms_all[(n, m)], low_terms)
        else:
            val = _inner_product_terms(gm_terms_all[(n, m)], high_terms)
        gm_out[(n, m)] = float(val)

    mpmath.mp.dps = old_dps
    return gm_out


# =============================================================================
# SECTION 6 -- POLAR GRID AND VERGENCE COMPUTATION
# =============================================================================
# All numerical maps live on a uniform polar grid (rho, theta) on the unit
# disk.  The "vergence" map is the curvature of the wavefront in diopters:
#
#       V(rho, theta) = -(1 / (rho * R^2)) * dW/drho
#
# where W is in meters, R = pupil_diameter / 2 in meters, and rho in [0, 1].
# The 1/rho factor is handled with a safe division (no operation at rho = 0).
# =============================================================================

def build_polar_grid(nr: int = GRID_NR, nt: int = GRID_NT
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (r_grid, t_grid, mask), all shape (nr, nt).

    The mask is True on the unit disk (rho <= 1).
    """
    r_vals = np.linspace(0.0, 1.0, nr)
    t_vals = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
    t_grid, r_grid = np.meshgrid(t_vals, r_vals)
    r_grid = r_grid.T
    t_grid = t_grid.T
    mask = (r_grid <= 1.0)
    return r_grid, t_grid, mask


def compute_vergence(wavefront_expr,
                     radius_mm: float,
                     r_grid: np.ndarray,
                     t_grid: np.ndarray,
                     mask: np.ndarray) -> np.ndarray:
    """Compute the vergence map (diopters) from a wavefront W(r, theta) in um.

    Numerical notes
    ---------------
    * The expression involves a 1/rho term; we therefore use a safe division
      (only where rho > 0) and explicitly set the origin to 0.
    * Any residual non-finite values are zeroed before masking.
    """
    f_diff = sp.diff(wavefront_expr, R_SYM)
    diff_fun = sp.lambdify((R_SYM, T_SYM), f_diff, "numpy")

    grad_um = np.asarray(diff_fun(r_grid, t_grid), dtype=float)
    grad_m  = grad_um * 1.0e-6                       # microns -> meters
    radius_m = float(radius_mm) * 1.0e-3             # mm -> meters

    out = np.zeros_like(r_grid, dtype=float)
    denom = (r_grid * radius_m) * radius_m           # rho * R^2 (rho dimensionless)
    np.divide(-grad_m, denom, out=out, where=(r_grid > 0))
    out[~np.isfinite(out)] = 0.0
    return np.where(mask, out, np.nan)


# =============================================================================
# SECTION 7 -- VERGENCE BASES (V AND V-TILDE) + PTV NORMALIZATIONS
# =============================================================================
# Two decompositions of the vergence map V(rho, theta) over the unit disk:
#
#   * V basis (non-orthogonal):
#         - For n = 2 the V mode is rotation-only (radial-independent), so V_L
#           is the unique fit that reproduces the *local* refraction near the
#           pupil center.  V_H is then the remainder V_total - V_L.
#         - For n >= 3 the V modes carry a radial weight that vanishes near
#           the center and grows toward the rim; they are projected on the
#           full disk.
#
#   * V-tilde basis (orthogonal over the full disk for ALL n):
#         - Fully analogous to the Zernike basis but for vergence; gives a
#           pure orthogonal decomposition with no centered-vs-rim split. 
#
# Both bases use the convention:  (n, +m) -> cos(m theta), (n, -m) -> sin(m theta).
# =============================================================================

def build_V_basis() -> Dict[Tuple[int, int], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Non-orthogonal vergence basis V(n, m), n = 2..6.

    Numerical convention: returns a function of (rho, theta) returning a
    real array of the same shape.
    """
    sqrt2  = np.sqrt(2)
    sqrt3  = np.sqrt(3)
    sqrt5  = np.sqrt(5)
    sqrt6  = np.sqrt(6)
    sqrt8  = np.sqrt(8)
    sqrt10 = np.sqrt(10)

    return {
        # n = 2
        (2, -2): lambda r, t: sqrt2 * np.sin(2 * t),
        (2,  0): lambda r, t: np.ones_like(r),
        (2,  2): lambda r, t: sqrt2 * np.cos(2 * t),
        # n = 3
        (3, -3): lambda r, t: 2 * r * np.sin(3 * t),
        (3, -1): lambda r, t: 2 * r * np.sin(t),
        (3,  1): lambda r, t: 2 * r * np.cos(t),
        (3,  3): lambda r, t: 2 * r * np.cos(3 * t),
        # n = 4
        (4, -4): lambda r, t: sqrt6 * r ** 2 * np.sin(4 * t),
        (4, -2): lambda r, t: sqrt6 * r ** 2 * np.sin(2 * t),
        (4,  0): lambda r, t: sqrt3 * r ** 2,
        (4,  2): lambda r, t: sqrt6 * r ** 2 * np.cos(2 * t),
        (4,  4): lambda r, t: sqrt6 * r ** 2 * np.cos(4 * t),
        # n = 5
        (5, -5): lambda r, t: 2 * sqrt2 * r ** 3 * np.sin(5 * t),
        (5, -3): lambda r, t: 2 * sqrt2 * (3 * r ** 3 - 2 * r) * np.sin(3 * t),
        (5, -1): lambda r, t: 2 * sqrt2 * (3 * r ** 3 - 2 * r) * np.sin(t),
        (5,  1): lambda r, t: 2 * sqrt2 * (3 * r ** 3 - 2 * r) * np.cos(t),
        (5,  3): lambda r, t: 2 * sqrt2 * (3 * r ** 3 - 2 * r) * np.cos(3 * t),
        (5,  5): lambda r, t: 2 * sqrt2 * r ** 3 * np.cos(5 * t),
        # n = 6
        (6, -6): lambda r, t: sqrt10 * r ** 4 * np.sin(6 * t),
        (6, -4): lambda r, t: sqrt10 * (4 * r ** 4 - 3 * r ** 2) * np.sin(4 * t),
        (6, -2): lambda r, t: sqrt10 * (4 * r ** 4 - 3 * r ** 2) * np.sin(2 * t),
        (6,  0): lambda r, t: sqrt5  * (4 * r ** 4 - 3 * r ** 2),
        (6,  2): lambda r, t: sqrt10 * (4 * r ** 4 - 3 * r ** 2) * np.cos(2 * t),
        (6,  4): lambda r, t: sqrt10 * (4 * r ** 4 - 3 * r ** 2) * np.cos(4 * t),
        (6,  6): lambda r, t: sqrt10 * r ** 4 * np.cos(6 * t),
    }


def build_Vtilde_basis() -> Dict[Tuple[int, int], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Orthogonal vergence basis V_tilde(n, m), n = 2..6 (full-disk orthonormal)."""
    sqrt2  = np.sqrt(2)
    sqrt3  = np.sqrt(3)
    sqrt5  = np.sqrt(5)
    sqrt6  = np.sqrt(6)
    sqrt10 = np.sqrt(10)

    return {
        (2, -2): lambda r, t: sqrt2 * np.sin(2 * t),
        (2,  0): lambda r, t: np.ones_like(r),
        (2,  2): lambda r, t: sqrt2 * np.cos(2 * t),

        (3, -3): lambda r, t: 2 * r * np.sin(3 * t),
        (3, -1): lambda r, t: 2 * r * np.sin(t),
        (3,  1): lambda r, t: 2 * r * np.cos(t),
        (3,  3): lambda r, t: 2 * r * np.cos(3 * t),

        (4, -4): lambda r, t: sqrt6 * r ** 2 * np.sin(4 * t),
        (4, -2): lambda r, t: sqrt6 * (2 * r ** 2 - 1) * np.sin(2 * t),
        (4,  0): lambda r, t: sqrt3 * (2 * r ** 2 - 1),
        (4,  2): lambda r, t: sqrt6 * (2 * r ** 2 - 1) * np.cos(2 * t),
        (4,  4): lambda r, t: sqrt6 * r ** 2 * np.cos(4 * t),

        (5, -5): lambda r, t: 2 * sqrt2 * r ** 3 * np.sin(5 * t),
        (5, -3): lambda r, t: 2 * sqrt2 * (3 * r ** 3 - 2 * r) * np.sin(3 * t),
        (5, -1): lambda r, t: 2 * sqrt2 * (3 * r ** 3 - 2 * r) * np.sin(t),
        (5,  1): lambda r, t: 2 * sqrt2 * (3 * r ** 3 - 2 * r) * np.cos(t),
        (5,  3): lambda r, t: 2 * sqrt2 * (3 * r ** 3 - 2 * r) * np.cos(3 * t),
        (5,  5): lambda r, t: 2 * sqrt2 * r ** 3 * np.cos(5 * t),

        (6, -6): lambda r, t: sqrt10 * r ** 4 * np.sin(6 * t),
        (6, -4): lambda r, t: sqrt10 * (4 * r ** 4 - 3 * r ** 2) * np.sin(4 * t),
        (6, -2): lambda r, t: sqrt10 * (6 * r ** 4 - 6 * r ** 2 + 1) * np.sin(2 * t),
        (6,  0): lambda r, t: sqrt5  * (6 * r ** 4 - 6 * r ** 2 + 1),
        (6,  2): lambda r, t: sqrt10 * (6 * r ** 4 - 6 * r ** 2 + 1) * np.cos(2 * t),
        (6,  4): lambda r, t: sqrt10 * (4 * r ** 4 - 3 * r ** 2) * np.cos(4 * t),
        (6,  6): lambda r, t: sqrt10 * r ** 4 * np.cos(6 * t),
    }


# Peak-to-valley (PTV) constants of each V / V_tilde mode over the full unit
# disk.  These are used to convert raw vergence coefficients into PTV-
# normalized units (npvV / npvVT).
V_PTV: Dict[Tuple[int, int], float] = {
    (2, -2): 2 * np.sqrt(2), (2, 0): 0.0,          (2, 2): 2 * np.sqrt(2),
    (3, -3): 4.0,            (3, -1): 4.0,         (3, 1): 4.0,             (3, 3): 4.0,
    (4, -4): 2 * np.sqrt(6), (4, -2): 2 * np.sqrt(6),
    (4,  0): np.sqrt(3),
    (4,  2): 2 * np.sqrt(6), (4,  4): 2 * np.sqrt(6),
    (5, -5): 4 * np.sqrt(2), (5, -3): 4 * np.sqrt(2), (5, -1): 4 * np.sqrt(2),
    (5,  1): 4 * np.sqrt(2), (5,  3): 4 * np.sqrt(2), (5,  5): 4 * np.sqrt(2),
    (6, -6): 2 * np.sqrt(10),
    (6, -4): 2 * np.sqrt(10), (6, -2): 2 * np.sqrt(10),
    (6,  0): 25 * np.sqrt(5) / 16,
    (6,  2): 2 * np.sqrt(10), (6,  4): 2 * np.sqrt(10), (6, 6): 2 * np.sqrt(10),
}

VT_PTV: Dict[Tuple[int, int], float] = {
    (2, -2): 2 * np.sqrt(2), (2, 0): 0.0,          (2, 2): 2 * np.sqrt(2),
    (3, -3): 4.0,            (3, -1): 4.0,         (3, 1): 4.0,             (3, 3): 4.0,
    (4, -4): 2 * np.sqrt(6),
    (4, -2): 2 * np.sqrt(6),
    (4,  0): 2 * np.sqrt(3),
    (4,  2): 2 * np.sqrt(6),
    (4,  4): 2 * np.sqrt(6),
    (5, -5): 4 * np.sqrt(2), (5, -3): 4 * np.sqrt(2), (5, -1): 4 * np.sqrt(2),
    (5,  1): 4 * np.sqrt(2), (5,  3): 4 * np.sqrt(2), (5,  5): 4 * np.sqrt(2),
    (6, -6): 2 * np.sqrt(10),
    (6, -4): 2 * np.sqrt(10),
    (6, -2): 2 * np.sqrt(10),
    (6,  0): 3 * np.sqrt(5) / 2,
    (6,  2): 2 * np.sqrt(10),
    (6,  4): 2 * np.sqrt(10),
    (6,  6): 2 * np.sqrt(10),
}


def build_normalized_basis(basis: Dict[Tuple[int, int], Callable],
                           ptv: Dict[Tuple[int, int], float]
                           ) -> Dict[Tuple[int, int], Callable]:
    """Return basis with each mode divided by its PTV (skipping PTV = 0)."""
    out = {}
    for nm, fn in basis.items():
        p = float(ptv.get(nm, 0.0))
        if p == 0.0:
            out[nm] = fn
        else:
            def _make(fn_inner, p_inner):
                return lambda r, t: fn_inner(r, t) / p_inner
            out[nm] = _make(fn, p)
    return out


# Mode lists used throughout the pipeline.
VERGENCE_MODES = [(n, m) for n in range(2, 7) for m in range(-n, n + 1, 2)]   # 25 modes
HALF_PYRAMID_MODES = [
    (2, 0), (2, 2),
    (3, 1), (3, 3),
    (4, 0), (4, 2), (4, 4),
    (5, 1), (5, 3), (5, 5),
    (6, 0), (6, 2), (6, 4), (6, 6),
]
MAG_AXIS_MODES = [(2, 2), (3, 1), (3, 3), (4, 2), (4, 4),
                  (5, 1), (5, 3), (5, 5), (6, 2), (6, 4), (6, 6)]


# =============================================================================
# SECTION 8 -- PROJECTIONS ON VERGENCE BASES (VL / VH FIT, TILDE FIT)
# =============================================================================
# Two projection styles are implemented:
#
#   * Centered fit (for V_L, n = 2 only): least-squares of the vergence map
#     on V(n, m) over a *central physical disk* of radius
#     vl_fit_radius_mm.  The radius is independent of the pupil under
#     analysis (anchored on the 6 mm reference).
#
#   * Full-disk fit (for V_H, n >= 3, and for all tilde modes): the same
#     least-squares projection but over the full unit disk.
# =============================================================================

def _projection_lsq(vergence_map: np.ndarray,
                    basis_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                    r_grid: np.ndarray, t_grid: np.ndarray,
                    region_mask: np.ndarray, nr: int, nt: int) -> float:
    """Least-squares coefficient = int(V * Vb * dA) / int(Vb^2 * dA), over region_mask."""
    dr = 1.0 / (nr - 1)
    dtheta = 2.0 * np.pi / nt
    area = r_grid * dr * dtheta
    v_map = np.where(region_mask, vergence_map, 0.0)
    v_b = np.where(region_mask, basis_func(r_grid, t_grid), 0.0)
    num = np.sum(v_map * v_b * area)
    den = np.sum(v_b ** 2 * area)
    if den < 1e-15:
        return 0.0
    return float(num / den)


def _discrete_projection(vergence_map: np.ndarray,
                         basis_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                         r_grid: np.ndarray, t_grid: np.ndarray,
                         region_mask: np.ndarray, nr: int, nt: int) -> float:
    """(1/pi) * integral over region_mask of V * Vb dA (used for orthogonal projection)."""
    dr = 1.0 / (nr - 1)
    dtheta = 2.0 * np.pi / nt
    area = r_grid * dr * dtheta
    v_map = np.where(region_mask, vergence_map, 0.0)
    v_b = np.where(region_mask, basis_func(r_grid, t_grid), 0.0)
    return float(np.sum(v_map * v_b * area) / math.pi)


def decompose_VL_VH(vergence_total: np.ndarray,
                    pupil_diameter_mm: float,
                    r_grid: np.ndarray, t_grid: np.ndarray,
                    mask: np.ndarray,
                    vl_fit_radius_mm: Optional[float] = None,
                    nr: int = GRID_NR, nt: int = GRID_NT
                    ) -> Tuple[Dict[Tuple[int, int], float], np.ndarray, np.ndarray]:
    """Decompose the total vergence map onto the V basis (non-orthogonal).

    Returns
    -------
    alpha    : dict {(n, m) -> coefficient (D)}, for all (n, m) with n in 2..6.
    vl_map   : the V_L map (low-degree fit, n = 2) on the full disk (nan outside).
    vh_map   : the V_H map (V_total - V_L) on the full disk (nan outside).
    """
    V = build_V_basis()
    radius_mm = pupil_diameter_mm / 2.0

    if vl_fit_radius_mm is None:
        # Default: 0.20 * (6 mm / 2) = 0.6 mm physical fit radius.
        vl_fit_radius_mm = VL_CENTER_FRACTION_REF * (VL_REF_DIAMETER_MM / 2.0)

    # ---- V_L: least-squares on a central disk of radius vl_fit_radius_mm ----
    if radius_mm <= 0:
        center_mask = np.zeros_like(r_grid, dtype=bool)
    else:
        center_radius_norm = min(1.0, vl_fit_radius_mm / radius_mm)
        center_mask = (r_grid <= center_radius_norm)

    alpha: Dict[Tuple[int, int], float] = {}
    for nm in [(2, -2), (2, 0), (2, 2)]:
        alpha[nm] = _projection_lsq(
            vergence_total, V[nm], r_grid, t_grid, center_mask, nr, nt
        )

    # ---- Reconstruct V_L map on the full disk ----
    vl_map = (
        alpha[(2, -2)] * V[(2, -2)](r_grid, t_grid)
        + alpha[(2, 0)] * V[(2, 0)](r_grid, t_grid)
        + alpha[(2, 2)] * V[(2, 2)](r_grid, t_grid)
    )
    vl_map = np.where(mask, vl_map, np.nan)

    # ---- V_H = V_total - V_L, then project V_H onto V(n>=3) on the full disk ---
    vh_map = vergence_total - vl_map
    for n in range(3, 7):
        for m in range(-n, n + 1, 2):
            alpha[(n, m)] = _projection_lsq(
                vh_map, V[(n, m)], r_grid, t_grid, mask, nr, nt
            )

    return alpha, vl_map, vh_map


def decompose_tilde(vergence_total: np.ndarray,
                    r_grid: np.ndarray, t_grid: np.ndarray,
                    mask: np.ndarray,
                    nr: int = GRID_NR, nt: int = GRID_NT
                    ) -> Tuple[Dict[Tuple[int, int], float], np.ndarray, np.ndarray]:
    """Decompose the total vergence map onto the V-tilde basis (orthogonal).

    Returns
    -------
    alpha_tilde : dict {(n, m) -> tilde coefficient (D)}, n in 2..6.
    vl_tilde    : the low-degree map (n = 2), full disk.
    vh_tilde    : the high-degree map (n in 3..6), full disk.
    """
    VT = build_Vtilde_basis()

    alpha_tilde: Dict[Tuple[int, int], float] = {}
    for nm in VERGENCE_MODES:
        alpha_tilde[nm] = _discrete_projection(
            vergence_total, VT[nm], r_grid, t_grid, mask, nr, nt
        )

    # Reconstruct VL and VH (tilde) maps.
    vl_tilde = np.zeros_like(r_grid)
    for n in [2]:
        for m in range(-n, n + 1, 2):
            vl_tilde += alpha_tilde[(n, m)] * VT[(n, m)](r_grid, t_grid)
    vl_tilde = np.where(mask, vl_tilde, np.nan)

    vh_tilde = np.zeros_like(r_grid)
    for n in range(3, 7):
        for m in range(-n, n + 1, 2):
            vh_tilde += alpha_tilde[(n, m)] * VT[(n, m)](r_grid, t_grid)
    vh_tilde = np.where(mask, vh_tilde, np.nan)

    return alpha_tilde, vl_tilde, vh_tilde


def scale_alpha_by_PTV(alpha: Dict[Tuple[int, int], float],
                       ptv: Dict[Tuple[int, int], float]
                       ) -> Dict[Tuple[int, int], float]:
    """Return a PTV-scaled copy of `alpha` (coefficients in PTV-normalized units)."""
    out: Dict[Tuple[int, int], float] = {}
    for nm, c in alpha.items():
        p = float(ptv.get(nm, 0.0))
        out[nm] = float(c) if p == 0.0 else float(c) * p
    return out


# =============================================================================
# SECTION 9 -- REFRACTION (S, C, A) AT CORNEA AND SPECTACLE PLANE
# =============================================================================
# Given the low-degree vergence coefficients (n = 2), the sphero-cylindrical
# refraction is obtained by:
#
#       A = 0.5 * atan2(c(2,-2), c(2,2))            (axis in degrees, in [0, 180))
#       amp2 = sqrt(c(2,-2)^2 + c(2,2)^2)
#       C    = -2 * sqrt(2) * amp2                  (cylinder, negative convention)
#       S    = c(2, 0) + sqrt(2) * amp2             (sphere)
#
# The convention follows the original OPDscan pipeline.  The vertex conversion
# converts the corneal-plane (S, C, A) to the spectacle plane at 12 mm.
# =============================================================================

def extract_sca(alpha: Dict[Tuple[int, int], float]) -> Tuple[float, float, float]:
    """Return (S, C, A) in (D, D, deg) from low-degree V coefficients."""
    c_2m2 = float(alpha.get((2, -2), 0.0))
    c_20  = float(alpha.get((2,  0), 0.0))
    c_22  = float(alpha.get((2,  2), 0.0))
    amp2 = math.sqrt(c_2m2 ** 2 + c_22 ** 2)
    C = -2.0 * math.sqrt(2) * amp2
    S =  c_20 + math.sqrt(2) * amp2
    A = math.degrees(0.5 * math.atan2(c_2m2, c_22)) % 180.0
    return float(S), float(C), float(round(A))


def vertex_convert_power(power_D: float, vertex_m: float = VERTEX_M) -> float:
    """Convert a single power (cornea -> spectacle plane at distance vertex_m)."""
    power_D = float(power_D)
    denom = 1.0 + vertex_m * power_D
    if abs(denom) < 1e-12:
        return float("nan")
    return power_D / denom


def vertex_convert_sca(S: float, C: float, A_deg: float,
                       vertex_m: float = VERTEX_M) -> Tuple[float, float, float]:
    """Convert (S, C, A) from the corneal plane to the spectacle plane.

    Uses the principal-meridian approach: convert the two principal powers
    independently, then recompose as (S', C', A') with negative-cylinder
    convention.
    """
    if S is None or C is None or A_deg is None:
        return float("nan"), float("nan"), float("nan")
    S = float(S); C = float(C); A = float(A_deg) % 180.0

    F_axis = S
    F_perp = S + C
    F_axis_v = vertex_convert_power(F_axis, vertex_m)
    F_perp_v = vertex_convert_power(F_perp, vertex_m)

    S_v = F_axis_v
    C_v = F_perp_v - F_axis_v
    A_v = A
    # Force negative cylinder convention
    if C_v > 0.0:
        S_v = S_v + C_v
        C_v = -C_v
        A_v = (A_v + 90.0) % 180.0
    return S_v, C_v, A_v


# =============================================================================
# SECTION 10 -- STATISTICS AND ANGLE / MAGNITUDE HELPERS
# =============================================================================

def canonical_axis_deg(a_cos: float, b_sin: float, m: int) -> float:
    """Canonical axis (deg) for a*cos(m theta) + b*sin(m theta) = A*cos(m(theta - axis))."""
    eps = 1e-15
    if int(m) <= 0:
        return 0.0
    if math.hypot(a_cos, b_sin) < eps:
        return 0.0
    phi = math.degrees(math.atan2(float(b_sin), float(a_cos))) % 360.0
    period = 360.0 / float(m)
    return (phi / float(m)) % period


def magnitude_axis(alpha: Dict[Tuple[int, int], float], n: int, m: int) -> Tuple[float, float]:
    """Return (magnitude, axis_deg) for mode (n, m) with m > 0."""
    if m == 0:
        return float(alpha.get((n, 0), 0.0)), 0.0
    a = float(alpha.get((n,  m), 0.0))
    b = float(alpha.get((n, -m), 0.0))
    return math.hypot(a, b), canonical_axis_deg(a, b, m)


def map_stats(arr2d: np.ndarray, remove_origin: bool = False) -> Dict[str, float]:
    """Return min/q1/median/q3/max + mean/std/rms over the valid (non-NaN) values."""
    arr = np.array(arr2d, dtype=float, copy=True)
    if remove_origin:
        # Set the central column (rho == 0) to NaN to avoid the 1/rho
        # singularity contaminating the descriptive statistics.
        # NOTE: with build_polar_grid() the grid is transposed so that
        # arr[i, j] sits at (theta_i, rho_j); rho == 0 is therefore the
        # column index 0 (NOT the row index 0).
        try:
            arr[:, 0] = np.nan
        except Exception:
            pass
    vals = arr[~np.isnan(arr)]
    if vals.size == 0:
        return dict(min=0.0, q1=0.0, median=0.0, q3=0.0, max=0.0,
                    mean=0.0, std=0.0, rms=0.0)
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    return dict(
        min=float(np.min(vals)),  q1=float(q1),    median=float(med),
        q3=float(q3),             max=float(np.max(vals)),
        mean=float(np.mean(vals)), std=float(np.std(vals)),
        rms=float(np.sqrt(np.mean(vals ** 2))),
    )


def snap_quarter(x: float) -> float:
    """Snap a value to the nearest 1/8 (0.125)."""
    return 0.125 * round(x / 0.125)


def smart_ticks(vmin: float, vmax: float, max_ticks: int = 50) -> np.ndarray:
    """Generate a clean array of major colorbar ticks between vmin and vmax."""
    amplitude = vmax - vmin
    if amplitude <= 0:
        return np.array([vmin, vmax])
    if   amplitude <=  3: step = 0.25
    elif amplitude <=  6: step = 0.25
    elif amplitude <= 12: step = 0.5
    elif amplitude <= 24: step = 1.0
    else:                 step = 2.0
    n_ticks = int(amplitude / step) + 1
    if n_ticks > max_ticks:
        # Fallback for very wide ranges
        candidates = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        step = min(candidates, key=lambda x: abs(x - amplitude / (max_ticks - 1)))
    start = np.ceil(vmin / step) * step
    end   = np.floor(vmax / step) * step
    ticks = np.arange(start, end + step / 2, step)
    if vmin < 0 < vmax and 0.0 not in ticks:
        ticks = np.sort(np.append(ticks, 0.0))
    return ticks[(ticks >= vmin) & (ticks <= vmax)]


# =============================================================================
# SECTION 11 -- SHARED FIGURE HELPERS
# =============================================================================
# Small utilities reused across all figure-generation functions: patient info
# box, pupil ruler / angular ticks.  They keep the visual style consistent
# (angular tick marks, dashed astigmatism axes, etc.).
# =============================================================================

def _info_text(case_id: str, pupil_diameter_mm: float) -> str:
    """Compact identification box for figures."""
    return f"Case: {case_id}\nPupil diameter: {pupil_diameter_mm:.2f} mm"


def _draw_pupil_ruler(ax, radius_mm: float,
                      angle_label_size: int = 12,
                      astig_axis_deg: Optional[float] = None) -> None:
    """Draw the pupil rim + angular ticks (and optional astigmatism axes)."""
    circle = plt.Circle((0, 0), radius_mm, color="black", fill=False, linewidth=1)
    ax.add_patch(circle)
    # Fine angular ticks every 5 deg
    for angle in range(0, 360, 5):
        rad = np.deg2rad(angle)
        line_length = radius_mm * (0.95 if angle % 10 else 0.9)
        ax.plot([0, line_length * np.cos(rad)], [0, line_length * np.sin(rad)],
                color="black", linewidth=0.05,
                linestyle="--" if angle % 10 else "-")
        if angle % 10 == 0:
            xl = 1.1 * radius_mm * np.cos(rad)
            yl = 1.1 * radius_mm * np.sin(rad)
            ax.text(xl, yl, f"{angle} deg", ha="center", va="center",
                    fontsize=angle_label_size)
    for angle in (0, 90, 180, 270):
        rad = np.deg2rad(angle)
        ax.plot([0, radius_mm * np.cos(rad)], [0, radius_mm * np.sin(rad)],
                color="black", linewidth=1.0)
    if astig_axis_deg is not None:
        rad_a   = np.deg2rad(astig_axis_deg)
        rad_a90 = np.deg2rad((astig_axis_deg + 90) % 180)
        ax.plot([-0.7 * radius_mm * np.cos(rad_a),   0.7 * radius_mm * np.cos(rad_a)],
                [-0.7 * radius_mm * np.sin(rad_a),   0.7 * radius_mm * np.sin(rad_a)],
                color="#4A4A4A", linewidth=2, linestyle="--")
        ax.plot([-0.5 * radius_mm * np.cos(rad_a90), 0.5 * radius_mm * np.cos(rad_a90)],
                [-0.5 * radius_mm * np.sin(rad_a90), 0.5 * radius_mm * np.sin(rad_a90)],
                color="#A9A9A9", linewidth=2, linestyle="--")


# =============================================================================
# SECTION 12 -- FIGURE: WAVEFRONT (Z vs GM) TRIPTYCHS
# =============================================================================
# Two figures:
#   (A) Z (total/low/high) vs GM (total/low/high), before piston/tilt removal.
#   (B) GM (no piston / no tilt) total/low/high.
# =============================================================================

def plot_wavefront_z_vs_gm(z_dict: Dict[Tuple[int, int], float],
                           gm_dict: Dict[Tuple[int, int], float],
                           gm_noPT_dict: Dict[Tuple[int, int], float],
                           radius_mm: float,
                           r_grid: np.ndarray, t_grid: np.ndarray, mask: np.ndarray,
                           output_dir: Path,
                           case_id: str,
                           pupil_diameter_mm: float,
                           wavefront_cmap) -> None:
    """Save  Wavefront_Z_vs_GM_{case_id}.png  and  Wavefront_GM_noPT_{case_id}.png."""
    x_grid = r_grid * radius_mm * np.cos(t_grid)
    y_grid = r_grid * radius_mm * np.sin(t_grid)

    def eval_wf(expr):
        fn = sp.lambdify((R_SYM, T_SYM), expr, "numpy")
        out = np.asarray(fn(r_grid, t_grid), dtype=float)
        return np.where(mask, out, np.nan)

    # Z wavefronts
    fz_tot = eval_wf(wavefront_from_zernike(z_dict))
    fz_low = eval_wf(wavefront_from_zernike({k: v for k, v in z_dict.items() if k[0] <= 2}))
    fz_high = eval_wf(wavefront_from_zernike({k: v for k, v in z_dict.items() if k[0] >= 3}))
    # GM wavefronts (full)
    fg_tot = eval_wf(wavefront_from_gm(gm_dict))
    fg_low = eval_wf(wavefront_from_gm(gm_dict, n_max=2))
    fg_high = eval_wf(wavefront_from_gm(gm_dict, n_min=3))
    # GM wavefronts (no piston / no tilt)
    fgn_tot = eval_wf(wavefront_from_gm(gm_noPT_dict))
    fgn_low = eval_wf(wavefront_from_gm(gm_noPT_dict, n_max=2))
    fgn_high = eval_wf(wavefront_from_gm(gm_noPT_dict, n_min=3))

    all_wf = np.concatenate([
        fz_tot.ravel(), fz_low.ravel(), fz_high.ravel(),
        fg_tot.ravel(), fg_low.ravel(), fg_high.ravel(),
        fgn_tot.ravel(), fgn_low.ravel(), fgn_high.ravel(),
    ])
    ab_wf = max(abs(np.nanmin(all_wf)), abs(np.nanmax(all_wf)))
    ab_wf = max(ab_wf, 1e-6)
    levels = np.linspace(-ab_wf, ab_wf, 80)

    def make_contour(ax, fig, arr, title):
        ctf = ax.contourf(x_grid, y_grid, arr, levels=levels,
                          cmap=wavefront_cmap, extend="both")
        cs = ax.contour(x_grid, y_grid, arr, levels=levels,
                        colors="k", linewidths=0.5)
        ax.clabel(cs, inline=True, fmt="%.2f")
        ax.set_title(title)
        fig.colorbar(ctf, ax=ax, format="%.2f")
        ax.axis("off"); ax.set_aspect("equal")

    # ---------- Figure A: Z vs GM (full coefficients) ----------
    figA, axsA = plt.subplots(2, 3, figsize=(18, 10))
    figA.suptitle("Wavefronts: Zernike vs Gatinel-Malet (before piston/tilt removal)",
                  fontsize=14)
    make_contour(axsA[0, 0], figA, fz_tot, "Zernike total (um)")
    make_contour(axsA[0, 1], figA, fz_low, "Zernike low-degree (n <= 2)")
    make_contour(axsA[0, 2], figA, fz_high, "Zernike high-degree (n >= 3)")
    make_contour(axsA[1, 0], figA, fg_tot, "GM total (um)")
    make_contour(axsA[1, 1], figA, fg_low, "GM low-degree (n <= 2)")
    make_contour(axsA[1, 2], figA, fg_high, "GM high-degree (n >= 3)")
    figA.text(0.02, 0.98, _info_text(case_id, pupil_diameter_mm),
              fontsize=10, va="top", ha="left",
              bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    plt.tight_layout()
    figA.savefig(output_dir / f"Wavefront_Z_vs_GM_{case_id}.png")
    plt.close(figA)

    # ---------- Figure B: GM no piston / no tilt ----------
    figB, axB = plt.subplots(1, 3, figsize=(18, 5))
    figB.suptitle("GM wavefront (after removing piston and tilt)", fontsize=14)
    make_contour(axB[0], figB, fgn_tot,  "GM total (no PT)")
    make_contour(axB[1], figB, fgn_low,  "GM low-degree (no PT)")
    make_contour(axB[2], figB, fgn_high, "GM high-degree (no PT)")
    figB.text(0.02, 0.98, _info_text(case_id, pupil_diameter_mm),
              fontsize=10, va="top", ha="left",
              bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    plt.tight_layout()
    figB.savefig(output_dir / f"Wavefront_GM_noPT_{case_id}.png")
    plt.close(figB)


# =============================================================================
# SECTION 13 -- FIGURE: VERGENCE TRIPTYCH (TOT / VL / VH)
# =============================================================================
# Renders one decomposition (either VL/VH or tilde) as a 3-panel figure.
# =============================================================================

def _draw_vergence_panel(ax, fig, data_map, title, vmin, vmax, x_grid, y_grid,
                         radius_mm,
                         show_boxplot: bool = False,
                         remove_origin: bool = False,
                         astig_axis_deg: Optional[float] = None,
                         subplot_title_size: int = 20,
                         contour_label_size: int = 15,
                         colorbar_label_size: int = 15,
                         colorbar_tick_size: int = 14,
                         boxplot_text_size: int = 20) -> Dict[str, float]:
    """Render a single vergence map onto `ax`.  Returns descriptive stats."""
    arr = data_map.copy()
    if remove_origin:
        # rho == 0 is the column index 0 in the transposed (theta, rho) grid
        try:
            arr[:, 0] = np.nan
        except Exception:
            pass
    ab_ = max(abs(vmin), abs(vmax))
    ab_ = max(ab_, 0.125)
    ab_ = snap_quarter(ab_)
    vmin, vmax = -ab_, ab_

    levels = np.arange(-ab_, ab_ + 0.125, 0.125)
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    ctf = ax.contourf(x_grid, y_grid, arr, levels=levels,
                      cmap="jet_r", norm=norm, extend="both")

    fine_start = np.ceil(vmin / 0.25) * 0.25
    fine_end   = np.floor(vmax / 0.25) * 0.25
    fine_levels = np.arange(fine_start, fine_end + 0.125, 0.25)
    ax.contour(x_grid, y_grid, arr, levels=fine_levels,
               colors="lightgray", linewidths=0.4, alpha=0.7)

    coarse_levels = smart_ticks(vmin, vmax, max_ticks=20)
    cs = ax.contour(x_grid, y_grid, arr, levels=coarse_levels,
                    colors="k", linewidths=0.5)
    ax.clabel(cs, inline=True, fmt="%.2f", fontsize=contour_label_size)

    ax.set_title(title, pad=40, fontsize=subplot_title_size)
    _draw_pupil_ruler(ax, radius_mm, astig_axis_deg=astig_axis_deg)
    ax.axis("off"); ax.set_aspect("equal")

    cb = fig.colorbar(ctf, ax=ax, format="%.2f", pad=0.12)
    cb_ticks = smart_ticks(vmin, vmax, max_ticks=50)
    cb.set_ticks(cb_ticks)
    cb.set_label("Power (D)", labelpad=15, fontsize=colorbar_label_size)
    cb.ax.tick_params(labelsize=colorbar_tick_size)

    stats = map_stats(arr, remove_origin=False)  # already removed if needed
    if show_boxplot:
        valid = arr[~np.isnan(arr)]
        if valid.size > 0:
            box_ax = cb.ax.inset_axes([-0.8, 0.0, 0.4, 1.0], transform=cb.ax.transAxes)
            box_ax.set_ylim([vmin, vmax])
            box_ax.boxplot(
                valid, vert=True, whis=[0, 100], widths=0.8,
                patch_artist=True,
                boxprops=dict(facecolor="white", edgecolor="red", alpha=0.6, linewidth=2),
                medianprops=dict(color="red", linewidth=2),
                whiskerprops=dict(color="red", linewidth=1.5),
                capprops=dict(color="red", linewidth=1.5),
            )
            box_ax.yaxis.tick_right()
            box_ax.set_xticks([])
            box_ax.set_facecolor("none")
            for sp_ in box_ax.spines.values():
                sp_.set_visible(False)
            text_box_ax = ax.inset_axes([0.92, -0.75, 0.25, 0.25], transform=ax.transAxes)
            box_text = (f"Min: {stats['min']:.2f} | Q1: {stats['q1']:.2f}\n"
                        f"Med: {stats['median']:.2f}\n"
                        f"Q3: {stats['q3']:.2f} | Max: {stats['max']:.2f}")
            text_box_ax.text(0.5, 0.5, box_text, transform=text_box_ax.transAxes,
                             ha="center", va="center",
                             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                                       edgecolor="red", linewidth=2),
                             fontsize=boxplot_text_size)
            text_box_ax.axis("off")
    return stats


def plot_decomposition(vergence_total: np.ndarray,
                       vl_map: np.ndarray, vh_map: np.ndarray,
                       refraction_text: Optional[str],
                       astig_axis_deg: Optional[float],
                       radius_mm: float,
                       r_grid: np.ndarray, t_grid: np.ndarray,
                       output_dir: Path,
                       case_id: str,
                       pupil_diameter_mm: float,
                       basis_label: str = "VL-VH") -> None:
    """Save a 3-panel figure of the vergence decomposition (total / VL / VH)."""
    x_grid = r_grid * radius_mm * np.cos(t_grid)
    y_grid = r_grid * radius_mm * np.sin(t_grid)

    # Auto-scale colormap limits
    def _limits(arr):
        v = arr[~np.isnan(arr)]
        if v.size == 0: return -1.0, 1.0
        ab = max(abs(np.min(v)), abs(np.max(v)))
        ab = max(ab, 0.125)
        return -snap_quarter(ab), snap_quarter(ab)

    tot_noc = vergence_total.copy()
    tot_noc[:, 0] = np.nan          # remove rho == 0 column (transposed grid)
    vmin, vmax = _limits(np.concatenate([tot_noc.ravel(), vl_map.ravel()]))
    hvmin, hvmax = _limits(vh_map)

    fig, axes = plt.subplots(1, 3, figsize=(28, 14))
    fig.suptitle(f"Final decomposition ({basis_label})", fontsize=28)

    _draw_vergence_panel(axes[0], fig, vergence_total, "TOT",
                         vmin, vmax, x_grid, y_grid, radius_mm,
                         show_boxplot=True, remove_origin=True)
    if refraction_text:
        axes[1].text(0.5, -0.2, refraction_text, transform=axes[1].transAxes,
                     ha="center", va="top", color="black",
                     bbox=dict(facecolor="lightgrey", alpha=0.7, boxstyle="round,pad=0.3"),
                     fontsize=26)
    _draw_vergence_panel(axes[1], fig, vl_map, r"$V_L$",
                         vmin, vmax, x_grid, y_grid, radius_mm,
                         astig_axis_deg=astig_axis_deg)
    stats_h = _draw_vergence_panel(axes[2], fig, vh_map, r"$V_H$",
                                   hvmin, hvmax, x_grid, y_grid, radius_mm,
                                   show_boxplot=True)
    axes[2].text(0.5, -0.25,
                 (rf"$\bf{{RMS={stats_h['rms']:.3f}}}$"
                  "\n"
                  rf"Mean $\pm$ SD = {stats_h['mean']:.3f} $\pm$ {stats_h['std']:.3f}"),
                 transform=axes[2].transAxes, ha="center", va="top", color="white",
                 bbox=dict(facecolor="black", alpha=0.4, boxstyle="round,pad=0.3"),
                 fontsize=22)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.30, wspace=0.3)
    fig.text(0.02, 0.98, _info_text(case_id, pupil_diameter_mm),
             fontsize=18, va="top", ha="left",
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    safe_label = basis_label.replace(" ", "_").replace("/", "_")
    fig.savefig(output_dir / f"Decomposition_{safe_label}_{case_id}.png", dpi=SAVE_DPI)
    plt.close(fig)


# =============================================================================
# SECTION 14 -- FIGURES: PYRAMIDS (FULL + HALF)
# =============================================================================
# Generic pyramid renderer used by:
#       * V (VL-VH)             -> plot_pyramid_full(...)
#       * V_tilde                -> plot_pyramid_full(...)
#       * V normalized (npvV)    -> plot_pyramid_full(...)
#       * V_tilde normalized (npvVT)
# The half-pyramid (m >= 0) version combines (cos, sin) into magnitude/axis
# for each (n, m > 0) mode.
# =============================================================================

FULL_PYRAMID_LAYOUT = [
    [None, None, None, None, (2, -2), None, (2, 0),
     None, (2, 2), None, None, None, None],
    [None, None, None, (3, -3), None, (3, -1), None,
     (3, 1), None, (3, 3), None, None, None],
    [None, None, (4, -4), None, (4, -2), None, (4, 0),
     None, (4, 2), None, (4, 4), None, None],
    [None, (5, -5), None, (5, -3), None, (5, -1), None,
     (5, 1), None, (5, 3), None, (5, 5), None],
    [(6, -6), None, (6, -4), None, (6, -2), None,
     (6, 0), None, (6, 2), None, (6, 4), None, (6, 6)],
]

HALF_PYRAMID_LAYOUT = [
    [(2, 0), None, (2, 2), None, None, None, None],
    [None, (3, 1), None, (3, 3), None, None, None],
    [(4, 0), None, (4, 2), None, (4, 4), None, None],
    [None, (5, 1), None, (5, 3), None, (5, 5), None],
    [(6, 0), None, (6, 2), None, (6, 4), None, (6, 6)],
]


def _pyramid_scale(rho_grid, theta_grid, alpha, basis_dict, layout, combine_half=False):
    """Compute the global colormap range needed across all subplots."""
    arrays = []
    for row in layout:
        for nm in row:
            if nm is None or nm == (2, 0):
                continue
            n, m = nm
            if combine_half:
                a = alpha.get((n,  m), 0.0)
                b = alpha.get((n, -m), 0.0) if m > 0 else 0.0
                arr = a * basis_dict[(n, m)](rho_grid, theta_grid)
                if m > 0:
                    arr = arr + b * basis_dict[(n, -m)](rho_grid, theta_grid)
            else:
                arr = alpha.get((n, m), 0.0) * basis_dict[(n, m)](rho_grid, theta_grid)
            arr = np.where(rho_grid <= 1, arr, np.nan)
            arrays.append(arr.flatten())
    if not arrays:
        return -0.25, 0.25
    stacked = np.concatenate(arrays)
    zmin, zmax = np.nanmin(stacked), np.nanmax(stacked)
    ab_ = max(abs(zmin), abs(zmax))
    if ab_ < 0.25:
        ab_ = 0.25
    ab_ = np.ceil(ab_ / 0.25) * 0.25
    return -ab_, ab_


def plot_pyramid_full(alpha: Dict[Tuple[int, int], float],
                      basis_dict: Dict[Tuple[int, int], Callable],
                      title: str,
                      filename: str,
                      output_dir: Path,
                      case_id: str,
                      pupil_diameter_mm: float,
                      symbol_latex: str = r"\mathbf{V}",
                      v20_clip_D: float = 15.0) -> None:
    """Render the full pyramid (cos + sin sub-panels) for n = 2..6."""
    rr = np.linspace(0, 1, PYRAMID_NPT)
    tt = np.linspace(0, 2 * np.pi, PYRAMID_NPT)
    RHO, THETA = np.meshgrid(rr, tt)
    XX = RHO * np.cos(THETA)
    YY = RHO * np.sin(THETA)

    c20 = alpha.get((2, 0), 0.0)
    others_vmin, others_vmax = _pyramid_scale(RHO, THETA, alpha, basis_dict,
                                              FULL_PYRAMID_LAYOUT, combine_half=False)
    others_levels = np.arange(others_vmin, others_vmax + 0.25, 0.25)
    others_norm = TwoSlopeNorm(vcenter=0, vmin=others_vmin, vmax=others_vmax)

    v20_levels = np.arange(-v20_clip_D, v20_clip_D + 0.25, 0.25)
    v20_norm = TwoSlopeNorm(vcenter=0, vmin=-v20_clip_D, vmax=v20_clip_D)

    nrow = len(FULL_PYRAMID_LAYOUT)
    ncol = len(FULL_PYRAMID_LAYOUT[-1])
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 12))
    fig.suptitle(title, fontsize=15)

    last_v20_ctf = None
    last_others_ctf = None
    for i, row in enumerate(FULL_PYRAMID_LAYOUT):
        for j, nm in enumerate(row):
            ax = axes[i][j]
            if nm is None:
                ax.set_visible(False); continue
            n, m = nm
            c = alpha.get(nm, 0.0)
            arr = c * basis_dict[nm](RHO, THETA)
            if nm == (2, 0):
                arr = arr - c20         # center on 0 for V(2,0) display
                levels = v20_levels; norm = v20_norm
            else:
                levels = others_levels; norm = others_norm
            arr = np.where(RHO <= 1, arr, np.nan)
            ctf = ax.contourf(XX, YY, arr, levels=levels,
                              cmap="jet_r", norm=norm, extend="both")
            cs = ax.contour(XX, YY, arr, levels=levels[::4],
                            colors="k", linewidths=0.5)
            ax.clabel(cs, inline=True, fmt="%.2f")
            if nm == (2, 0):  last_v20_ctf = ctf
            else:             last_others_ctf = ctf
            ax.set_title(rf"${symbol_latex}_{{{n},{m}}}$", fontsize=10)
            ax.text(0.5, -0.1, rf"$\mathbf{{c={c:.2f}}}$",
                    transform=ax.transAxes, ha="center", va="top",
                    color="white", fontsize=11,
                    bbox=dict(facecolor="gray", alpha=0.7))
            ax.axis("off"); ax.set_aspect("equal")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    if last_v20_ctf and last_others_ctf:
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(last_v20_ctf, cax=cax)
        cbar.set_label("Power (D) for V(2,0)", labelpad=15)
        cbar.set_ticks(np.arange(-int(v20_clip_D), int(v20_clip_D) + 1, 5))
        cbar.ax.hlines(c20, 0.0, 1.2, color="red", lw=2,
                       transform=cbar.ax.get_yaxis_transform())
        cbar.ax.text(-0.05, c20, f"{c20:.2f}D", color="red", va="center", ha="right",
                     transform=cbar.ax.get_yaxis_transform(), fontsize=10)
        cax2 = cax.twinx()
        cax2.set_ylabel("Power (D) for other modes", labelpad=15)
        cax2.set_yticks(np.arange(others_vmin, others_vmax + 0.125, 0.25))
        cax2.set_yticklabels([f"{v:.2f}"
                              for v in np.arange(others_vmin, others_vmax + 0.125, 0.25)])
        cax2.set_ylim(others_vmin, others_vmax)

    fig.text(0.02, 0.98, _info_text(case_id, pupil_diameter_mm),
             fontsize=10, va="top", ha="left",
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.savefig(output_dir / filename)
    plt.close(fig)


def plot_pyramid_half(alpha: Dict[Tuple[int, int], float],
                      basis_dict: Dict[Tuple[int, int], Callable],
                      title: str,
                      filename: str,
                      output_dir: Path,
                      case_id: str,
                      pupil_diameter_mm: float,
                      symbol_latex: str = r"\vec{\mathbf{V}}",
                      v20_clip_D: float = 15.0) -> None:
    """Render the half-pyramid (m >= 0) view, combining (cos, sin) per mode."""
    rr = np.linspace(0, 1, PYRAMID_NPT)
    tt = np.linspace(0, 2 * np.pi, PYRAMID_NPT)
    RHO, THETA = np.meshgrid(rr, tt)
    XX = RHO * np.cos(THETA)
    YY = RHO * np.sin(THETA)

    c20 = alpha.get((2, 0), 0.0)
    others_vmin, others_vmax = _pyramid_scale(RHO, THETA, alpha, basis_dict,
                                              HALF_PYRAMID_LAYOUT, combine_half=True)
    others_levels = np.arange(others_vmin, others_vmax + 0.25, 0.25)
    others_norm = TwoSlopeNorm(vcenter=0, vmin=others_vmin, vmax=others_vmax)
    v20_levels = np.arange(-v20_clip_D, v20_clip_D + 0.25, 0.25)
    v20_norm = TwoSlopeNorm(vcenter=0, vmin=-v20_clip_D, vmax=v20_clip_D)

    nrow = len(HALF_PYRAMID_LAYOUT)
    ncol = len(HALF_PYRAMID_LAYOUT[0])
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3 * nrow))
    fig.suptitle(title, fontsize=15)

    last_v20_ctf = None
    last_others_ctf = None
    for i in range(nrow):
        for j in range(ncol):
            ax = axes[i][j]
            nm = HALF_PYRAMID_LAYOUT[i][j]
            if nm is None:
                ax.set_visible(False); continue
            n, m = nm
            a = alpha.get((n, m), 0.0)
            b = alpha.get((n, -m), 0.0) if m > 0 else 0.0
            arr = a * basis_dict[(n, m)](RHO, THETA)
            if m > 0:
                arr = arr + b * basis_dict[(n, -m)](RHO, THETA)
            if nm == (2, 0):
                arr = arr - c20
                levels = v20_levels; norm = v20_norm
            else:
                levels = others_levels; norm = others_norm
            arr = np.where(RHO <= 1, arr, np.nan)
            ctf = ax.contourf(XX, YY, arr, levels=levels,
                              cmap="jet_r", norm=norm, extend="both")
            cs = ax.contour(XX, YY, arr, levels=levels[::4],
                            colors="k", linewidths=0.5)
            ax.clabel(cs, inline=True, fmt="%.2f")
            if nm == (2, 0):  last_v20_ctf = ctf
            else:             last_others_ctf = ctf
            ax.set_title(rf"${symbol_latex}_{{{n},{m}}}$", fontsize=10)
            zmin = np.nanmin(arr); zmax = np.nanmax(arr)
            if m == 0:
                text_str = (rf"$\mathbf{{c={a:.2f}}}$" "\n"
                            rf"$\mathbf{{min={zmin:.2f}\,D}}$" "\n"
                            rf"$\mathbf{{max={zmax:.2f}\,D}}$")
            else:
                mag = math.hypot(a, b)
                axis_deg = canonical_axis_deg(a, b, m)
                text_str = (rf"$\mathbf{{sin={b:.2f}}}$" "\n"
                            rf"$\mathbf{{cos={a:.2f}}}$" "\n"
                            rf"$\mathbf{{magnitude={mag:.2f}}}$" "\n"
                            rf"$\mathbf{{axis={axis_deg:.1f}^\circ}}$" "\n"
                            rf"$\mathbf{{min={zmin:.2f}\,D}}$" "\n"
                            rf"$\mathbf{{max={zmax:.2f}\,D}}$")
            ax.text(0.5, -0.3, text_str, transform=ax.transAxes,
                    ha="center", va="top", color="white", fontsize=10,
                    bbox=dict(facecolor="gray", alpha=0.7))
            ax.axis("off"); ax.set_aspect("equal")

    plt.subplots_adjust(wspace=0.1, hspace=0.3, right=0.85)
    if last_v20_ctf and last_others_ctf:
        cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(last_v20_ctf, cax=cax)
        cbar.set_label("Power (D) for V(2,0)", labelpad=15)
        cbar.set_ticks(np.arange(-int(v20_clip_D), int(v20_clip_D) + 1, 5))
        cbar.ax.hlines(c20, 0.0, 1.2, color="red", lw=2,
                       transform=cbar.ax.get_yaxis_transform())
        cbar.ax.text(-0.05, c20, f"{c20:.2f}D", color="red", va="center", ha="right",
                     transform=cbar.ax.get_yaxis_transform(), fontsize=10)
        cax2 = cax.twinx()
        cax2.set_ylabel("Power (D) for other modes", labelpad=15)
        cax2.set_yticks(np.arange(others_vmin, others_vmax + 0.125, 0.25))
        cax2.set_yticklabels([f"{v:.2f}"
                              for v in np.arange(others_vmin, others_vmax + 0.125, 0.25)])
        cax2.set_ylim(others_vmin, others_vmax)

    fig.text(0.02, 0.98, _info_text(case_id, pupil_diameter_mm),
             fontsize=10, va="top", ha="left",
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.savefig(output_dir / filename)
    plt.close(fig)


# =============================================================================
# SECTION 15 -- FIGURES: HISTOGRAMS
# =============================================================================
# Six families of bar-chart histograms used to summarize the coefficients:
#   * plain bar chart per alpha dict (4 variants)
#   * half-pyramid magnitude histogram (cos + sin -> magnitude per mode)
#   * magnitude-unit histogram (m > 0 modes)
#   * PTV-normalized vector histogram
#   * all-alphas comparison
#   * VL-VH vs tilde comparison
# =============================================================================

def plot_histogram_alpha(alpha: Dict[Tuple[int, int], float],
                         output_dir: Path, case_id: str,
                         pupil_diameter_mm: float,
                         label: str,
                         ylabel: str = "Coefficient value") -> None:
    """Plain bar chart of all coefficients in one alpha dict."""
    keys = sorted(alpha.keys())
    vals = [alpha[k] for k in keys]
    labels = [f"({n},{m})" for n, m in keys]
    colors = ["red" if v < 0 else "blue" for v in vals]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_title(f"Histogram of {label} - {case_id}")
    ax.set_xlabel("Mode (n, m)"); ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    for bar, val in zip(bars, vals):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}",
                ha="center", va="bottom" if h >= 0 else "top", fontsize=9)
    fig.text(0.02, 0.98, _info_text(case_id, pupil_diameter_mm),
             fontsize=9, va="top", ha="left",
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    plt.tight_layout()
    fig.savefig(output_dir / f"Histogram_{label}_{case_id}.png")
    plt.close(fig)


def plot_histogram_pyramid_coefficients(coeffs: Dict[Tuple[int, int], float],
                                         output_dir: Path,
                                         case_id: str,
                                         pupil_diameter_mm: float,
                                         label: str,
                                         ylabel: str = "Coefficient value (um)",
                                         symbol_latex: str = "Z",
                                         n_max: int = N_MAX) -> None:
    """Bar histogram of the 45 pyramid coefficients (Zernike or GM).

    Bars are drawn in standard pyramid order (n = 0, 1, ..., n_max), with a
    light vertical separator between successive orders and an "n = k" label
    underneath each block.  Used for both the input Zernike coefficients and
    the output GM coefficients.
    """
    nm_list = pyramid_indices(n_max)
    values = [float(coeffs.get(nm, 0.0)) for nm in nm_list]
    labels = [f"({n},{m})" for n, m in nm_list]
    colors = ["red" if v < 0 else "blue" for v in values]

    fig, ax = plt.subplots(figsize=(18, 6))
    fig.suptitle(f"Histogram of {label} coefficients - {case_id}", fontsize=14)
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)

    # Vertical separators between successive orders n.
    current_n = nm_list[0][0]
    starts = {current_n: 0}
    for i, (n, _) in enumerate(nm_list[1:], 1):
        if n != current_n:
            ax.axvline(x=i - 0.5, color="#B0B0B0", linewidth=0.8, alpha=0.7, zorder=0)
            starts[n] = i
            current_n = n

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_xlabel("Mode (n, m)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Value above / below each bar
    y_range = max(abs(min(values + [0])), abs(max(values + [0])), 1e-6)
    for bar, val in zip(bars, values):
        if abs(val) < 1e-6:
            continue
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + (0.02 if h >= 0 else -0.02) * y_range,
                f"{val:.3f}",
                ha="center", va="bottom" if h >= 0 else "top", fontsize=7, rotation=0)

    # "n = k" annotation under each block
    y_pos = ax.get_ylim()[0] - 0.10 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for n, start in starts.items():
        end = start + (n + 1) - 1
        center = (start + end) / 2.0
        ax.text(center, y_pos, rf"$n={n}$", ha="center", va="top",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor="gray", alpha=0.9))

    fig.text(0.02, 0.98, _info_text(case_id, pupil_diameter_mm),
             fontsize=9, va="top", ha="left",
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(output_dir / f"Histogram_{label}_{case_id}.png")
    plt.close(fig)


def plot_half_pyramid_histogram(alpha: Dict[Tuple[int, int], float],
                                output_dir: Path, case_id: str,
                                pupil_diameter_mm: float,
                                title_suffix: str = "VL-VH",
                                is_orthogonal: bool = False) -> None:
    """Bar chart over half-pyramid modes (m >= 0), excluding (2, 0) (shown as text)."""
    values: List[float] = []
    labels: List[str] = []
    colors: List[str] = []
    c20 = alpha.get((2, 0), 0.0)
    for row in HALF_PYRAMID_LAYOUT:
        for nm in row:
            if nm is None or nm == (2, 0):
                continue
            n, m = nm
            a = alpha.get((n, m), 0.0)
            symbol = r"\tilde{V}" if is_orthogonal else "V"
            if m == 0:
                value = a
                label = rf"$\mathbf{{{symbol}}}_{{{n},{m}}}$"
                color = "red" if value < 0 else "blue"
            else:
                b = alpha.get((n, -m), 0.0)
                magnitude = math.hypot(a, b)
                axis_deg = canonical_axis_deg(a, b, m)
                value = magnitude
                label = rf"$\mathbf{{{symbol}}}_{{{n},{m}}}\ ({axis_deg:.0f}^\circ)$"
                color = "blue"
            values.append(value); labels.append(label); colors.append(color)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Half-pyramid coefficients ({title_suffix}) - {case_id}", fontsize=14)
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="black")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Coefficient magnitude (D)"); ax.set_xlabel("Mode")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    symbol = r"\tilde{V}" if is_orthogonal else "V"
    fig.text(0.05, 0.95, rf"$\mathbf{{{symbol}}}_{{2,0}} = {c20:.3f}\ \mathrm{{D}}$",
             fontsize=10, va="top", ha="left",
             bbox=dict(facecolor="white", alpha=0.8))
    fig.text(0.02, 0.98, _info_text(case_id, pupil_diameter_mm),
             fontsize=9, va="top", ha="left",
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / f"Half_Pyramid_Histogram_{title_suffix.replace(' ', '_')}_{case_id}.png")
    plt.close(fig)


def plot_magnitude_unit_histogram(alpha: Dict[Tuple[int, int], float],
                                  output_dir: Path, case_id: str,
                                  pupil_diameter_mm: float,
                                  title_suffix: str = "VL-VH",
                                  is_orthogonal: bool = False) -> None:
    """Magnitude/axis histogram for modes with m > 0 only."""
    values: List[float] = []
    labels: List[str] = []
    for n, m in MAG_AXIS_MODES:
        if m <= 0: continue
        a = alpha.get((n,  m), 0.0)
        b = alpha.get((n, -m), 0.0)
        magnitude = math.hypot(a, b)
        axis_deg = canonical_axis_deg(a, b, m) if magnitude > 0 else 0.0
        symbol = r"\tilde{V}" if is_orthogonal else "V"
        values.append(magnitude)
        labels.append(rf"$\mathbf{{{symbol}}}_{{{n},{m}}}\ ({axis_deg:.0f}^\circ)$")

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Magnitude units ({title_suffix}) - {case_id}", fontsize=14)
    bars = ax.bar(range(len(values)), values, color="blue", edgecolor="black")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Magnitude (D)"); ax.set_xlabel("Mode")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.text(0.02, 0.98, _info_text(case_id, pupil_diameter_mm),
             fontsize=9, va="top", ha="left",
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    plt.tight_layout()
    fig.savefig(output_dir / f"Mag_Unit_Histogram_{title_suffix.replace(' ', '_')}_{case_id}.png")
    plt.close(fig)


def plot_ptv_normalized_vector_histogram(alpha: Dict[Tuple[int, int], float],
                                         output_dir: Path, case_id: str,
                                         pupil_diameter_mm: float,
                                         title_suffix: str = "VL-VH",
                                         is_orthogonal: bool = False) -> None:
    """Bar chart of PTV-normalized vector magnitudes (m >= 0 modes)."""
    values: List[float] = []
    labels: List[str] = []
    axis_labels: List[Optional[str]] = []
    colors: List[str] = []

    for row in HALF_PYRAMID_LAYOUT:
        for nm in row:
            if nm is None: continue
            n, m = nm
            symbol_v = r"\vec{\tilde{V}}" if is_orthogonal else r"\vec{\mathbf{V}}"
            if m == 0:
                v = float(alpha.get((n, 0), 0.0))
                ax_lbl = None
                col = "red" if v < 0 else "blue"
            else:
                a = float(alpha.get((n,  m), 0.0))
                b = float(alpha.get((n, -m), 0.0))
                v = math.hypot(a, b)
                ax_lbl = f"({canonical_axis_deg(a, b, m):.1f} deg)"
                col = "blue"
            values.append(v)
            labels.append(rf"${symbol_v}_{{{n},{m}}}$")
            axis_labels.append(ax_lbl); colors.append(col)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(f"PTV-normalized vector coefficient magnitudes ({title_suffix}) - {case_id}",
                 fontsize=14)
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Coefficient value (D)"); ax.set_xlabel("Mode")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    y_min = min([0.0] + values); y_max = max([0.0] + values)
    span = max(y_max - y_min, 1e-12)
    ax.set_ylim(y_min - 0.18 * span, y_max + 0.12 * span)
    for bar, v, ax_lbl in zip(bars, values, axis_labels):
        xm = bar.get_x() + bar.get_width() / 2
        y_txt = v + (0.02 if v >= 0 else -0.02) * span
        ax.text(xm, y_txt, f"{v:.3f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        if ax_lbl:
            ax.text(xm, -0.14, ax_lbl, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8)
    fig.text(0.02, 0.98, _info_text(case_id, pupil_diameter_mm),
             fontsize=9, va="top", ha="left",
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_dir / f"PTV_Normalized_Vector_{title_suffix.replace(' ', '_').replace('-', '_')}_{case_id}.png")
    plt.close(fig)


def plot_all_alpha_histogram(alpha_final, alpha_tilde, alpha_npvV, alpha_npvVT,
                             output_dir: Path, case_id: str,
                             pupil_diameter_mm: float) -> None:
    """Side-by-side bars for all 4 coefficient sets."""
    keys = sorted(alpha_final.keys())
    labels = [f"({n},{m})" for n, m in keys]
    v1 = [alpha_final.get(k, 0.0) for k in keys]
    v2 = [alpha_tilde.get(k, 0.0) for k in keys]
    v3 = [alpha_npvV.get(k, 0.0) for k in keys]
    v4 = [alpha_npvVT.get(k, 0.0) for k in keys]

    x = np.arange(len(labels)); w = 0.2
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.suptitle(f"All coefficients histogram - {case_id}", fontsize=14)
    ax.bar(x - 1.5 * w, v1, w, label="VL-VH",    color="blue")
    ax.bar(x - 0.5 * w, v2, w, label="Tilde",    color="green")
    ax.bar(x + 0.5 * w, v3, w, label="npvV",     color="red")
    ax.bar(x + 1.5 * w, v4, w, label="npvVT",    color="purple")
    ax.set_xlabel("Mode (n, m)"); ax.set_ylabel("Coefficient value")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(); ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    fig.text(0.02, 0.98, _info_text(case_id, pupil_diameter_mm),
             fontsize=9, va="top", ha="left",
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    plt.tight_layout()
    fig.savefig(output_dir / f"All_Alpha_Histogram_{case_id}.png")
    plt.close(fig)


def plot_vlvh_vs_tilde_comparison(alpha_final, alpha_tilde,
                                  output_dir: Path, case_id: str,
                                  pupil_diameter_mm: float) -> None:
    """Side-by-side comparison of VL-VH and tilde coefficients."""
    modes = VERGENCE_MODES
    vlvh = [alpha_final.get(k, 0.0) for k in modes]
    orth = [alpha_tilde.get(k, 0.0) for k in modes]
    labels = [f"({n},{m})" for n, m in modes]

    fig, ax = plt.subplots(figsize=(20, 15), facecolor="white")
    ax.set_facecolor("white")
    fig.suptitle("VL-VH vs Orthogonal (tilde) coefficient comparison", fontsize=30)

    x = np.arange(len(labels)); w = 0.35; spacing = 0.05

    # Vertical separators on n changes
    current_n = modes[0][0]
    for i, (n, _) in enumerate(modes[1:], 1):
        if n != current_n:
            ax.axvline(x=i - 0.5, color="#D3D3D3", linewidth=0.9, alpha=0.7, zorder=1)
            current_n = n

    pos_v, neg_v = "#2E86AB", "#E63946"
    pos_t, neg_t = "#7FCDFF", "#FF9999"
    bars1 = ax.bar(x - w / 2 - spacing / 2, vlvh, w, label="VL-VH",
                   color=[pos_v if v >= 0 else neg_v for v in vlvh],
                   edgecolor="black", linewidth=1.2, zorder=2)
    bars2 = ax.bar(x + w / 2 + spacing / 2, orth, w, label="Tilde",
                   color=[pos_t if v >= 0 else neg_t for v in orth],
                   edgecolor="black", linewidth=1.2, hatch="xx", zorder=2)
    for bars, vals in [(bars1, vlvh), (bars2, orth)]:
        for bar, v in zip(bars, vals):
            h = bar.get_height()
            if abs(h) < 0.01: continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + (0.02 if h >= 0 else -0.02),
                    f"{v:.2f}", ha="center",
                    va="bottom" if h >= 0 else "top",
                    fontsize=17, rotation=45)

    ax.set_xlabel("Mode (n, m)", fontsize=18)
    ax.set_ylabel("Coefficient value (D)", fontsize=18)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    legend_items = [
        Patch(facecolor=pos_v, edgecolor="black", label="VL-VH (positive)"),
        Patch(facecolor=neg_v, edgecolor="black", label="VL-VH (negative)"),
        Patch(facecolor=pos_t, edgecolor="black", label="Tilde (positive)", hatch="xx"),
        Patch(facecolor=neg_t, edgecolor="black", label="Tilde (negative)", hatch="xx"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=18, framealpha=0.9)

    all_v = vlvh + orth
    y_max = max(abs(min(all_v)), max(all_v)) * 1.2 if all_v else 0.5
    y_max = np.ceil(y_max * 4) / 4
    ax.set_yticks(np.arange(-y_max, y_max + 0.125, 0.25))
    ax.set_ylim(-y_max, y_max)
    ax.axhline(0, color="black", linewidth=1.5, zorder=2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, which="major", zorder=0)

    fig.text(0.01, 0.99, _info_text(case_id, pupil_diameter_mm),
             fontsize=17, va="top", ha="left",
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    vlvh_rms = float(np.sqrt(np.mean(np.array(vlvh) ** 2))) if vlvh else 0.0
    tilde_rms = float(np.sqrt(np.mean(np.array(orth) ** 2))) if orth else 0.0
    corr = float(np.corrcoef(vlvh, orth)[0, 1]) if len(vlvh) > 1 else 0.0
    fig.text(0.85, 0.15,
             f"VL-VH RMS: {vlvh_rms:.3f} D\nTilde RMS: {tilde_rms:.3f} D\nCorrelation: {corr:.2f}",
             fontsize=18, va="bottom", ha="left",
             bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="black", linewidth=1.2))
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output_dir / f"VLVH_vs_Tilde_Comparison_{case_id}.png",
                dpi=SAVE_DPI, facecolor="white")
    plt.close(fig)


# =============================================================================
# SECTION 16 -- INPUT / OUTPUT (SINGLE-CASE)
# =============================================================================
# Read a Zernike vector in pyramid order from CSV (two accepted formats).
# Write the full coefficient dictionary to a tabular CSV summary.
# =============================================================================

def read_zernike_vector(path: Path, n_max: int = N_MAX
                        ) -> Dict[Tuple[int, int], float]:
    """Read 45 Zernike coefficients from CSV; return {(n, m): value}.

    Two formats are supported:

    Format A -- one coefficient per line, in pyramid order:

        0.0
        0.1
        -0.05
        ...
        (45 lines for n_max = 8)

    Format B -- 3 columns with header n,m,coeff (any row order):

        n,m,coeff
        0,0,0.0
        1,-1,0.1
        1,1,-0.05
        2,-2,0.02
        ...
    """
    path = Path(path)
    nm_list = pyramid_indices(n_max)
    n_expected = len(nm_list)

    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        text = fh.read()

    # Detect format B by header keywords.
    first_line = text.splitlines()[0].strip().lower() if text.strip() else ""
    is_format_b = ("n" in first_line and "m" in first_line and
                   ("coeff" in first_line or "value" in first_line))

    if is_format_b:
        reader = csv.DictReader(text.splitlines())
        # Accept any column-name variations.
        n_col = m_col = c_col = None
        for fld in reader.fieldnames or []:
            ll = fld.strip().lower()
            if ll == "n": n_col = fld
            elif ll == "m": m_col = fld
            elif ll in ("coeff", "coefficient", "value"): c_col = fld
        if not (n_col and m_col and c_col):
            raise ValueError(
                f"CSV header must include columns 'n', 'm', 'coeff' (got {reader.fieldnames})."
            )
        out: Dict[Tuple[int, int], float] = {nm: 0.0 for nm in nm_list}
        for row in reader:
            n = int(row[n_col]); m = int(row[m_col])
            if (n, m) not in out:
                continue
            out[(n, m)] = float(row[c_col])
        return out

    # Format A: one coefficient per line, in pyramid order.
    raw = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    # Tolerate the user putting all 45 values comma-separated on a single line.
    if len(raw) == 1 and ("," in raw[0] or ";" in raw[0] or "\t" in raw[0]):
        sep = "," if "," in raw[0] else (";" if ";" in raw[0] else "\t")
        raw = [v.strip() for v in raw[0].split(sep) if v.strip()]
    if len(raw) != n_expected:
        raise ValueError(
            f"Expected {n_expected} Zernike coefficients in pyramid order, got {len(raw)}."
        )
    coeffs = [float(v) for v in raw]
    return vector_to_dict(coeffs, n_max=n_max)


def write_coefficients_csv(case_id: str,
                            pupil_diameter_mm: float,
                            zernike: Dict[Tuple[int, int], float],
                            gm: Dict[Tuple[int, int], float],
                            alpha_vlvh: Dict[Tuple[int, int], float],
                            alpha_tilde: Dict[Tuple[int, int], float],
                            alpha_npvV: Dict[Tuple[int, int], float],
                            alpha_npvVT: Dict[Tuple[int, int], float],
                            ref_vlvh: Tuple[float, float, float],
                            ref_vlvh_v12: Tuple[float, float, float],
                            ref_tilde: Tuple[float, float, float],
                            ref_tilde_v12: Tuple[float, float, float],
                            stats_vlvh: Dict[str, Dict[str, float]],
                            stats_tilde: Dict[str, Dict[str, float]],
                            output_dir: Path) -> Path:
    """Write a tabular CSV with all derived coefficients and refractions."""
    output_dir = Path(output_dir)
    out_path = output_dir / f"{case_id}_coefficients.csv"

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# case_id", case_id])
        w.writerow(["# pupil_diameter_mm", f"{pupil_diameter_mm:.4f}"])
        w.writerow([])

        # Section: Zernike
        w.writerow(["section", "n", "m", "value_um"])
        for nm in pyramid_indices(N_MAX):
            n, m = nm
            w.writerow(["Zernike", n, m, f"{zernike.get(nm, 0.0):.10g}"])
        w.writerow([])

        # Section: GM
        w.writerow(["section", "n", "m", "value_um"])
        for nm in pyramid_indices(N_MAX):
            n, m = nm
            w.writerow(["GM", n, m, f"{gm.get(nm, 0.0):.10g}"])
        w.writerow([])

        # Section: VL-VH, tilde, npvV, npvVT
        w.writerow(["section", "n", "m", "value_D"])
        for nm in VERGENCE_MODES:
            n, m = nm
            w.writerow(["VL-VH",  n, m, f"{alpha_vlvh.get(nm, 0.0):.6f}"])
            w.writerow(["Tilde",  n, m, f"{alpha_tilde.get(nm, 0.0):.6f}"])
            w.writerow(["npvV",   n, m, f"{alpha_npvV.get(nm, 0.0):.6f}"])
            w.writerow(["npvVT",  n, m, f"{alpha_npvVT.get(nm, 0.0):.6f}"])
        w.writerow([])

        # Section: refractions
        w.writerow(["section", "plane", "S_D", "C_D", "A_deg"])
        w.writerow(["Refraction VL-VH",  "cornea",     *[f"{v:.3f}" for v in ref_vlvh]])
        w.writerow(["Refraction VL-VH",  "spectacle",  *[f"{v:.3f}" for v in ref_vlvh_v12]])
        w.writerow(["Refraction Tilde",  "cornea",     *[f"{v:.3f}" for v in ref_tilde]])
        w.writerow(["Refraction Tilde",  "spectacle",  *[f"{v:.3f}" for v in ref_tilde_v12]])
        w.writerow([])

        # Section: map statistics
        w.writerow(["section", "basis", "component", "min", "q1", "median", "q3",
                    "max", "mean", "std", "rms"])
        for basis_name, stats in [("VL-VH", stats_vlvh), ("Tilde", stats_tilde)]:
            for comp in ("TOT", "VL", "VH"):
                s = stats[comp]
                w.writerow(["Map stats", basis_name, comp,
                            f"{s['min']:.4f}", f"{s['q1']:.4f}",
                            f"{s['median']:.4f}", f"{s['q3']:.4f}",
                            f"{s['max']:.4f}", f"{s['mean']:.4f}",
                            f"{s['std']:.4f}", f"{s['rms']:.4f}"])
    return out_path


def write_summary_txt(case_id: str,
                       pupil_diameter_mm: float,
                       ref_vlvh: Tuple[float, float, float],
                       ref_vlvh_v12: Tuple[float, float, float],
                       ref_tilde: Tuple[float, float, float],
                       ref_tilde_v12: Tuple[float, float, float],
                       stats_vlvh: Dict[str, Dict[str, float]],
                       stats_tilde: Dict[str, Dict[str, float]],
                       output_dir: Path) -> Path:
    """Write a short human-readable summary."""
    out_path = Path(output_dir) / f"{case_id}_summary.txt"
    lines = []
    lines.append(f"Case: {case_id}")
    lines.append(f"Pupil diameter (mm): {pupil_diameter_mm:.3f}")
    lines.append("")
    lines.append("Refraction (VL-VH basis):")
    lines.append(f"   cornea     S = {ref_vlvh[0]:+.2f} D   C = {ref_vlvh[1]:+.2f} D   A = {ref_vlvh[2]:.0f} deg")
    lines.append(f"   spectacle  S = {ref_vlvh_v12[0]:+.2f} D   C = {ref_vlvh_v12[1]:+.2f} D   A = {ref_vlvh_v12[2]:.0f} deg")
    lines.append("")
    lines.append("Refraction (tilde basis):")
    lines.append(f"   cornea     S = {ref_tilde[0]:+.2f} D   C = {ref_tilde[1]:+.2f} D   A = {ref_tilde[2]:.0f} deg")
    lines.append(f"   spectacle  S = {ref_tilde_v12[0]:+.2f} D   C = {ref_tilde_v12[1]:+.2f} D   A = {ref_tilde_v12[2]:.0f} deg")
    lines.append("")
    lines.append("VH (high-order) map statistics, VL-VH basis:")
    s = stats_vlvh["VH"]
    lines.append(f"   RMS = {s['rms']:.3f} D    Mean +/- SD = {s['mean']:.3f} +/- {s['std']:.3f} D")
    lines.append("VH (high-order) map statistics, tilde basis:")
    s = stats_tilde["VH"]
    lines.append(f"   RMS = {s['rms']:.3f} D    Mean +/- SD = {s['mean']:.3f} +/- {s['std']:.3f} D")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# =============================================================================
# SECTION 17 -- OPTIONAL: SUB-PUPIL ZERNIKE TRANSFORM (UTILITY)
# =============================================================================
# This block is *not* used by the main single-case pipeline.  It is kept here
# so that a future multi-diameter sweep (e.g., evaluate the same wavefront on
# 6.0 / 5.0 / 4.0 mm) can be added without re-introducing OPDscan-specific
# code.
#
# Convention
# ----------
# Output pupil is always the unit disk (x^2 + y^2 <= 1).
# Old coordinates expressed in the *old* pupil normalization:
#       x_old = S * x + dx
#       y_old = S * y + dy
# with:
#       S  = S2 / S1                  (ratio of pupil radii)
#       dx = decenter_x_mm / (S1 / 2)
#       dy = decenter_y_mm / (S1 / 2)
# =============================================================================

_SUBPUPIL_T_CACHE: Dict[Tuple[int, int, str, str, str], Tuple[List[Tuple[int, int]], List[List[Any]]]] = {}


def _shift_terms(terms_in, S, dx, dy):
    out = {}
    for (px, py), c in terms_in.items():
        for i in range(px + 1):
            ci = mpmath.mpf(math.comb(px, i)) * (S ** i) * (dx ** (px - i))
            for j in range(py + 1):
                cj = mpmath.mpf(math.comb(py, j)) * (S ** j) * (dy ** (py - j))
                key = (i, j)
                out[key] = out.get(key, mpmath.mpf("0")) + c * ci * cj
    return out


def subpupil_transform_matrix(n_max: int,
                              S1_mm: float, S2_mm: float,
                              decenter_mm: float = 0.0,
                              decenter_angle_deg: float = 0.0,
                              dps: int = MP_DPS):
    """Build the matrix T mapping input Zernike coeffs -> sub-pupil coeffs."""
    if S1_mm <= 0 or S2_mm <= 0:
        raise ValueError("S1_mm and S2_mm must be > 0.")

    S = mpmath.mpf(S2_mm) / mpmath.mpf(S1_mm)
    a = math.radians(float(decenter_angle_deg))
    R1 = float(S1_mm) / 2.0
    dx = mpmath.mpf(float(decenter_mm) * math.cos(a) / R1)
    dy = mpmath.mpf(float(decenter_mm) * math.sin(a) / R1)

    if float(S + mpmath.sqrt(dx * dx + dy * dy)) > 1.0 + 1e-12:
        raise ValueError("Sub-pupil does not fit inside the original pupil.")

    key = (int(n_max), int(dps),
           mpmath.nstr(S, 40), mpmath.nstr(dx, 40), mpmath.nstr(dy, 40))
    if key in _SUBPUPIL_T_CACHE:
        return _SUBPUPIL_T_CACHE[key]

    old_dps = mpmath.mp.dps; mpmath.mp.dps = dps
    nm_list = pyramid_indices(n_max)
    z_terms = _get_zernike_poly_terms(n_max, dps=dps)
    shifted = {nm: _shift_terms(z_terms[nm], S, dx, dy) for nm in nm_list}
    N = len(nm_list)
    T = [[mpmath.mpf("0")] * N for _ in range(N)]
    for i, nm_out in enumerate(nm_list):
        terms_out = z_terms[nm_out]
        for j, nm_in in enumerate(nm_list):
            T[i][j] = _inner_product_terms(terms_out, shifted[nm_in])
    _SUBPUPIL_T_CACHE[key] = (nm_list, T)
    mpmath.mp.dps = old_dps
    return nm_list, T


def apply_subpupil_transform(z_dict: Dict[Tuple[int, int], float],
                             S1_mm: float, S2_mm: float,
                             decenter_mm: float = 0.0,
                             decenter_angle_deg: float = 0.0,
                             n_max: int = N_MAX) -> Dict[Tuple[int, int], float]:
    """Recompute Zernike coefficients on a (possibly decentered) sub-pupil.

    Exact analytic transform (no numerical quadrature).  Returns a dict of
    Zernike coefficients on the new pupil.
    """
    nm_list, T = subpupil_transform_matrix(n_max, S1_mm, S2_mm,
                                           decenter_mm, decenter_angle_deg,
                                           dps=MP_DPS)
    old_dps = mpmath.mp.dps; mpmath.mp.dps = MP_DPS
    c_in = [mpmath.mpf(z_dict.get(nm, 0.0)) for nm in nm_list]
    c_out = []
    for i in range(len(nm_list)):
        s = mpmath.mpf("0")
        row = T[i]
        for j in range(len(nm_list)):
            s += row[j] * c_in[j]
        c_out.append(float(s))
    mpmath.mp.dps = old_dps
    return {nm_list[i]: c_out[i] for i in range(len(nm_list))}


# =============================================================================
# SECTION 18 -- SINGLE-CASE PIPELINE
# =============================================================================
# The function below is the *only* entry point a third-party integrator needs.
# It runs the whole math (Z -> GM -> vergence -> VL/VH -> tilde -> refractions)
# and saves all artifacts (CSV + figures + summary) to `output_dir`.
# =============================================================================

def process_case(zernike: Dict[Tuple[int, int], float],
                 pupil_diameter_mm: float,
                 output_dir: Path,
                 case_id: str,
                 generate_figures: bool = True,
                 vl_fit_radius_mm: Optional[float] = None) -> Dict[str, Any]:
    """Run the complete single-case pipeline.

    Parameters
    ----------
    zernike            : OSA Zernike coefficients (microns), {(n, m): value}.
    pupil_diameter_mm  : Diameter (mm) at which the Zernike expansion was
                         measured.  The vergence maps use radius = diameter / 2.
    output_dir         : Directory where all artifacts are saved.
    case_id            : Short identifier used to name output files.
    generate_figures   : If False, only numerical outputs (CSV) are saved.
    vl_fit_radius_mm   : Override the default central V_L fit radius (default
                         is 0.6 mm = 20 % of a 6 mm reference pupil).

    Returns
    -------
    A dictionary holding every computed quantity, useful for tests and for
    chaining downstream analyses.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    radius_mm = float(pupil_diameter_mm) / 2.0

    # ----- 1. GM coefficients from Zernike (exact analytic projection) ------
    gm = project_zernike_to_gm(zernike, n_max=N_MAX, dps=MP_DPS)
    gm_noPT = dict(gm)
    for nm in [(0, 0), (1, -1), (1, 1)]:
        gm_noPT[nm] = 0.0

    # ----- 2. Wavefront expression (microns) and vergence map (D) -----------
    # CRITICAL: piston and tilt must be removed in the GM basis, NOT in the
    # Zernike basis.  Reason: high-order Zernike modes carry lower-degree
    # radial terms.  For instance,
    #       Z(4, 0) = sqrt(5) * (6 r^4 - 6 r^2 + 1)
    # contains a constant term and a r^2 term.  Zeroing only Z(0, 0) and
    # Z(1, +/-1) therefore leaves parasitic central content embedded in
    # Z(4, 0), Z(6, 0), ...  Under the 1/rho factor in compute_vergence(),
    # this content would contaminate the central vergence.
    #
    # In the GM basis the piston/defocus content carried by all Z(n, 0) modes
    # is redistributed into GM(0, 0) and GM(2, 0).  Zeroing GM(0, 0) and
    # GM(1, +/-1) and then re-expanding the wavefront on the GM basis removes
    # piston and tilt cleanly without touching the higher-order content.
    W_noPT_expr = wavefront_from_gm(gm_noPT)

    r_grid, t_grid, mask = build_polar_grid(GRID_NR, GRID_NT)
    V_total = compute_vergence(W_noPT_expr, radius_mm, r_grid, t_grid, mask)
    V_total = np.nan_to_num(V_total, nan=0.0)
    V_total = np.where(mask, V_total, np.nan)

    # ----- 3. Decompositions (VL-VH and tilde) -----------------------------
    alpha_vlvh, vl_map, vh_map = decompose_VL_VH(
        np.where(mask, V_total, 0.0),
        pupil_diameter_mm,
        r_grid, t_grid, mask,
        vl_fit_radius_mm=vl_fit_radius_mm,
        nr=GRID_NR, nt=GRID_NT,
    )
    alpha_tilde, vl_tilde, vh_tilde = decompose_tilde(
        np.where(mask, V_total, 0.0),
        r_grid, t_grid, mask,
        nr=GRID_NR, nt=GRID_NT,
    )

    # ----- 4. PTV-normalized coefficients ----------------------------------
    alpha_npvV  = scale_alpha_by_PTV(alpha_vlvh,  V_PTV)
    alpha_npvVT = scale_alpha_by_PTV(alpha_tilde, VT_PTV)

    # ----- 5. Refractions (corneal plane + spectacle plane at 12 mm) -------
    # VL-VH refraction: derived directly from the low-degree VL coefficients
    # (n = 2) returned by decompose_VL_VH().
    S, C, A = extract_sca(alpha_vlvh)
    S_v, C_v, A_v = vertex_convert_sca(S, C, A, vertex_m=VERTEX_M)
    A_v_int = int(round(A_v)) % 180

    # Tilde refraction: matches the original OPDscan pipeline, which
    # reconstructs the full tilde map (sum over n = 2..6) and then re-projects
    # this reconstructed map onto VT(2, -2), VT(2, 0), VT(2, 2) via a
    # full-disk LSQ fit.  Because VT is orthonormal, the result is numerically
    # very close to alpha_tilde[(2, m)] but can differ by one degree on the
    # rounded axis due to integration sampling -- so we reproduce the exact
    # same step here.
    VT_dict = build_Vtilde_basis()
    tilde_tot_map = np.zeros_like(r_grid)
    for nm in VERGENCE_MODES:
        n, m = nm
        tilde_tot_map = tilde_tot_map + alpha_tilde[nm] * VT_dict[nm](r_grid, t_grid)
    tilde_tot_map = np.where(mask, tilde_tot_map, np.nan)

    c2m2_t = _projection_lsq(tilde_tot_map, VT_dict[(2, -2)],
                             r_grid, t_grid, mask, GRID_NR, GRID_NT)
    c20_t  = _projection_lsq(tilde_tot_map, VT_dict[(2,  0)],
                             r_grid, t_grid, mask, GRID_NR, GRID_NT)
    c22_t  = _projection_lsq(tilde_tot_map, VT_dict[(2,  2)],
                             r_grid, t_grid, mask, GRID_NR, GRID_NT)
    amp2_t = math.sqrt(c2m2_t ** 2 + c22_t ** 2)
    C_t = -2.0 * math.sqrt(2) * amp2_t
    S_t = c20_t + math.sqrt(2) * amp2_t
    A_t = math.degrees(0.5 * math.atan2(c2m2_t, c22_t)) % 180.0
    A_t = float(round(A_t))
    S_tv, C_tv, A_tv = vertex_convert_sca(S_t, C_t, A_t, vertex_m=VERTEX_M)
    A_tv_int = int(round(A_tv)) % 180

    ref_text = (
        f"S = {S:.2f}, C = {C:.2f} x A = {int(round(A)) % 180} deg\n"
        f"(12 mm: S = {S_v:.2f}, C = {C_v:.2f} x A = {A_v_int} deg)"
    )

    # ----- 6. Descriptive map statistics -----------------------------------
    stats_vlvh = {
        "TOT": map_stats(np.where(mask, V_total, np.nan), remove_origin=True),
        "VL":  map_stats(vl_map),
        "VH":  map_stats(vh_map),
    }
    stats_tilde = {
        "TOT": map_stats(np.where(mask, V_total, np.nan), remove_origin=True),
        "VL":  map_stats(vl_tilde),
        "VH":  map_stats(vh_tilde),
    }

    # ----- 7. Write coefficients and summary -------------------------------
    csv_path = write_coefficients_csv(
        case_id=case_id,
        pupil_diameter_mm=pupil_diameter_mm,
        zernike=zernike, gm=gm,
        alpha_vlvh=alpha_vlvh, alpha_tilde=alpha_tilde,
        alpha_npvV=alpha_npvV,  alpha_npvVT=alpha_npvVT,
        ref_vlvh=(S, C, int(round(A)) % 180),
        ref_vlvh_v12=(S_v, C_v, A_v_int),
        ref_tilde=(S_t, C_t, int(round(A_t)) % 180),
        ref_tilde_v12=(S_tv, C_tv, A_tv_int),
        stats_vlvh=stats_vlvh, stats_tilde=stats_tilde,
        output_dir=output_dir,
    )
    txt_path = write_summary_txt(
        case_id=case_id,
        pupil_diameter_mm=pupil_diameter_mm,
        ref_vlvh=(S, C, int(round(A)) % 180),
        ref_vlvh_v12=(S_v, C_v, A_v_int),
        ref_tilde=(S_t, C_t, int(round(A_t)) % 180),
        ref_tilde_v12=(S_tv, C_tv, A_tv_int),
        stats_vlvh=stats_vlvh, stats_tilde=stats_tilde,
        output_dir=output_dir,
    )

    # ----- 8. Figures ------------------------------------------------------
    if generate_figures:
        wave_cmap = LinearSegmentedColormap.from_list(
            "centered_cmap", ["blue", "cyan", "green", "yellow", "orange", "red"]
        )

        # 8a. Wavefronts (Z vs GM + GM no piston/tilt)
        plot_wavefront_z_vs_gm(
            zernike, gm, gm_noPT,
            radius_mm=radius_mm,
            r_grid=r_grid, t_grid=t_grid, mask=mask,
            output_dir=output_dir, case_id=case_id,
            pupil_diameter_mm=pupil_diameter_mm,
            wavefront_cmap=wave_cmap,
        )

        # 8b. Vergence triptychs (VL-VH and tilde)
        plot_decomposition(
            np.where(mask, V_total, np.nan), vl_map, vh_map,
            refraction_text=ref_text,
            astig_axis_deg=int(round(A)) % 180,
            radius_mm=radius_mm,
            r_grid=r_grid, t_grid=t_grid,
            output_dir=output_dir, case_id=case_id,
            pupil_diameter_mm=pupil_diameter_mm,
            basis_label="VL-VH",
        )
        plot_decomposition(
            np.where(mask, V_total, np.nan), vl_tilde, vh_tilde,
            refraction_text=(
                f"S = {S_t:.2f}, C = {C_t:.2f} x A = {int(round(A_t)) % 180} deg\n"
                f"(12 mm: S = {S_tv:.2f}, C = {C_tv:.2f} x A = {A_tv_int} deg)"
            ),
            astig_axis_deg=int(round(A_t)) % 180,
            radius_mm=radius_mm,
            r_grid=r_grid, t_grid=t_grid,
            output_dir=output_dir, case_id=case_id,
            pupil_diameter_mm=pupil_diameter_mm,
            basis_label="Tilde",
        )

        # 8c. Pyramids (full + half) for each of the four bases
        V_dict   = build_V_basis()
        VT_dict  = build_Vtilde_basis()
        npvV_dict  = build_normalized_basis(V_dict,  V_PTV)
        npvVT_dict = build_normalized_basis(VT_dict, VT_PTV)

        plot_pyramid_full(alpha_vlvh, V_dict,
                          title="Pyramid: VL-VH decomposition (n = 2..6)",
                          filename=f"Pyramid_VL_VH_n2_6_{case_id}.png",
                          output_dir=output_dir, case_id=case_id,
                          pupil_diameter_mm=pupil_diameter_mm,
                          symbol_latex=r"\mathbf{V}")
        plot_pyramid_half(alpha_vlvh, V_dict,
                          title="Half pyramid (m >= 0, VL-VH)",
                          filename=f"Half_pyramid_VL_VH_{case_id}.png",
                          output_dir=output_dir, case_id=case_id,
                          pupil_diameter_mm=pupil_diameter_mm,
                          symbol_latex=r"\vec{\mathbf{V}}")

        plot_pyramid_full(alpha_tilde, VT_dict,
                          title="Pyramid: tilde (orthogonal) decomposition (n = 2..6)",
                          filename=f"Pyramid_Tilde_n2_6_{case_id}.png",
                          output_dir=output_dir, case_id=case_id,
                          pupil_diameter_mm=pupil_diameter_mm,
                          symbol_latex=r"\tilde{V}")
        plot_pyramid_half(alpha_tilde, VT_dict,
                          title="Half pyramid (m >= 0, tilde)",
                          filename=f"Half_pyramid_Tilde_{case_id}.png",
                          output_dir=output_dir, case_id=case_id,
                          pupil_diameter_mm=pupil_diameter_mm,
                          symbol_latex=r"\vec{\tilde{V}}")

        plot_pyramid_full(alpha_npvV, npvV_dict,
                          title="Pyramid: PTV-normalized VL-VH (npvV)",
                          filename=f"Pyramid_npvV_n2_6_{case_id}.png",
                          output_dir=output_dir, case_id=case_id,
                          pupil_diameter_mm=pupil_diameter_mm,
                          symbol_latex=r"\mathbf{V}")
        plot_pyramid_half(alpha_npvV, npvV_dict,
                          title="Half pyramid (m >= 0, PTV-normalized VL-VH)",
                          filename=f"Half_pyramid_npvV_{case_id}.png",
                          output_dir=output_dir, case_id=case_id,
                          pupil_diameter_mm=pupil_diameter_mm,
                          symbol_latex=r"\vec{\mathbf{V}}")
        plot_pyramid_full(alpha_npvVT, npvVT_dict,
                          title="Pyramid: PTV-normalized tilde (npvVT)",
                          filename=f"Pyramid_npvVT_n2_6_{case_id}.png",
                          output_dir=output_dir, case_id=case_id,
                          pupil_diameter_mm=pupil_diameter_mm,
                          symbol_latex=r"\tilde{V}")
        plot_pyramid_half(alpha_npvVT, npvVT_dict,
                          title="Half pyramid (m >= 0, PTV-normalized tilde)",
                          filename=f"Half_pyramid_npvVT_{case_id}.png",
                          output_dir=output_dir, case_id=case_id,
                          pupil_diameter_mm=pupil_diameter_mm,
                          symbol_latex=r"\vec{\tilde{V}}")

        # 8d. Histograms

        # Input Zernike (45 coefficients, microns) and derived GM (45, microns).
        plot_histogram_pyramid_coefficients(
            zernike, output_dir, case_id, pupil_diameter_mm,
            label="Zernike", ylabel="Zernike coefficient (um)", symbol_latex="Z",
        )
        plot_histogram_pyramid_coefficients(
            gm, output_dir, case_id, pupil_diameter_mm,
            label="GM", ylabel="GM coefficient (um)", symbol_latex="GM",
        )

        plot_histogram_alpha(alpha_vlvh,  output_dir, case_id, pupil_diameter_mm, "alpha_VLVH")
        plot_histogram_alpha(alpha_tilde, output_dir, case_id, pupil_diameter_mm, "alpha_Tilde")
        plot_histogram_alpha(alpha_npvV,  output_dir, case_id, pupil_diameter_mm, "alpha_npvV",
                             ylabel="Coefficient value (PTV-normalized)")
        plot_histogram_alpha(alpha_npvVT, output_dir, case_id, pupil_diameter_mm, "alpha_npvVT",
                             ylabel="Coefficient value (PTV-normalized)")
        plot_half_pyramid_histogram(alpha_vlvh,  output_dir, case_id, pupil_diameter_mm,
                                    title_suffix="VL-VH",      is_orthogonal=False)
        plot_half_pyramid_histogram(alpha_tilde, output_dir, case_id, pupil_diameter_mm,
                                    title_suffix="Tilde",      is_orthogonal=True)
        plot_half_pyramid_histogram(alpha_npvV,  output_dir, case_id, pupil_diameter_mm,
                                    title_suffix="npvV",       is_orthogonal=False)
        plot_half_pyramid_histogram(alpha_npvVT, output_dir, case_id, pupil_diameter_mm,
                                    title_suffix="npvVT",      is_orthogonal=True)
        plot_magnitude_unit_histogram(alpha_vlvh,  output_dir, case_id, pupil_diameter_mm,
                                      title_suffix="VL-VH", is_orthogonal=False)
        plot_magnitude_unit_histogram(alpha_tilde, output_dir, case_id, pupil_diameter_mm,
                                      title_suffix="Tilde", is_orthogonal=True)
        plot_ptv_normalized_vector_histogram(alpha_npvV,  output_dir, case_id, pupil_diameter_mm,
                                             title_suffix="PTV VL-VH", is_orthogonal=False)
        plot_ptv_normalized_vector_histogram(alpha_npvVT, output_dir, case_id, pupil_diameter_mm,
                                             title_suffix="PTV Tilde", is_orthogonal=True)
        plot_all_alpha_histogram(alpha_vlvh, alpha_tilde, alpha_npvV, alpha_npvVT,
                                  output_dir, case_id, pupil_diameter_mm)
        plot_vlvh_vs_tilde_comparison(alpha_vlvh, alpha_tilde,
                                       output_dir, case_id, pupil_diameter_mm)

    return {
        "case_id": case_id,
        "pupil_diameter_mm": float(pupil_diameter_mm),
        "zernike": zernike,
        "gm": gm,
        "gm_noPT": gm_noPT,
        "alpha_vlvh": alpha_vlvh,
        "alpha_tilde": alpha_tilde,
        "alpha_npvV": alpha_npvV,
        "alpha_npvVT": alpha_npvVT,
        "refraction_vlvh": {
            "cornea":    {"S": S,   "C": C,   "A": int(round(A))   % 180},
            "spectacle": {"S": S_v, "C": C_v, "A": A_v_int},
        },
        "refraction_tilde": {
            "cornea":    {"S": S_t,  "C": C_t,  "A": int(round(A_t)) % 180},
            "spectacle": {"S": S_tv, "C": C_tv, "A": A_tv_int},
        },
        "map_stats_vlvh":  stats_vlvh,
        "map_stats_tilde": stats_tilde,
        "files": {"csv": str(csv_path), "summary_txt": str(txt_path)},
    }


# =============================================================================
# SECTION 19 -- BATCH HOOK (STUB FOR FUTURE EXTENSION)
# =============================================================================
# Single-case for now.  Below is a placeholder: a future implementation can
# iterate over a directory of CSV inputs and call `process_case` on each.
# =============================================================================

def process_batch(input_dir: Path,
                  output_dir: Path,
                  pupil_diameter_mm: float,
                  generate_figures: bool = True) -> List[Dict[str, Any]]:
    """Run `process_case` on every *.csv file in `input_dir`.

    The case_id is taken from the file stem.  The summary CSV produced for
    each case is left in `output_dir/<case_id>/`.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sorted(input_dir.glob("*.csv")):
        case_id = path.stem
        case_dir = output_dir / case_id
        zernike = read_zernike_vector(path)
        results.append(
            process_case(zernike, pupil_diameter_mm, case_dir, case_id,
                         generate_figures=generate_figures)
        )
    return results


# =============================================================================
# SECTION 20 -- COMMAND-LINE INTERFACE
# =============================================================================
# Two modes:
#   * `--zernike <file>` to run a single case.
#   * `--batch <dir>`    to run all CSV files in a directory.
# =============================================================================

# -----------------------------------------------------------------------------
# Default paths used when the script is run without any CLI argument.
# These are resolved relative to this file's location so the script works
# out of the box when launched from any IDE (Spyder, PyCharm, VSCode...) or
# directly from the terminal:
#       python tracey_zernike_to_vergence_v1.py
# -----------------------------------------------------------------------------
SCRIPT_DIR             = Path(__file__).resolve().parent
DEFAULT_ZERNIKE_CSV    = SCRIPT_DIR / "fake_zernike_test.csv"
DEFAULT_DIAMETER_MM    = 6.0
DEFAULT_CASE_ID        = "FakeTest"
DEFAULT_OUTPUT_DIR     = SCRIPT_DIR / "out_FakeTest"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tracey Zernike -> Vergence pipeline (single case or batch).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--zernike", type=Path, default=None,
                     help="Path to a single Zernike CSV (45 coefficients, pyramid order). "
                          f"Defaults to {DEFAULT_ZERNIKE_CSV.name} next to this script.")
    src.add_argument("--batch", type=Path, default=None,
                     help="Path to a directory of Zernike CSV files (batch mode).")
    parser.add_argument("--diameter-mm", type=float, default=DEFAULT_DIAMETER_MM,
                        help="Pupil diameter (mm) at which the Zernike expansion was measured.")
    parser.add_argument("--case-id", type=str, default=DEFAULT_CASE_ID,
                        help="Identifier used to name output files (single-case mode).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory where artifacts are saved.")
    parser.add_argument("--no-figures", action="store_true",
                        help="If set, only numerical outputs are produced (no PNG figures).")
    parser.add_argument("--vl-fit-radius-mm", type=float, default=None,
                        help="Override the default V_L central fit radius (default = 0.6 mm).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # If nothing was passed, fall back to the demonstration default: process
    # `fake_zernike_test.csv` sitting next to this script.
    if args.zernike is None and args.batch is None:
        args.zernike = DEFAULT_ZERNIKE_CSV
        print(f"[INFO] No --zernike / --batch passed; using default demo file: "
              f"{args.zernike}")

    if args.zernike is not None:
        if not Path(args.zernike).is_file():
            print(f"[ERROR] Zernike CSV not found: {args.zernike}", file=sys.stderr)
            return 1
        zernike = read_zernike_vector(args.zernike)
        process_case(
            zernike=zernike,
            pupil_diameter_mm=args.diameter_mm,
            output_dir=args.output_dir,
            case_id=args.case_id,
            generate_figures=not args.no_figures,
            vl_fit_radius_mm=args.vl_fit_radius_mm,
        )
        print(f"[OK] Case '{args.case_id}' processed. Output: {args.output_dir}")
        return 0

    # Batch mode
    if not Path(args.batch).is_dir():
        print(f"[ERROR] Batch directory not found: {args.batch}", file=sys.stderr)
        return 1
    results = process_batch(
        input_dir=args.batch,
        output_dir=args.output_dir,
        pupil_diameter_mm=args.diameter_mm,
        generate_figures=not args.no_figures,
    )
    print(f"[OK] Processed {len(results)} case(s) from {args.batch}. Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
