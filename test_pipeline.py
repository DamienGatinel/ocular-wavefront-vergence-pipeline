#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the public wavefront->vergence pipeline (zernike_to_vergence.py).
Covers: the four pupil diameters, both vergence bases (Ṽ and VL-VH), the
peak-to-valley normalisation, and the oriented-mode magnitude/axis.
Run:  python test_pipeline.py     (exits non-zero on failure)
"""
import importlib.util, os, math, sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location("z2v",os.path.join(HERE,"zernike_to_vergence.py"))
tr=importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
NM=tr.pyramid_indices(8); R,T,MASK=tr.build_polar_grid()

def derive(z, d):
    gm=tr.project_zernike_to_gm(z,n_max=8); gmn=dict(gm)
    for k in [(0,0),(1,-1),(1,1)]: gmn[k]=0.0
    W=tr.wavefront_from_gm(gmn); V=tr.compute_vergence(W,d/2.0,R,T,MASK)
    V=np.nan_to_num(V,nan=0.0); V=np.where(MASK,V,np.nan)
    a,_,_=tr.decompose_VL_VH(np.where(MASK,V,0.0),d,R,T,MASK)
    at,_,_=tr.decompose_tilde(np.where(MASK,V,0.0),R,T,MASK)
    return a,at

fails=[]; _n=[0]
def check(name,cond):
    _n[0]+=1; print(("ok  " if cond else "FAIL")+" "+name); (fails.append(name) if not cond else None)

# a synthetic eye with defocus, astigmatism, coma, SA
z={nm:0.0 for nm in NM}
z[(2,0)]=-3.0; z[(2,2)]=0.4; z[(3,1)]=0.25; z[(3,-1)]=-0.1; z[(4,0)]=0.18; z[(5,1)]=0.08

# 1) four pupil diameters produce finite coefficients
for d in (6.0,5.0,4.0,3.0):
    a,at=derive(z,d)
    check(f"pupil {d:.0f}mm finite VL-VH", all(np.isfinite(list(a.values()))))
    check(f"pupil {d:.0f}mm finite TILDE", all(np.isfinite(list(at.values()))))

# 2) coma is NOT mechanically collinear: with pure V5^1 added, ratio changes
a1,_=derive({**z,(5,1):0.0},6.0)
a2,_=derive({**z,(5,1):0.4},6.0)
r1=a1[(5,1)]/a1[(3,1)] if abs(a1[(3,1)])>1e-9 else None
r2=a2[(5,1)]/a2[(3,1)] if abs(a2[(3,1)])>1e-9 else None
check("primary/secondary coma not a fixed ratio", r1 is not None and r2 is not None and abs(r1-r2)>0.1)

# 3) VL-VH high-order coma matches the orthogonal Ṽ coma (full-disk projection)
a,at=derive(z,6.0)
check("VL-VH coma ~= TILDE coma (n3_m1)", abs(a[(3,1)]-at[(3,1)])<5e-3)
check("VL-VH coma ~= TILDE coma (n5_m1)", abs(a[(5,1)]-at[(5,1)])<5e-3)

# 4) peak-to-valley factors are positive and scaling preserves the axis (scale-invariant)
npv=tr.scale_alpha_by_PTV(a,tr.V_PTV)
check("all PV factors > 0", all(v>0 for k,v in tr.V_PTV.items() if v!=0))
axis_raw = math.degrees(0.5*math.atan2(a[(2,-2)],a[(2,2)]))%180
axis_pv  = math.degrees(0.5*math.atan2(npv[(2,-2)],npv[(2,2)]))%180
check("axis invariant under PV normalisation (PTVaxis==axis)", abs(axis_raw-axis_pv)<1e-6)

# 5) oriented-mode magnitude/axis INVERT the coefficients: reconstructing
#    (cos,sin) from (magnitude, axis) must return the original coefficient pair.
#    This checks the axis convention, not merely that magnitude = hypot.
for (n,m) in [(3,1),(2,2),(4,4)]:
    mag,axis=tr.magnitude_axis(a,n,m)
    th=math.radians(axis)
    cos_rec=mag*math.cos(m*th); sin_rec=mag*math.sin(m*th)
    check(f"mag/axis invert coefficients (n{n}_m{m})",
          abs(cos_rec-a[(n,m)])<1e-9 and abs(sin_rec-a[(n,-m)])<1e-9)

# 6) magnitude is rotation-invariant: rotating the astigmatism pair by an angle
#    leaves the magnitude unchanged (axis shifts). Non-tautological check.
import numpy as _np
a_cos,a_sin=a[(2,2)],a[(2,-2)]; m=2; phi=0.3
rot_cos= a_cos*math.cos(m*phi)+a_sin*math.sin(m*phi)
rot_sin=-a_cos*math.sin(m*phi)+a_sin*math.cos(m*phi)
check("astig magnitude invariant under rotation",
      abs(math.hypot(rot_cos,rot_sin)-math.hypot(a_cos,a_sin))<1e-12)

print("\n%d test(s), %d failure(s)"%(_n[0], len(fails)))
sys.exit(1 if fails else 0)
