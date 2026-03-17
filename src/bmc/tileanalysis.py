########################################################################
# class to support BMC tile analysis
#
# Michael Feig
# mfeiglab@gmail.com
#
# 2025-2026
#
########################################################################

from __future__ import annotations

import logging
import re
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path

import gemmi
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from scipy.spatial.transform import Rotation as Rot
from scipy.special import logsumexp

logging.disable(logging.CRITICAL)

kb = 0.008314462618
T = 300

hhmarkers = []
hhmarkers += [
    {
        "bend": 146.6,
        "twist": 2.95,
        "rot": 5.67,
        "dist": 6.46,
        "col": "red",
        "label": "HTP HH(P)",
        "pos": 1,
    }
]
hhmarkers += [
    {
        "bend": 175.3,
        "twist": -0.05,
        "rot": 0.00,
        "dist": 6.69,
        "col": "magenta",
        "label": "HTP HH(T)",
        "pos": 1,
    }
]
hhmarkers += [
    {
        "bend": 138.4,
        "twist": 5.54,
        "rot": 2.11,
        "dist": 6.15,
        "col": "#800000",
        "label": "T3 HH",
        "pos": -1,
    }
]
hhmarkers += [
    {
        "bend": 144.3,
        "twist": 3.48,
        "rot": 5.04,
        "dist": 6.47,
        "col": "#800000",
        "label": "T4 HH",
        "pos": -1,
    }
]

hpmarkers = []
hpmarkers += [
    {
        "bend": 150.9,
        "twist": -3.76,
        "rot": 13.69,
        "dist": 5.77,
        "col": "purple",
        "label": "HTP HP",
        "pos": 1,
    }
]
hpmarkers += [
    {
        "bend": 143.0,
        "twist": -5.59,
        "rot": 15.33,
        "dist": 5.68,
        "col": "purple",
        "label": "HP T3 HP",
        "pos": -1,
    }
]
hpmarkers += [
    {
        "bend": 148.6,
        "twist": -3.07,
        "rot": 12.91,
        "dist": 5.94,
        "col": "purple",
        "label": "HP T4 HP",
        "pos": 1,
    }
]

htmarkers = []
htmarkers += [
    {
        "bend": 159.6,
        "twist": -6.63,
        "rot": -0.74,
        "dist": 6.66,
        "col": "#8000F0",
        "label": "HTP HT",
        "pos": 1,
    }
]

tics = {}
tics["bend"] = [90, 120, 150, 180, 210]
tics["twist"] = [-40, -20, 0, 20, 40]
tics["dist"] = [5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2]

minmax = {}
minmax["bend"] = [60, 230]
minmax["twist"] = [-50, 50]
minmax["dist"] = [5.5, 7.2]

label = {}
label["bend"] = "Planar angle [deg]"
label["twist"] = "Twisting angle [deg]"
label["dist"] = "Distance [nm]"

colors1d = ["blue", "red", "green", "orange", "magenta", "cyan", "brown", "pink", "lime"]

plt.rcParams.update(
    {
        "font.size": 20,
        "font.family": "monospace",
        "font.weight": "normal",
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
        "figure.titlesize": 20,
    }
)

# tile geometry analysis


def _normalize_rows(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    return v / n


def orthonormalize_batch(F):
    U, _, Vt = np.linalg.svd(F)
    M = U @ Vt
    neg = np.linalg.det(M) < 0
    if np.any(neg):
        U[neg, :, -1] *= -1
        M = U @ Vt
    return M


def eulerAngles(xA, yA, zA, xB, yB, zB, seq="YXZ", degrees=True):
    xA = _normalize_rows(xA)
    yA = _normalize_rows(yA)
    zA = _normalize_rows(zA)
    xB = _normalize_rows(xB)
    yB = _normalize_rows(yB)
    zB = _normalize_rows(zB)
    A = np.stack([xA, yA, zA], axis=-1)
    B = np.stack([xB, yB, zB], axis=-1)
    A = orthonormalize_batch(A)
    B = orthonormalize_batch(B)
    R_A = np.transpose(A, (0, 2, 1)) @ B
    return Rot.from_matrix(R_A).as_euler(seq, degrees=degrees)


def indices_of_chain(structure, chain_name, resmin=1, resmax=9999):
    m = structure[0]
    idx = []
    i = 0
    for ch in m:
        for res in ch:
            for a in res:
                if ch.name == chain_name and res.seqid.num >= resmin and res.seqid.num <= resmax:
                    idx.append(i)
                i += 1
    return idx


def hh_dimeridx(pdb1, chlist1, pdb2, chlist2, resmin=3, resmax=88):
    s1 = gemmi.read_structure(pdb1)
    s2 = gemmi.read_structure(pdb2)

    xyzc1 = {}
    c1c = {}
    for k in chlist1:
        alist = [a for res in s1[0][k] for a in res if a.name == "CA"]
        xyzc1[k] = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)
        c1c[k] = np.average(xyzc1[k], axis=0)

    xyzc2 = {}
    c2c = {}
    for k in chlist2:
        alist = [a for res in s2[0][k] for a in res if a.name == "CA"]
        xyzc2[k] = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)
        c2c[k] = np.average(xyzc2[k], axis=0)

    dmat = {(k1, k2): np.linalg.norm(c2c[k2] - c1c[k1]) for k1 in chlist1 for k2 in chlist2}

    (c1alabel, c2alabel), _ = min(dmat.items(), key=lambda kv: kv[1])

    n1 = len(chlist1)
    n2 = len(chlist2)
    idx1 = {k: i for i, k in enumerate(chlist1)}
    idx2 = {k: i for i, k in enumerate(chlist2)}

    def at1(base, off):
        return chlist1[(idx1[base] + off) % n1]

    def at2(base, off):
        return chlist2[(idx2[base] + off) % n2]

    def prev1(k):
        return at1(k, -1)

    def next1(k):
        return at1(k, +1)

    def next2(k):
        return at2(k, -1)

    def prev2(k):
        return at2(k, +1)

    c1prev, c1next = prev1(c1alabel), next1(c1alabel)
    c2prev, c2next = prev2(c2alabel), next2(c2alabel)

    dpp = dmat[(c1prev, c2prev)]
    dpn = dmat[(c1prev, c2next)]
    dnp = dmat[(c1next, c2prev)]
    dnn = dmat[(c1next, c2next)]

    if dpp <= dpn and dpp <= dnp and dpp <= dnn:
        off1, off2 = (range(2, 8), range(0, 6))
    elif dpn <= dpp and dpn <= dnp and dpn <= dnn:
        off1, off2 = (range(2, 8), range(-1, 5))
    elif dnp <= dpp and dnp <= dpn and dnp <= dnn:
        off1, off2 = (range(3, 9), range(0, 6))
    else:
        off1, off2 = (range(3, 9), range(-1, 5))

    idxlist1 = [indices_of_chain(s1, at1(c1alabel, o), resmin, resmax) for o in off1]
    idxlist2 = [indices_of_chain(s2, at2(c2alabel, o), resmin, resmax) for o in off2]

    return [idxlist1, idxlist2]


def ph_dimeridx(pdb1, chlist1, pdb2, chlist2, resmin1=1, resmax1=95, resmin2=3, resmax2=88):
    s1 = gemmi.read_structure(pdb1)
    s2 = gemmi.read_structure(pdb2)

    xyzc1 = {}
    c1c = {}
    for k in chlist1:
        alist = [a for res in s1[0][k] for a in res if a.name == "CA"]
        xyzc1[k] = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)
        c1c[k] = np.average(xyzc1[k], axis=0)

    xyzc2 = {}
    c2c = {}
    for k in chlist2:
        alist = [a for res in s2[0][k] for a in res if a.name == "CA"]
        xyzc2[k] = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)
        c2c[k] = np.average(xyzc2[k], axis=0)

    dmat = {(k1, k2): np.linalg.norm(c2c[k2] - c1c[k1]) for k1 in chlist1 for k2 in chlist2}

    (c1alabel, c2alabel), _ = min(dmat.items(), key=lambda kv: kv[1])

    n1 = len(chlist1)
    n2 = len(chlist2)
    idx1 = {k: i for i, k in enumerate(chlist1)}
    idx2 = {k: i for i, k in enumerate(chlist2)}

    def at1(base, off):
        return chlist1[(idx1[base] + off) % n1]

    def at2(base, off):
        return chlist2[(idx2[base] + off) % n2]

    def next2(k):
        return at2(k, -1)

    def prev2(k):
        return at2(k, +1)

    c2prev, c2next = prev2(c2alabel), next2(c2alabel)

    if dmat[(c1alabel, c2next)] < dmat[(c1alabel, c2prev)]:
        off1, off2 = (range(2, 7), range(-1, 5))
    else:
        off1, off2 = (range(2, 7), range(0, 6))

    idxlist1 = [indices_of_chain(s1, at1(c1alabel, o), resmin1, resmax1) for o in off1]
    idxlist2 = [indices_of_chain(s2, at2(c2alabel, o), resmin2, resmax2) for o in off2]

    return [idxlist1, idxlist2]


def th_dimeridx(pdb1, chlist1, pdb2, chlist2, resmin1=5, resmax1=205, resmin2=3, resmax2=88):
    s1 = gemmi.read_structure(pdb1)
    s2 = gemmi.read_structure(pdb2)

    xyzc1 = {}
    c1c = {}
    for k in chlist1:
        alist = [a for res in s1[0][k] for a in res if a.name == "CA"]
        xyzc1[k] = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)
        c1c[k] = np.average(xyzc1[k], axis=0)

    xyzc2 = {}
    c2c = {}
    for k in chlist2:
        alist = [a for res in s2[0][k] for a in res if a.name == "CA"]
        xyzc2[k] = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)
        c2c[k] = np.average(xyzc2[k], axis=0)

    dmat = {(k1, k2): np.linalg.norm(c2c[k2] - c1c[k1]) for k1 in chlist1 for k2 in chlist2}

    (c1alabel, c2alabel), _ = min(dmat.items(), key=lambda kv: kv[1])

    n1 = len(chlist1)
    n2 = len(chlist2)
    idx1 = {k: i for i, k in enumerate(chlist1)}
    idx2 = {k: i for i, k in enumerate(chlist2)}

    def at1(base, off):
        return chlist1[(idx1[base] + off) % n1]

    def at2(base, off):
        return chlist2[(idx2[base] + off) % n2]

    def prev1(k):
        return at1(k, -1)

    def next1(k):
        return at1(k, +1)

    def next2(k):
        return at2(k, -1)

    def prev2(k):
        return at2(k, +1)

    c1prev, c1next = prev1(c1alabel), next1(c1alabel)
    c2prev, c2next = prev2(c2alabel), next2(c2alabel)

    dpp = dmat[(c1prev, c2prev)]
    dpn = dmat[(c1prev, c2next)]
    dnp = dmat[(c1next, c2prev)]
    dnn = dmat[(c1next, c2next)]

    if dpp <= dpn and dpp <= dnp and dpp <= dnn:
        off1, off2 = (range(1, 4), range(0, 6))
    elif dpn <= dpp and dpn <= dnp and dpn <= dnn:
        off1, off2 = (range(1, 4), range(-1, 5))
    elif dnp <= dpp and dnp <= dpn and dnp <= dnn:
        off1, off2 = (range(2, 5), range(0, 6))
    else:
        off1, off2 = (range(2, 5), range(-1, 5))

    idxlist1 = [indices_of_chain(s1, at1(c1alabel, o), resmin1, resmax1) for o in off1]
    idxlist2 = [indices_of_chain(s2, at2(c2alabel, o), resmin2, resmax2) for o in off2]

    return [idxlist1, idxlist2]


def dimergeom(c1, c2, x1, y1, z1, x2, y2, z2):
    d = c2 - c1
    dist = np.linalg.norm(d, axis=1, keepdims=True)  # in nm

    dot = np.sum(z1 * z2, axis=1, keepdims=True)
    cross = np.linalg.norm(np.cross(z1, z2, axis=1), axis=1, keepdims=True)
    ang = np.degrees(np.arctan2(cross, dot))

    euler = eulerAngles(x1, y1, z1, x2, y2, z2)

    x1 /= np.linalg.norm(x1, axis=1, keepdims=True)
    y1 /= np.linalg.norm(y1, axis=1, keepdims=True)
    z1 /= np.linalg.norm(z1, axis=1, keepdims=True)

    shiftz = np.sum(d * z1, axis=1, keepdims=True)
    shiftx = np.sum(d * x1, axis=1, keepdims=True)
    shifty = np.sum(d * y1, axis=1, keepdims=True)

    return dist, ang, euler, shiftx, shifty, shiftz


def hh_dimergeom(traj1, idx1, traj2, idx2):
    c1list = [x for c in idx1 for x in c]
    c1 = np.average(traj1.xyz[:, c1list], axis=1)

    c2list = [x for c in idx2 for x in c]
    c2 = np.average(traj2.xyz[:, c2list], axis=1)

    c1c = {}
    for kidx, lidx in enumerate(idx1):
        c1c[kidx] = np.average(traj1.xyz[:, lidx], axis=1)

    c2c = {}
    for kidx, lidx in enumerate(idx2):
        c2c[kidx] = np.average(traj2.xyz[:, lidx], axis=1)

    x1 = (c1c[0] + c1c[1]) - (c1c[3] + c1c[4])
    y1 = c1c[2] - c1c[5]
    z1 = np.cross(x1, y1, axis=1)

    x2 = (c2c[0] + c2c[1]) - (c2c[3] + c2c[4])
    y2 = c2c[2] - c2c[5]
    z2 = np.cross(x2, y2, axis=1)

    return dimergeom(c1, c2, x1, y1, z1, x2, y2, z2)


def ph_dimergeom(traj1, idx1, traj2, idx2):
    c1list = [x for c in idx1 for x in c]
    c1 = np.average(traj1.xyz[:, c1list], axis=1)

    c2list = [x for c in idx2 for x in c]
    c2 = np.average(traj2.xyz[:, c2list], axis=1)

    c1c = {}
    for kidx, lidx in enumerate(idx1):
        c1c[kidx] = np.average(traj1.xyz[:, lidx], axis=1)

    c2c = {}
    for kidx, lidx in enumerate(idx2):
        c2c[kidx] = np.average(traj2.xyz[:, lidx], axis=1)

    x1 = (c1c[0] - c1 + c1c[1] - c1) - (c1c[3] - c1)
    y1 = c1c[2] - c1c[4]
    z1 = np.cross(x1, y1, axis=1)

    x2 = (c2c[0] + c2c[1]) - (c2c[3] + c2c[4])
    y2 = c2c[2] - c2c[5]
    z2 = np.cross(x2, y2, axis=1)

    return dimergeom(c1, c2, x1, y1, z1, x2, y2, z2)


def th_dimergeom(traj1, idx1, traj2, idx2):
    c1list = [x for c in idx1 for x in c]
    c1 = np.average(traj1.xyz[:, c1list], axis=1)

    c2list = [x for c in idx2 for x in c]
    c2 = np.average(traj2.xyz[:, c2list], axis=1)

    c1c = {}
    for kidx, lidx in enumerate(idx1):
        c1c[kidx] = np.average(traj1.xyz[:, lidx], axis=1)

    c2c = {}
    for kidx, lidx in enumerate(idx2):
        c2c[kidx] = np.average(traj2.xyz[:, lidx], axis=1)

    x1 = (c1c[0] - c1) - (c1c[1] - c1 + c1c[2] - c1)
    y1 = c1c[1] - c1c[2]
    z1 = np.cross(x1, y1, axis=1)

    x2 = (c2c[0] + c2c[1]) - (c2c[3] + c2c[4])
    y2 = c2c[2] - c2c[5]
    z2 = np.cross(x2, y2, axis=1)

    return dimergeom(c1, c2, x1, y1, z1, x2, y2, z2)


def hh_analysis(dir=".", *, caname="CA.pdb", trajname="CA.xtc"):
    p = Path(dir)
    capdb = p / caname

    clist = hh_dimeridx(
        str(capdb), ["A", "B", "E", "F", "C", "D"], str(capdb), ["G", "H", "K", "L", "I", "J"]
    )

    t = md.load(str(p / trajname), top=str(capdb))
    d, ang, eu, sx, sy, sz = hh_dimergeom(t, clist[0], t, clist[1])

    d, ang, sx, sy, sz = map(np.ravel, (d, ang, sx, sy, sz))
    eu = np.asarray(eu)

    df = pd.DataFrame(
        {
            "dist": d,
            "angle": 180.0 - ang,
            "bend": 180.0 - eu[:, 0],
            "twist": eu[:, 1],
            "rot": eu[:, 2],
            "shiftx": sx,
            "shifty": sy,
            "shiftz": sz,
        }
    )
    return df


def ph_analysis(dir=".", *, caname="CA.pdb", trajname="CA.xtc"):
    p = Path(dir)
    capdb = p / caname
    clist = ph_dimeridx(
        str(capdb), ["A", "B", "C", "D", "E"], str(capdb), ["F", "G", "J", "K", "H", "I"]
    )

    t = md.load(str(p / trajname), top=str(capdb))
    d, ang, eu, sx, sy, sz = ph_dimergeom(t, clist[0], t, clist[1])

    d, ang, sx, sy, sz = map(np.ravel, (d, ang, sx, sy, sz))
    eu = np.asarray(eu)

    df = pd.DataFrame(
        {
            "dist": d,
            "angle": 180.0 - ang,
            "bend": 180.0 - eu[:, 0],
            "twist": eu[:, 1],
            "rot": eu[:, 2],
            "shiftx": sx,
            "shifty": sy,
            "shiftz": sz,
        }
    )
    return df


def th_analysis(dir=".", *, caname="CA.pdb", trajname="CA.xtc"):
    p = Path(dir)
    capdb = p / caname
    clist = th_dimeridx(str(capdb), ["A", "C", "B"], str(capdb), ["D", "E", "H", "I", "F", "G"])

    t = md.load(str(p / trajname), top=str(capdb))
    d, ang, eu, sx, sy, sz = th_dimergeom(t, clist[0], t, clist[1])

    d, ang, sx, sy, sz = map(np.ravel, (d, ang, sx, sy, sz))
    eu = np.asarray(eu)

    df = pd.DataFrame(
        {
            "dist": d,
            "angle": 180.0 - ang,
            "bend": 180.0 - eu[:, 0],
            "twist": eu[:, 1],
            "rot": eu[:, 2],
            "shiftx": sx,
            "shifty": sy,
            "shiftz": sz,
        }
    )
    return df


def hhh_analysis(dir=".", *, caname="CA.pdb", trajname="CA.xtc"):
    p = Path(dir)
    capdb = p / caname

    clist12 = hh_dimeridx(
        str(capdb), ["A", "B", "E", "F", "C", "D"], str(capdb), ["G", "H", "K", "L", "I", "J"]
    )
    clist13 = hh_dimeridx(
        str(capdb), ["A", "B", "E", "F", "C", "D"], str(capdb), ["M", "N", "Q", "R", "O", "P"]
    )
    clist23 = hh_dimeridx(
        str(capdb), ["G", "H", "K", "L", "I", "J"], str(capdb), ["M", "N", "Q", "R", "O", "P"]
    )

    t = md.load(str(p / trajname), top=str(capdb))

    d, ang, eu, sx, sy, sz = hh_dimergeom(t, clist12[0], t, clist12[1])
    d, ang, sx, sy, sz = map(np.ravel, (d, ang, sx, sy, sz))
    eu = np.asarray(eu)
    df12 = pd.DataFrame(
        {
            "dist12": d,
            "angle12": 180.0 - ang,
            "bend12": 180.0 - eu[:, 0],
            "twist12": eu[:, 1],
            "rot12": eu[:, 2],
            "shiftx12": sx,
            "shifty12": sy,
            "shiftz12": sz,
        }
    )

    d, ang, eu, sx, sy, sz = hh_dimergeom(t, clist13[0], t, clist13[1])
    d, ang, sx, sy, sz = map(np.ravel, (d, ang, sx, sy, sz))
    eu = np.asarray(eu)
    df13 = pd.DataFrame(
        {
            "dist13": d,
            "angle13": 180.0 - ang,
            "bend13": 180.0 - eu[:, 0],
            "twist13": eu[:, 1],
            "rot13": eu[:, 2],
            "shiftx13": sx,
            "shifty13": sy,
            "shiftz13": sz,
        }
    )

    d, ang, eu, sx, sy, sz = hh_dimergeom(t, clist23[0], t, clist23[1])
    d, ang, sx, sy, sz = map(np.ravel, (d, ang, sx, sy, sz))
    eu = np.asarray(eu)
    df23 = pd.DataFrame(
        {
            "dist23": d,
            "angle23": 180.0 - ang,
            "bend23": 180.0 - eu[:, 0],
            "twist23": eu[:, 1],
            "rot23": eu[:, 2],
            "shiftx23": sx,
            "shifty23": sy,
            "shiftz23": sz,
        }
    )

    df123 = pd.merge(df12, df13, left_index=True, right_index=True, how="inner")
    df = pd.merge(df123, df23, left_index=True, right_index=True, how="inner")
    return df


def phh_analysis(dir=".", *, caname="CA.pdb", trajname="CA.xtc"):
    p = Path(dir)
    capdb = p / caname

    clist12 = ph_dimeridx(
        str(capdb), ["A", "B", "C", "D", "E"], str(capdb), ["F", "G", "J", "K", "H", "I"]
    )
    clist13 = ph_dimeridx(
        str(capdb), ["A", "B", "C", "D", "E"], str(capdb), ["L", "M", "P", "Q", "N", "O"]
    )
    clist23 = hh_dimeridx(
        str(capdb), ["F", "G", "J", "K", "H", "I"], str(capdb), ["L", "M", "P", "Q", "N", "O"]
    )

    t = md.load(str(p / trajname), top=str(capdb))

    d, ang, eu, sx, sy, sz = ph_dimergeom(t, clist12[0], t, clist12[1])
    d, ang, sx, sy, sz = map(np.ravel, (d, ang, sx, sy, sz))
    eu = np.asarray(eu)
    df12 = pd.DataFrame(
        {
            "dist12": d,
            "angle12": 180.0 - ang,
            "bend12": 180.0 - eu[:, 0],
            "twist12": eu[:, 1],
            "rot12": eu[:, 2],
            "shiftx12": sx,
            "shifty12": sy,
            "shiftz12": sz,
        }
    )

    d, ang, eu, sx, sy, sz = ph_dimergeom(t, clist13[0], t, clist13[1])
    d, ang, sx, sy, sz = map(np.ravel, (d, ang, sx, sy, sz))
    eu = np.asarray(eu)
    df13 = pd.DataFrame(
        {
            "dist13": d,
            "angle13": 180.0 - ang,
            "bend13": 180.0 - eu[:, 0],
            "twist13": eu[:, 1],
            "rot13": eu[:, 2],
            "shiftx13": sx,
            "shifty13": sy,
            "shiftz13": sz,
        }
    )

    d, ang, eu, sx, sy, sz = hh_dimergeom(t, clist23[0], t, clist23[1])
    d, ang, sx, sy, sz = map(np.ravel, (d, ang, sx, sy, sz))
    eu = np.asarray(eu)
    df23 = pd.DataFrame(
        {
            "dist23": d,
            "angle23": 180.0 - ang,
            "bend23": 180.0 - eu[:, 0],
            "twist23": eu[:, 1],
            "rot23": eu[:, 2],
            "shiftx23": sx,
            "shifty23": sy,
            "shiftz23": sz,
        }
    )

    df123 = pd.merge(df12, df13, left_index=True, right_index=True, how="inner")
    df = pd.merge(df123, df23, left_index=True, right_index=True, how="inner")
    return df


def thh_analysis(dir=".", *, caname="CA.pdb", trajname="CA.xtc"):
    p = Path(dir)
    capdb = p / caname

    clist12 = th_dimeridx(str(capdb), ["A", "C", "B"], str(capdb), ["D", "E", "H", "I", "F", "G"])
    clist13 = th_dimeridx(str(capdb), ["A", "C", "B"], str(capdb), ["J", "K", "N", "O", "L", "M"])
    clist23 = hh_dimeridx(
        str(capdb), ["D", "E", "H", "I", "F", "G"], str(capdb), ["J", "K", "N", "O", "L", "M"]
    )

    t = md.load(str(p / trajname), top=str(capdb))

    d, ang, eu, sx, sy, sz = th_dimergeom(t, clist12[0], t, clist12[1])
    d, ang, sx, sy, sz = map(np.ravel, (d, ang, sx, sy, sz))
    eu = np.asarray(eu)
    df12 = pd.DataFrame(
        {
            "dist12": d,
            "angle12": 180.0 - ang,
            "bend12": 180.0 - eu[:, 0],
            "twist12": eu[:, 1],
            "rot12": eu[:, 2],
            "shiftx12": sx,
            "shifty12": sy,
            "shiftz12": sz,
        }
    )

    d, ang, eu, sx, sy, sz = th_dimergeom(t, clist13[0], t, clist13[1])
    d, ang, sx, sy, sz = map(np.ravel, (d, ang, sx, sy, sz))
    eu = np.asarray(eu)
    df13 = pd.DataFrame(
        {
            "dist13": d,
            "angle13": 180.0 - ang,
            "bend13": 180.0 - eu[:, 0],
            "twist13": eu[:, 1],
            "rot13": eu[:, 2],
            "shiftx13": sx,
            "shifty13": sy,
            "shiftz13": sz,
        }
    )

    d, ang, eu, sx, sy, sz = hh_dimergeom(t, clist23[0], t, clist23[1])
    d, ang, sx, sy, sz = map(np.ravel, (d, ang, sx, sy, sz))
    eu = np.asarray(eu)
    df23 = pd.DataFrame(
        {
            "dist23": d,
            "angle23": 180.0 - ang,
            "bend23": 180.0 - eu[:, 0],
            "twist23": eu[:, 1],
            "rot23": eu[:, 2],
            "shiftx23": sx,
            "shifty23": sy,
            "shiftz23": sz,
        }
    )

    df123 = pd.merge(df12, df13, left_index=True, right_index=True, how="inner")
    df = pd.merge(df123, df23, left_index=True, right_index=True, how="inner")
    return df


def tile_analysis(tag="hh", *, dir=".", path=None):
    if path is None:
        path = ["set1"]
    data = {}
    for p in path:
        base = str(Path(dir) / p)
        if tag == "hh":
            data[p] = hh_analysis(base, trajname="CAwrapped.xtc")
        elif tag == "th":
            data[p] = th_analysis(base, trajname="CAwrapped.xtc")
        elif tag == "ph":
            data[p] = ph_analysis(base, trajname="CAwrapped.xtc")
        elif tag == "hhh":
            data[p] = hhh_analysis(base, trajname="CAwrapped.xtc")
        elif tag == "thh":
            data[p] = thh_analysis(base, trajname="CAwrapped.xtc")
        elif tag == "phh":
            data[p] = phh_analysis(base, trajname="CAwrapped.xtc")
        else:
            raise ValueError(f"Unknown tag: {tag}")
    return data


# read data from metadynamics sampling


def _parse_fields_from_header(fname):
    with open(fname, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s.startswith("#"):
                # reached data before finding FIELDS
                return None
            if re.match(r"^#\s*!\s*FIELDS\b", s, flags=re.IGNORECASE):
                toks = s.split()
                try:
                    i = [t.upper() for t in toks].index("FIELDS")
                    return toks[i + 1 :]
                except ValueError:
                    if toks and toks[0].startswith("#"):
                        toks = toks[1:]
                    if toks and toks[0] == "!":
                        toks = toks[1:]
                    if toks and toks[0].upper() == "FIELDS":
                        return toks[1:]
                    return None
    return None


def _infer_plumed_index_col(cols):
    lower = {c.lower(): c for c in cols}
    if "time" in lower:
        return lower["time"]
    if "step" in lower:
        return lower["step"]
    return None


def read_plumed_data(dir=".", path=["set1"], logname="plumed.log", verbose=False):
    frames = []
    for p in path:
        fname = Path(dir) / p / logname
        if not fname.exists():
            continue

        cols = _parse_fields_from_header(fname)
        if cols is None or len(cols) == 0:
            print(f"{fname} is not plumed log file")
            continue

        tcol = _infer_plumed_index_col(cols)

        dtype = {c: float for c in cols}
        if tcol is not None and tcol.lower() == "step":
            dtype[tcol] = int

        df = pd.read_csv(
            fname,
            sep=r"\s+",
            engine="python",
            names=cols,
            usecols=range(len(cols)),
            comment="#",
            header=None,
            dtype=dtype,
            na_values=["nan", "NaN", "INF", "inf", "-inf"],
            on_bad_lines="skip",
        )

        df.insert(0, "set", p)

        if df.empty:
            if verbose:
                print(f"read {fname} (0 rows)")
            continue

        if tcol is None or tcol not in df.columns:
            warnings.warn(
                f"{fname}: no 'time' or 'step' field; using row number as 'frame'",
                RuntimeWarning,
                stacklevel=2,
            )
            df.insert(1, "frame", np.arange(len(df), dtype=int))
            idx_cols = ["set", "frame"]
        else:
            idx_cols = ["set", tcol]

        df = df.drop_duplicates(subset=idx_cols, keep="first")
        df = df.set_index(idx_cols).sort_index()

        frames.append(df)
        if verbose:
            print(f"read {fname} ({len(df)} rows)")

    if not frames:
        return {}

    out = pd.concat(frames, axis=0)
    return {u: g.droplevel(0) for u, g in out.groupby(level=0)}


def process_meta(tag="hh", *, dir=".", path=None, verbose=False):
    if path is None:
        path = ["set1"]

    pindiv = read_plumed_data(dir, path, verbose=verbose, logname="plumed.log")
    pcomb1 = read_plumed_data(dir, path, verbose=verbose, logname="comb/plumed1.log")
    pcomb2 = read_plumed_data(dir, path, verbose=verbose, logname="comb/plumed2.log")

    if pcomb2:
        pcomb = {p: pd.concat([pcomb1[p], pcomb2[p]], axis=0, ignore_index=True) for p in path}
    else:
        pcomb = pcomb1

    tiledata = tile_analysis(tag, dir=dir, path=path)
    data = {
        p: pd.merge(pindiv[p], tiledata[p], left_index=True, right_index=True, how="inner")
        for p in path
    }

    mask = {}
    for p in path:
        mask[p] = data[p]["uwall.bias"] + data[p]["lwall.bias"] < 1
        data[p] = data[p].loc[mask[p]].copy()
        data[p].reset_index(drop=True, inplace=True)

    for p in path:
        wham = unbias_wham(np.array([data[p]["metad.bias"]]).T)
        data[p]["ww"] = pd.DataFrame(np.exp(wham["logW"]) / np.sum(np.exp(wham["logW"])))

    data["comb"] = pd.concat([data[p] for p in path], ignore_index=True)

    combmask = pd.concat([mask[p] for p in path], ignore_index=True)
    bias_matrix = np.column_stack(
        [np.asarray(pcomb[p]["metad.bias"].loc[combmask], dtype=float) for p in path]
    )
    counts = [len(data[p]) for p in path]
    mbar = unbias_mbar(bias_matrix, counts=counts)
    wham = unbias_wham(bias_matrix)

    data["comb"]["wwmbar"] = pd.DataFrame(mbar["ww"])
    data["comb"]["wwwham"] = pd.DataFrame(wham["ww"])
    data["comb"]["ww"] = data["comb"]["wwmbar"]

    data["mbar"] = mbar
    data["wham"] = wham
    data["bias_matrix"] = bias_matrix

    data["sets"] = path

    return data


# umbrella sampling


def read_umbrella_bias(dir, umbrellas, *, verbose=False):
    cols = ["step", "xbias", "ybias", "zbias", "anglebias", "torsionbias", "rotbias"]
    dtype = {
        "step": int,
        "xbias": float,
        "ybias": float,
        "zbias": float,
        "anglebias": float,
        "torsionbias": float,
        "rotbias": float,
    }

    frames = []
    dir = Path(dir)

    for u in umbrellas:
        base = dir / u / "bias.dat"
        candidates = [base, Path(str(base) + ".gz")]

        fname = None
        for p in candidates:
            if p.exists():
                fname = p
                break

        if fname is None:
            if verbose:
                print(f"WARNING: no bias.dat or bias.dat.gz for umbrella {u}")
            continue

        df = pd.read_csv(
            fname,
            sep=r"\s+",
            engine="python",
            names=cols,
            usecols=range(len(cols)),
            comment="#",
            skiprows=1,
            dtype=dtype,
            na_values=["nan", "NaN", "INF", "inf", "-inf"],
            on_bad_lines="skip",
        )
        df["ubias"] = (
            df["xbias"]
            + df["ybias"]
            + df["zbias"]
            + df["anglebias"]
            + df["torsionbias"]
            + df["rotbias"]
        )
        df["obias"] = df["anglebias"] + df["torsionbias"] + df["rotbias"]
        df.insert(0, "umbrella", u)
        frames.append(df)

        if verbose:
            print(f"read {fname} ({len(df)} rows)")

    if not frames:
        if verbose:
            print("No bias data read from any umbrella.")
        return {}

    out = pd.concat(frames, ignore_index=True).set_index(["umbrella", "step"]).sort_index()

    return {u: g.droplevel(0) for u, g in out.groupby(level=0)}


def read_umbrella_geometry(fname, *, verbose=False):
    cols = ["gstep", "gxdist", "gydist", "gzdist", "gangle", "gtorsion", "grot1", "grot2"]
    dtype = {
        "gstep": int,
        "gxdist": float,
        "gydist": float,
        "gzdist": float,
        "gangle": float,
        "gtorsion": float,
        "grot1": float,
        "grot2": float,
    }

    fname = Path(fname)
    # Try the given name first, then "<fname>.gz"
    if fname.exists():
        chosen = fname
    else:
        gz_name = Path(str(fname) + ".gz")
        if gz_name.exists():
            chosen = gz_name
        else:
            if verbose:
                print(f"WARNING: no {fname} or {gz_name} found")
            return None

    if verbose:
        print(f"reading geometry from {chosen}")

    df = pd.read_csv(
        chosen,
        sep=r"\s+",
        engine="python",
        names=cols,
        usecols=range(len(cols)),
        comment="#",
        skiprows=1,
        dtype=dtype,
        na_values=["nan", "NaN", "INF", "inf", "-inf"],
        on_bad_lines="skip",
    )
    return df



_RUN_RE = re.compile(r"^run_(\d+\.\d{1,2})(?:_(.+))?$")


def find_run_dirs(dir: str) -> list[str]:
    base = Path(dir)
    path: list[str] = []

    for p in base.iterdir():
        if p.is_dir() and _RUN_RE.match(p.name):
            path.append(p.name)

    def sort_key(name: str) -> tuple[float, str]:
        m = _RUN_RE.match(name)
        assert m is not None
        num = float(m.group(1))
        suffix = m.group(2) or ""
        return num, suffix

    return sorted(path, key=sort_key)


def process_umbrella(
    tag="hh",
    *,
    dir=".",
    path=None,
    verbose=False,
    biasval="xbias",
    obiaslimit=None,
    skip=0,
    trajname="CA.xtc",
):
    if path is None:
        path = find_run_dirs(dir)

    geo = read_umbrella_geometry(dir + "/" + path[0] + "/geometry.dat")

    if tag == "hh":
        dimer = hh_analysis(dir, trajname=trajname)
    elif tag == "ph":
        dimer = ph_analysis(dir, trajname=trajname)
    elif tag == "th":
        dimer = th_analysis(dir, trajname=trajname)
    else:
        print(f"unknown tag {tag}")
        return

    df = pd.merge(geo, dimer, left_index=True, right_index=True, how="inner")

    nwin = len(path)
    nper = len(df) // nwin
    data = {path[i]: df.iloc[i * nper : (i + 1) * nper].reset_index(drop=True) for i in range(nwin)}

    bias = read_umbrella_bias(dir, path, verbose=verbose)
    for i in range(nwin):
        bindiv = bias[path[i]].iloc[i * nper : (i + 1) * nper].reset_index(drop=True)
        data[path[i]] = pd.merge(
            data[path[i]], bindiv, left_index=True, right_index=True, how="inner"
        )

    mask = {}
    for p in path:
        if obiaslimit is not None:
            mask[p] = data[p]["obias"] < obiaslimit
        else:
            mask[p] = pd.Series(True, index=data[p].index)
        if skip > 0:
            mask[p].iloc[:skip] = False
        data[p] = data[p].loc[mask[p]].copy()
        data[p].reset_index(drop=True, inplace=True)

    for p in path:
        wham = unbias_wham(np.array([data[p][biasval]]).T)
        data[p]["ww"] = pd.DataFrame(np.exp(wham["logW"]) / np.sum(np.exp(wham["logW"])))

    combmask = pd.concat([mask[p] for p in path], ignore_index=True)
    data["comb"] = df.loc[combmask].copy()

    bias_matrix = np.column_stack(
        [np.asarray(bias[p][biasval].loc[combmask], dtype=float) for p in path]
    )
    counts = [len(data[p]) for p in path]

    mbar = unbias_mbar(bias_matrix, counts=counts)

    # wham=unbias_wham(bias_matrix)
    # data['comb']['wwmbar']=pd.DataFrame(mbar['ww'])
    # data['comb']['wwwham']=pd.DataFrame(wham['ww'])
    # data['comb']['ww']=data['comb']['wwmbar']
    # data['wham']=wham

    data["comb"]["ww"] = pd.DataFrame(mbar["ww"])

    data["mbar"] = mbar
    data["bias_matrix"] = bias_matrix

    data["bias"] = bias
    data["counts"] = counts
    data["sets"] = path

    return data


# unbiasing and projecting onto reaction coordinates


def unbias_wham(
    bias,
    *,
    kT: float = kb * T,
    frame_weight=None,
    traj_weight=None,
    maxiter: int = 1000,
    threshold: float = 1e-20,
    verbose: bool = False,
):

    nframes = bias.shape[0]
    ntraj = bias.shape[1]

    if frame_weight is None:
        frame_weight = np.ones(nframes)
    if traj_weight is None:
        traj_weight = np.ones(ntraj)

    assert len(traj_weight) == ntraj
    assert len(frame_weight) == nframes

    shifted_bias = bias / kT

    shifts0 = np.min(shifted_bias, axis=0)
    shifted_bias -= shifts0[np.newaxis, :]
    shifts1 = np.min(shifted_bias, axis=1)
    shifted_bias -= shifts1[:, np.newaxis]

    expv = np.exp(-shifted_bias)

    Z = np.ones(ntraj)

    Zold = Z.copy()

    if verbose:
        sys.stderr.write("WHAM: start\n")
    for nit in range(maxiter):
        weight = 1.0 / np.matmul(expv, traj_weight / Z) * frame_weight
        Z = np.matmul(weight, expv)
        Z /= np.sum(Z * traj_weight)
        ratio = np.maximum(Z, 1e-300) / np.maximum(Zold, 1e-300)
        eps = np.sum(np.log(ratio) ** 2)
        Zold = Z.copy()
        if verbose:
            sys.stderr.write("WHAM: iteration " + str(nit) + " eps " + str(eps) + "\n")
        if eps < threshold:
            break
    logW = np.log(weight) + shifts1

    if verbose:
        sys.stderr.write("WHAM: end")

    return {
        "logW": logW,
        "logZ": np.log(Z) - shifts0,
        "nit": nit,
        "eps": eps,
        "ww": np.exp(logW) / np.sum(np.exp(logW)),
    }


def unbias_mbar(bias, *, kT=kb * T, counts=None, verbose=False):
    beta = 1.0 / (kT) if kT is not None else 1.0
    u_kn = (beta * np.asarray(bias, float)).T  # (K,N)

    if counts is None:
        N, K = bias.shape
        n_per = N // K
        state_of_sample = np.repeat(np.arange(K, dtype=int), n_per)
        counts = np.bincount(state_of_sample, minlength=K)

    from pymbar import MBAR

    mbar = MBAR(u_kn, counts, verbose=verbose, maximum_iterations=500)

    log_den = logsumexp(mbar.f_k[:, None] - mbar.u_kn + np.log(mbar.N_k)[:, None], axis=0)
    logw = -log_den
    ww = np.exp(logw - logsumexp(logw))

    return {"mbar": mbar, "logW": logw, "ww": ww}


def pmf1d_mbar(mbar, data, tag, *, kT=kb * T, nbins=100, rang=None, verbose=False):
    """1D PMF via PyMBAR FES (histogram) with safe handling of empty bins.

    Notes
    -----
    PyMBAR's FES.get_fes() raises KeyError when queried at histogram bins with
    zero occupancy. We therefore query only occupied bins and fill the rest
    with NaN.
    """
    if "mbar" in mbar:
        mbar = mbar["mbar"]

    x = np.asarray(data[tag], float).ravel()
    u_n = np.zeros(x.shape[0], float)

    if rang is None:
        eps = 1e-12 * (float(x.max()) - float(x.min()) + 1.0)
        edges_1d = np.linspace(float(x.min()), float(x.max()) + eps, nbins + 1)
    else:
        xmin, xmax = float(rang[0]), float(rang[1])
        edges_1d = np.linspace(xmin, xmax, nbins + 1)

    centers = 0.5 * (edges_1d[:-1] + edges_1d[1:])

    from pymbar import FES

    fes = FES(
        mbar.u_kn,
        mbar.N_k,
        mbar_options=dict(verbose=verbose, maximum_iterations=500),
    )
    _ = fes.generate_fes(
        u_n,
        x[:, None],
        fes_type="histogram",
        histogram_parameters={"bin_edges": [edges_1d]},
    )

    hist, _ = np.histogram(x, bins=edges_1d)
    occ = hist > 0
    if not np.any(occ):
        raise ValueError("pmf1d_mbar: no occupied bins (check input data/range)")

    centers_occ = centers[occ]
    out = fes.get_fes(
        centers_occ,
        reference_point="from-lowest",
        uncertainty_method="analytical",
    )

    f_i = np.full(centers.shape, np.nan, dtype=float)
    df_i = np.full(centers.shape, np.nan, dtype=float)
    f_i[occ] = np.asarray(out["f_i"], float).ravel()
    if out.get("df_i") is not None:
        df_i[occ] = np.asarray(out["df_i"], float).ravel()

    F_kT = f_i * float(kT)
    dF_kT = df_i * float(kT)

    if np.any(np.isfinite(F_kT)):
        F_kT = F_kT - np.nanmin(F_kT)

    idx = pd.Index(np.arange(nbins), name="x")
    pmf1d = pd.DataFrame({f"{tag}": F_kT}, index=idx)
    dpmf = pd.DataFrame({f"{tag}": dF_kT}, index=idx)
    ranges = pd.DataFrame({tag: centers}, index=idx)

    return dict(
        edges=[edges_1d],
        centers=centers,
        F_kT=F_kT,
        dF_kT=dF_kT,
        pmf=pmf1d,
        dpmf=dpmf,
        ranges=ranges,
    )


def pmf2d_from_weights(data, tag, *, wtag="ww", kT=kb * T, nbins=(100, 100), rang=None):
    """
    Project onto a 2D reaction coordinate (x,y) using weights

    Parameters
    ----------
    data   : DataFrame
        Panda data frame object
    tag    : (str,str)
        keys to access x and y values
    wtag   : str
        key to access weights, default: 'ww'
    nbins : (int, int)
        Number of histogram bins along x and y.
    rang : ((float, float), (float,float))
        min/max values for two dimensions, default is to use min/max values from data

    Returns
    -------
    result : dict
        {
          'x_edges','y_edges',        # length nx+1, ny+1
          'x_centers','y_centers',    # length nx, ny
          'F_kT',                     # shape (nx, ny), np.nan where empty
          'P',                        # normalized probability over bins (nx,ny), np.nan where empty
        }
    """

    x = np.asarray(data[tag[0]], float).ravel()
    y = np.asarray(data[tag[1]], float).ravel()
    w = np.asarray(data[wtag], float).ravel()

    assert x.shape == y.shape == w.shape
    if rang is None:

        def pad(a):
            return 1e-12 * (a.max() - a.min() + 1.0)

        x_edges = np.linspace(x.min(), x.max() + pad(x), nbins[0] + 1)
        y_edges = np.linspace(y.min(), y.max() + pad(y), nbins[1] + 1)
    else:
        (xmin, xmax), (ymin, ymax) = rang
        x_edges = np.linspace(xmin, xmax, nbins[0] + 1)
        y_edges = np.linspace(ymin, ymax, nbins[1] + 1)

    H, xe, ye = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=w)
    P = H / H.sum() if H.sum() > 0 else H
    with np.errstate(divide="ignore", invalid="ignore"):
        F = -np.log(P)
    finite = np.isfinite(F)
    if np.any(finite):
        F -= np.nanmin(F)

    F_kT = F * kT

    xcen = 0.5 * (xe[:-1] + xe[1:])
    ycen = 0.5 * (ye[:-1] + ye[1:])

    idx = pd.MultiIndex.from_product([np.arange(nbins[0]), np.arange(nbins[1])], names=["x", "y"])
    pmf2d = pd.DataFrame({tag[0] + "." + tag[1]: F_kT.ravel()}, index=idx)

    ranges = pd.DataFrame({tag[0]: xcen, tag[1]: ycen})

    return dict(
        x_edges=xe,
        y_edges=ye,
        x_centers=xcen,
        y_centers=ycen,
        F_kT=F * kT,
        P=P,
        pmf=pmf2d,
        ranges=ranges,
    )


def pmf1d_from_weights(data, tag, *, wtag="ww", kT=kb * T, nbins=100, rang=None):
    """
    Project onto a 1D reaction coordinate x using weights.

    Parameters
    ----------
    data  : DataFrame
        Pandas DataFrame holding coordinate and weights.
    tag   : str
        Column key for x values (e.g., 'bend').
    wtag  : str
        Column key for weights (default: 'ww').
    kT    : float
        Thermal energy for scaling free energy to physical units (k_B T).
    nbins : int
        Number of histogram bins along x.
    rang  : (float, float) or None
        (xmin, xmax). If None, use min/max from data (with tiny pad).

    Returns
    -------
    dict with:
      'x_edges'   : length nbins+1
      'x_centers' : length nbins
      'F_kT'      : (nbins,) free energy [same units as kT], NaN where empty
      'P'         : (nbins,) normalized probability, 0 where empty
      'pmf'       : DataFrame (index=bin id), column name f"{tag}", values F_kT
      'ranges'    : DataFrame with column tag giving bin centers
    """
    x = np.asarray(data[tag], float).ravel()
    w = np.asarray(data[wtag], float).ravel()
    assert x.shape == w.shape, "x and weights must have same length"

    if rang is None:
        pad = 1e-12 * (x.max() - x.min() + 1.0)
        x_edges = np.linspace(x.min(), x.max() + pad, nbins + 1)
    else:
        xmin, xmax = rang
        x_edges = np.linspace(xmin, xmax, nbins + 1)

    H, xe = np.histogram(x, bins=x_edges, weights=w)
    H = H.astype(float)
    Hsum = H.sum()
    P = H / Hsum if Hsum > 0 else H

    with np.errstate(divide="ignore", invalid="ignore"):
        F = -np.log(P)
    finite = np.isfinite(F)
    if np.any(finite):
        F -= np.nanmin(F)
    F_kT = F * kT

    x_centers = 0.5 * (xe[:-1] + xe[1:])

    idx = pd.Index(np.arange(nbins), name="x")
    pmf1d = pd.DataFrame({f"{tag}": F_kT}, index=idx)
    ranges = pd.DataFrame({tag: x_centers})

    return dict(edges=xe, centers=x_centers, F_kT=F_kT, P=P, pmf=pmf1d, dpmf=None, ranges=ranges)


# plotting


def dist1D(
    data: pd.DataFrame,
    ranges: pd.DataFrame,
    *,
    err: None,
    fmin=0.0,
    fmax=20.0,
    size: int = 1,
    label=label,
    minmax=minmax,
    tics=tics,
    colors=colors1d,
    lw=2,
    key=None,
    markers=None,
    tag="dist",
    mode="together",
    horizontal=None,
    vertical=None,
    save=None,
) -> None:

    if mode == "together":
        nplots = 1
        rows = 1
        cols = 1
    else:
        nplots = len(data)
        rows = int((nplots + 1) / 2)
        cols = 2

    if key is not None:
        xoff = 3
    else:
        xoff = 1

    fig, ax = plt.subplots(
        rows,
        cols,
        figsize=(cols * 5 * size + xoff, rows * 4 * size + 1),
        dpi=75,
        constrained_layout=True,
    )

    xlabel = "distance [nm]"
    ylabel = "[kJ/mol]"

    xmin = 5.0
    xmax = 10.0

    if tag is not None:
        if label is not None and tag in label:
            xlabel = label[tag]
        if tics is not None and tag in tics:
            xtics = tics[tag]
        else:
            xtics = None
        if minmax is not None and tag in minmax:
            xmin = minmax[tag][0]
            xmax = minmax[tag][1]
            if xtics is not None:
                xtics = [x for x in xtics if x >= xmin and x <= xmax]

    if nplots > 1:
        ax = ax.ravel()

    for i, d in enumerate(data):
        X = ranges[i][tag]
        Y = d[tag]

        if mode == "together":
            axi = ax
        else:
            axi = ax[i]

        if i < len(colors):
            linecolor = colors[i]
        else:
            linecolor = (0.5, 0.5, 0.5)

        if key is not None and i < len(key):
            keyname = key[i]
        else:
            keyname = ""

        axi.plot(X, Y, color=linecolor, label=keyname, linewidth=lw)
        axi.set_xlabel(xlabel)  # , fontsize=20)
        axi.set_ylabel(ylabel)  # , fontsize=20)
        axi.set_xlim(xmin, xmax)
        axi.set_ylim(fmin, fmax)

        if err is not None and err[i] is not None:
            axi.fill_between(X, Y - err[i][tag], Y + err[i][tag], alpha=0.3, color=linecolor)

        if xtics is not None:
            axi.set_xticks(xtics)

        if markers is not None:
            for m in markers:
                axi.plot(
                    m[tag], 1.0, "x", color=m["col"], markersize=int(12 * size), markeredgewidth=4
                )
                if len(m) > 3:
                    axi.annotate(
                        m["label"],
                        xy=(m[tag], 1.0),
                        xytext=(0, 16 * m["pos"] * size + 6 * size),
                        color=m["col"],
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=int(14 * size),
                    )

        if vertical is not None:
            axi.axvline(x=vertical, color="#808080", linestyle="--", linewidth=3)
        if horizontal is not None:
            axi.axhline(y=horizontal, color="#808080", linestyle="--", linewidth=3)

    if nplots > 1:
        for i in range(nplots, rows * cols):
            ax[i].remove()
    else:
        if key is not None:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)

    if save:
        fig.savefig(save, dpi=300)

    plt.show()


def dist2D(
    data: pd.DataFrame,
    ranges: pd.DataFrame,
    *,
    nlevels: int = 51,
    threshold: float = 25.0,
    colorbar: bool = True,
    cmap=None,
    size: int = 1,
    label=None,
    minmax=None,
    tics=None,
    vertical=None,
    horizontal=None,
    markers=None,
    xtag="bend",
    ytag="dist",
    save=None,
) -> None:

    nplots = len(data)
    rows = int((nplots + 1) / 2)
    cols = 2
    fig, ax = plt.subplots(
        rows,
        cols,
        figsize=(cols * 5 * size + 1, rows * 4 * size + 1),
        dpi=75,
        constrained_layout=True,
    )

    if cmap is None:
        cmap = plt.get_cmap("terrain")

    xlabel = "Planar angle [deg]"
    ylabel = "distance [nm]"

    xmin = 90.0
    xmax = 180.0
    ymin = 5.0
    ymax = 8.0

    if xtag is not None:
        if label is not None and xtag in label:
            xlabel = label[xtag]
        if tics is not None and xtag in tics:
            xtics = tics[xtag]
        else:
            xtics = None
        if minmax is not None and xtag in minmax:
            xmin = minmax[xtag][0]
            xmax = minmax[xtag][1]
            if xtics is not None:
                xtics = [x for x in xtics if x >= xmin and x <= xmax]

    if ytag is not None:
        if label is not None and ytag in label:
            ylabel = label[ytag]
        if tics is not None and ytag in tics:
            ytics = tics[ytag]
        else:
            ytics = None
        if minmax is not None and ytag in minmax:
            ymin = minmax[ytag][0]
            ymax = minmax[ytag][1]
            if ytics is not None:
                ytics = [y for y in ytics if y >= ymin and y <= ymax]

    ax = ax.ravel()
    for i, d in enumerate(data):
        k = next(iter(d.keys()))
        kx, ky = k.split(".")

        X = np.broadcast_to(ranges[i][kx], d[k].unstack().shape)
        Y = np.broadcast_to(ranges[i][ky], d[k].unstack().shape).T
        Z_raw = d[k].unstack().values.T

        if threshold is not None:
            Z = np.minimum(Z_raw, threshold)
        else:
            Z = Z_raw

        Z_masked = np.ma.masked_invalid(Z)
        zmin = np.nanmin(Z_masked)
        zmax = np.nanmax(Z_masked)

        if threshold is not None and zmax < threshold:
            zmax = threshold  # extend range to threshold if not reached naturally

        levels = np.linspace(zmin, zmax, nlevels)
        norm = BoundaryNorm(levels, ncolors=cmap.N)
        cf = ax[i].contourf(X, Y, Z_masked, cmap=cmap, levels=levels, norm=norm, extend="max")
        ax[i].contour(
            X, Y, Z_masked, colors="k", levels=levels, linewidths=0.5, linestyles="dotted"
        )

        ax[i].set_xlabel(xlabel)  # , fontsize=20)
        ax[i].set_ylabel(ylabel)  # , fontsize=20)
        ax[i].set_xlim(xmin, xmax)
        ax[i].set_ylim(ymin, ymax)

        if xtics is not None:
            ax[i].set_xticks(xtics)

        if ytics is not None:
            ax[i].set_yticks(ytics)

        if markers is not None:
            for m in markers:
                ax[i].plot(
                    m[xtag],
                    m[ytag],
                    "x",
                    color=m["col"],
                    markersize=int(12 * size),
                    markeredgewidth=4,
                )
                if len(m) > 3:
                    ax[i].annotate(
                        m["label"],
                        xy=(m[xtag], m[ytag]),
                        xytext=(0, 16 * m["pos"] * size + 6 * size),
                        color=m["col"],
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=int(14 * size),
                    )
        if vertical is not None:
            ax[i].axvline(x=vertical, color="#808080", linestyle="--", linewidth=3)
        if horizontal is not None:
            ax[i].axhline(y=horizontal, color="#808080", linestyle="--", linewidth=3)

    for i in range(nplots, rows * cols):
        ax[i].remove()

    if colorbar:
        cbar = fig.colorbar(cf, ax=fig.axes, shrink=0.95)
        cbar.ax.set_ylabel("[kJ/mol]", rotation=90)

    if save:
        fig.savefig(save, dpi=300)

    plt.show()


def plot2D_combined(
    df,
    xtag="bend",
    ytag="dist",
    *,
    minmax=minmax,
    tics=tics,
    label=label,
    kbT=kb * T,
    size=1.5,
    markers=None,
    vertical=None,
    horizontal=None,
    nbins=(100, 100),
    save=None,
):
    if isinstance(xtag, list) and isinstance(ytag, list):
        sxtag = xtag[0].rstrip("0123456789")
        sytag = ytag[0].rstrip("0123456789")
        dplotlist = []
        for i in range(len(xtag)):
            dp = df["comb"][[xtag[i], ytag[i], "ww"]].fillna(0)
            dp.columns = [sxtag, sytag, "ww"]
            dplotlist += [dp]
        dplot = pd.concat(dplotlist)
        res = pmf2d_from_weights(dplot, [sxtag, sytag], nbins=nbins)
    else:
        sxtag = xtag.rstrip("0123456789")
        sytag = ytag.rstrip("0123456789")
        dplot = df["comb"][[xtag, ytag, "ww"]].fillna(0)
        res = pmf2d_from_weights(dplot, [xtag, ytag], nbins=nbins)

    dist2D(
        [res["pmf"]],
        [res["ranges"]],
        colorbar=True,
        size=size,
        markers=markers,
        xtag=sxtag,
        ytag=sytag,
        minmax=minmax,
        tics=tics,
        label=label,
        vertical=vertical,
        horizontal=horizontal,
        save=save,
    )


def plot2D_individual(
    df,
    xtag="bend",
    ytag="dist",
    *,
    setlist=None,
    minmax=minmax,
    tics=tics,
    label=label,
    kbT=kb * T,
    size=1.0,
    markers=None,
    vertical=None,
    nbins=(100, 100),
    horizontal=None,
    save=None,
):

    if isinstance(xtag, list) and isinstance(ytag, list):
        sxtag = xtag[0].rstrip("0123456789")
        sytag = ytag[0].rstrip("0123456789")
    else:
        sxtag = xtag.rstrip("0123456789")
        sytag = ytag.rstrip("0123456789")

    if setlist is None:
        setlist = df["sets"]

    pmf = []
    rang = []
    for p in setlist:
        if isinstance(xtag, list) and isinstance(ytag, list):
            dplotlist = []
            for k in range(len(xtag)):
                dp = df[p][[xtag[k], ytag[k], "ww"]].fillna(0)
                dp.columns = [sxtag, sytag, "ww"]
                dplotlist += [dp]
            dplot = pd.concat(dplotlist)
        else:
            dplot = df[p][[xtag, ytag, "ww"]].fillna(0)

        res = pmf2d_from_weights(dplot, [xtag, ytag], nbins=nbins)

        pmf += [res["pmf"]]
        rang += [res["ranges"]]

    dist2D(
        pmf,
        rang,
        colorbar=False,
        size=size,
        markers=markers,
        xtag=sxtag,
        ytag=sytag,
        minmax=minmax,
        tics=tics,
        label=label,
        vertical=vertical,
        horizontal=horizontal,
        save=save,
    )


def _is_tiledata_dict(obj) -> bool:
    return isinstance(obj, dict) and "comb" in obj


def _nanmean_axis(a: np.ndarray, axis: int = 0) -> np.ndarray:
    a = np.asarray(a, float)
    mask = np.isfinite(a)
    n = np.sum(mask, axis=axis)
    s = np.sum(np.where(mask, a, 0.0), axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s / n
    mean = np.where(n > 0, mean, np.nan)
    return mean


def _nansem_axis(a: np.ndarray, axis: int = 0) -> np.ndarray:
    a = np.asarray(a, float)
    mask = np.isfinite(a)
    n = np.sum(mask, axis=axis)

    s = np.sum(np.where(mask, a, 0.0), axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s / n
    mean = np.where(n > 0, mean, np.nan)

    mean_exp = np.expand_dims(mean, axis=axis)
    dev = np.where(mask, a - mean_exp, 0.0)
    ss = np.sum(dev * dev, axis=axis)

    with np.errstate(divide="ignore", invalid="ignore"):
        var = ss / (n - 1)
    var = np.where(n > 1, var, np.nan)

    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        sem = sd / np.sqrt(n)
    sem = np.where(n > 1, sem, np.nan)
    return sem


def _interp_to_grid(x_ref: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.size == 0 or y.size == 0:
        return np.full(x_ref.shape, np.nan, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    with np.errstate(invalid="ignore"):
        return np.interp(x_ref, x, y, left=np.nan, right=np.nan)


def average_pmf1d(pmf, ranges, tag, *, method="linear", kT=kb * T):
    """Average 1D PMFs on a common grid.

    Parameters
    ----------
    pmf : list[pd.DataFrame]
        Each dataframe has a column `tag` with free energies (kJ/mol).
    ranges : list[pd.DataFrame]
        Each dataframe has a column `tag` with bin centers.
    tag : str
        Column name for x / PMF.
    method : {"linear","boltzmann"}
        Linear averages free energies; boltzmann averages exp(-F/kT).
    kT : float
        Thermal energy in same units as free energies (kJ/mol).

    Returns
    -------
    dict with keys: pmf, dpmf, ranges
        dpmf contains SEM (per bin) across the input PMFs.
    """
    if len(pmf) != len(ranges):
        raise ValueError("average_pmf1d: pmf and ranges length mismatch")
    if len(pmf) == 0:
        raise ValueError("average_pmf1d: empty input")

    m = str(method).lower().strip()
    if m in {"linear", "fe", "free_energy"}:
        mode = "linear"
    elif m in {"boltzmann", "boltz", "exp"}:
        mode = "boltzmann"
    else:
        raise ValueError(f"average_pmf1d: unknown method {method!r}")

    x_ref = np.asarray(ranges[0][tag], float).ravel()
    if x_ref.size == 0:
        raise ValueError("average_pmf1d: empty grid")

    y_stack = []
    for p, r in zip(pmf, ranges):
        x = np.asarray(r[tag], float).ravel()
        y = np.asarray(p[tag], float).ravel()
        if x.shape != x_ref.shape or not np.allclose(x, x_ref, atol=1e-9, rtol=0.0):
            y = _interp_to_grid(x_ref, x, y)
        y_stack.append(y)

    Y = np.vstack(y_stack)  # (nrep, nbins)

    if mode == "linear":
        mean = _nanmean_axis(Y, axis=0)
        sem = _nansem_axis(Y, axis=0)
    else:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            W = np.exp(-Y / float(kT))
        mean_w = _nanmean_axis(W, axis=0)
        sem_w = _nansem_axis(W, axis=0)

        mean = np.full(mean_w.shape, np.nan, dtype=float)
        sem = np.full(mean_w.shape, np.nan, dtype=float)

        ok = np.isfinite(mean_w) & (mean_w > 0.0)
        mean[ok] = -float(kT) * np.log(mean_w[ok])
        sem[ok] = float(kT) * sem_w[ok] / mean_w[ok]

    idx = pmf[0].index
    pmf_avg = pd.DataFrame({tag: mean}, index=idx)
    dpmf_avg = pd.DataFrame({tag: sem}, index=idx)
    ranges_avg = pd.DataFrame({tag: x_ref}, index=idx)

    return {"pmf": pmf_avg, "dpmf": dpmf_avg, "ranges": ranges_avg}


def plot1D_grouped(
    groups,
    tag="dist",
    *,
    average="boltzmann",
    usembar=False,
    minmax=minmax,
    tics=tics,
    label=label,
    colors=colors1d,
    key=None,
    kbT=kb * T,
    nbins=50,
    fmin=0.0,
    fmax=20.0,
    size=1.5,
    markers=None,
    offset=None,
    matchflat=None,
    matchzero=False,
    average_overlay=False,
    vertical=None,
    horizontal=None,
    save=None,
):
    """Average replicate PMFs per group and plot group averages together.

    `groups` can be:
      - list[list[tiledata]]  (each inner list is a group of replicate datasets)
      - dict[str, list[tiledata]]  (group name -> replicate datasets)

    When `average_overlay=True`, individual replicate PMFs are plotted (no error
    shading) along with the group averages (with SEM shading).
    """
    if isinstance(groups, dict) and "comb" not in groups:
        group_names = list(groups.keys())
        group_list = list(groups.values())
    else:
        group_names = None
        group_list = list(groups) if isinstance(groups, (list, tuple)) else [groups]

    norm_groups = []
    for g in group_list:
        if _is_tiledata_dict(g):
            norm_groups.append([g])
        else:
            norm_groups.append(list(g))

    for g in norm_groups:
        if not g:
            raise ValueError("plot1D_grouped: empty group")
        for d in g:
            if not _is_tiledata_dict(d):
                raise TypeError("plot1D_grouped: group entries must be tiledata dicts")

    n_groups = len(norm_groups)
    if group_names is None:
        if key is not None:
            if len(key) != n_groups:
                raise ValueError("plot1D_grouped: key length mismatch")
            group_names = list(key)
        else:
            group_names = [f"group{i+1}" for i in range(n_groups)]
    else:
        if key is not None:
            if len(key) != n_groups:
                raise ValueError("plot1D_grouped: key length mismatch")
            group_names = list(key)

    reps = []
    rep_group = []
    for gi, g in enumerate(norm_groups):
        for d in g:
            reps.append(d)
            rep_group.append(gi)

    if isinstance(tag, list):
        taglist = list(tag)
        stag = taglist[0].rstrip("0123456789")
    else:
        taglist = None
        stag = str(tag).rstrip("0123456789")

    vals = []
    for d in reps:
        if taglist is None:
            vals.append(np.asarray(d["comb"][tag], float).ravel())
        else:
            for t in taglist:
                vals.append(np.asarray(d["comb"][t], float).ravel())

    x = np.concatenate(vals)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("plot1D_grouped: no finite values for common range")

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    pad = 1e-12 * (xmax - xmin + 1.0)
    rang = (xmin, xmax + pad)

    pmf_rep = []
    ranges_rep = []
    err_rep = []

    for d in reps:
        if taglist is not None:
            dplotlist = []
            for t in taglist:
                dp = d["comb"][[t, "ww"]].fillna(0).copy()
                dp.columns = [stag, "ww"]
                dplotlist.append(dp)
            dplot = pd.concat(dplotlist, ignore_index=True)
            res = pmf1d_from_weights(dplot, stag, nbins=nbins, kT=kbT, rang=rang)
        else:
            if usembar:
                res = pmf1d_mbar(
                    d["mbar"],
                    d["comb"],
                    tag,
                    nbins=nbins,
                    kT=kbT,
                    rang=rang,
                )
                if stag != tag:
                    res["pmf"] = res["pmf"].rename(columns={tag: stag})
                    if res["dpmf"] is not None:
                        res["dpmf"] = res["dpmf"].rename(columns={tag: stag})
                    res["ranges"] = res["ranges"].rename(columns={tag: stag})
            else:
                dplot = d["comb"][[tag, "ww"]].fillna(0).copy()
                if stag != tag:
                    dplot = dplot.rename(columns={tag: stag})
                res = pmf1d_from_weights(dplot, stag, nbins=nbins, kT=kbT, rang=rang)

        pmf_rep.append(res["pmf"])
        ranges_rep.append(res["ranges"])
        err_rep.append(res["dpmf"])

    n_rep = len(pmf_rep)

    base_rep = [0.0] * n_rep
    base_grp = [0.0] * n_groups
    if offset is None:
        pass
    elif isinstance(offset, (float, int)):
        base_rep = [float(offset)] * n_rep
    else:
        off = list(offset)
        if len(off) == n_rep:
            base_rep = [float(off[i]) for i in range(n_rep)]
        elif len(off) == n_groups:
            base_grp = [float(off[i]) for i in range(n_groups)]
        else:
            base_rep = [float(off[i]) if i < len(off) else 0.0 for i in range(n_rep)]

    for p, o in zip(pmf_rep, base_rep):
        p[stag] = p[stag] + float(o)

    extra = [0.0] * n_rep
    if matchflat is not None and len(matchflat) == 2:
        mmin, mmax = float(matchflat[0]), float(matchflat[1])

        means = []
        for p, r in zip(pmf_rep, ranges_rep):
            mask = r[stag].between(mmin, mmax, inclusive="both")
            m = p[stag][mask].mean()
            means.append(float(m) if pd.notna(m) else 0.0)

        if matchzero:
            extra = [-m for m in means]
        else:
            mmax_val = max(means) if means else 0.0
            extra = [mmax_val - m for m in means]

    for p, o in zip(pmf_rep, extra):
        p[stag] = p[stag] + float(o)

    pmf_grp = []
    ranges_grp = []
    err_grp = []
    for gi in range(n_groups):
        idxs = [i for i, g in enumerate(rep_group) if g == gi]
        pmfs = [pmf_rep[i] for i in idxs]
        rngs = [ranges_rep[i] for i in idxs]

        avg = average_pmf1d(pmfs, rngs, stag, method=average, kT=kbT)
        if base_grp[gi] != 0.0:
            avg["pmf"][stag] = avg["pmf"][stag] + float(base_grp[gi])

        pmf_grp.append(avg["pmf"])
        ranges_grp.append(avg["ranges"])
        err_grp.append(avg["dpmf"])

    if average_overlay:
        pmf_plot = []
        ranges_plot = []
        err_plot = []
        key_plot = []
        colors_plot = []

        for gi in range(n_groups):
            idxs = [i for i, g in enumerate(rep_group) if g == gi]
            col = colors[gi] if gi < len(colors) else (0.5, 0.5, 0.5)

            for i in idxs:
                pmf_plot.append(pmf_rep[i])
                ranges_plot.append(ranges_rep[i])
                err_plot.append(None)
                key_plot.append("_nolegend_")
                colors_plot.append(col)

            pmf_plot.append(pmf_grp[gi])
            ranges_plot.append(ranges_grp[gi])
            err_plot.append(err_grp[gi])
            key_plot.append(group_names[gi])
            colors_plot.append(col)

        dist1D(
            pmf_plot,
            ranges_plot,
            err=err_plot,
            size=size,
            markers=markers,
            tag=stag,
            minmax=minmax,
            tics=tics,
            label=label,
            fmin=fmin,
            fmax=fmax,
            colors=colors_plot,
            key=key_plot,
            vertical=vertical,
            horizontal=horizontal,
            save=save,
        )
        return

    dist1D(
        pmf_grp,
        ranges_grp,
        err=err_grp,
        size=size,
        markers=markers,
        tag=stag,
        minmax=minmax,
        tics=tics,
        label=label,
        fmin=fmin,
        fmax=fmax,
        colors=colors,
        key=group_names,
        vertical=vertical,
        horizontal=horizontal,
        save=save,
    )


def plot1D_combined(
    df,
    tag="dist",
    *,
    usembar=False,
    minmax=minmax,
    tics=tics,
    label=label,
    colors=colors1d,
    key=None,
    kbT=kb * T,
    nbins=50,
    fmin=0.0,
    fmax=20.0,
    size=1.5,
    markers=None,
    offset=None,
    matchflat=None,
    matchzero=False,
    average=None,
    average_overlay=False,
    average_key="avg",
    vertical=None,
    horizontal=None,
    save=None,
):
    """Plot 1D PMFs.

    Input handling
    --------------
    - plot1D_combined([d1, d2, d3], ...) plots each entry as an individual PMF.
    - plot1D_combined([[d1, d2, d3]], ...) averages that group and plots the mean PMF.
    - plot1D_combined([[d1, d2], d3, [d4, d5]], ...) averages groups and plots singles.

    Averaging
    ---------
    - Explicit groups (list/tuple entries inside the top-level list) are averaged using
      `average` if given, otherwise "boltzmann".
    - If no explicit groups are present, setting `average` averages all provided PMFs
      (optionally overlaying individuals via `average_overlay`).
    """
    if isinstance(df, dict) and "comb" not in df:
        avg_mode = average if average is not None else "boltzmann"
        plot1D_grouped(
            df,
            tag,
            average=avg_mode,
            usembar=usembar,
            minmax=minmax,
            tics=tics,
            label=label,
            colors=colors,
            key=key,
            kbT=kbT,
            nbins=nbins,
            fmin=fmin,
            fmax=fmax,
            size=size,
            markers=markers,
            offset=offset,
            matchflat=matchflat,
            matchzero=matchzero,
            average_overlay=average_overlay,
            vertical=vertical,
            horizontal=horizontal,
            save=save,
        )
        return

    top = list(df) if isinstance(df, (list, tuple)) else [df]

    groups: list[list[dict]] = []
    is_group: list[bool] = []
    for it in top:
        if _is_tiledata_dict(it):
            groups.append([it])
            is_group.append(False)
            continue
        if isinstance(it, (list, tuple)):
            g = list(it)
            if not g:
                raise ValueError("plot1D_combined: empty group")
            for d in g:
                if not _is_tiledata_dict(d):
                    raise TypeError("plot1D_combined: group entries must be tiledata dicts")
            groups.append(g)
            is_group.append(True)
            continue
        raise TypeError("plot1D_combined: df must be tiledata dicts or lists of them")

    has_groups = any(is_group)
    if has_groups:
        avg_mode = average if average is not None else "boltzmann"

        n_items = len(groups)
        if key is not None and len(key) != n_items:
            raise ValueError("plot1D_combined: key length must match top-level entries")

        if isinstance(tag, list):
            taglist = list(tag)
            stag = taglist[0].rstrip("0123456789")
        else:
            taglist = None
            stag = str(tag).rstrip("0123456789")

        reps: list[dict] = []
        rep_item: list[int] = []
        for gi, g in enumerate(groups):
            for d in g:
                reps.append(d)
                rep_item.append(gi)

        vals = []
        for d in reps:
            if taglist is None:
                vals.append(np.asarray(d["comb"][tag], float).ravel())
            else:
                for t in taglist:
                    vals.append(np.asarray(d["comb"][t], float).ravel())

        x = np.concatenate(vals)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("plot1D_combined: no finite values for common range")

        xmin = float(np.min(x))
        xmax = float(np.max(x))
        pad = 1e-12 * (xmax - xmin + 1.0)
        rang = (xmin, xmax + pad)

        pmf_rep = []
        ranges_rep = []
        err_rep = []
        for d in reps:
            if taglist is not None:
                dplotlist = []
                for t in taglist:
                    dp = d["comb"][[t, "ww"]].fillna(0).copy()
                    dp.columns = [stag, "ww"]
                    dplotlist.append(dp)
                dplot = pd.concat(dplotlist, ignore_index=True)
                res = pmf1d_from_weights(dplot, stag, nbins=nbins, kT=kbT, rang=rang)
            else:
                if usembar:
                    res = pmf1d_mbar(
                        d["mbar"],
                        d["comb"],
                        tag,
                        nbins=nbins,
                        kT=kbT,
                        rang=rang,
                    )
                    if stag != tag:
                        res["pmf"] = res["pmf"].rename(columns={tag: stag})
                        if res["dpmf"] is not None:
                            res["dpmf"] = res["dpmf"].rename(columns={tag: stag})
                        res["ranges"] = res["ranges"].rename(columns={tag: stag})
                else:
                    dplot = d["comb"][[tag, "ww"]].fillna(0).copy()
                    if stag != tag:
                        dplot = dplot.rename(columns={tag: stag})
                    res = pmf1d_from_weights(dplot, stag, nbins=nbins, kT=kbT, rang=rang)

            pmf_rep.append(res["pmf"])
            ranges_rep.append(res["ranges"])
            err_rep.append(res["dpmf"])

        n_rep = len(pmf_rep)
        base_rep = [0.0] * n_rep
        if offset is None:
            pass
        elif isinstance(offset, (float, int)):
            base_rep = [float(offset)] * n_rep
        else:
            off = list(offset)
            if len(off) == n_items:
                base_rep = [float(off[rep_item[i]]) for i in range(n_rep)]
            elif len(off) == n_rep:
                base_rep = [float(off[i]) for i in range(n_rep)]
            else:
                base_rep = [float(off[i]) if i < len(off) else 0.0 for i in range(n_rep)]

        for p, o in zip(pmf_rep, base_rep):
            p[stag] = p[stag] + float(o)

        extra = [0.0] * n_rep
        if matchflat is not None and len(matchflat) == 2:
            mmin, mmax = float(matchflat[0]), float(matchflat[1])
            means = []
            for p, r in zip(pmf_rep, ranges_rep):
                mask = r[stag].between(mmin, mmax, inclusive="both")
                m = p[stag][mask].mean()
                means.append(float(m) if pd.notna(m) else 0.0)

            if matchzero:
                extra = [-m for m in means]
            else:
                mmax_val = max(means) if means else 0.0
                extra = [mmax_val - m for m in means]

        for p, o in zip(pmf_rep, extra):
            p[stag] = p[stag] + float(o)

        pmf_plot = []
        ranges_plot = []
        err_plot = []
        colors_plot = []
        key_plot = [] if key is not None else None

        for gi in range(n_items):
            idxs = [i for i, g in enumerate(rep_item) if g == gi]
            col = colors[gi] if gi < len(colors) else (0.5, 0.5, 0.5)

            if is_group[gi]:
                if average_overlay:
                    for ri in idxs:
                        pmf_plot.append(pmf_rep[ri])
                        ranges_plot.append(ranges_rep[ri])
                        err_plot.append(None)
                        colors_plot.append(col)
                        if key_plot is not None:
                            key_plot.append("_nolegend_")

                avg = average_pmf1d(
                    [pmf_rep[i] for i in idxs],
                    [ranges_rep[i] for i in idxs],
                    stag,
                    method=avg_mode,
                    kT=kbT,
                )
                pmf_plot.append(avg["pmf"])
                ranges_plot.append(avg["ranges"])
                err_plot.append(avg["dpmf"])
                colors_plot.append(col)
                if key_plot is not None:
                    key_plot.append(key[gi])
            else:
                ri = idxs[0]
                pmf_plot.append(pmf_rep[ri])
                ranges_plot.append(ranges_rep[ri])
                err_plot.append(err_rep[ri])
                colors_plot.append(col)
                if key_plot is not None:
                    key_plot.append(key[gi])

        dist1D(
            pmf_plot,
            ranges_plot,
            err=err_plot,
            size=size,
            markers=markers,
            tag=stag,
            minmax=minmax,
            tics=tics,
            label=label,
            fmin=fmin,
            fmax=fmax,
            colors=colors_plot,
            key=key_plot,
            vertical=vertical,
            horizontal=horizontal,
            save=save,
        )
        return

    # no explicit groups: preserve the old behavior (plot individual or average all)
    pmf = []
    ranges = []
    err = []

    dflist = top

    if isinstance(tag, list):
        taglist = list(tag)
        stag = taglist[0].rstrip("0123456789")
    else:
        taglist = None
        stag = str(tag).rstrip("0123456789")

    rang = None
    if average is not None:
        vals = []
        for d in dflist:
            if taglist is None:
                vals.append(np.asarray(d["comb"][tag], float).ravel())
            else:
                for t in taglist:
                    vals.append(np.asarray(d["comb"][t], float).ravel())
        x = np.concatenate(vals)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("plot1D_combined: no finite values for common range")
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        pad = 1e-12 * (xmax - xmin + 1.0)
        rang = (xmin, xmax + pad)

    for d in dflist:
        if taglist is not None:
            dplotlist = []
            for t in taglist:
                dp = d["comb"][[t, "ww"]].fillna(0).copy()
                dp.columns = [stag, "ww"]
                dplotlist.append(dp)
            dplot = pd.concat(dplotlist, ignore_index=True)
            res = pmf1d_from_weights(dplot, stag, nbins=nbins, kT=kbT, rang=rang)
        else:
            if usembar:
                res = pmf1d_mbar(
                    d["mbar"],
                    d["comb"],
                    tag,
                    nbins=nbins,
                    kT=kbT,
                    rang=rang,
                )
                if stag != tag:
                    res["pmf"] = res["pmf"].rename(columns={tag: stag})
                    if res["dpmf"] is not None:
                        res["dpmf"] = res["dpmf"].rename(columns={tag: stag})
                    res["ranges"] = res["ranges"].rename(columns={tag: stag})
            else:
                dplot = d["comb"][[tag, "ww"]].fillna(0).copy()
                if stag != tag:
                    dplot = dplot.rename(columns={tag: stag})
                res = pmf1d_from_weights(dplot, stag, nbins=nbins, kT=kbT, rang=rang)

        pmf.append(res["pmf"])
        ranges.append(res["ranges"])
        err.append(res["dpmf"])

    n = len(pmf)

    if offset is None:
        base = [0.0] * n
    elif isinstance(offset, (float, int)):
        base = [float(offset)] * n
    else:
        base = [float(offset[i]) if i < len(offset) else 0.0 for i in range(n)]

    extra = [0.0] * n
    if matchflat is not None and len(matchflat) == 2:
        mmin, mmax = float(matchflat[0]), float(matchflat[1])

        means = []
        for p, r in zip(pmf, ranges):
            mask = r[stag].between(mmin, mmax, inclusive="both")
            m = p[stag][mask].mean()
            means.append(float(m) if pd.notna(m) else 0.0)

        if matchzero:
            extra = [-m for m in means]
        else:
            mmax_val = max(means) if means else 0.0
            extra = [mmax_val - m for m in means]

    total = [b + e for b, e in zip(base, extra)]
    for p, o in zip(pmf, total):
        p[stag] = p[stag] + float(o)

    if average is not None:
        avg = average_pmf1d(pmf, ranges, stag, method=average, kT=kbT)
        if average_overlay:
            pmf_plot = pmf + [avg["pmf"]]
            ranges_plot = ranges + [avg["ranges"]]
            err_plot = [None] * len(pmf) + [avg["dpmf"]]

            key_plot = None
            if key is not None:
                key_plot = list(key) + [average_key]
        else:
            pmf_plot = [avg["pmf"]]
            ranges_plot = [avg["ranges"]]
            err_plot = [avg["dpmf"]]

            key_plot = None
            if key is not None:
                key_plot = [average_key]
        colors_plot = colors
    else:
        pmf_plot = pmf
        ranges_plot = ranges
        err_plot = err
        key_plot = key
        colors_plot = colors

    dist1D(
        pmf_plot,
        ranges_plot,
        err=err_plot,
        size=size,
        markers=markers,
        tag=stag,
        minmax=minmax,
        tics=tics,
        label=label,
        fmin=fmin,
        fmax=fmax,
        colors=colors_plot,
        key=key_plot,
        vertical=vertical,
        horizontal=horizontal,
        save=save,
    )


def plot_series(
    s, *, title=None, xlabel=None, ylabel=None, logx=False, logy=False, save=None, size=1
):
    fig, ax = plt.subplots(figsize=(4 * size, 3 * size))
    ax.plot(s.index, s.values)
    ax.set_title(title or "")
    ax.set_xlabel(xlabel or s.index.name or "x")
    ax.set_ylabel(ylabel or s.name or "value")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300)
    plt.show()


def plot_hist_overlap(
    frames: dict[str, pd.DataFrame],
    col: str,
    *,
    wcol: str | None = None,
    keys: Iterable[str] | None = None,
    bins: int = 60,
    rang: tuple[float, float] | None = None,
    density: bool = True,
    cmap: str = "viridis",
    alpha: float = 0.35,
    lw: float = 1.3,
    figsize: tuple[float, float] = (14.0, 4.2),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    legend_max: int = 18,
    sort_keys: bool = True,
    save: str | None = None,
) -> None:
    """
    Overlay per-dict-entry histograms for a given dataframe column.

    Parameters
    ----------
    frames : dict[str, DataFrame]
        Mapping like {"run_6.20": df, ...}. Non-DataFrame entries are ignored.
    col : str
        Column to histogram (e.g., "gxdist").
    wcol : str | None
        Optional per-sample weights column (e.g., "ww").
    keys : iterable[str] | None
        Subset/order of keys to plot. Defaults to all DataFrame keys.
    bins : int
        Number of bins.
    rang : (float, float) | None
        Histogram x-range. If None, computed from all data.
    density : bool
        Plot probability density (recommended for overlap checks).
    cmap : str
        Matplotlib colormap name.
    alpha : float
        Line fill alpha.
    lw : float
        Line width.
    figsize : (float, float)
        Wide figure size.
    legend : bool
        Show legend (auto-limited via legend_max).
    legend_max : int
        Max legend entries (avoids huge legends for 50 windows).
    sort_keys : bool
        Sort keys (lexicographic) if keys is None.
    save : str | None
        Save figure path if given.
    """
    df_keys = [k for k, v in frames.items() if isinstance(v, pd.DataFrame)]
    if keys is None:
        use_keys = sorted(df_keys) if sort_keys else list(df_keys)
    else:
        use_keys = [k for k in keys if k in frames and isinstance(frames[k], pd.DataFrame)]

    if not use_keys:
        raise ValueError("No DataFrame entries to plot.")

    xs: list[np.ndarray] = []
    ws: list[np.ndarray | None] = []
    for k in use_keys:
        s = pd.to_numeric(frames[k][col], errors="coerce").to_numpy()
        mask = np.isfinite(s)
        s = s[mask]
        if s.size == 0:
            continue
        xs.append(s)
        if wcol is None:
            ws.append(None)
        else:
            w = pd.to_numeric(frames[k][wcol], errors="coerce").to_numpy()
            w = w[mask]
            w = w[np.isfinite(w)]
            if w.size != s.size:
                w = None
            ws.append(w)

    if not xs:
        raise ValueError(f"No finite data found for column {col!r}.")

    if rang is None:
        xmin = min(float(np.min(a)) for a in xs)
        xmax = max(float(np.max(a)) for a in xs)
        pad = 1e-12 * (xmax - xmin + 1.0)
        rang = (xmin, xmax + pad)

    edges = np.linspace(rang[0], rang[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=figsize, dpi=100, constrained_layout=True)
    cm = plt.get_cmap(cmap)
    n = len(xs)
    colors = [cm(i / max(n - 1, 1)) for i in range(n)]

    shown = 0
    for i, (k, x, w) in enumerate(zip(use_keys, xs, ws)):
        h, _ = np.histogram(x, bins=edges, weights=w, density=density)
        ax.plot(centers, h, color=colors[i], lw=lw, alpha=0.95, label=k)
        ax.fill_between(centers, 0.0, h, color=colors[i], alpha=alpha, linewidth=0.0)
        shown += 1

    # ax.set_title(title or f"Histogram overlap: {col}")
    ax.set_xlabel(xlabel or col)
    ax.set_ylabel(ylabel or ("Density" if density else "Count"))
    ax.set_xlim(rang[0], rang[1])
    ax.grid(True, linestyle="--", alpha=0.35)

    if legend and shown <= legend_max:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    if save:
        fig.savefig(save, dpi=300)
    plt.show()
