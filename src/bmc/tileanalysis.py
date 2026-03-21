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

import contextlib
import gzip
import logging
import os
import re
import sys
import warnings
from collections.abc import Iterable, Iterator
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


@contextlib.contextmanager
def _suppress_c_stdout_stderr(enabled: bool = True) -> Iterator[None]:
    if not enabled:
        yield
        return

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_out = os.dup(1)
    old_err = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)
        os.close(devnull)


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

    with _suppress_c_stdout_stderr():
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

    with _suppress_c_stdout_stderr():
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

    with _suppress_c_stdout_stderr():
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


def _open_text_auto(fname):
    path = Path(fname)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _choose_optional_gz(fname):
    fname = Path(fname)
    if fname.exists():
        return fname

    gz_name = Path(str(fname) + ".gz")
    if gz_name.exists():
        return gz_name
    return None


def _parse_simple_header(fname, known_cols):
    with _open_text_auto(fname) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue

            tokens = [tok.lstrip("#!").strip().lower() for tok in s.split()]
            tokens = [tok for tok in tokens if tok]
            if tokens and tokens[0] == "fields":
                tokens = tokens[1:]

            if tokens and all(tok in known_cols for tok in tokens):
                return tokens
            return None
    return None


def _count_data_columns(fname, *, skiprows=1):
    with _open_text_auto(fname) as fh:
        for i, line in enumerate(fh):
            if i < skiprows:
                continue
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            return len(s.split())
    return None


def _read_umbrella_table(
    fname,
    *,
    required_cols,
    optional_cols=(),
    verbose=False,
):
    chosen = _choose_optional_gz(fname)
    if chosen is None:
        if verbose:
            gz_name = Path(str(Path(fname)) + ".gz")
            print(f"WARNING: no {fname} or {gz_name} found")
        return None

    if verbose:
        print(f"reading umbrella data from {chosen}")

    known_cols = set(required_cols) | set(optional_cols)
    cols = _parse_simple_header(chosen, known_cols)
    if cols is None:
        ncols = _count_data_columns(chosen, skiprows=1)
        if ncols is None:
            return pd.DataFrame(columns=list(required_cols) + list(optional_cols))
        if ncols == len(required_cols):
            cols = list(required_cols)
        elif ncols == len(required_cols) + len(optional_cols):
            cols = list(required_cols) + list(optional_cols)
        else:
            raise ValueError(
                f"Could not determine columns for {chosen}: found {ncols} data columns"
            )

    dtype = {}
    for col in cols:
        dtype[col] = int if col.endswith("step") else float

    return pd.read_csv(
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


def read_umbrella_bias(dir, umbrellas, *, verbose=False):
    required_cols = [
        "step",
        "xbias",
        "ybias",
        "zbias",
        "anglebias",
        "torsionbias",
        "rotbias",
    ]
    optional_cols = ["dihbias"]

    frames = []
    dir = Path(dir)

    for u in umbrellas:
        fname = dir / u / "bias.dat"
        df = _read_umbrella_table(
            fname,
            required_cols=required_cols,
            optional_cols=optional_cols,
            verbose=False,
        )

        if df is None:
            if verbose:
                print(f"WARNING: no bias.dat or bias.dat.gz for umbrella {u}")
            continue

        dihbias = df["dihbias"] if "dihbias" in df.columns else 0.0
        df["ubias"] = (
            df["xbias"]
            + df["ybias"]
            + df["zbias"]
            + df["anglebias"]
            + df["torsionbias"]
            + df["rotbias"]
            + dihbias
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
    required_cols = [
        "gstep",
        "gxdist",
        "gydist",
        "gzdist",
        "gangle",
        "gtorsion",
        "grot1",
        "grot2",
    ]
    optional_cols = ["gdihbias"]

    df = _read_umbrella_table(
        fname,
        required_cols=required_cols,
        optional_cols=optional_cols,
        verbose=verbose,
    )
    return df


_RUN_RE = re.compile(r"^run_" r"(\d+(?:\.\d+)?)" r"(?:_(\d+(?:\.\d+)?))?$")


def find_run_dirs(dir: str) -> list[str]:
    base = Path(dir)
    paths: list[str] = []

    for p in base.iterdir():
        if p.is_dir() and _RUN_RE.match(p.name):
            paths.append(p.name)

    def sort_key(name: str) -> tuple[float, int, float]:
        m = _RUN_RE.match(name)
        assert m is not None
        first = float(m.group(1))
        second_s = m.group(2)

        if second_s is None:
            return first, 0, 0.0
        return first, 1, float(second_s)

    return sorted(paths, key=sort_key)


def _normalize_bias_tags(
    biasval: str | list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    if isinstance(biasval, str):
        tags = [biasval]
    else:
        tags = list(biasval)

    if not tags:
        raise ValueError("biasval must contain at least one tag")

    out: list[str] = []
    for tag in tags:
        if tag not in out:
            out.append(tag)
    return tuple(out)


def _combined_bias_tag(tags: tuple[str, ...]) -> str:
    if len(tags) == 1:
        return tags[0]
    return "_".join(tags)


def _sum_bias_columns(df: pd.DataFrame, tags: tuple[str, ...]) -> pd.Series:
    missing = [tag for tag in tags if tag not in df.columns]
    if missing:
        raise KeyError(f"Missing bias columns: {missing}")
    return df.loc[:, list(tags)].sum(axis=1)


def _ensure_combined_bias_column(df: pd.DataFrame, tags: tuple[str, ...]) -> str:
    bias_key = _combined_bias_tag(tags)
    if len(tags) > 1 or bias_key not in df.columns:
        df[bias_key] = _sum_bias_columns(df, tags)
    return bias_key


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

    bias_tags = _normalize_bias_tags(biasval)
    bias_key = _combined_bias_tag(bias_tags)

    if path is None:
        path = find_run_dirs(dir)

    geo = read_umbrella_geometry(dir + "/" + path[0] + "/geometry.dat", verbose=verbose)

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
    for p in path:
        _ensure_combined_bias_column(bias[p], bias_tags)

    for i in range(nwin):
        bindiv = bias[path[i]].iloc[i * nper : (i + 1) * nper].reset_index(drop=True)
        data[path[i]] = pd.merge(
            data[path[i]], bindiv, left_index=True, right_index=True, how="inner"
        )
        _ensure_combined_bias_column(data[path[i]], bias_tags)

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
        wham = unbias_wham(np.asarray(data[p][[bias_key]], dtype=float))
        data[p]["ww"] = pd.DataFrame(np.exp(wham["logW"]) / np.sum(np.exp(wham["logW"])))

    combmask = pd.concat([mask[p] for p in path], ignore_index=True)
    combmask_arr = combmask.to_numpy(dtype=bool)
    data["comb"] = pd.concat([data[p] for p in path], ignore_index=True)

    bias_matrix = np.column_stack(
        [np.asarray(bias[p][bias_key].iloc[combmask_arr], dtype=float) for p in path]
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


def _is_energy_vector(obj) -> bool:
    if isinstance(obj, pd.Series):
        return True
    if isinstance(obj, np.ndarray):
        return obj.ndim <= 1
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return True
        return np.isscalar(obj[0])
    return False


def _validate_energy_offset(energy_offset, nframes: int, *, name="energy_offset") -> np.ndarray:
    arr = np.asarray(energy_offset, float).ravel()
    if arr.size != nframes:
        raise ValueError(f"{name} length mismatch: got {arr.size}, expected {nframes}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _get_frame_energy(
    data: pd.DataFrame,
    *,
    energy_offset=None,
    energy_col=None,
) -> np.ndarray | None:
    if energy_offset is not None and energy_col is not None:
        raise ValueError("Provide either energy_offset or energy_col, not both")
    if energy_col is not None:
        if energy_col not in data.columns:
            raise KeyError(f"Missing energy column: {energy_col}")
        energy_offset = data[energy_col]
    if energy_offset is None:
        return None
    return _validate_energy_offset(energy_offset, len(data))


def _apply_energy_offset_to_weights(
    w: np.ndarray,
    energy_offset=None,
    *,
    kT: float = kb * T,
) -> np.ndarray:
    w = np.asarray(w, float).ravel()
    if energy_offset is None:
        return w

    de = _validate_energy_offset(energy_offset, w.size)
    shift = float(np.min(de))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        fac = np.exp(-(de - shift) / float(kT))
    return w * fac


def _split_combined_energy(df, energy_offset) -> dict[str, np.ndarray]:
    arr = _validate_energy_offset(energy_offset, len(df["comb"]))
    out: dict[str, np.ndarray] = {}
    start = 0
    for p in df["sets"]:
        nrow = len(df[p])
        out[p] = arr[start : start + nrow]
        start += nrow

    if start != arr.size:
        raise ValueError("combined energy_offset split mismatch")
    return out


def _normalize_set_energy_offsets(
    df, energy_offset, *, setlist=None
) -> dict[str, np.ndarray | None]:
    if setlist is None:
        setlist = df["sets"]

    if energy_offset is None:
        return {p: None for p in setlist}

    if isinstance(energy_offset, dict):
        if "comb" in energy_offset and len(energy_offset) == 1:
            by_set = _split_combined_energy(df, energy_offset["comb"])
            return {p: by_set[p] for p in setlist}

        missing = [p for p in setlist if p not in energy_offset]
        if missing:
            raise KeyError(f"Missing energy_offset for sets: {missing}")

        return {
            p: _validate_energy_offset(
                energy_offset[p],
                len(df[p]),
                name=f"energy_offset[{p}]",
            )
            for p in setlist
        }

    if _is_energy_vector(energy_offset):
        by_set = _split_combined_energy(df, energy_offset)
        return {p: by_set[p] for p in setlist}

    if not isinstance(energy_offset, (list, tuple)):
        raise TypeError("energy_offset must be an array or a per-set container")

    if len(energy_offset) != len(setlist):
        raise ValueError("energy_offset length must match number of plotted sets")

    return {
        p: _validate_energy_offset(eo, len(df[p]), name=f"energy_offset[{p}]")
        for p, eo in zip(setlist, energy_offset)
    }


def _normalize_energy_structure(obj, energy_offset):
    if _is_tiledata_dict(obj):
        return energy_offset

    if isinstance(obj, (list, tuple)):
        items = list(obj)
    else:
        raise TypeError("energy_offset shape does not match input data")

    if energy_offset is None:
        return [_normalize_energy_structure(it, None) for it in items]

    if len(items) == 1 and _is_tiledata_dict(items[0]) and _is_energy_vector(energy_offset):
        return [_normalize_energy_structure(items[0], energy_offset)]

    if not isinstance(energy_offset, (list, tuple)):
        raise ValueError("energy_offset must mirror the input data structure")

    if len(energy_offset) != len(items):
        raise ValueError("energy_offset length must match the input data structure")

    return [_normalize_energy_structure(it, eo) for it, eo in zip(items, energy_offset)]


def _pmf2d_from_frame_data(
    data: pd.DataFrame,
    xtag,
    ytag,
    *,
    nbins=(100, 100),
    kT=kb * T,
    rang=None,
    energy_offset=None,
    energy_col=None,
):
    energy = _get_frame_energy(
        data,
        energy_offset=energy_offset,
        energy_col=energy_col,
    )

    if isinstance(xtag, list) and isinstance(ytag, list):
        sxtag = xtag[0].rstrip("0123456789")
        sytag = ytag[0].rstrip("0123456789")

        dplotlist = []
        eplotlist = []
        for xkey, ykey in zip(xtag, ytag):
            dp = data[[xkey, ykey, "ww"]].fillna(0).copy()
            dp.columns = [sxtag, sytag, "ww"]
            dplotlist.append(dp)
            if energy is not None:
                eplotlist.append(energy)

        dplot = pd.concat(dplotlist, ignore_index=True)
        denergy = None if energy is None else np.concatenate(eplotlist)

        res = pmf2d_from_weights(
            dplot,
            [sxtag, sytag],
            nbins=nbins,
            kT=kT,
            rang=rang,
            energy_offset=denergy,
        )
        return res, sxtag, sytag

    sxtag = str(xtag).rstrip("0123456789")
    sytag = str(ytag).rstrip("0123456789")

    dplot = data[[xtag, ytag, "ww"]].fillna(0).copy()
    res = pmf2d_from_weights(
        dplot,
        [xtag, ytag],
        nbins=nbins,
        kT=kT,
        rang=rang,
        energy_offset=energy,
    )
    return res, sxtag, sytag


def _pmf1d_from_tiledata(
    data,
    tag="dist",
    *,
    usembar=False,
    kT=kb * T,
    nbins=50,
    rang=None,
    energy_offset=None,
    energy_col=None,
):
    if isinstance(tag, list):
        taglist = list(tag)
        stag = taglist[0].rstrip("0123456789")
    else:
        taglist = None
        stag = str(tag).rstrip("0123456789")

    energy = _get_frame_energy(
        data["comb"],
        energy_offset=energy_offset,
        energy_col=energy_col,
    )

    if taglist is not None:
        dplotlist = []
        eplotlist = []
        for t in taglist:
            dp = data["comb"][[t, "ww"]].fillna(0).copy()
            dp.columns = [stag, "ww"]
            dplotlist.append(dp)
            if energy is not None:
                eplotlist.append(energy)

        dplot = pd.concat(dplotlist, ignore_index=True)
        denergy = None if energy is None else np.concatenate(eplotlist)
        return pmf1d_from_weights(
            dplot,
            stag,
            nbins=nbins,
            kT=kT,
            rang=rang,
            energy_offset=denergy,
        )

    if usembar:
        res = pmf1d_mbar(
            data["mbar"],
            data["comb"],
            tag,
            nbins=nbins,
            kT=kT,
            rang=rang,
            energy_offset=energy,
        )
        if stag != tag:
            res["pmf"] = res["pmf"].rename(columns={tag: stag})
            if res["dpmf"] is not None:
                res["dpmf"] = res["dpmf"].rename(columns={tag: stag})
            res["ranges"] = res["ranges"].rename(columns={tag: stag})
        return res

    dplot = data["comb"][[tag, "ww"]].fillna(0).copy()
    if stag != tag:
        dplot = dplot.rename(columns={tag: stag})
    return pmf1d_from_weights(
        dplot,
        stag,
        nbins=nbins,
        kT=kT,
        rang=rang,
        energy_offset=energy,
    )


def pmf1d_mbar(
    mbar,
    data,
    tag,
    *,
    kT=kb * T,
    nbins=100,
    rang=None,
    verbose=False,
    energy_offset=None,
    energy_col=None,
):
    """1D PMF via PyMBAR FES with optional target-state reweighting."""
    if "mbar" in mbar:
        mbar = mbar["mbar"]

    x = np.asarray(data[tag], float).ravel()
    energy = _get_frame_energy(
        data,
        energy_offset=energy_offset,
        energy_col=energy_col,
    )
    if energy is None:
        u_n = np.zeros(x.shape[0], float)
    else:
        u_n = energy / float(kT)

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


def pmf2d_from_weights(
    data,
    tag,
    *,
    wtag="ww",
    kT=kb * T,
    nbins=(100, 100),
    rang=None,
    energy_offset=None,
    energy_col=None,
):
    """
    Project onto a 2D reaction coordinate (x,y) using weights.

    If `energy_offset` is given, it is interpreted as a per-frame target-state
    energy offset in kJ/mol and applied as exp(-energy_offset / kT) before
    histogramming.
    """

    x = np.asarray(data[tag[0]], float).ravel()
    y = np.asarray(data[tag[1]], float).ravel()
    w = np.asarray(data[wtag], float).ravel()
    energy = _get_frame_energy(
        data,
        energy_offset=energy_offset,
        energy_col=energy_col,
    )
    w = _apply_energy_offset_to_weights(w, energy, kT=kT)

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


def pmf1d_from_weights(
    data,
    tag,
    *,
    wtag="ww",
    kT=kb * T,
    nbins=100,
    rang=None,
    energy_offset=None,
    energy_col=None,
):
    """
    Project onto a 1D reaction coordinate x using weights.

    If `energy_offset` is given, it is interpreted as a per-frame target-state
    energy offset in kJ/mol and applied as exp(-energy_offset / kT) before
    histogramming.
    """
    x = np.asarray(data[tag], float).ravel()
    w = np.asarray(data[wtag], float).ravel()
    energy = _get_frame_energy(
        data,
        energy_offset=energy_offset,
        energy_col=energy_col,
    )
    w = _apply_energy_offset_to_weights(w, energy, kT=kT)

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

    return dict(
        edges=xe,
        centers=x_centers,
        F_kT=F_kT,
        P=P,
        pmf=pmf1d,
        dpmf=None,
        ranges=ranges,
    )


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
    energy_offset=None,
    energy_col=None,
    save=None,
):
    res, sxtag, sytag = _pmf2d_from_frame_data(
        df["comb"],
        xtag,
        ytag,
        nbins=nbins,
        kT=kbT,
        energy_offset=energy_offset,
        energy_col=energy_col,
    )

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
    energy_offset=None,
    energy_col=None,
    save=None,
):

    if setlist is None:
        setlist = df["sets"]

    if energy_offset is not None and energy_col is not None:
        raise ValueError("Provide either energy_offset or energy_col, not both")

    energy_by_set = _normalize_set_energy_offsets(
        df,
        energy_offset,
        setlist=setlist,
    )

    pmf = []
    rang = []
    sxtag = None
    sytag = None
    for p in setlist:
        res, sxtag, sytag = _pmf2d_from_frame_data(
            df[p],
            xtag,
            ytag,
            nbins=nbins,
            kT=kbT,
            energy_offset=energy_by_set[p],
            energy_col=energy_col,
        )
        pmf.append(res["pmf"])
        rang.append(res["ranges"])

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
    energy_offset=None,
    energy_col=None,
    save=None,
):
    """Average replicate PMFs per group and plot group averages together."""
    if isinstance(groups, dict) and "comb" not in groups:
        group_names = list(groups.keys())
        group_list = list(groups.values())
        if isinstance(energy_offset, dict):
            missing = [k for k in group_names if k not in energy_offset]
            if missing:
                raise KeyError(f"Missing energy_offset groups: {missing}")
            energy_input = [energy_offset[k] for k in group_names]
        else:
            energy_input = energy_offset
    else:
        group_names = None
        group_list = list(groups) if isinstance(groups, (list, tuple)) else [groups]
        energy_input = energy_offset

    energy_groups = _normalize_energy_structure(group_list, energy_input)

    norm_groups = []
    norm_energy_groups = []
    for g, ge in zip(group_list, energy_groups):
        if _is_tiledata_dict(g):
            norm_groups.append([g])
            norm_energy_groups.append([ge])
        else:
            norm_groups.append(list(g))
            if ge is None:
                ge = [None] * len(g)
            elif not isinstance(ge, (list, tuple)) or len(ge) != len(g):
                raise ValueError("energy_offset must mirror grouped input data")
            norm_energy_groups.append(list(ge))

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
    rep_energy = []
    for gi, (g, ge) in enumerate(zip(norm_groups, norm_energy_groups)):
        for d, eo in zip(g, ge):
            reps.append(d)
            rep_group.append(gi)
            rep_energy.append(eo)

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
    for d, eo in zip(reps, rep_energy):
        res = _pmf1d_from_tiledata(
            d,
            tag,
            usembar=usembar,
            kT=kbT,
            nbins=nbins,
            rang=rang,
            energy_offset=eo,
            energy_col=energy_col,
        )
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
    energy_offset=None,
    energy_col=None,
    save=None,
):
    """Plot 1D PMFs with optional per-frame target-state reweighting."""
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
            energy_offset=energy_offset,
            energy_col=energy_col,
            save=save,
        )
        return

    top = list(df) if isinstance(df, (list, tuple)) else [df]
    energy_top = _normalize_energy_structure(top, energy_offset)

    groups: list[list[dict]] = []
    group_energy = []
    is_group: list[bool] = []
    for it, eo in zip(top, energy_top):
        if _is_tiledata_dict(it):
            groups.append([it])
            group_energy.append(eo)
            is_group.append(False)
            continue
        if isinstance(it, (list, tuple)):
            g = list(it)
            if not g:
                raise ValueError("plot1D_combined: empty group")
            for d in g:
                if not _is_tiledata_dict(d):
                    raise TypeError("plot1D_combined: group entries must be tiledata dicts")
            if eo is None:
                eo = [None] * len(g)
            elif not isinstance(eo, (list, tuple)) or len(eo) != len(g):
                raise ValueError("energy_offset must mirror grouped input data")
            groups.append(g)
            group_energy.append(list(eo))
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
        rep_energy = []
        for gi, (g, ge) in enumerate(zip(groups, group_energy)):
            if is_group[gi]:
                for d, eo in zip(g, ge):
                    reps.append(d)
                    rep_item.append(gi)
                    rep_energy.append(eo)
            else:
                reps.append(g[0])
                rep_item.append(gi)
                rep_energy.append(ge)

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
        for d, eo in zip(reps, rep_energy):
            res = _pmf1d_from_tiledata(
                d,
                tag,
                usembar=usembar,
                kT=kbT,
                nbins=nbins,
                rang=rang,
                energy_offset=eo,
                energy_col=energy_col,
            )
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

    pmf = []
    ranges = []
    err = []

    dflist = top
    energy_list = list(energy_top)

    if isinstance(tag, list):
        stag = tag[0].rstrip("0123456789")
    else:
        stag = str(tag).rstrip("0123456789")

    rang = None
    if average is not None:
        vals = []
        if isinstance(tag, list):
            for d in dflist:
                for t in tag:
                    vals.append(np.asarray(d["comb"][t], float).ravel())
        else:
            for d in dflist:
                vals.append(np.asarray(d["comb"][tag], float).ravel())
        x = np.concatenate(vals)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("plot1D_combined: no finite values for common range")
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        pad = 1e-12 * (xmax - xmin + 1.0)
        rang = (xmin, xmax + pad)

    for d, eo in zip(dflist, energy_list):
        res = _pmf1d_from_tiledata(
            d,
            tag,
            usembar=usembar,
            kT=kbT,
            nbins=nbins,
            rang=rang,
            energy_offset=eo,
            energy_col=energy_col,
        )
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
