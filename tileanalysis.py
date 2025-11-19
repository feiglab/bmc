########################################################################
# class to support BMC tile analysis
#
# Michael Feig
# mfeiglab@gmail.com
#
# 2025
#
########################################################################

from __future__ import print_function, division

import sys
import os
import math
import re

from pathlib import Path

import numpy as np
import pandas as pd
import mdtraj as md

import gemmi

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from scipy.spatial.transform import Rotation as Rot
from scipy.special import logsumexp
           
import logging
logging.disable(logging.CRITICAL)

from pymbar import MBAR, FES

kb=0.008314462618
T=300

hhmarkers=[]
hhmarkers+=[{'bend': 146.6, 'twist':  2.95, 'rot': 5.67, 'dist': 6.46, 'col': 'red',    'label': 'HTP HH(P)','pos': 1}]
hhmarkers+=[{'bend': 175.3, 'twist': -0.05, 'rot': 0.00, 'dist': 6.69, 'col': 'magenta','label': 'HTP HH(T)','pos': 1}]
hhmarkers+=[{'bend': 138.4, 'twist':  5.54, 'rot': 2.11, 'dist': 6.15, 'col': '#800000','label': 'T3 HH',    'pos': -1}]
hhmarkers+=[{'bend': 144.3, 'twist':  3.48, 'rot': 5.04, 'dist': 6.47, 'col': '#800000','label': 'T4 HH',    'pos': -1}]

hpmarkers=[]
hpmarkers+=[{'bend': 150.9, 'twist': -3.76, 'rot': 13.69, 'dist': 5.77, 'col': 'purple','label': 'HTP HP',    'pos': 1}]
hpmarkers+=[{'bend': 143.0, 'twist': -5.59, 'rot': 15.33, 'dist': 5.68, 'col': 'purple','label': 'HP T3 HP',  'pos': -1}]
hpmarkers+=[{'bend': 148.6, 'twist': -3.07, 'rot': 12.91, 'dist': 5.94, 'col': 'purple','label': 'HP T4 HP',  'pos': 1}]

htmarkers=[]
htmarkers+=[{'bend': 159.6, 'twist': -6.63, 'rot': -0.74, 'dist': 6.66, 'col': '#8000F0','label': 'HTP HT',   'pos': 1}]

tics={}
tics['bend']=[90,120,150,180,210]
tics['twist']=[-40,-20,0,20,40]
tics['dist']=[5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2]

minmax={}
minmax['bend']=[60,230]
minmax['twist']=[-50,50]
minmax['dist']=[5.5,7.2]

label={}
label['bend']='Planar angle [deg]'
label['twist']='Twisting angle [deg]'
label['dist']='Distance [nm]'

colors1d=['blue','red','green','orange','magenta']

plt.rcParams.update({
   "font.size" : 20,
   "font.family" : 'monospace',
   "font.weight" : 'normal',
   "axes.titlesize": 24,
   "axes.labelsize": 22,
   "xtick.labelsize": 20,
   "ytick.labelsize": 20,
   "legend.fontsize": 18,
   "figure.titlesize": 20,
})

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


def eulerAngles(xA, yA, zA, xB, yB, zB, seq='YXZ', degrees=True):
    xA = _normalize_rows(xA); yA = _normalize_rows(yA); zA = _normalize_rows(zA)
    xB = _normalize_rows(xB); yB = _normalize_rows(yB); zB = _normalize_rows(zB)
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
                if ch.name == chain_name and res.seqid.num>=resmin and res.seqid.num<=resmax:
                    idx.append(i)
                i += 1
    return idx


def hh_dimeridx(pdb1,chlist1,pdb2,chlist2,resmin=3,resmax=88):
    s1=gemmi.read_structure(pdb1)
    s2=gemmi.read_structure(pdb2)

    alist=[a for ch in chlist1 for res in s1[0][ch] for a in res if a.name =='CA']
    xyz1_ca=np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)

    alist=[a for ch in chlist2 for res in s2[0][ch] for a in res if a.name =='CA']
    xyz2_ca=np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)

    c1=np.average(xyz1_ca,axis=0)
    c2=np.average(xyz2_ca,axis=0)
    
    xyzc1={}
    c1c={}
    for k in chlist1:
        alist=[a for res in s1[0][k] for a in res if a.name =='CA']
        xyzc1[k]=np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)
        c1c[k]=np.average(xyzc1[k],axis=0)

    xyzc2={}
    c2c={}
    for k in chlist2:
        alist=[a for res in s2[0][k] for a in res if a.name =='CA']
        xyzc2[k]=np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)
        c2c[k]=np.average(xyzc2[k],axis=0)

    dmat = {(k1, k2): np.linalg.norm(c2c[k2] - c1c[k1]) for k1 in chlist1 for k2 in chlist2}
    
    (c1alabel, c2alabel), _ = min(dmat.items(), key=lambda kv: kv[1])

    n1 = len(chlist1)
    n2 = len(chlist2)
    idx1 = {k: i for i, k in enumerate(chlist1)}
    idx2 = {k: i for i, k in enumerate(chlist2)}
    
    def at1(base, off): return chlist1[(idx1[base] + off) % n1]
    def at2(base, off): return chlist2[(idx2[base] + off) % n2]

    def prev1(k): return at1(k, -1)
    def next1(k): return at1(k, +1)
    def next2(k): return at2(k, -1)
    def prev2(k): return at2(k, +1)

    c1prev, c1next = prev1(c1alabel), next1(c1alabel)
    c2prev, c2next = prev2(c2alabel), next2(c2alabel)

    dpp=dmat[(c1prev,c2prev)]
    dpn=dmat[(c1prev,c2next)]
    dnp=dmat[(c1next,c2prev)]
    dnn=dmat[(c1next,c2next)]

    if dpp<=dpn and dpp<=dnp and dpp<=dnn:
        off1, off2 = (range(2, 8),  range(0, 6)) 
    if dpn<=dpp and dpn<=dnp and dpn<=dnn:
        off1, off2 = (range(2, 8),  range(-1, 5))
    if dnp<=dpp and dnp<=dpn and dnp<=dnn:
        off1, off2 = (range(3, 9),  range(0, 6))
    if dnn<=dpp and dnn<=dnp and dnn<=dnp:
        off1, off2 = (range(3, 9),  range(-1, 5))

    idxlist1 = [indices_of_chain(s1, at1(c1alabel, o), resmin, resmax) for o in off1]
    idxlist2 = [indices_of_chain(s2, at2(c2alabel, o), resmin, resmax) for o in off2]

    return [idxlist1,idxlist2]


def ph_dimeridx(pdb1,chlist1,pdb2,chlist2,resmin1=1,resmax1=95,resmin2=3,resmax2=88):
    s1=gemmi.read_structure(pdb1)
    s2=gemmi.read_structure(pdb2)

    alist=[a for ch in chlist1 for res in s1[0][ch] for a in res if a.name =='CA']
    xyz1_ca=np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)

    alist=[a for ch in chlist2 for res in s2[0][ch] for a in res if a.name =='CA']
    xyz2_ca=np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)

    c1=np.average(xyz1_ca,axis=0)
    c2=np.average(xyz2_ca,axis=0)
    
    xyzc1={}
    c1c={}
    for k in chlist1:
        alist=[a for res in s1[0][k] for a in res if a.name =='CA']
        xyzc1[k]=np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)
        c1c[k]=np.average(xyzc1[k],axis=0)

    xyzc2={}
    c2c={}
    for k in chlist2:
        alist=[a for res in s2[0][k] for a in res if a.name =='CA']
        xyzc2[k]=np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)
        c2c[k]=np.average(xyzc2[k],axis=0)

    dmat = {(k1, k2): np.linalg.norm(c2c[k2] - c1c[k1]) for k1 in chlist1 for k2 in chlist2}
    
    (c1alabel, c2alabel), _ = min(dmat.items(), key=lambda kv: kv[1])

    n1 = len(chlist1)
    n2 = len(chlist2)
    idx1 = {k: i for i, k in enumerate(chlist1)}
    idx2 = {k: i for i, k in enumerate(chlist2)}
    
    def at1(base, off): return chlist1[(idx1[base] + off) % n1]
    def at2(base, off): return chlist2[(idx2[base] + off) % n2]

    def next2(k): return at2(k, -1)
    def prev2(k): return at2(k, +1)

    c2prev, c2next = prev2(c2alabel), next2(c2alabel)

    if dmat[(c1alabel, c2next)] < dmat[(c1alabel, c2prev)]:
        off1, off2 = (range(2, 7),  range(-1, 5)) 
    else:
        off1, off2 = (range(2, 7),  range(0, 6))

    idxlist1 = [indices_of_chain(s1, at1(c1alabel, o), resmin1, resmax1) for o in off1]
    idxlist2 = [indices_of_chain(s2, at2(c2alabel, o), resmin2, resmax2) for o in off2]

    return [idxlist1,idxlist2]


def th_dimeridx(pdb1,chlist1,pdb2,chlist2,resmin1=5,resmax1=205,resmin2=3,resmax2=88):
    s1=gemmi.read_structure(pdb1)
    s2=gemmi.read_structure(pdb2)

    alist=[a for ch in chlist1 for res in s1[0][ch] for a in res if a.name =='CA']
    xyz1_ca=np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)

    alist=[a for ch in chlist2 for res in s2[0][ch] for a in res if a.name =='CA']
    xyz2_ca=np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)

    c1=np.average(xyz1_ca,axis=0)
    c2=np.average(xyz2_ca,axis=0)
    
    xyzc1={}
    c1c={}
    for k in chlist1:
        alist=[a for res in s1[0][k] for a in res if a.name =='CA']
        xyzc1[k]=np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)
        c1c[k]=np.average(xyzc1[k],axis=0)

    xyzc2={}
    c2c={}
    for k in chlist2:
        alist=[a for res in s2[0][k] for a in res if a.name =='CA']
        xyzc2[k]=np.array([[a.pos.x, a.pos.y, a.pos.z] for a in alist], dtype=float)
        c2c[k]=np.average(xyzc2[k],axis=0)

    dmat = {(k1, k2): np.linalg.norm(c2c[k2] - c1c[k1]) for k1 in chlist1 for k2 in chlist2}
    
    (c1alabel, c2alabel), _ = min(dmat.items(), key=lambda kv: kv[1])

    n1 = len(chlist1)
    n2 = len(chlist2)
    idx1 = {k: i for i, k in enumerate(chlist1)}
    idx2 = {k: i for i, k in enumerate(chlist2)}
    
    def at1(base, off): return chlist1[(idx1[base] + off) % n1]
    def at2(base, off): return chlist2[(idx2[base] + off) % n2]

    def prev1(k): return at1(k, -1)
    def next1(k): return at1(k, +1)
    def next2(k): return at2(k, -1)
    def prev2(k): return at2(k, +1)

    c1prev, c1next = prev1(c1alabel), next1(c1alabel)
    c2prev, c2next = prev2(c2alabel), next2(c2alabel)

    dpp=dmat[(c1prev,c2prev)]
    dpn=dmat[(c1prev,c2next)]
    dnp=dmat[(c1next,c2prev)]
    dnn=dmat[(c1next,c2next)]

    if dpp<=dpn and dpp<=dnp and dpp<=dnn:
        off1, off2 = (range(1, 4),  range(0, 6)) 
    if dpn<=dpp and dpn<=dnp and dpn<=dnn:
        off1, off2 = (range(1, 4),  range(-1, 5))
    if dnp<=dpp and dnp<=dpn and dnp<=dnn:
        off1, off2 = (range(2, 5),  range(0, 6))
    if dnn<=dpp and dnn<=dnp and dnn<=dnp:
        off1, off2 = (range(2, 5),  range(-1, 5))

    idxlist1 = [indices_of_chain(s1, at1(c1alabel, o), resmin1, resmax1) for o in off1]
    idxlist2 = [indices_of_chain(s2, at2(c2alabel, o), resmin2, resmax2) for o in off2]

    return [idxlist1,idxlist2]


def dimergeom(c1,c2,x1,y1,z1,x2,y2,z2):
    d=c2-c1
    dist=np.linalg.norm(d,axis=1,keepdims=True) # in nm

    dot = np.sum(z1*z2, axis=1, keepdims=True)
    cross=np.linalg.norm(np.cross(z1,z2,axis=1),axis=1,keepdims=True)
    ang=np.degrees(np.arctan2(cross,dot))
    
    euler=eulerAngles(x1,y1,z1,x2,y2,z2)

    x1/=np.linalg.norm(x1,axis=1,keepdims=True)
    y1/=np.linalg.norm(y1,axis=1,keepdims=True)
    z1/=np.linalg.norm(z1,axis=1,keepdims=True)
    
    shiftz=np.sum(d*z1,axis=1,keepdims=True) 
    shiftx=np.sum(d*x1,axis=1,keepdims=True)
    shifty=np.sum(d*y1,axis=1,keepdims=True)
    
    return dist,ang,euler,shiftx,shifty,shiftz    


def hh_dimergeom(traj1,idx1, traj2, idx2):
    c1list = [x for c in idx1 for x in c]    
    c1=np.average(traj1.xyz[:,c1list],axis=1)
    
    c2list = [x for c in idx2 for x in c]
    c2=np.average(traj2.xyz[:,c2list],axis=1)

    c1c={}
    for k,l in enumerate(idx1):
        c1c[k]=np.average(traj1.xyz[:,l],axis=1)

    c2c={}
    for k,l in enumerate(idx2):
        c2c[k]=np.average(traj2.xyz[:,l],axis=1)
    
    x1 = (c1c[0] + c1c[1]) - (c1c[3] + c1c[4])
    y1 =  c1c[2] - c1c[5]
    z1 = np.cross(x1, y1,axis=1)

    x2 = (c2c[0] + c2c[1]) - (c2c[3] + c2c[4])
    y2 =  c2c[2] - c2c[5]
    z2 = np.cross(x2, y2,axis=1)

    return dimergeom(c1,c2,x1,y1,z1,x2,y2,z2)


def ph_dimergeom(traj1,idx1, traj2, idx2):
    c1list = [x for c in idx1 for x in c]    
    c1=np.average(traj1.xyz[:,c1list],axis=1)
    
    c2list = [x for c in idx2 for x in c]
    c2=np.average(traj2.xyz[:,c2list],axis=1)

    c1c={}
    for k,l in enumerate(idx1):
        c1c[k]=np.average(traj1.xyz[:,l],axis=1)

    c2c={}
    for k,l in enumerate(idx2):
        c2c[k]=np.average(traj2.xyz[:,l],axis=1)
    
    x1 =  (c1c[0]-c1 + c1c[1]-c1) - (c1c[3]-c1)
    y1 =  c1c[2] - c1c[4]
    z1 = np.cross(x1, y1,axis=1)

    x2 = (c2c[0] + c2c[1]) - (c2c[3] + c2c[4])
    y2 =  c2c[2] - c2c[5]
    z2 = np.cross(x2, y2,axis=1)

    return dimergeom(c1,c2,x1,y1,z1,x2,y2,z2)

    
def th_dimergeom(traj1,idx1, traj2, idx2):
    c1list = [x for c in idx1 for x in c]    
    c1=np.average(traj1.xyz[:,c1list],axis=1)
    
    c2list = [x for c in idx2 for x in c]
    c2=np.average(traj2.xyz[:,c2list],axis=1)

    c1c={}
    for k,l in enumerate(idx1):
        c1c[k]=np.average(traj1.xyz[:,l],axis=1)

    c2c={}
    for k,l in enumerate(idx2):
        c2c[k]=np.average(traj2.xyz[:,l],axis=1)
    
    x1 =  (c1c[0]-c1) - (c1c[1]-c1 + c1c[2]-c1)
    y1 =  c1c[1] - c1c[2]
    z1 = np.cross(x1, y1,axis=1)

    x2 = (c2c[0] + c2c[1]) - (c2c[3] + c2c[4])
    y2 =  c2c[2] - c2c[5]
    z2 = np.cross(x2, y2,axis=1)

    return dimergeom(c1,c2,x1,y1,z1,x2,y2,z2)


def hh_analysis(dir=".",*,caname="CA.pdb",xtcname="CA.xtc"):
    p=Path(dir)
    capdb=p / caname

    clist=hh_dimeridx(str(capdb),['A','B','E','F','C','D'], str(capdb),['G','H','K','L','I','J'])

    t=md.load(str(p/xtcname), top=str(capdb))
    d,ang,eu,sx,sy,sz=hh_dimergeom(t,clist[0],t,clist[1])
    
    d,ang,sx,sy,sz=map(np.ravel, (d,ang,sx,sy,sz))
    eu=np.asarray(eu)

    df=pd.DataFrame({ 'dist' : d, 'angle' : 180.0-ang, 
                      'bend' : 180.0-eu[:,0], 'twist' : eu[:,1], 'rot' : eu[:,2],
                      'shiftx': sx, 'shifty': sy, 'shiftz': sz }) 
    return df


def ph_analysis(dir=".",*,caname="CA.pdb",xtcname="CA.xtc"):
    p=Path(dir)
    capdb=p/caname
    clist=ph_dimeridx(str(capdb),['A','B','C','D','E'],str(capdb),['F','G','J','K','H','I'])

    t=md.load(str(p/xtcname), top=str(capdb))
    d,ang,eu,sx,sy,sz=ph_dimergeom(t,clist[0],t,clist[1])

    d,ang,sx,sy,sz=map(np.ravel, (d,ang,sx,sy,sz))
    eu=np.asarray(eu)

    df=pd.DataFrame({ 'dist' : d, 'angle' : 180.0-ang,
                      'bend' : 180.0-eu[:,0], 'twist' : eu[:,1], 'rot' : eu[:,2],
                      'shiftx': sx, 'shifty': sy, 'shiftz': sz })
    return df


def th_analysis(dir=".",*,caname="CA.pdb",xtcname="CA.xtc"):
    p=Path(dir)
    capdb=p/caname
    clist=th_dimeridx(str(capdb),['A','C','B'],str(capdb),['D','E','H','I','F','G'])

    t=md.load(str(p/xtcname), top=str(capdb))
    d,ang,eu,sx,sy,sz=th_dimergeom(t,clist[0],t,clist[1])

    d,ang,sx,sy,sz=map(np.ravel, (d,ang,sx,sy,sz))
    eu=np.asarray(eu)

    df=pd.DataFrame({ 'dist' : d, 'angle' : 180.0-ang,
                      'bend' : 180.0-eu[:,0], 'twist' : eu[:,1], 'rot' : eu[:,2],
                      'shiftx': sx, 'shifty': sy, 'shiftz': sz })
    return df


def hhh_analysis(dir=".",*,caname="CA.pdb",xtcname="CA.xtc"):
    p=Path(dir)
    capdb=p / caname

    clist12=hh_dimeridx(str(capdb),['A','B','E','F','C','D'],str(capdb),['G','H','K','L','I','J'])
    clist13=hh_dimeridx(str(capdb),['A','B','E','F','C','D'],str(capdb),['M','N','Q','R','O','P'])
    clist23=hh_dimeridx(str(capdb),['G','H','K','L','I','J'],str(capdb),['M','N','Q','R','O','P'])

    t=md.load(str(p/xtcname), top=str(capdb))

    d,ang,eu,sx,sy,sz=hh_dimergeom(t,clist12[0],t,clist12[1])
    d,ang,sx,sy,sz=map(np.ravel, (d,ang,sx,sy,sz))
    eu=np.asarray(eu)
    df12=pd.DataFrame({ 'dist12' : d, 'angle12' : 180.0-ang,
                        'bend12' : 180.0-eu[:,0], 'twist12' : eu[:,1], 'rot12' : eu[:,2],
                        'shiftx12': sx, 'shifty12': sy, 'shiftz12': sz })

    d,ang,eu,sx,sy,sz=hh_dimergeom(t,clist13[0],t,clist13[1])
    d,ang,sx,sy,sz=map(np.ravel, (d,ang,sx,sy,sz))
    eu=np.asarray(eu)
    df13=pd.DataFrame({ 'dist13' : d, 'angle13' : 180.0-ang,
                        'bend13' : 180.0-eu[:,0], 'twist13' : eu[:,1], 'rot13' : eu[:,2],
                        'shiftx13': sx, 'shifty13': sy, 'shiftz13': sz })

    d,ang,eu,sx,sy,sz=hh_dimergeom(t,clist23[0],t,clist23[1])
    d,ang,sx,sy,sz=map(np.ravel, (d,ang,sx,sy,sz))
    eu=np.asarray(eu)
    df23=pd.DataFrame({ 'dist23' : d, 'angle23' : 180.0-ang,
                        'bend23' : 180.0-eu[:,0], 'twist23' : eu[:,1], 'rot23' : eu[:,2],
                        'shiftx23': sx, 'shifty23': sy, 'shiftz23': sz })

    df123=pd.merge(df12,df13,left_index=True,right_index=True,how='inner')
    df=pd.merge(df123,df23,left_index=True,right_index=True,how='inner')
    return df


def phh_analysis(dir=".",*,caname="CA.pdb",xtcname="CA.xtc"):
    p=Path(dir)
    capdb=p / caname

    clist12=ph_dimeridx(str(capdb),['A','B','C','D','E'],str(capdb),['F','G','J','K','H','I'])
    clist13=ph_dimeridx(str(capdb),['A','B','C','D','E'],str(capdb),['L','M','P','Q','N','O'])
    clist23=hh_dimeridx(str(capdb),['F','G','J','K','H','I'],str(capdb),['L','M','P','Q','N','O'])

    t=md.load(str(p/xtcname), top=str(capdb))

    d,ang,eu,sx,sy,sz=ph_dimergeom(t,clist12[0],t,clist12[1])
    d,ang,sx,sy,sz=map(np.ravel, (d,ang,sx,sy,sz))
    eu=np.asarray(eu)
    df12=pd.DataFrame({ 'dist12' : d, 'angle12' : 180.0-ang,
                        'bend12' : 180.0-eu[:,0], 'twist12' : eu[:,1], 'rot12' : eu[:,2],
                        'shiftx12': sx, 'shifty12': sy, 'shiftz12': sz })

    d,ang,eu,sx,sy,sz=ph_dimergeom(t,clist13[0],t,clist13[1])
    d,ang,sx,sy,sz=map(np.ravel, (d,ang,sx,sy,sz))
    eu=np.asarray(eu)
    df13=pd.DataFrame({ 'dist13' : d, 'angle13' : 180.0-ang,
                        'bend13' : 180.0-eu[:,0], 'twist13' : eu[:,1], 'rot13' : eu[:,2],
                        'shiftx13': sx, 'shifty13': sy, 'shiftz13': sz })

    d,ang,eu,sx,sy,sz=hh_dimergeom(t,clist23[0],t,clist23[1])
    d,ang,sx,sy,sz=map(np.ravel, (d,ang,sx,sy,sz))
    eu=np.asarray(eu)
    df23=pd.DataFrame({ 'dist23' : d, 'angle23' : 180.0-ang,
                        'bend23' : 180.0-eu[:,0], 'twist23' : eu[:,1], 'rot23' : eu[:,2],
                        'shiftx23': sx, 'shifty23': sy, 'shiftz23': sz })

    df123=pd.merge(df12,df13,left_index=True,right_index=True,how='inner')
    df=pd.merge(df123,df23,left_index=True,right_index=True,how='inner')
    return df


def thh_analysis(dir=".",*,caname="CA.pdb",xtcname="CA.xtc"):
    p=Path(dir)
    capdb=p / caname

    clist12=th_dimeridx(str(capdb),['A','C','B'],str(capdb),['D','E','H','I','F','G'])
    clist13=th_dimeridx(str(capdb),['A','C','B'],str(capdb),['J','K','N','O','L','M'])
    clist23=hh_dimeridx(str(capdb),['D','E','H','I','F','G'],str(capdb),['J','K','N','O','L','M'])

    t=md.load(str(p/xtcname), top=str(capdb))

    d,ang,eu,sx,sy,sz=th_dimergeom(t,clist12[0],t,clist12[1])
    d,ang,sx,sy,sz=map(np.ravel, (d,ang,sx,sy,sz))
    eu=np.asarray(eu)
    df12=pd.DataFrame({ 'dist12' : d, 'angle12' : 180.0-ang,
                        'bend12' : 180.0-eu[:,0], 'twist12' : eu[:,1], 'rot12' : eu[:,2],
                        'shiftx12': sx, 'shifty12': sy, 'shiftz12': sz })

    d,ang,eu,sx,sy,sz=th_dimergeom(t,clist13[0],t,clist13[1])
    d,ang,sx,sy,sz=map(np.ravel, (d,ang,sx,sy,sz))
    eu=np.asarray(eu)
    df13=pd.DataFrame({ 'dist13' : d, 'angle13' : 180.0-ang,
                        'bend13' : 180.0-eu[:,0], 'twist13' : eu[:,1], 'rot13' : eu[:,2],
                        'shiftx13': sx, 'shifty13': sy, 'shiftz13': sz })

    d,ang,eu,sx,sy,sz=hh_dimergeom(t,clist23[0],t,clist23[1])
    d,ang,sx,sy,sz=map(np.ravel, (d,ang,sx,sy,sz))
    eu=np.asarray(eu)
    df23=pd.DataFrame({ 'dist23' : d, 'angle23' : 180.0-ang,
                        'bend23' : 180.0-eu[:,0], 'twist23' : eu[:,1], 'rot23' : eu[:,2],
                        'shiftx23': sx, 'shifty23': sy, 'shiftz23': sz })

    df123=pd.merge(df12,df13,left_index=True,right_index=True,how='inner')
    df=pd.merge(df123,df23,left_index=True,right_index=True,how='inner')
    return df


def tile_analysis(tag='hh',*,dir=".", path=['set1']):
    data={}
    for p in path:
        if tag == 'hh':
           data[p]=hh_analysis(dir+p,xtcname="CAwrapped.xtc")
        if tag == 'th':
           data[p]=th_analysis(dir+p,xtcname="CAwrapped.xtc")
        if tag == 'ph':
           data[p]=ph_analysis(dir+p,xtcname="CAwrapped.xtc")
        if tag == 'hhh':
           data[p]=hhh_analysis(dir+p,xtcname="CAwrapped.xtc")
        if tag == 'thh':
           data[p]=thh_analysis(dir+p,xtcname="CAwrapped.xtc")
        if tag == 'phh':
           data[p]=phh_analysis(dir+p,xtcname="CAwrapped.xtc")
    return data


# read data from metadynamics sampling

def _parse_fields_from_header(fname):
    with open(fname, 'r', encoding='utf-8') as fh:
        for line in fh:
            s = line.strip()
            if not s.startswith('#'):
                # reached data before finding FIELDS
                return None
            if re.match(r'^#\s*!\s*FIELDS\b', s, flags=re.IGNORECASE):
                toks = s.split()
                try:
                    i = [t.upper() for t in toks].index('FIELDS')
                    return toks[i+1:]
                except ValueError:
                    if toks and toks[0].startswith('#'):
                        toks = toks[1:]
                    if toks and toks[0] == '!':
                        toks = toks[1:]
                    if toks and toks[0].upper() == 'FIELDS':
                        return toks[1:]
                    return None
    return None

def read_plumed_data(dir=".", path=[ 'set1' ], logname='plumed.log', verbose=False):
    frames = []
    for p in path:
        fname = Path(dir) / p / logname
        if not fname.exists():
            continue

        cols = _parse_fields_from_header(fname)
        if cols is None or len(cols) == 0:
            print(f"{fname} is not plumed log file")
            continue

        dtype = {c: float for c in cols}

        df = pd.read_csv( fname, sep=r"\s+", engine="python", names=cols, usecols=range(len(cols)), comment="#", header=None,
            dtype=dtype, na_values=['nan','NaN','INF','inf','-inf'], on_bad_lines='skip')

        df.insert(0,'set',p)
        df = df.set_index(['set', 'time']).sort_index()
        frames.append(df)
        if verbose:
            print(f"read {fname} ({len(df)} rows)")

    if not frames:
        empty = pd.DataFrame()
        return {}

    out = pd.concat(frames, axis=0)

    return {u: g.droplevel(0) for u, g in out.groupby(level=0)}


def process_meta(tag='hh',*,dir=".",path=None,verbose=False):
    if path is None:
       path = ['set1']

    pindiv = read_plumed_data(dir,path,verbose=verbose,logname='plumed.log')
    pcomb1 = read_plumed_data(dir,path,verbose=verbose,logname='comb/plumed1.log')
    pcomb2 = read_plumed_data(dir,path,verbose=verbose,logname='comb/plumed2.log')

    if pcomb2:
        pcomb={p: pd.concat([pcomb1[p],pcomb2[p]], axis=0, ignore_index=True) for p in path}
    else:
        pcomb=pcomb1

    tiledata=tile_analysis(tag,dir=dir,path=path)
    data={p: pd.merge(pindiv[p],tiledata[p],left_index=True,right_index=True,how='inner') for p in path}

    mask={}
    for p in path:
        mask[p]=(data[p]['uwall.bias']+data[p]['lwall.bias']<1)
        data[p]=data[p].loc[mask[p]].copy()
        data[p].reset_index(drop=True,inplace=True)
        
    for p in path:
        wham=unbias_wham(np.array([data[p]['metad.bias']]).T)
        data[p]['ww']=pd.DataFrame(np.exp(wham["logW"])/np.sum(np.exp(wham["logW"])))
    
    data['comb']=pd.concat([data[p] for p in path],ignore_index=True)

    combmask=pd.concat([mask[p] for p in path], ignore_index=True)
    bias_matrix=np.column_stack([np.asarray(pcomb[p]['metad.bias'].loc[combmask],dtype=float) for p in path])    
    counts=[len(data[p]) for p in path]
    mbar=unbias_mbar(bias_matrix,counts=counts)
    wham=unbias_wham(bias_matrix)

    data['comb']['wwmbar']=pd.DataFrame(mbar['ww'])
    data['comb']['wwwham']=pd.DataFrame(wham['ww'])
    data['comb']['ww']=data['comb']['wwmbar']

    data['mbar']=mbar
    data['wham']=wham
    data['bias_matrix']=bias_matrix

    data['sets']=path
        
    return data


# umbrella sampling

def read_umbrella_bias(dir, umbrellas, *, verbose=False):
    cols = ['step','xbias','ybias','zbias','anglebias','torsionbias','rotbias']
    dtype = {'step': int, 'xbias': float, 'ybias': float, 'zbias': float,
             'anglebias': float, 'torsionbias': float, 'rotbias': float}

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
            sep=r'\s+',
            engine='python',
            names=cols,
            usecols=range(len(cols)),
            comment='#',
            skiprows=1,
            dtype=dtype,
            na_values=['nan','NaN','INF','inf','-inf'],
            on_bad_lines='skip',
        )
        df['ubias'] = (
            df['xbias'] + df['ybias'] + df['zbias']
            + df['anglebias'] + df['torsionbias'] + df['rotbias']
        )
        df['obias'] = df['anglebias'] + df['torsionbias'] + df['rotbias']
        df.insert(0, 'umbrella', u)
        frames.append(df)

        if verbose:
            print(f"read {fname} ({len(df)} rows)")

    if not frames:
        if verbose:
            print("No bias data read from any umbrella.")
        return {}

    out = (
        pd.concat(frames, ignore_index=True)
          .set_index(['umbrella', 'step'])
          .sort_index()
    )

    return {u: g.droplevel(0) for u, g in out.groupby(level=0)}


def read_umbrella_geometry(fname, *, verbose=False):
    cols = ['gstep','gxdist','gydist','gzdist','gangle','gtorsion','grot1','grot2']
    dtype = {
        'gstep': int,
        'gxdist': float,
        'gydist': float,
        'gzdist': float,
        'gangle': float,
        'gtorsion': float,
        'grot1': float,
        'grot2': float,
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
        sep=r'\s+',
        engine='python',
        names=cols,
        usecols=range(len(cols)),
        comment='#',
        skiprows=1,
        dtype=dtype,
        na_values=['nan','NaN','INF','inf','-inf'],
        on_bad_lines='skip',
    )
    return df

def process_umbrella(tag='hh',*,dir=".",path=None,verbose=False, biasval='xbias', obiaslimit=None, skip=0):
    if path is None:
       path=[]
       for r in range(10,99):
          fdir=f'run_{r/10:.1f}'
          if Path(dir+"/"+fdir).exists():
             path+=[fdir]
    
    geo=read_umbrella_geometry(dir+"/"+path[0]+"/geometry.dat")
    dimer=hh_analysis(dir)
    df=pd.merge(geo,dimer,left_index=True,right_index=True,how='inner')

    nwin=len(path)
    nper=len(df) // nwin
    data={path[i]: df.iloc[i*nper : (i+1)*nper].reset_index(drop=True) for i in range(nwin)}

    bias=read_umbrella_bias(dir,path,verbose=verbose)
    for i in range(nwin):
        bindiv=bias[path[i]].iloc[i*nper:(i+1)*nper].reset_index(drop=True)
        data[path[i]]=pd.merge(data[path[i]],bindiv,left_index=True,right_index=True,how='inner')

    mask={}
    for p in path:
        if obiaslimit is not None:
           mask[p]=(data[p]['obias']<obiaslimit)
        else:
           mask[p]=pd.Series(True,index=data[p].index)
        if skip>0:
           mask[p].iloc[:skip]=False
        data[p]=data[p].loc[mask[p]].copy()
        data[p].reset_index(drop=True,inplace=True)

    for p in path:
        wham=unbias_wham(np.array([data[p][biasval]]).T)
        data[p]['ww']=pd.DataFrame(np.exp(wham["logW"])/np.sum(np.exp(wham["logW"])))

    combmask=pd.concat([mask[p] for p in path], ignore_index=True) 
    data['comb']=df.loc[combmask].copy()

    bias_matrix=np.column_stack([np.asarray(bias[p][biasval].loc[combmask],dtype=float) for p in path])
    counts=[len(data[p]) for p in path]

    mbar=unbias_mbar(bias_matrix, counts=counts)

    #wham=unbias_wham(bias_matrix)
    #data['comb']['wwmbar']=pd.DataFrame(mbar['ww'])
    #data['comb']['wwwham']=pd.DataFrame(wham['ww'])
    #data['comb']['ww']=data['comb']['wwmbar']
    #data['wham']=wham
    
    data['comb']['ww']=pd.DataFrame(mbar['ww'])
    data['mbar']=mbar
    data['bias_matrix']=bias_matrix
    
    data['bias']=bias
    data['counts']=counts
    data['sets']=path

    return data
    
# unbiasing and projecting onto reaction coordinates

def unbias_wham(bias,*,kT: float = kb*T, 
         frame_weight=None,traj_weight=None, 
         maxiter: int = 1000, threshold: float = 1e-20, 
         verbose: bool = False):

    nframes = bias.shape[0]
    ntraj = bias.shape[1]

    if frame_weight is None:
        frame_weight = np.ones(nframes)
    if traj_weight is None:
        traj_weight = np.ones(ntraj)

    assert len(traj_weight) == ntraj
    assert len(frame_weight) == nframes

    shifted_bias = bias/kT

    shifts0 = np.min(shifted_bias, axis=0)
    shifted_bias -= shifts0[np.newaxis,:]
    shifts1 = np.min(shifted_bias, axis=1)
    shifted_bias -= shifts1[:,np.newaxis]

    expv = np.exp(-shifted_bias)

    Z = np.ones(ntraj)

    Zold = Z

    if verbose:
        sys.stderr.write("WHAM: start\n")
    for nit in range(maxiter):
        weight = 1.0/np.matmul(expv, traj_weight/Z)*frame_weight
        Z = np.matmul(weight, expv)
        Z /= np.sum(Z*traj_weight)
        eps = np.sum(np.log(Z/Zold)**2)
        Zold = Z
        if verbose:
            sys.stderr.write("WHAM: iteration "+str(nit)+" eps "+str(eps)+"\n")
        if eps < threshold:
            break
    nfev=nit
    logW = np.log(weight) + shifts1

    if verbose:
        sys.stderr.write("WHAM: end")

    return {"logW":logW, "logZ":np.log(Z)-shifts0, "nit":nit, "eps":eps, "ww": np.exp(logW)/np.sum(np.exp(logW))}

def unbias_mbar(bias, *, kT=kb*T, counts=None, verbose=False):
    beta = 1.0/(kT) if kT is not None else 1.0
    u_kn = (beta * np.asarray(bias, float)).T          # (K,N)

    if counts is None:
       N, K = bias.shape
       n_per = N // K
       state_of_sample = np.repeat(np.arange(K, dtype=int), n_per)
       counts = np.bincount(state_of_sample, minlength=K)

    mbar = MBAR(u_kn, counts, verbose=verbose, maximum_iterations=500)

    log_den = logsumexp(mbar.f_k[:,None] - mbar.u_kn + np.log(mbar.N_k)[:,None], axis=0)
    logw = -log_den
    ww = np.exp(logw-logsumexp(logw))

    return {"mbar":mbar, "logW":logw, "ww":ww}


def pmf1d_mbar(mbar,data,tag,*,kT=kb*T,nbins=100,verbose=False):
    if 'mbar' in mbar:
       mbar=mbar['mbar']
    
    x = np.asarray(data[tag], float).ravel()
    u_n = np.zeros(x.shape[0], float)
    eps = 1e-12 * (x.max() - x.min() + 1.0)
    edges = [np.linspace(x.min(), x.max() + eps, nbins + 1)]
    centers = 0.5*(edges[0][:-1] + edges[0][1:])

    fes=FES(mbar.u_kn,mbar.N_k,mbar_options=dict(verbose=verbose,maximum_iterations=500))
    _ = fes.generate_fes(u_n, x[:, None], fes_type='histogram',
                 histogram_parameters={'bin_edges': edges})
    out = fes.get_fes(centers,reference_point='from-lowest',uncertainty_method='analytical')
    
    F_kT=out['f_i']*kT
    dF_kT=out.get('df_i')*kT

    idx = pd.Index(np.arange(nbins), name='x')
    pmf1d = pd.DataFrame({f"{tag}": F_kT}, index=idx)
    dpmf = pd.DataFrame({f"{tag}": dF_kT}, index=idx)
    ranges = pd.DataFrame({tag: centers})

    return dict(
        edges=edges,
        centers=centers,
        F_kT=F_kT,
        dF_kT=dF_kT,
        pmf=pmf1d,
        dpmf=dpmf,
        ranges=ranges
    )
    
    return centers, (F_kT-np.nanmin(F_kT))*kT, dF_kT*kT

def pmf2d_from_weights(data,tag, *, wtag='ww', kT=kb*T, nbins=(100,100), rang=None):
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

    x = np.asarray(data[tag[0]], float).ravel(); 
    y = np.asarray(data[tag[1]], float).ravel(); 
    w = np.asarray(data[wtag], float).ravel()

    assert x.shape == y.shape == w.shape
    if rang is None:
        pad = lambda a: 1e-12*(a.max()-a.min()+1.0)
        x_edges = np.linspace(x.min(), x.max()+pad(x), nbins[0]+1)
        y_edges = np.linspace(y.min(), y.max()+pad(y), nbins[1]+1)
    else:
        (xmin,xmax),(ymin,ymax) = rang
        x_edges = np.linspace(xmin, xmax, nbins[0]+1)
        y_edges = np.linspace(ymin, ymax, nbins[1]+1)

    H, xe, ye = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=w)
    P = H / H.sum() if H.sum() > 0 else H
    with np.errstate(divide='ignore', invalid='ignore'):
        F = -np.log(P)
    finite = np.isfinite(F)
    if np.any(finite):
        F -= np.nanmin(F)

    F_kT=F*kT

    xcen=0.5*(xe[:-1]+xe[1:])
    ycen=0.5*(ye[:-1]+ye[1:])

    idx = pd.MultiIndex.from_product([np.arange(nbins[0]), np.arange(nbins[1])], names=['x','y'])
    pmf2d = pd.DataFrame({tag[0]+"."+tag[1]: F_kT.ravel()}, index=idx)

    ranges=pd.DataFrame({tag[0]: xcen, tag[1]: ycen})

    return dict(
        x_edges=xe, y_edges=ye,
        x_centers=xcen, y_centers=ycen,
        F_kT=F*kT, P=P,
        pmf=pmf2d, ranges=ranges
    )

def pmf1d_from_weights(data, tag, *, wtag='ww', kT=kb*T, nbins=100, rang=None):
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

    with np.errstate(divide='ignore', invalid='ignore'):
        F = -np.log(P)
    finite = np.isfinite(F)
    if np.any(finite):
        F -= np.nanmin(F)
    F_kT = F * kT

    x_centers = 0.5 * (xe[:-1] + xe[1:])

    idx = pd.Index(np.arange(nbins), name='x')
    pmf1d = pd.DataFrame({f"{tag}": F_kT}, index=idx)
    ranges = pd.DataFrame({tag: x_centers})

    return dict(
        edges=xe,
        centers=x_centers,
        F_kT=F_kT,
        P=P,
        pmf=pmf1d,
        dpmf=None,
        ranges=ranges
    )

# plotting

def dist1D(data: pd.DataFrame, ranges: pd.DataFrame, *, err: None,
           fmin=0.0,fmax=20.0, size: int = 1, label=label, minmax=minmax, tics=tics, colors=colors1d, lw=2, key=None,
           markers=None, tag='dist', mode='together', horizontal=None, vertical=None, save=None) -> plt.Figure:

    if mode == 'together':
       nplots=1
       rows=1
       cols=1
    else:
       nplots = len(data)
       rows=int((nplots+1)/2)
       cols=2

    if key is not None: 
       xoff=3
    else:
       xoff=1

    fig,ax = plt.subplots(rows,cols,figsize=(cols*5*size+xoff,rows*4*size+1),dpi=75,constrained_layout=True)
    
    xlabel='distance [nm]'
    ylabel='[kJ/mol]'

    xmin=5.0
    xmax=10.0

    if tag is not None:
       if label is not None and tag in label:
          xlabel=label[tag]
       if tics is not None and tag in tics:
          xtics=tics[tag]
       else:
          xtics=None
       if minmax is not None and tag in minmax:
          xmin=minmax[tag][0]
          xmax=minmax[tag][1]
          if xtics is not None:
             xtics=[x for x in xtics if x>=xmin and x<=xmax]

    if nplots>1:
        ax=ax.ravel()

    for i, d in enumerate(data):
        X = ranges[i][tag]
        Y = d[tag]

        if mode == 'together':
           axi=ax
        else:
           axi=ax[i]

        if i<len(colors):
           linecolor=colors[i]
        else:
           linecolor='(0.5, 0.5, 0.5)'

        if key is not None and i<len(key):
           keyname=key[i]
        else:
           keyname=""

        axi.plot(X,Y,color=linecolor,label=keyname,linewidth=lw)
        axi.set_xlabel(xlabel) #, fontsize=20)
        axi.set_ylabel(ylabel) #, fontsize=20)
        axi.set_xlim(xmin,xmax)
        axi.set_ylim(fmin,fmax)

        if err is not None and err[i] is not None:
           axi.fill_between(X,Y-err[i][tag],Y+err[i][tag],alpha=0.3,color=linecolor) 
        
        if xtics is not None:
           axi.set_xticks(xtics)

        if markers is not None:
           for m in markers:
              axi.plot(m[tag],1.0,"x",color=m['col'], markersize=int(12*size), markeredgewidth=4)
              if (len(m)>3):
                 axi.annotate(m['label'],xy=(m[tag],1.0), xytext=(0,16*m['pos']*size+6*size), 
                      color=m['col'], textcoords="offset points", ha="center", va="top",fontsize=int(14*size))

        if vertical is not None:
           axi.axvline(x=vertical, color="#808080", linestyle="--", linewidth=3)
        if horizontal is not None:
           axi.axhline(y=horizontal, color="#808080", linestyle="--", linewidth=3)

    if nplots>1:
       for i in range(nplots,rows*cols):
          ax[i].remove()
    else:
       if key is not None:
          ax.legend(loc='upper left', bbox_to_anchor=(1.02,1),borderaxespad=0.)
        
    if save: fig.savefig(save, dpi=300)

    plt.show()


def dist2D(data: pd.DataFrame, ranges: pd.DataFrame, *,
           nlevels: int = 51, threshold: float = 25.0,
           colorbar: bool = True, cmap=None,
           size: int = 1, label=None, minmax=None, tics=None,
           vertical=None, horizontal=None,
           markers=None, xtag='bend', ytag='dist', save=None) -> plt.Figure:

    nplots = len(data)
    rows=int((nplots+1)/2)
    cols=2
    fig,ax = plt.subplots(rows,cols,figsize=(cols*5*size+1,rows*4*size+1),dpi=75,constrained_layout=True)
    
    if cmap is None:
        cmap = plt.get_cmap('terrain')

    xlabel='Planar angle [deg]'
    ylabel='distance [nm]'

    xmin=90.0
    xmax=180.0
    ymin=5.0
    ymax=8.0

    if xtag is not None:
       if label is not None and xtag in label:
          xlabel=label[xtag]
       if tics is not None and xtag in tics:
          xtics=tics[xtag]
       else:
          xtics=None
       if minmax is not None and xtag in minmax:
          xmin=minmax[xtag][0]
          xmax=minmax[xtag][1]
          if xtics is not None:
             xtics=[x for x in xtics if x>=xmin and x<=xmax]

    if ytag is not None:
       if label is not None and ytag in label:
          ylabel=label[ytag]
       if tics is not None and ytag in tics:
          ytics=tics[ytag]
       else:
          ytics=None
       if minmax is not None and ytag in minmax:
          ymin=minmax[ytag][0]
          ymax=minmax[ytag][1]
          if ytics is not None:
             ytics=[y for y in ytics if y>=ymin and y<=ymax]

    ax=ax.ravel()
    for i, d in enumerate(data):
        k=d.keys()[0]
        kx, ky = k.split('.')

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
        cm = ax[i].contourf(X, Y, Z_masked, cmap=cmap, levels=levels, norm=norm, extend='max')
        ax[i].contour(X, Y, Z_masked, colors='k', levels=levels, linewidths=0.5, linestyles='dotted')

        ax[i].set_xlabel(xlabel) #, fontsize=20)
        ax[i].set_ylabel(ylabel) #, fontsize=20)
        ax[i].set_xlim(xmin,xmax)
        ax[i].set_ylim(ymin,ymax)
        
        if xtics is not None:
           ax[i].set_xticks(xtics)

        if ytics is not None:
           ax[i].set_yticks(ytics)

        if markers is not None:
           for m in markers:
              ax[i].plot(m[xtag],m[ytag],"x",color=m['col'], markersize=int(12*size), markeredgewidth=4)
              if (len(m)>3):
                 ax[i].annotate(m['label'],xy=(m[xtag],m[ytag]), xytext=(0,16*m['pos']*size+6*size), 
                       color=m['col'], textcoords="offset points", ha="center", va="top",fontsize=int(14*size))
        if vertical is not None:
           ax[i].axvline(x=vertical, color="#808080", linestyle="--", linewidth=3)
        if horizontal is not None:
           ax[i].axhline(y=horizontal, color="#808080", linestyle="--", linewidth=3)

    for i in range(nplots,rows*cols):
        ax[i].remove()
        
    if colorbar:
        cbar = fig.colorbar(cm, ax=fig.axes, shrink=0.95)
        cbar.ax.set_ylabel('[kJ/mol]', rotation=90)

    if save: fig.savefig(save, dpi=300)
     
    plt.show()

    
def plot2D_combined(df,xtag='bend',ytag='dist',*, 
                  minmax=minmax,tics=tics, label=label, kbT=kb*T, size=1.5, markers=None, vertical=None, horizontal=None,
                  nbins=(100,100), save=None):
    if isinstance(xtag,list) and isinstance(ytag,list):
       sxtag=xtag[0].rstrip('0123456789')
       sytag=ytag[0].rstrip('0123456789')
       dplotlist=[]
       for i in range(len(xtag)):
           dp=df['comb'][[xtag[i],ytag[i],'ww']].fillna(0)
           dp.columns=[sxtag,sytag,'ww']
           dplotlist+=[dp]
       dplot=pd.concat(dplotlist)
       res=pmf2d_from_weights(dplot,[sxtag,sytag],nbins=nbins) 
    else:
       sxtag=xtag.rstrip('0123456789')
       sytag=ytag.rstrip('0123456789')
       dplot=df['comb'][[xtag,ytag,'ww']].fillna(0)
       res=pmf2d_from_weights(dplot,[xtag,ytag],nbins=nbins) 

    dist2D([res['pmf']], [res['ranges']], colorbar=True ,size=size, markers=markers, xtag=sxtag, ytag=sytag,
           minmax=minmax, tics=tics, label=label, vertical=vertical, horizontal=horizontal, save=save)


def plot2D_individual(df,xtag='bend',ytag='dist',*, 
               setlist=None, minmax=minmax,tics=tics, label=label, kbT=kb*T, size=1.0, 
               markers=None, vertical=None, nbins=(100,100), horizontal=None,save=None):

    if isinstance(xtag,list) and isinstance(ytag,list):
       sxtag=xtag[0].rstrip('0123456789')
       sytag=ytag[0].rstrip('0123456789')
    else:
       sxtag=xtag.rstrip('0123456789')
       sytag=ytag.rstrip('0123456789')

    if setlist is None:
       setlist=df['sets']

    pmf=[]
    rang=[]
    for p in setlist:
       if isinstance(xtag,list) and isinstance(ytag,list):
          dplotlist=[]
          for k in range(len(xtag)):
              dp=df[p][[xtag[k],ytag[k],'ww']].fillna(0)
              dp.columns=[sxtag,sytag,'ww']
              dplotlist+=[dp]
          dplot=pd.concat(dplotlist)
       else:
          dplot=df[p][[xtag,ytag,'ww']].fillna(0)

       res=pmf2d_from_weights(dplot,[xtag,ytag],nbins=nbins) 

       pmf+=[res['pmf']]
       rang+=[res['ranges']]

    dist2D(pmf,rang, colorbar=False ,size=size, markers=markers, xtag=sxtag, ytag=sytag,
           minmax=minmax, tics=tics, label=label, vertical=vertical, horizontal=horizontal, save=save)

def plot1D_combined(df,tag='dist',*, usembar=False, minmax=minmax,tics=tics, label=label, colors=colors1d, key=None,
                    kbT=kb*T, nbins=50, fmin=0.0, fmax=20.0,
                    size=1.5, markers=None, offset=None, matchflat=None, matchzero=False,
                    vertical=None, horizontal=None, save=None):
    pmf=[]
    ranges=[]
    err=[]

    if isinstance(df,list):
       dflist=df
    else:
       dflist=[df]

    for i,d in enumerate(dflist):
       if isinstance(tag,list):
          stag=tag[0].rstrip('0123456789')
          dplotlist=[]
          for i in range(len(tag)):
              dp=d['comb'][[tag[i],'ww']].fillna(0)
              dp.columns=[stag,'ww']
              dplotlist+=[dp]
          dplot=pd.concat(dplotlist)
          res=pmf1d_from_weights(dplot,stag) 
       else:
          stag=tag.rstrip('0123456789')
          if usembar:
              res=pmf1d_mbar(d['mbar'], d['comb'], tag,nbins=nbins)
          else:
              dplot=d['comb'][[tag,'ww']].fillna(0)
              res=pmf1d_from_weights(dplot,tag,nbins=nbins) 

       pmf+=[res['pmf']]
       ranges+=[res['ranges']]
       err+=[res['dpmf']]

    n=len(pmf)

    base=[offset[i] if (offset is not None and i<len(offset)) else 0.0 for i in range(n)]

    extra=[0.0]*n
    if matchflat is not None and len(matchflat)==2:
        mmin,mmax=matchflat

        means=[]
        for p, r in zip(pmf,ranges):
            mask=r[tag].between(mmin,mmax,inclusive='both')
            m=p[tag][mask].mean()
            means.append(float(m) if pd.notna(m) else 0.0)

        if matchzero:
           extra = [-m for m in means]
        else:
           mmax_val=max(means) if means else 0.0
           extra=[mmax_val-m for m in means]

    total = [b+e for b,e in zip(base,extra)]

    for p, o in zip(pmf,total):
        p[tag]=p[tag]+o

    dist1D(pmf, ranges, err=err, size=size, markers=markers, tag=stag, minmax=minmax, tics=tics, 
           label=label, fmin=fmin, fmax=fmax, colors=colors, key=key, vertical=vertical, horizontal=horizontal, save=save )


def plot_series(s, *, title=None, xlabel=None, ylabel=None, logx=False, logy=False, save=None, size=1):
    fig, ax = plt.subplots(figsize=(4*size,3*size))
    ax.plot(s.index, s.values)
    ax.set_title(title or "")
    ax.set_xlabel(xlabel or s.index.name or "x")
    ax.set_ylabel(ylabel or s.name or "value")
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save: fig.savefig(save, dpi=300)
    plt.show()
