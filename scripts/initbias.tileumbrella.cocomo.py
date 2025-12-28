#!/usr/bin/env python

# usage:
#
#  initbias.tileumbrella.cocomo.py [args]
#      args:   .                         : working directory
#              dimer.ca.pdb              : initial structure
#
#              A:B:C:D:E:F.2-91          : reference selection
#              G:H:I:J:K:L.2-91          : other selection
#
#              A:G                       : anchor points
#              90:90                     : rotation angle reference
#
#              6.0[:9.0[:0.1]]           : biasmin[:biasmax[:biasdelta]]
#              500:200[:200[:200[:200]]] : kinit:kbias[:kdist[:kcent[:kangle]]]
#
#              0                         : device
#
# hexamer-hexamer:
#    A:B:C:D:E:F.2-91 G:H:I:J:K:L.2-91 A:G 90:90 6.0:9.0:0.1
# hexmer-pentamer:
#    F:G:H:I:J:K.2-91 A:B:C:D:E.1-95 H:C 120:90 6.0:9.0:0.1
# hexamer-trimer:
#    D:E:F:G:H:I.2-91 A:B:C.19-205 D:C 60:85 6.2:9.0:0.1
#

import sys
from pathlib import Path
import numpy as np

from cocomo import Assembly, COCOMO
from mdsim import PDBReader, StructureSelector

def _argv(i: int, default: str) -> str:
    """Return argv[i] if present and non-empty; otherwise default."""
    return sys.argv[i] if len(sys.argv) > i and str(sys.argv[i]).strip() != "" else default

def _as_float(name: str, s: str) -> float:
    try:
        return float(s)
    except Exception as e:
        raise SystemExit(f"ERROR: {name} must be a float, got {s!r}") from e

def _as_int(name: str, s: str) -> int:
    try:
        return int(s)
    except Exception as e:
        raise SystemExit(f"ERROR: {name} must be an int, got {s!r}") from e

def _parse_floats(s, defaults, n_out=5):
    vals = [float(x) for x in s.split(":") if x]
    out = []
    last = None

    for i in range(n_out):
        if i < len(vals):
            last = vals[i]
            out.append(last)
        elif i < len(defaults):
            last = defaults[i]
            out.append(last)
        else:
            out.append(last if last is not None else 0.0)

    return out

def _find(tdir: Path, filename: str) -> Path:
    tdir = Path(tdir).expanduser().resolve()

    for d in [tdir, *tdir.parents]:
        candidate = d / filename
        if candidate.is_file():
            return candidate.resolve()

    cwd_candidate = Path.cwd() / filename
    if cwd_candidate.is_file():
        return cwd_candidate.resolve()

    raise FileNotFoundError(
        f"Could not find '{filename}' in {tdir} or its parent directories or CWD"
    )

def main() -> None:
    default_tdir = "."
    default_pdb = "dimer.ca.pdb"

    default_reftile = "A:B:C:D:E:F.2-91"
    default_othertile = "G:H:I:J:K:L.2-91"

    default_anchor = "A:G"

    default_rotstr = "90:90"
    default_refrot = 90

    default_kstr = "500:200"
    default_kinit = 500
    default_kbias = 200

    default_bstr = "6.0:9.0:0.1"
    default_bmin = 6.0
    default_bmax = 9.0
    default_bdel = 0.1

    default_device = "0"

    tdir = Path(_argv(1, default_tdir)).expanduser().resolve()
    pdb_arg = _argv(2, default_pdb)

    tdir.mkdir(parents=True, exist_ok=True)
    pdb_path = _find(tdir,pdb_arg)

    refsel = _argv(3, default_reftile)
    othersel = _argv(4, default_othertile)
    anchor = _argv(5, default_anchor).split(":")

    refrot1, refrot2 = _parse_floats(
       _argv(6, default_rotstr),
       [default_refrot, default_refrot],
       n_out=2,
    )

    bmin, bmax, bdel = _parse_floats(
       _argv(7, default_bstr), 
       [default_bmin, default_bmax, default_bdel],
       n_out=3,
    )

    kinit, kbias, kdist, kcent, kangle = _parse_floats(
       _argv(8, default_kstr), 
       [default_kinit, default_kbias],
       n_out=5,
    )
    
    device = _as_int("device", _argv(9, default_device))

    base, dot, suffix = refsel.partition(".")
    tiles = base.split(":")
    suf = f".{suffix}" if dot else ""
    reftile = [f"{t}{suf}" for t in tiles]

    tlen = len(tiles)
    i = tiles.index(anchor[0])
    asel1 = f"{tiles[i]}:{tiles[(i+1) % tlen]}{suf}"
    asel2 = f"{tiles[(i+2) % tlen]}:{tiles[(i+3) % tlen]}{suf}"
    aselt = f"{tiles[i]}:{tiles[(i+2) % tlen]}:{tiles[(i+3) % tlen]}{suf}"

    base, dot, suffix = othersel.partition(".")
    tiles = base.split(":")
    suf = f".{suffix}" if dot else ""

    i = tiles.index(anchor[1])
    tlen=len(tiles)
    if tlen == 6:
        bsel1 = f"{tiles[i]}:{tiles[(i+1)%tlen]}{suf}"
        bsel2 = f"{tiles[(i+2)%tlen]}:{tiles[(i+3)%tlen]}{suf}"
        bselt = f"{tiles[i]}:{tiles[(i+2)%tlen]}:{tiles[(i+3)%tlen]}{suf}"
    elif tlen == 5:
        bsel1 = f"{tiles[i]}:{tiles[(i+1)%tlen]}{suf}"
        bsel2 = f"{tiles[(i+3)%tlen]}:{tiles[(i+4)%tlen]}{suf}"
        bselt = f"{tiles[i]}:{tiles[(i+1)%tlen]}:{tiles[(i+4)%tlen]}{suf}"
    elif tlen == 3:
        bsel1 = f"{tiles[i]}{suf}"
        bsel2 = f"{tiles[(i+1)%tlen]}{suf}"
        bselt = f"{tiles[i]}:{tiles[(i+1)%tlen]}{suf}"
    else:
        raise ValueError(f"invalid length of other selection") 

    s=PDBReader(str(_pdb_path))

    aca=StructureSelector(refsel+".CA").atom_indices(s)
    bca=StructureSelector(othersel+".CA").atom_indices(s)

    rc=[StructureSelector(t+".CA").atom_indices(s) for t in reftile]

    aca1=StructureSelector(asel1+".CA").atom_indices(s)
    aca2=StructureSelector(asel2+".CA").atom_indices(s)
    acat1=StructureSelector(aselt+".CA").atom_indices(s)

    bca1=StructureSelector(bsel1+".CA").atom_indices(s)
    bca2=StructureSelector(bsel2+".CA").atom_indices(s)
    bcat1=StructureSelector(bselt+".CA").atom_indices(s)

    restart=edir / "equi_298npt.xml"
    for biasval in np.arange(bmin, bmax+1.0E-8, bdel):
       tag=f"{biasval:.2f}"

       bdir=Path(f"run_{tag}")
       bdir.mkdir(parents=True, exist_ok=True)

       sim=MDSim(xml=str(sdir / "system.xml"),restart=str(restart))

       sim.set_umbrella_xyz_distance(aca,bca,direction="x",target=biasval,k=kinit)
       sim.set_umbrella_xyz_distance(aca,bca,direction="y",target=0.0,k=kinit)
       sim.set_umbrella_xyz_distance(aca,bca,direction="z",target=0.0,k=kinit)
       sim.set_umbrella_center(rc,k=kinit)
       sim.set_umbrella_angle_norm(aca,aca1,aca2,bca,bca1,bca2,k=kinit)
       sim.set_umbrella_dihedral(acat1,aca,bca,bcat1,k=kinit)
       sim.set_umbrella_angle(aca,bca,bcat1,target=np.radians(refrot1),k=kinit)
       sim.set_umbrella_angle(acat1,aca,bca,target=np.radians(refrot2),k=kinit)
       sim.set_force_groups()

       sim.write_system(str(bdir / f"bias_system_{tag}.xml"))

       sim.setup_simulation(resources='CUDA', device=device, temperature=298, tstep=0.003, gamma=0.1)
       sim.simulate(nstep=10000,nout=1000,
                    logfile = str(bdir / f"biasinit_{tag}.log"),
                    dcdfile = str(bdir / f"biasinit_{tag}.dcd"))
       biasinitxml = bdir / f"biasinit_{tag}.xml"
       sim.write_state(str(biasinitxml))

       sim.update_umbrella_xyz_distance("x",kbias)
       sim.update_umbrella_xyz_distance("y",kdist)
       sim.update_umbrella_xyz_distance("z",kdist)
       sim.update_umbrella_center(kcent)
       sim.update_umbrella_angle_norm(kangle)
       sim.update_umbrella_dihedral(kangle)
       sim.update_umbrella_angle(kangle)

       sim.simulate(nstep=5000,logfile=str(bdir / f"biasprod_0.log"))
       biasprodxml=bdir / f"biasprod_0.xml"
       sim.write_state(str(biasprodxml))

       restart=biasinitxml
       
       print(f"finished {tag}")

if __name__ == "__main__":
    main()

