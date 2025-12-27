#!/usr/bin/env python3

# usage:
#
#  prep.bmctileumbrella.cocomo.py [args]
#      args:   .                 : directory
#              dimer.ca.pdb      : input PDB
#              D:E:F:G:H:I.2-91  : reference selection
#              A:B:C.19-205      : other selection

import os
import sys
from pathlib import Path

from cocomo import Assembly,COCOMO
from mdsim import PDBReader, StructureSelector, plane_normal

from openmm.unit import kilojoule, mole, kelvin, nanometer
import numpy as np

def _argv(i: int, default: str) -> str:
    """Return argv[i] if present and non-empty; otherwise default."""
    return sys.argv[i] if len(sys.argv) > i and str(sys.argv[i]).strip() != "" else default

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
    default_tdir = "." # example: H_P/horizonal/set1
    default_pdb = "dimer.ca.pdb"
    default_reftile = "A:B:C:D:E:F.2-91"
    default_othertile = "G:H:I:J:K:L.2-91"

    tdir = Path(_argv(1, default_tdir)).expanduser().resolve()
    tdir.mkdir(parents=True, exist_ok=True)

    pdb_arg = _argv(2, default_pdb)
    pdb_path=_find(tdir,pdb_arg)

    refsel = _argv(3, default_reftile)
    base, dot, suffix = refsel.partition(".")
    tiles = base.split(":")
    reftile = [f"{t}.{suffix}" for t in tiles] if dot else tiles

    othersel = _argv(4, default_othertile)

    orient_reftile=True

    s=PDBReader(str(pdb_path))
    ch=s.center(StructureSelector(refsel).atom_indices(s))[0]

    if orient_reftile:
        pts = [s.center(StructureSelector(h).atom_indices(s))[0] for h in reftile]

        n = plane_normal(pts)  
        phiy = float(np.arctan2(-n[0], n[2]))
        s.rotate_about_y(phiy, anchor=ch)
        print(f"rotated around Y by {np.degrees(phiy):.3f} degrees")

        pts = [s.center(StructureSelector(h).atom_indices(s))[0] for h in reftile]
        n = plane_normal(pts)
        phix = float(np.arctan2(n[1], np.hypot(n[0], n[2])))
        s.rotate_about_x(phix, anchor=ch)
        print(f"rotated around X by {np.degrees(phix):.3f} degrees")

        co = s.center(StructureSelector(othersel).atom_indices(s))[0]
        v = (co - ch).value_in_unit(nanometer)
        phiz = float(np.arctan2(v[1], v[0]))
        s.rotate_about_z(-phiz, anchor=ch)
        print(f"rotated around Z by {np.degrees(phiz):.3f} degrees")

    xref=20.0*nanometer
    yref=10.0*nanometer
    zref=10.0*nanometer

    translate = [ xref/4.0-ch[0], yref/2.0-ch[1], zref/2.0-ch[2] ]
    s.translate(translate)
    
    t_nm = [float(x.value_in_unit(nanometer)) for x in translate]
    print(
        f"translated center by "
        f"({t_nm[0]:.3f}, {t_nm[1]:.3f}, {t_nm[2]:.3f}) nm"
    )

    s.write_pdb(str(tdir / "dimer.ca.positioned.pdb"))

    components=_find(tdir,"dimer.components")
    component_types="component_types_files"
    interactions=_find(tdir,"interactions")

    asm=Assembly(components,component_types,structure=s,interactions=interactions)
    sim=COCOMO(asm,version=2)
    sim.setup_simulation(resources='CPU')
    print(f"initial energy: {sim.get_potentialEnergy()}")
    sim.write_system(str(tdir / "system.xml"))

    sim.set_position_restraint(selection="name CA",k=10.0)
    sim.setup_simulation(resources='CPU',tstep=0.03)

    sim.minimize(nstep=1000)
    print(f"minimized energy: {sim.get_potentialEnergy()}")

    sim.set_velocities()
    sim.simulate(nstep=1000)
    print(f"energy after simulation: {sim.get_potentialEnergy()}")

    sim.write_state(str(tdir / "initial.xml"))

if __name__ == "__main__":
    main()

