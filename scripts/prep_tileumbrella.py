#!/usr/bin/env python

# usage:
#
#  prep.tileumbrella.py [args]
#      args:   allatom           : mode 'allatom' or 'cocomo'
#              setup             : directory
#              dimer.pdb         : input PDB
#              D:E:F:G:H:I.2-91  : reference selection
#              A:B:C.19-205      : other selection
#              22:11:9           : box in [nm]
#              100               : NaCl concentration [mM]
#

import sys
from pathlib import Path
from mdsim import PDBReader, MDSim, StructureSelector, solvate, ion_counts, plane_normal
from cocomo import Assembly, COCOMO

import numpy as np
from openmm.unit import nanometer, kilojoule, mole, kelvin

def _argv(i: int, default: str) -> str:
    """Return argv[i] if present and non-empty; otherwise default."""
    return sys.argv[i] if len(sys.argv) > i and str(sys.argv[i]).strip() != "" else default

def _as_float(name: str, s: str) -> float:
    try:
        return float(s)
    except Exception as e:
        raise SystemExit(f"ERROR: {name} must be a float, got {s!r}") from e

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
    default_reftile = "A:B:C:D:E:F.2-91"
    default_othertile = "G:H:I:J:K:L.2-91"
    default_conc = "100"

    mode = _argv(1, 'allatom')
    tdir = Path(_argv(2, "setup")).expanduser().resolve()

    if mode.lower() == 'cocomo':
        pdb_arg = _argv(3, "dimer.ca.pdb")
    elif mode.lower() == 'allatom':
        pdb_arg = _argv(3, "dimer.pdb")
    else:
        raise SystemExit(f"invalid mode {mode}")

    refsel = _argv(4, default_reftile)
    base, dot, suffix = refsel.partition(".")
    tiles = base.split(":")
    reftile = [f"{t}.{suffix}" for t in tiles] if dot else tiles

    othersel = _argv(5, default_othertile)

    if mode.lower() == 'allatom':
        boxstr = _argv(6, "22:11:9")
    elif mode.lower() == 'cocomo':
        boxstr = _argv(6, "100")
    else:
        raise SystemExit(f"invalide mode {mode}")

    bsplit=boxstr.split(":")
    if len(bsplit) == 1:
       boxx = _as_float("box", boxstr) * nanometer
       boxy = boxx
       boxz = boxx
    elif len(bsplit) == 2:
       boxx = _as_float("boxx", bsplit[0]) * nanometer
       boxy = _as_float("boxy", bsplit[1]) * nanometer
       boxz = boxy
    else:
       boxx = _as_float("boxx", bsplit[0]) * nanometer
       boxy = _as_float("boxy", bsplit[1]) * nanometer
       boxz = _as_float("boxz", bsplit[2]) * nanometer
 
    conc = _as_float("conc", _argv(7, default_conc))

    orient_reftile=True

    if mode.lower() == 'allatom':
        home = Path.home()
        ffdir = home / "ff" / "openmm"
        ff = [str(ffdir / "c36m.xml"), str(ffdir / "waters_ions_default.xml")]

        missing_ff = [f for f in ff if not Path(f).is_file()]
        if missing_ff:
            raise SystemExit(
                "ERROR: forcefield file(s) not found:\n  " + "\n  ".join(missing_ff)
            )

    # Ensure output directory exists
    tdir.mkdir(parents=True, exist_ok=True)

    # Resolve PDB (try as-given, then with directory prefix)

    pdb_path = _find(tdir, pdb_arg)

    s=PDBReader(str(pdb_path))
    ch=s.center(StructureSelector(refsel).atom_indices(s))[0]

    if orient_reftile:
        pts = [s.center(StructureSelector(h).atom_indices(s))[0] for h in reftile]

        n = plane_normal(pts)  
        phiy = float(np.arctan2(-n[0], n[2]))
        s.rotate_about_y(phiy, anchor=ch)

        pts = [s.center(StructureSelector(h).atom_indices(s))[0] for h in reftile]
        n = plane_normal(pts)
        phix = float(np.arctan2(n[1], np.hypot(n[0], n[2])))
        s.rotate_about_x(phix, anchor=ch)

        co = s.center(StructureSelector(othersel).atom_indices(s))[0]
        v = (co - ch).value_in_unit(nanometer)
        phiz = float(np.arctan2(v[1], v[0]))
        s.rotate_about_z(-phiz, anchor=ch)

        rx=np.degrees(phix)
        ry=np.degrees(phiy)
        rz=np.degrees(phiz)
        print(f"rotated Y: {ry:.3f} X: {rx:.3f} Z: {rz:.3f} degrees")

    translate = [ boxx/4.0-ch[0], boxy/2.0-ch[1], boxz/2.0-ch[2] ]
    s.translate(translate)
    
    t_nm = [float(x.value_in_unit(nanometer)) for x in translate]
    print(
        f"translated center by "
        f"({t_nm[0]:.3f}, {t_nm[1]:.3f}, {t_nm[2]:.3f}) nm"
    )

    s.write_pdb(str(tdir / "dimer.protein.pdb"))

    if mode.lower() == 'allatom':
        nion, nsod, ncla = ion_counts(boxx, boxy, boxz, conc, s[0].nominal_charge())
        solvated, wbox= solvate(s[0],box_min=(0.0, 0.0, 0.0),box_max=(boxx,boxy,boxz),
                                ions={"SOD": nsod, "CLA": ncla})

        print(f"solvated system with {nsod} Na and {ncla} Cl ions")
        print(f"box size: {wbox} nm^3")

        solvated.write_pdb(str(tdir / "dimer.solvated.pdb"))

        sim=MDSim(model=solvated,ff=ff,box=(boxx, boxy, boxz),hmass=True,switching="openmm")
    elif mode.lower() == 'cocomo':
        components=_find(tdir,"dimer.components")
        component_types=_find(tdir,"component_types_files")
        interactions=_find(tdir,"interactions")

        asm=Assembly(components,component_types,structure=s,interactions=interactions)
        sim=COCOMO(asm,box=(boxx, boxy, boxz),version=2)
    else:
        raise SystemExit(f"invalid mode {mode}")
        
    sim.setup_simulation()
    print(f"openmm energy: {sim.get_potentialEnergy()}")

    sim.write_system(str(tdir / "system.xml"))
    sim.write_state(str(tdir / "initial.xml"))

if __name__ == "__main__":
    main()
