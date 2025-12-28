#!/usr/bin/env python

# usage:
#
#  prep.tileumbrella.allatom.py [args]
#      args:   setup             : directory
#              dimer.pdb         : input PDB
#              D:E:F:G:H:I.2-91  : reference selection
#              A:B:C.19-205      : other selection
#              22 11 9           : box in [nm]
#              100               : NaCl concentration [mM]
#

import sys
from pathlib import Path
from mdsim import PDBReader, MDSim, StructureSelector, solvate, ion_counts, plane_normal
import numpy as np
from openmm.unit import nanometer

def _argv(i: int, default: str) -> str:
    """Return argv[i] if present and non-empty; otherwise default."""
    return sys.argv[i] if len(sys.argv) > i and str(sys.argv[i]).strip() != "" else default

def _as_float(name: str, s: str) -> float:
    try:
        return float(s)
    except Exception as e:
        raise SystemExit(f"ERROR: {name} must be a float, got {s!r}") from e


def resolve_pdb(pdb_arg: str, tdir: Path) -> Path:
    """
    Resolve PDB path:
      1) as given
      2) relative to tdir (directory prefix)
    """
    p = Path(pdb_arg)
    if p.is_file():
        return p

    p2 = tdir / pdb_arg
    if p2.is_file():
        return p2

    raise SystemExit(
        "ERROR: input PDB file not found.\n"
        f"  Tried: {str(p)}\n"
        f"  Tried: {str(p2)}"
    )


def main() -> None:
    default_tdir = "setup"
    default_pdb = "dimer.pdb"
    default_reftile = "A:B:C:D:E:F.2-91"
    default_othertile = "G:H:I:J:K:L.2-91"
    default_boxx = "22.0"
    default_boxy = "11.0"
    default_boxz = "9.0"
    default_conc = "100"

    tdir = Path(_argv(1, default_tdir)).expanduser().resolve()
    pdb_arg = _argv(2, default_pdb)

    refsel = _argv(3, default_reftile)
    base, dot, suffix = refsel.partition(".")
    tiles = base.split(":")
    reftile = [f"{t}.{suffix}" for t in tiles] if dot else tiles

    othersel = _argv(4, default_othertile)

    boxx = _as_float("boxx", _argv(5, default_boxx)) * nanometer
    boxy = _as_float("boxy", _argv(6, default_boxy)) * nanometer
    boxz = _as_float("boxz", _argv(7, default_boxz)) * nanometer
    conc = _as_float("conc", _argv(8, default_conc))

    orient_reftile=True

    home = Path.home()
    ffdir = home / "ff" / "openmm"
    ff = [str(ffdir / "c36m.xml"), str(ffdir / "waters_ions_default.xml")]

    # fail early if forcefield files are missing
    missing_ff = [f for f in ff if not Path(f).is_file()]
    if missing_ff:
        raise SystemExit(
            "ERROR: forcefield file(s) not found:\n  " + "\n  ".join(missing_ff)
        )

    # Ensure output directory exists
    tdir.mkdir(parents=True, exist_ok=True)

    # Resolve PDB (try as-given, then with directory prefix)
    pdb_path = resolve_pdb(pdb_arg, tdir)

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

    translate = [ boxx/4.0-ch[0], boxy/2.0-ch[1], boxz/2.0-ch[2] ]
    s.translate(translate)
    
    t_nm = [float(x.value_in_unit(nanometer)) for x in translate]
    print(
        f"translated center by "
        f"({t_nm[0]:.3f}, {t_nm[1]:.3f}, {t_nm[2]:.3f}) nm"
    )

    s.write_pdb(str(tdir / "dimer.protein.pdb"))

    nion, nsod, ncla = ion_counts(boxx, boxy, boxz, conc, s[0].nominal_charge())
    solvated, wbox= solvate(s[0],box_min=(0.0, 0.0, 0.0),box_max=(boxx,boxy,boxz),ions={"SOD": nsod, "CLA": ncla})

    print(f"solvated system with {nsod} Na and {ncla} Cl ions")
    print(f"box size: {wbox} nm^3")

    solvated.write_pdb(str(tdir / "dimer.solvated.pdb"))

    sim=MDSim(model=solvated,ff=ff,box=(boxx, boxy, boxz),hmass=True,switching="openmm")
    sim.setup_simulation()
    print(f"openmm energy: {sim.get_potentialEnergy()}")

    sim.write_system(str(tdir / "system.xml"))
    sim.write_state(str(tdir / "initial.xml"))


if __name__ == "__main__":
    main()
