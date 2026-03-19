#!/usr/bin/env python3
#
# hexamer-hexamer:
#    A:B:C:D:E:F.2-91 G:H:I:J:K:L.2-91
# hexmer-pentamer:
#    F:G:H:I:J:K.2-91 A:B:C:D:E.1-95
# hexamer-trimer:
#    D:E:F:G:H:I.2-91 A:B:C.19-205
#

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from cocomo import COCOMO, Assembly
from mdsim import (
    MDSim,
    PDBReader,
    StructureSelector,
    ion_counts,
    plane_normal,
    solvate,
)
from openmm.unit import nanometer

from .tile_config import read_config
from .tileumbrella_shared import (
    find_input_file,
    parse_args,
    parse_config_path,
    split_tile_selection,
    write_args_config,
)


def _as_float(name: str, s: str) -> float:
    try:
        return float(s)
    except Exception as exc:
        raise SystemExit(f"ERROR: {name} must be a float, got {s!r}") from exc


@dataclass(frozen=True)
class BoxNM:
    x: float
    y: float
    z: float

    def as_units(self) -> tuple:
        return (
            self.x * nanometer,
            self.y * nanometer,
            self.z * nanometer,
        )


_HELP_OPTIONS = (
    "mode",
    "setup",
    "pdb",
    "refsel",
    "othersel",
    "box",
    "biasdir",
    "biasangle",
    "conc",
    "surf",
    "orient",
    "ff",
    "config",
    "write_config",
)


def _parse_box_nm(s: str) -> BoxNM:
    parts = [p.strip() for p in s.split(":") if p.strip() != ""]
    if len(parts) == 1:
        x = _as_float("box", parts[0])
        return BoxNM(x, x, x)
    if len(parts) == 2:
        x = _as_float("boxx", parts[0])
        y = _as_float("boxy", parts[1])
        return BoxNM(x, y, y)
    if len(parts) == 3:
        x = _as_float("boxx", parts[0])
        y = _as_float("boxy", parts[1])
        z = _as_float("boxz", parts[2])
        return BoxNM(x, y, z)
    raise SystemExit("ERROR: --box must be 'x', 'x:y', or 'x:y:z' in nm " "(e.g. 22:11:9)")


def _expand_forcefields(paths: Sequence[str]) -> list[str]:
    return [str(Path(p).expanduser().resolve()) for p in paths]


def _default_forcefields() -> list[str]:
    ffdir = Path.home() / "ff" / "openmm"
    return _expand_forcefields(
        [
            str(ffdir / "c36m.xml"),
            str(ffdir / "waters_ions_default.xml"),
        ]
    )


def _validate_forcefields(ff: Sequence[str]) -> None:
    missing = [p for p in ff if not Path(p).is_file()]
    if missing:
        msg = "ERROR: forcefield file(s) not found:\n  " + "\n  ".join(missing)
        raise SystemExit(msg)


def main() -> None:
    cfg_path = parse_config_path()
    cfg = read_config(cfg_path)

    args = parse_args(cfg, _HELP_OPTIONS, prog="prep_tileumbrella.py")

    mode = str(args.mode).lower()
    tdir = Path(args.setup).expanduser().resolve()
    tdir.mkdir(parents=True, exist_ok=True)

    if args.pdb is None:
        pdb_arg = "dimer.pdb" if mode == "allatom" else "dimer.ca.pdb"
    else:
        pdb_arg = str(args.pdb)

    biasdir = str(args.biasdir)
    if args.box is None:
        if mode == "allatom":
            if biasdir == "x":
                if args.biasangle is None:
                    box_str = "22:11:9"
                else:
                    box_str = "22:11:20"
            elif biasdir == "y":
                box_str = "11:22:9"
            elif biasdir == "z":
                box_str = "10:10:18"
            else:
                raise SystemExit(f"invalid biasdir {biasdir!r}; must be 'x', 'y', or 'z'")
        else:
            box_str = "100"
    else:
        box_str = str(args.box)

    box_nm = _parse_box_nm(box_str)
    boxx, boxy, boxz = box_nm.as_units()

    ff_val: Optional[list[str]] = None
    if mode == "allatom":
        if args.ff is None:
            ff_val = _default_forcefields()
        else:
            ff_val = _expand_forcefields(list(args.ff))
        _validate_forcefields(ff_val)

    cfg_path = Path(args.config)
    if bool(args.write_config):
        overrides: dict[str, object] = {
            "mode": mode,
            "pdb_in": pdb_arg,
            "box": box_str,
        }
        if ff_val is not None:
            overrides["ff"] = ff_val
        write_args_config(cfg_path, cfg, args, overrides=overrides)

    pdb_path = find_input_file(tdir, pdb_arg)
    s = PDBReader(str(pdb_path))

    refsel = str(args.refsel)
    othersel = str(args.othersel)
    reftile = split_tile_selection(refsel)

    ch = s.center(StructureSelector(refsel).atom_indices(s))[0]

    if bool(args.orient):
        biasdir = str(args.biasdir).lower()
        if biasdir not in {"x", "y", "z"}:
            raise SystemExit(f"invalid biasdir {biasdir!r}; must be 'x', 'y', or 'z'")

        pts = [s.center(StructureSelector(h).atom_indices(s))[0] for h in reftile]
        nrm = plane_normal(pts)
        phiy = float(np.arctan2(-nrm[0], nrm[2]))
        s.rotate_about_y(phiy, anchor=ch)

        pts = [s.center(StructureSelector(h).atom_indices(s))[0] for h in reftile]
        nrm = plane_normal(pts)
        phix = float(np.arctan2(nrm[1], np.hypot(nrm[0], nrm[2])))
        s.rotate_about_x(phix, anchor=ch)

        phiz = 0.0
        if biasdir in {"x", "y"}:
            co = s.center(StructureSelector(othersel).atom_indices(s))[0]
            vec = (co - ch).value_in_unit(nanometer)
            if biasdir == "x":
                phiz = float(np.arctan2(vec[1], vec[0]))
                s.rotate_about_z(-phiz, anchor=ch)
            else:
                phiz = float(np.arctan2(vec[0], vec[1]))
                s.rotate_about_z(+phiz, anchor=ch)

        rx = np.degrees(phix)
        ry = np.degrees(phiy)
        rz = np.degrees(phiz)
        print(f"rotated Y: {ry:.3f} X: {rx:.3f} Z: {rz:.3f} degrees")

    biasdir = str(args.biasdir).lower()
    if biasdir == "x":
        target = (boxx / 4.0, boxy / 2.0, boxz / 2.0)
    elif biasdir == "y":
        target = (boxx / 2.0, boxy / 4.0, boxz / 2.0)
    elif biasdir == "z":
        target = (boxx / 2.0, boxy / 2.0, boxz / 4.0)
    else:
        raise SystemExit(f"invalid biasdir {biasdir!r}")

    translate = [target[0] - ch[0], target[1] - ch[1], target[2] - ch[2]]
    s.translate(translate)

    t_nm = [float(x.value_in_unit(nanometer)) for x in translate]
    print(f"translated center by ({t_nm[0]:.3f}, {t_nm[1]:.3f}, " f"{t_nm[2]:.3f}) nm")

    s.write_pdb(str(tdir / "dimer.protein.pdb"))

    if mode == "allatom":
        conc = float(args.conc)
        _, nsod, ncla = ion_counts(boxx, boxy, boxz, conc, s[0].nominal_charge())
        solvated, wbox = solvate(
            s[0],
            box_min=(0.0, 0.0, 0.0),
            box_max=(boxx, boxy, boxz),
            ions={"SOD": nsod, "CLA": ncla},
        )
        print(f"solvated system with {nsod} Na and {ncla} Cl ions")
        print(f"box size: {wbox} nm^3")

        solvated.write_pdb(str(tdir / "dimer.solvated.pdb"))
        sim = MDSim(
            model=solvated,
            ff=ff_val,
            box=(boxx, boxy, boxz),
            hmass=True,
            switching="openmm",
        )
    elif mode == "cocomo":
        surf = float(args.surf)
        components = find_input_file(tdir, "dimer.components")
        component_types = find_input_file(tdir, "component_types_files")
        interactions = find_input_file(tdir, "interactions")
        asm = Assembly(
            components,
            component_types,
            structure=s,
            interactions=interactions,
        )
        sim = COCOMO(asm, box=(boxx, boxy, boxz), version=2, surfscale=surf)
    else:
        raise SystemExit(f"invalid mode {mode!r}")

    sim.setup_simulation()
    print(f"openmm energy: {sim.get_potentialEnergy()}")

    sim.write_system(str(tdir / "system.xml"))
    sim.write_state(str(tdir / "initial.xml"))


if __name__ == "__main__":
    main()
