#!/usr/bin/env python3
#
# hexamer-hexamer:
#    A:B:C:D:E:F.2-91 G:H:I:J:K:L.2-91 A:G 90:90 6.0:9.0:0.1
# hexmer-pentamer:
#    F:G:H:I:J:K.2-91 A:B:C:D:E.1-95 H:C 120:90 6.0:9.0:0.1
# hexamer-trimer:
#    D:E:F:G:H:I.2-91 A:B:C.19-205 D:C 60:85 6.2:9.0:0.1
#

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cocomo import COCOMO
from mdsim import MDSim, PDBReader, StructureSelector

from .tile_config import read_config
from .tileumbrella_shared import (
    build_anchor_selections,
    build_bias_pairs,
    find_input_file,
    format_bias_tag,
    parse_args,
    parse_config_path,
    parse_floats,
    split_tile_selection,
    write_args_config,
)


@dataclass(frozen=True)
class ModeParams:
    initsteps: int
    initout: int
    prodsteps: int
    prodout: int
    tstep: float
    gamma: float


_HELP_OPTIONS = (
    "mode",
    "setup",
    "equi",
    "refpdb",
    "refsel",
    "othersel",
    "anchor",
    "rot",
    "bias",
    "biasdir",
    "biasangle",
    "k",
    "kinit",
    "kbias",
    "kbiasangle",
    "biaspairs",
    "kdistx",
    "kdisty",
    "kdistz",
    "kdist",
    "kcent",
    "knorm",
    "kdihed",
    "krot",
    "flip",
    "normcap",
    "distcap",
    "device",
    "resources",
    "config",
    "write_config",
)


def _mode_params(mode: str) -> ModeParams:
    m = mode.lower()
    if m == "allatom":
        return ModeParams(
            initsteps=10000,
            initout=1000,
            prodsteps=5000,
            prodout=1000,
            tstep=0.003,
            gamma=0.1,
        )
    if m == "cocomo":
        return ModeParams(
            initsteps=1000,
            initout=200,
            prodsteps=200,
            prodout=100,
            tstep=0.03,
            gamma=1.0,
        )
    raise SystemExit(f"ERROR: unknown mode {mode!r}")


def main() -> None:
    cfg_path = parse_config_path()
    cfg = read_config(cfg_path)

    args = parse_args(cfg, _HELP_OPTIONS, prog="initbias_tileumbrella.py")
    params = _mode_params(str(args.mode))

    sdir = Path(args.setup).expanduser().resolve()
    edir = Path(args.equi).expanduser().resolve()

    if args.refpdb is None:
        refpdb = "dimer.protein.pdb"
    else:
        refpdb = str(args.refpdb)

    cfg_path = Path(args.config)
    if bool(args.write_config):
        write_args_config(
            cfg_path,
            cfg,
            args,
            overrides={
                "mode": str(args.mode),
                "refpdb": refpdb,
            },
        )

    pdb_path = find_input_file(sdir, refpdb)
    s = PDBReader(str(pdb_path))

    refsel = str(args.refsel)
    othersel = str(args.othersel)
    anchor = str(args.anchor)
    biasdir = str(args.biasdir)

    refrot1, refrot2 = parse_floats(str(args.rot), [90.0, 90.0], n_out=2)
    k_spec = "500.0:200.0" if args.k is None else str(args.k)
    kinit, kbias, kdist, kcent, kangle = parse_floats(
        k_spec,
        [500.0, 200.0],
        n_out=5,
    )

    if args.kinit is not None:
        kinit = float(args.kinit)

    if args.kbias is not None:
        kbias = float(args.kbias)

    if args.kdistx is not None:
        kdistx = float(args.kdistx)
    else:
        kdistx = kdist

    if args.kdisty is not None:
        kdisty = float(args.kdisty)
    else:
        kdisty = kdist

    if args.kdistz is not None:
        kdistz = float(args.kdistz)
    else:
        kdistz = kdist

    if args.kcent is not None:
        kcent = float(args.kcent)

    if args.knorm is not None:
        knorm = float(args.knorm)
    else:
        knorm = kangle

    if args.kdihed is not None:
        kdihed = float(args.kdihed)
    else:
        kdihed = kangle

    if args.krot is not None:
        krot = float(args.krot)
    else:
        krot = kangle

    if args.normcap is not None:
        normcap = float(args.normcap)
    else:
        normcap = None

    if args.distcap is not None:
        distcap = float(args.distcap)
        if args.kdist is not None:
            kdist = float(args.kdist)
    else:
        distcap = None

    bias_pairs = build_bias_pairs(
        bias=str(args.bias),
        biasangle=None if args.biasangle is None else str(args.biasangle),
        biaspairs=None if args.biaspairs is None else str(args.biaspairs),
    )

    if any(biasangleval is not None for _, biasangleval in bias_pairs):
        if args.kbiasangle is not None:
            kbiasangle = float(args.kbiasangle)
        else:
            kbiasangle = kbias
    else:
        kbiasangle = 0.0

    device = int(args.device)
    resources = str(args.resources)

    reftile = split_tile_selection(refsel)
    asel1, asel2, aselt, bsel1, bsel2, bselt, as11, as12, bselc = build_anchor_selections(
        refsel=refsel,
        othersel=othersel,
        anchor=anchor,
    )

    aca = StructureSelector(refsel + ".CA").atom_indices(s)
    bca = StructureSelector(othersel + ".CA").atom_indices(s)

    rc = [StructureSelector(tile + ".CA").atom_indices(s) for tile in reftile]
    rcref = [s.center(r)[0] for r in rc]
    print(reftile)
    print(rcref)

    aca1 = StructureSelector(asel1 + ".CA").atom_indices(s)
    aca2 = StructureSelector(asel2 + ".CA").atom_indices(s)
    acat1 = StructureSelector(aselt + ".CA").atom_indices(s)

    bca1 = StructureSelector(bsel1 + ".CA").atom_indices(s)
    bca2 = StructureSelector(bsel2 + ".CA").atom_indices(s)
    bcat1 = StructureSelector(bselt + ".CA").atom_indices(s)

    a1 = StructureSelector(as11 + ".CA").atom_indices(s)
    a2 = StructureSelector(as12 + ".CA").atom_indices(s)

    bcac = StructureSelector(bselc + ".CA").atom_indices(s)

    restart = edir / "equi_final.xml"
    mode = str(args.mode).lower()

    keps = 1e-8

    for biasval, biasangleval in bias_pairs:
        tag = format_bias_tag(biasval, biasangleval)

        bdir = Path(f"run_{tag}")
        bdir.mkdir(parents=True, exist_ok=True)

        if mode == "allatom":
            sim = MDSim(xml=str(sdir / "system.xml"), restart=str(restart))
        elif mode == "cocomo":
            sim = COCOMO(
                xml=str(sdir / "system.xml"),
                restart=str(restart),
                version=2,
            )
        else:
            raise SystemExit(f"ERROR: unknown mode {mode!r}")

        if biasdir == "x":
            sim.set_umbrella_xyz_distance(
                aca,
                bca,
                direction="x",
                target=biasval,
                k=kinit,
            )
            if kdisty > keps:
                sim.set_umbrella_xyz_distance(
                    aca,
                    bca,
                    direction="y",
                    target=0.0,
                    k=kinit,
                )
            if kdistz > keps:
                sim.set_umbrella_xyz_distance(
                    aca,
                    bca,
                    direction="z",
                    target=0.0,
                    k=kinit,
                )
        elif biasdir == "y":
            if kdistx > keps:
                sim.set_umbrella_xyz_distance(
                    aca,
                    bca,
                    direction="x",
                    target=0.0,
                    k=kinit,
                )
            sim.set_umbrella_xyz_distance(
                aca,
                bca,
                direction="y",
                target=biasval,
                k=kinit,
            )
            if kdistz > keps:
                sim.set_umbrella_xyz_distance(
                    aca,
                    bca,
                    direction="z",
                    target=0.0,
                    k=kinit,
                )
        elif biasdir == "z":
            if kdistx > keps:
                sim.set_umbrella_xyz_distance(
                    aca,
                    bca,
                    direction="x",
                    target=0.0,
                    k=kinit,
                )
            if kdisty > keps:
                sim.set_umbrella_xyz_distance(
                    aca,
                    bca,
                    direction="y",
                    target=0.0,
                    k=kinit,
                )
            sim.set_umbrella_xyz_distance(
                aca,
                bca,
                direction="z",
                target=biasval,
                k=kinit,
            )
        else:
            raise SystemExit(f"ERROR: invalid biasdir {biasdir}")

        if biasangleval is not None and kbiasangle > keps:
            sim.set_umbrella_dihedral(
                aca,
                a1,
                a2,
                bca,
                target=np.radians(biasangleval),
                k=kinit,
                tag="dihbias",
            )

        if kcent > keps:
            sim.set_umbrella_center(rc, k=kinit, target=rcref)

        if knorm > keps:
            if normcap is None:
                normtarget = np.radians(0)
                normmode = "both"
            else:
                normtarget = np.radians(normcap)
                normmode = "above"

            if bool(args.flip):
                sim.set_umbrella_angle_norm(
                    aca,
                    aca1,
                    aca2,
                    bca,
                    bca2,
                    bca1,
                    target=normtarget,
                    side=normmode,
                    k=kinit,
                )
            else:
                sim.set_umbrella_angle_norm(
                    aca,
                    aca1,
                    aca2,
                    bca,
                    bca1,
                    bca2,
                    target=normtarget,
                    side=normmode,
                    k=kinit,
                )

        if kdihed > keps:
            sim.set_umbrella_dihedral(acat1, aca, bca, bcat1, k=kinit)

        if krot > keps:
            sim.set_umbrella_angle(
                aca,
                bca,
                bcat1,
                target=np.radians(refrot1),
                k=kinit,
            )
            sim.set_umbrella_angle(
                acat1,
                aca,
                bca,
                target=np.radians(refrot2),
                k=kinit,
            )

        if distcap is not None and kdist > keps:
            sim.set_umbrella_distance(aca1, bcac, target=distcap, side="above", k=kinit)

        sim.set_force_groups()

        sim.write_system(str(bdir / f"bias_system_{tag}.xml"))

        sim.setup_simulation(
            resources=resources,
            device=device,
            temperature=298,
            tstep=params.tstep,
            gamma=params.gamma,
        )
        sim.simulate(
            nstep=params.initsteps,
            nout=params.initout,
            logfile=str(bdir / f"biasinit_{tag}.log"),
            dcdfile=str(bdir / f"biasinit_{tag}.dcd"),
        )

        biasinitxml = bdir / f"biasinit_{tag}.xml"
        sim.write_state(str(biasinitxml))

        if biasdir == "x":
            sim.update_umbrella_xyz_distance("x", kbias)
            if kdisty > keps:
                sim.update_umbrella_xyz_distance("y", kdisty)
            if kdistz > keps:
                sim.update_umbrella_xyz_distance("z", kdistz)
        elif biasdir == "y":
            if kdistx > keps:
                sim.update_umbrella_xyz_distance("x", kdistx)
            sim.update_umbrella_xyz_distance("y", kbias)
            if kdistz > keps:
                sim.update_umbrella_xyz_distance("z", kdistz)
        elif biasdir == "z":
            if kdistx > keps:
                sim.update_umbrella_xyz_distance("x", kdistx)
            if kdisty > keps:
                sim.update_umbrella_xyz_distance("y", kdisty)
            sim.update_umbrella_xyz_distance("z", kbias)
        else:
            raise SystemExit(f"ERROR: invalid biasdir {biasdir}")

        if biasangleval is not None and kbiasangle > keps:
            sim.update_umbrella_dihedral(kbiasangle, tag="dihbias")

        if kcent > keps:
            sim.update_umbrella_center(kcent)

        if knorm > keps:
            sim.update_umbrella_angle_norm(knorm)

        if kdihed > keps:
            sim.update_umbrella_dihedral(kdihed)

        if krot > keps:
            sim.update_umbrella_angle(krot)

        if distcap is not None and kdist > keps:
            sim.update_umbrella_distance(kdist)

        sim.simulate(
            nstep=params.prodsteps,
            nout=params.prodout,
            logfile=str(bdir / "biasprod_0.log"),
        )

        sim.write_state(str(bdir / "biasprod_0.xml"))

        restart = biasinitxml
        print(f"finished {tag}")


if __name__ == "__main__":
    main()
