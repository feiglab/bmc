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

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cocomo import COCOMO
from mdsim import MDSim, PDBReader, StructureSelector

from .tile_config import format_value, parse_bool, read_config, write_config


def _parse_floats(s: str, defaults: Sequence[float], n_out: int) -> list[float]:
    vals = [float(x) for x in s.split(":") if x.strip() != ""]
    out: list[float] = []
    last: float | None = None

    for i in range(n_out):
        if i < len(vals):
            last = vals[i]
            out.append(last)
        elif i < len(defaults):
            last = float(defaults[i])
            out.append(last)
        else:
            out.append(last if last is not None else 0.0)

    return out


def _find(tdir: Path, filename: str) -> Path:
    tdir = Path(tdir).expanduser().resolve()

    for d in (tdir, *tdir.parents):
        candidate = d / filename
        if candidate.is_file():
            return candidate.resolve()

    candidate = Path.cwd() / filename
    if candidate.is_file():
        return candidate.resolve()

    raise FileNotFoundError(f"Could not find '{filename}' in {tdir} or its parents or CWD")


def _split_tile_sel(sel: str) -> list[str]:
    base, dot, suffix = sel.partition(".")
    tiles = base.split(":")
    if dot:
        return [f"{t}.{suffix}" for t in tiles]
    return tiles


def _build_anchor_selections(refsel: str, othersel: str, anchor: str) -> tuple[str, ...]:
    a0, a1 = anchor.split(":")

    # reference tile
    base, dot, suffix = refsel.partition(".")
    tiles = base.split(":")
    suf = f".{suffix}" if dot else ""

    tlen = len(tiles)
    i = tiles.index(a0)
    asel1 = f"{tiles[i]}:{tiles[(i + 1) % tlen]}{suf}"
    asel2 = f"{tiles[(i + 2) % tlen]}:{tiles[(i + 3) % tlen]}{suf}"
    aselt = f"{tiles[i]}:{tiles[(i + 2) % tlen]}:{tiles[(i + 3) % tlen]}{suf}"
    as11 = f"{tiles[i]}{suf}"
    as12 = f"{tiles[(i + 1) % tlen]}{suf}"

    # other tile
    base, dot, suffix = othersel.partition(".")
    tiles = base.split(":")
    suf = f".{suffix}" if dot else ""

    tlen = len(tiles)
    i = tiles.index(a1)

    if tlen == 6:
        bsel1 = f"{tiles[i]}:{tiles[(i + 1) % tlen]}{suf}"
        bsel2 = f"{tiles[(i + 2) % tlen]}:{tiles[(i + 3) % tlen]}{suf}"
        bselt = f"{tiles[i]}:{tiles[(i + 2) % tlen]}:{tiles[(i + 3) % tlen]}{suf}"
    elif tlen == 5:
        bsel1 = f"{tiles[i]}:{tiles[(i + 1) % tlen]}{suf}"
        bsel2 = f"{tiles[(i + 3) % tlen]}:{tiles[(i + 4) % tlen]}{suf}"
        bselt = f"{tiles[i]}:{tiles[(i + 1) % tlen]}:{tiles[(i + 4) % tlen]}{suf}"
    elif tlen == 3:
        bsel1 = f"{tiles[i]}{suf}"
        bsel2 = f"{tiles[(i + 1) % tlen]}{suf}"
        bselt = f"{tiles[i]}:{tiles[(i + 1) % tlen]}{suf}"
    else:
        raise SystemExit("ERROR: invalid length of other selection")

    return asel1, asel2, aselt, bsel1, bsel2, bselt, as11, as12


def _parse_config_args(argv: Sequence[str] | None = None) -> Path:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config"),
        help="Config file (key/value) to read/write",
    )
    ns, _ = p.parse_known_args(argv)
    return Path(ns.config)


@dataclass(frozen=True)
class ModeParams:
    initsteps: int
    initout: int
    prodsteps: int
    prodout: int
    tstep: float
    gamma: float


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


def _apply_config_defaults(
    p: argparse.ArgumentParser,
    cfg: dict[str, str],
) -> None:
    defaults: dict[str, object] = {}

    if "mode" in cfg:
        defaults["mode"] = cfg["mode"]
    if "setup" in cfg:
        defaults["setup"] = Path(cfg["setup"])
    if "equi" in cfg:
        defaults["equi"] = Path(cfg["equi"])
    if "refpdb" in cfg:
        defaults["refpdb"] = cfg["refpdb"]
    if "refsel" in cfg:
        defaults["refsel"] = cfg["refsel"]
    if "othersel" in cfg:
        defaults["othersel"] = cfg["othersel"]
    if "anchor" in cfg:
        defaults["anchor"] = cfg["anchor"]
    if "rot" in cfg:
        defaults["rot"] = cfg["rot"]
    if "bias" in cfg:
        defaults["bias"] = cfg["bias"]
    if "biasangle" in cfg:
        defaults["biasangle"] = cfg["biasangle"]
    if "flip" in cfg:
        defaults["flip"] = parse_bool(cfg["flip"])
    if "k" in cfg:
        defaults["k"] = cfg["k"]
    if "kinit" in cfg:
        defaults["kinit"] = cfg["kinit"]
    if "kbias" in cfg:
        defaults["kbias"] = cfg["kbias"]
    if "kbiasangle" in cfg:
        defaults["kbiasangle"] = cfg["kbiasangle"]
    if "kdistx" in cfg:
        defaults["kdistx"] = cfg["kdistx"]
    if "kdisty" in cfg:
        defaults["kdisty"] = cfg["kdisty"]
    if "kdistz" in cfg:
        defaults["kdistz"] = cfg["kdistz"]
    if "kcent" in cfg:
        defaults["kcent"] = cfg["kcent"]
    if "knorm" in cfg:
        defaults["knorm"] = cfg["knorm"]
    if "kdihed" in cfg:
        defaults["kdihed"] = cfg["kdihed"]
    if "krot" in cfg:
        defaults["krot"] = cfg["krot"]
    if "biasdir" in cfg:
        defaults["biasdir"] = cfg["biasdir"]

    if defaults:
        p.set_defaults(**defaults)


def _parse_args(
    cfg: dict[str, str],
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="initbias_tileumbrella.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode_grp = p.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--mode",
        choices=["allatom", "cocomo"],
        default="allatom",
        help="Simulation mode",
    )
    mode_grp.add_argument(
        "--allatom",
        dest="mode",
        action="store_const",
        const="allatom",
        help="Shortcut for --mode allatom",
    )
    mode_grp.add_argument(
        "--cocomo",
        dest="mode",
        action="store_const",
        const="cocomo",
        help="Shortcut for --mode cocomo",
    )

    p.add_argument("--setup", type=Path, default=Path("setup"), help="Setup directory")
    p.add_argument("--equi", type=Path, default=Path("equi"), help="Equi directory")
    p.add_argument(
        "--refpdb",
        type=str,
        default="dimer.protein.pdb",
        help="Reference PDB (searched relative to --setup and parents, then CWD)",
    )

    p.add_argument(
        "--refsel",
        type=str,
        default="A:B:C:D:E:F.2-91",
        help="Reference selection",
    )
    p.add_argument(
        "--othersel",
        type=str,
        default="G:H:I:J:K:L.2-91",
        help="Other selection",
    )
    p.add_argument(
        "--anchor",
        type=str,
        default="A:G",
        help="Anchor points as 'X:Y' where X in ref tiles, Y in other tiles",
    )
    p.add_argument(
        "--rot",
        type=str,
        default="90:90",
        help="Rotation angle reference as 'rot1:rot2' in degrees",
    )
    p.add_argument(
        "--bias",
        type=str,
        default="6.0:9.0:0.1",
        help="Bias range as 'min:max:delta'",
    )
    p.add_argument(
        "--biasdir",
        type=str,
        default="x",
        help="Bias direction 'x', 'y', 'z'",
    )
    p.add_argument(
        "--biasangle",
        type=str,
        default=None,
        help="Bias angle range as 'min:max:delta'",
    )
    p.add_argument(
        "--k",
        type=str,
        default="500:200",
        help="Force constants as 'kinit:kbias[:kdist[:kcent[:kangle]]]'",
    )
    p.add_argument(
        "--kinit",
        type=float,
        default=None,
        help="Initial force constant",
    )
    p.add_argument(
        "--kbias",
        type=float,
        default=None,
        help="Force constant for distance bias",
    )
    p.add_argument(
        "--kbiasangle",
        type=float,
        default=None,
        help="Force constant for angle bias",
    )
    p.add_argument(
        "--kdistx",
        type=float,
        default=None,
        help="Force constant for distance x, if not bias",
    )
    p.add_argument(
        "--kdisty",
        type=float,
        default=None,
        help="Force constant for distance y, if not bias",
    )
    p.add_argument(
        "--kdistz",
        type=float,
        default=None,
        help="Force constant for distance z, if not bias",
    )
    p.add_argument(
        "--kcent",
        type=float,
        default=None,
        help="Force constant for central force",
    )
    p.add_argument(
        "--knorm",
        type=float,
        default=None,
        help="Force constant for angle norm restraint",
    )
    p.add_argument(
        "--kdihed",
        type=float,
        default=None,
        help="Force constant for dihedral twist restraint",
    )
    p.add_argument(
        "--krot",
        type=float,
        default=None,
        help="Force constant for rotation restraint",
    )

    flip_grp = p.add_mutually_exclusive_group()
    flip_grp.add_argument(
        "--flip",
        dest="flip",
        action="store_true",
        help="Flipped second ring",
    )
    flip_grp.add_argument(
        "--no-flip",
        dest="flip",
        action="store_false",
        help="No flipped second ring",
    )
    p.set_defaults(flip=False)

    p.add_argument("--device", type=int, default=0, help="Device index")
    p.add_argument("--resources", type=str, default="CUDA", help="OpenMM platform name")

    p.add_argument(
        "--config",
        type=Path,
        default=Path("config"),
        help="Config file (key/value) to read/write",
    )
    p.add_argument(
        "--no-write-config",
        dest="write_config",
        action="store_false",
        help="Disable writing updated config values",
    )
    p.set_defaults(write_config=True)

    _apply_config_defaults(p, cfg)
    return p.parse_args(argv)


def main() -> None:
    cfg_path = _parse_config_args()
    cfg = read_config(cfg_path)

    args = _parse_args(cfg)
    params = _mode_params(str(args.mode))

    sdir = Path(args.setup).expanduser().resolve()
    edir = Path(args.equi).expanduser().resolve()

    cfg_path = Path(args.config)
    if bool(args.write_config):
        cfg["mode"] = format_value(str(args.mode).lower())
        cfg["setup"] = format_value(args.setup)
        cfg["equi"] = format_value(args.equi)
        cfg["refpdb"] = format_value(args.refpdb)
        cfg["refsel"] = format_value(args.refsel)
        cfg["othersel"] = format_value(args.othersel)
        cfg["anchor"] = format_value(args.anchor)
        cfg["rot"] = format_value(args.rot)
        cfg["bias"] = format_value(args.bias)
        cfg["biasdir"] = format_value(args.biasdir)
        if args.biasangle is not None:
            cfg["biasangle"] = format_value(args.biasangle)
        cfg["k"] = format_value(args.k)
        if args.kinit is not None:
            cfg["kinit"] = format_value(args.kinit)
        if args.kbias is not None:
            cfg["kbias"] = format_value(args.kbias)
        if args.kbiasangle is not None:
            cfg["kbiasangle"] = format_value(args.kbiasangle)
        if args.kdistx is not None:
            cfg["kdistx"] = format_value(args.kdistx)
        if args.kdisty is not None:
            cfg["kdisty"] = format_value(args.kdisty)
        if args.kdistz is not None:
            cfg["kdistz"] = format_value(args.kdistz)
        if args.kcent is not None:
            cfg["kcent"] = format_value(args.kcent)
        if args.knorm is not None:
            cfg["knorm"] = format_value(args.knorm)
        if args.kdihed is not None:
            cfg["kdihed"] = format_value(args.kdihed)
        if args.krot is not None:
            cfg["krot"] = format_value(args.krot)

        cfg["flip"] = format_value(bool(args.flip))
        write_config(cfg_path, cfg)

    pdb_path = _find(sdir, str(args.refpdb))
    s = PDBReader(str(pdb_path))

    refsel = str(args.refsel)
    othersel = str(args.othersel)
    anchor = str(args.anchor)
    biasdir = str(args.biasdir)

    refrot1, refrot2 = _parse_floats(str(args.rot), [90.0, 90.0], n_out=2)
    bmin, bmax, bdel = _parse_floats(str(args.bias), [6.0, 9.0, 0.1], n_out=3)
    kinit, kbias, kdist, kcent, kangle = _parse_floats(
        str(args.k),
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

    if args.biasangle is not None:
        amin, amax, adel = _parse_floats(str(args.biasangle), [90.0, 180.0, 15.0], n_out=3)
        bias_pairs = [
            (biasval, biasangleval)
            for biasval in np.arange(bmin, bmax + 1.0e-8, bdel)
            for biasangleval in np.arange(amin, amax + 1.0e-8, adel)
        ]
        if args.kbiasangle is not None:
            kbiasangle = float(args.kbiasangle)
        else:
            kbiasangle = kbias
    else:
        bias_pairs = [(biasval, None) for biasval in np.arange(bmin, bmax + 1.0e-8, bdel)]
        kbiasangle = 0.0

    device = int(args.device)
    resources = str(args.resources)

    reftile = _split_tile_sel(refsel)
    asel1, asel2, aselt, bsel1, bsel2, bselt, as11, as12 = _build_anchor_selections(
        refsel=refsel,
        othersel=othersel,
        anchor=anchor,
    )

    aca = StructureSelector(refsel + ".CA").atom_indices(s)
    bca = StructureSelector(othersel + ".CA").atom_indices(s)

    rc = [StructureSelector(t + ".CA").atom_indices(s) for t in reftile]
    rcref = [s.center(r)[0] for r in rc]

    aca1 = StructureSelector(asel1 + ".CA").atom_indices(s)
    aca2 = StructureSelector(asel2 + ".CA").atom_indices(s)
    acat1 = StructureSelector(aselt + ".CA").atom_indices(s)

    bca1 = StructureSelector(bsel1 + ".CA").atom_indices(s)
    bca2 = StructureSelector(bsel2 + ".CA").atom_indices(s)
    bcat1 = StructureSelector(bselt + ".CA").atom_indices(s)

    a1 = StructureSelector(as11 + ".CA").atom_indices(s)
    a2 = StructureSelector(as12 + ".CA").atom_indices(s)

    restart = edir / "equi_final.xml"
    mode = str(args.mode).lower()

    keps = 1e-8

    for biasval, biasangleval in bias_pairs:
        if biasangleval is not None:
            tag = f"{biasval:.2f}_{biasangleval:.0f}"
        else:
            tag = f"{biasval:.2f}"

        bdir = Path(f"run_{tag}")
        bdir.mkdir(parents=True, exist_ok=True)

        if mode == "allatom":
            sim = MDSim(xml=str(sdir / "system.xml"), restart=str(restart))
        elif mode == "cocomo":
            sim = COCOMO(xml=str(sdir / "system.xml"), restart=str(restart), version=2)
        else:
            raise SystemExit(f"ERROR: unknown mode {mode!r}")

        if biasdir == "x":
            sim.set_umbrella_xyz_distance(aca, bca, direction="x", target=biasval, k=kinit)
            if kdisty > keps:
                sim.set_umbrella_xyz_distance(aca, bca, direction="y", target=0.0, k=kinit)
            if kdistz > keps:
                sim.set_umbrella_xyz_distance(aca, bca, direction="z", target=0.0, k=kinit)
        elif biasdir == "y":
            if kdistx > keps:
                sim.set_umbrella_xyz_distance(aca, bca, direction="x", target=0.0, k=kinit)
            sim.set_umbrella_xyz_distance(aca, bca, direction="y", target=biasval, k=kinit)
            if kdistz > keps:
                sim.set_umbrella_xyz_distance(aca, bca, direction="z", target=0.0, k=kinit)
        elif biasdir == "z":
            if kdistx > keps:
                sim.set_umbrella_xyz_distance(aca, bca, direction="x", target=0.0, k=kinit)
            if kdisty > keps:
                sim.set_umbrella_xyz_distance(aca, bca, direction="y", target=0.0, k=kinit)
            sim.set_umbrella_xyz_distance(aca, bca, direction="z", target=biasval, k=kinit)
        else:
            raise SystemExit(f"ERROR: invalid biasdir {biasdir}")

        if biasangleval is not None and kbiasangle > keps:
            sim.set_umbrella_dihedral(
                aca, a1, a2, bca, target=np.radians(biasangleval), k=kinit, tag="dihbias"
            )

        if kcent > keps:
            sim.set_umbrella_center(rc, k=kinit, target=rcref)

        if knorm > keps:
            if bool(args.flip):
                sim.set_umbrella_angle_norm(aca, aca1, aca2, bca, bca2, bca1, k=kinit)
            else:
                sim.set_umbrella_angle_norm(aca, aca1, aca2, bca, bca1, bca2, k=kinit)

        if kdihed > keps:
            sim.set_umbrella_dihedral(acat1, aca, bca, bcat1, k=kinit)

        if krot > keps:
            sim.set_umbrella_angle(aca, bca, bcat1, target=np.radians(refrot1), k=kinit)
            sim.set_umbrella_angle(acat1, aca, bca, target=np.radians(refrot2), k=kinit)

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
