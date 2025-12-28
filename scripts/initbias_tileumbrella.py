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
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from cocomo import COCOMO
from mdsim import MDSim, PDBReader, StructureSelector


def _as_int(name: str, s: str) -> int:
    try:
        return int(s)
    except Exception as e:
        raise SystemExit(f"ERROR: {name} must be an int, got {s!r}") from e


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

    raise FileNotFoundError(
        f"Could not find '{filename}' in {tdir} or its parents or CWD"
    )


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

    return asel1, asel2, aselt, bsel1, bsel2, bselt


@dataclass(frozen=True)
class ModeParams:
    initsteps: int
    initout: int
    prodsteps: int
    tstep: float
    gamma: float


def _mode_params(mode: str) -> ModeParams:
    m = mode.lower()
    if m == "allatom":
        return ModeParams(
            initsteps=10000,
            initout=1000,
            prodsteps=5000,
            tstep=0.003,
            gamma=0.1,
        )
    if m == "cocomo":
        return ModeParams(
            initsteps=1000,
            initout=200,
            prodsteps=100,
            tstep=0.03,
            gamma=1.0,
        )
    raise SystemExit(f"ERROR: unknown mode {mode!r}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
        "--pdb",
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
        help="Bias range as 'bmin:bmax:bdelta'",
    )
    p.add_argument(
        "--k",
        type=str,
        default="500:200",
        help="Force constants as 'kinit:kbias[:kdist[:kcent[:kangle]]]'",
    )
    p.add_argument("--device", type=int, default=0, help="Device index")
    p.add_argument("--resources", type=str, default="CUDA", help="OpenMM platform name")

    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()
    params = _mode_params(str(args.mode))

    sdir = Path(args.setup).expanduser().resolve()
    edir = Path(args.equi).expanduser().resolve()

    pdb_path = _find(sdir, str(args.pdb))
    s = PDBReader(str(pdb_path))

    refsel = str(args.refsel)
    othersel = str(args.othersel)
    anchor = str(args.anchor)

    refrot1, refrot2 = _parse_floats(str(args.rot), [90.0, 90.0], n_out=2)
    bmin, bmax, bdel = _parse_floats(str(args.bias), [6.0, 9.0, 0.1], n_out=3)
    kinit, kbias, kdist, kcent, kangle = _parse_floats(
        str(args.k),
        [500.0, 200.0],
        n_out=5,
    )

    device = int(args.device)
    resources = str(args.resources)

    reftile = _split_tile_sel(refsel)
    asel1, asel2, aselt, bsel1, bsel2, bselt = _build_anchor_selections(
        refsel=refsel,
        othersel=othersel,
        anchor=anchor,
    )

    aca = StructureSelector(refsel + ".CA").atom_indices(s)
    bca = StructureSelector(othersel + ".CA").atom_indices(s)

    rc = [StructureSelector(t + ".CA").atom_indices(s) for t in reftile]

    aca1 = StructureSelector(asel1 + ".CA").atom_indices(s)
    aca2 = StructureSelector(asel2 + ".CA").atom_indices(s)
    acat1 = StructureSelector(aselt + ".CA").atom_indices(s)

    bca1 = StructureSelector(bsel1 + ".CA").atom_indices(s)
    bca2 = StructureSelector(bsel2 + ".CA").atom_indices(s)
    bcat1 = StructureSelector(bselt + ".CA").atom_indices(s)

    restart = edir / "equi_final.xml"
    mode = str(args.mode).lower()

    for biasval in np.arange(bmin, bmax + 1.0e-8, bdel):
        tag = f"{biasval:.2f}"

        bdir = Path(f"run_{tag}")
        bdir.mkdir(parents=True, exist_ok=True)

        if mode == "allatom":
            sim = MDSim(xml=str(sdir / "system.xml"), restart=str(restart))
        elif mode == "cocomo":
            sim = COCOMO(xml=str(sdir / "system.xml"), restart=str(restart), version=2)
        else:
            raise SystemExit(f"ERROR: unknown mode {mode!r}")

        sim.set_umbrella_xyz_distance(aca, bca, direction="x", target=biasval, k=kinit)
        sim.set_umbrella_xyz_distance(aca, bca, direction="y", target=0.0, k=kinit)
        sim.set_umbrella_xyz_distance(aca, bca, direction="z", target=0.0, k=kinit)
        sim.set_umbrella_center(rc, k=kinit)
        sim.set_umbrella_angle_norm(aca, aca1, aca2, bca, bca1, bca2, k=kinit)
        sim.set_umbrella_dihedral(acat1, aca, bca, bcat1, k=kinit)
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

        sim.update_umbrella_xyz_distance("x", kbias)
        sim.update_umbrella_xyz_distance("y", kdist)
        sim.update_umbrella_xyz_distance("z", kdist)
        sim.update_umbrella_center(kcent)
        sim.update_umbrella_angle_norm(kangle)
        sim.update_umbrella_dihedral(kangle)
        sim.update_umbrella_angle(kangle)

        sim.simulate(nstep=params.prodsteps, logfile=str(bdir / "biasprod_0.log"))

        sim.write_state(str(bdir / "biasprod_0.xml"))

        restart = biasinitxml
        print(f"finished {tag}")


if __name__ == "__main__":
    main()

