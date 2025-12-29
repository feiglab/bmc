#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from mdsim import (
    PDBReader,
    StructureSelector,
    harmonic_energy_angle,
    harmonic_energy_dihedral,
    harmonic_energy_xyz,
    load_dcd,
)
from openmm.unit import degrees, kilojoule, mole, nanometer, radian

from .tile_config import format_value, read_config, write_config


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


def _apply_config_defaults(
    p: argparse.ArgumentParser,
    cfg: dict[str, str],
) -> None:
    defaults: dict[str, object] = {}

    if "refsel" in cfg:
        defaults["refsel"] = cfg["refsel"]
    if "othersel" in cfg:
        defaults["othersel"] = cfg["othersel"]
    if "anchor" in cfg:
        defaults["anchor"] = cfg["anchor"]
    if "capdb" in cfg:
        defaults["capdb"] = cfg["capdb"]
    if "cadcd" in cfg:
        defaults["cadcd"] = cfg["cadcd"]
    if "rot" in cfg:
        defaults["rot"] = cfg["rot"]
    if "k" in cfg:
        defaults["k"] = cfg["k"]
    if "bias" in cfg:
        defaults["bias"] = cfg["bias"]

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

    p.add_argument(
        "--capdb",
        type=str,
        default="CA.pdb",
        help="Reference PDB (CA only)",
    )

    p.add_argument(
        "--cadcd",
        type=str,
        default="ca.dcd",
        help="Reference PDB (CA only)",
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

    cfg_path = Path(args.config)
    if bool(args.write_config):
        cfg["capdb"] = format_value(args.capdb)
        cfg["cadcd"] = format_value(args.cadcd)
        cfg["refsel"] = format_value(args.refsel)
        cfg["othersel"] = format_value(args.othersel)
        cfg["anchor"] = format_value(args.anchor)
        cfg["rot"] = format_value(args.rot)
        cfg["bias"] = format_value(args.bias)
        cfg["k"] = format_value(args.k)
        write_config(cfg_path, cfg)

    pdb_path = _find(".", str(args.capdb))
    s = PDBReader(str(pdb_path)).select_CA()

    dcd_path = _find(".", str(args.cadcd))
    traj = load_dcd(str(dcd_path), s)

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

    asel1, asel2, aselt, bsel1, bsel2, bselt = _build_anchor_selections(
        refsel=refsel,
        othersel=othersel,
        anchor=anchor,
    )

    aca = StructureSelector(refsel + ".CA").atom_indices(s)
    bca = StructureSelector(othersel + ".CA").atom_indices(s)

    aca1 = StructureSelector(asel1 + ".CA").atom_indices(s)
    aca2 = StructureSelector(asel2 + ".CA").atom_indices(s)
    acat1 = StructureSelector(aselt + ".CA").atom_indices(s)

    bca1 = StructureSelector(bsel1 + ".CA").atom_indices(s)
    bca2 = StructureSelector(bsel2 + ".CA").atom_indices(s)
    bcat1 = StructureSelector(bselt + ".CA").atom_indices(s)

    distance_vector = traj.distance_vector(aca, bca)
    angle_norm = traj.angle_norm(aca, aca1, aca2, bca, bca1, bca2)
    dihedral = traj.dihedral(acat1, aca, bca, bcat1)
    angle1 = traj.angle(aca, bca, bcat1)
    angle2 = traj.angle(acat1, aca, bca)

    biasy = harmonic_energy_xyz(
        distance_vector, kdist * kilojoule / mole / nanometer**2, 0.0 * nanometer, axis="y"
    )
    biasz = harmonic_energy_xyz(
        distance_vector, kdist * kilojoule / mole / nanometer**2, 0.0 * nanometer, axis="z"
    )
    biasnorm = harmonic_energy_angle(
        angle_norm, kangle * kilojoule / mole / radian**2, 0.0 * radian
    )
    biastwist = harmonic_energy_dihedral(
        dihedral, kangle * kilojoule / mole / radian**2, 0.0 * radian
    )
    biasangle1 = harmonic_energy_angle(
        angle1, kangle * kilojoule / mole / radian**2, np.radians(refrot1) * radian
    )
    biasangle2 = harmonic_energy_angle(
        angle2, kangle * kilojoule / mole / radian**2, np.radians(refrot2) * radian
    )

    for biasval in np.arange(bmin, bmax + 1.0e-8, bdel):
        tag = f"{biasval:.2f}"

        bdir = Path(f"run_{tag}")

        biasx = harmonic_energy_xyz(
            distance_vector, kbias * kilojoule / mole / nanometer**2, biasval * nanometer, axis="x"
        )

        with open(str(bdir / "bias.dat"), "w") as f:
            f.write(
                "Step\tx_dist_bias[kJ/mol]\ty_bias_bias[kJ/mol]\tz_bias_bias[kJ/mol]\t"
                "angle_bias[kJ/mol]\ttorsion_bias[kJ/mol]\trot_angle_bias[kJ/mol]\n"
            )
            n = len(biasx)
            for i in range(n):
                bx = biasx[i].value_in_unit(kilojoule / mole)
                by = biasy[i].value_in_unit(kilojoule / mole)
                bz = biasz[i].value_in_unit(kilojoule / mole)
                ba = biasnorm[i].value_in_unit(kilojoule / mole)
                bt = biastwist[i].value_in_unit(kilojoule / mole)
                bra = (biasangle1[i] + biasangle2[i]).value_in_unit(kilojoule / mole)

                f.write(f"{i}\t" f"{bx}\t" f"{by}\t" f"{bz}\t" f"{ba}\t" f"{bt}\t" f"{bra}\n")

        with open(str(bdir / "geometry.dat"), "w") as f:
            f.write(
                "Step\tX_Distance[nm]\tY_Distance[nm]\tZ_Distance[nm]\t"
                "NormalAngle[deg]\tTorsionAngle[deg]\tRotAngle1[deg]\tRotAngle2[deg]\n"
            )
            n = len(distance_vector)
            for i in range(n):
                gx = distance_vector[i][0].value_in_unit(nanometer)
                gy = distance_vector[i][1].value_in_unit(nanometer)
                gz = distance_vector[i][2].value_in_unit(nanometer)
                gn = angle_norm[i].value_in_unit(degrees)
                gd = dihedral[i].value_in_unit(degrees)
                ga1 = angle1[i].value_in_unit(degrees)
                ga2 = angle2[i].value_in_unit(degrees)

                f.write(
                    f"{i}\t" f"{gx}\t" f"{gy}\t" f"{gz}\t" f"{gn}\t" f"{gd}\t" f"{ga1}\t" f"{ga2}\n"
                )

        print(f"finished {tag}")


if __name__ == "__main__":
    main()
