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

from .tile_config import format_value, parse_bool, read_config, write_config
from .tileumbrella_shared import (
    build_anchor_selections,
    build_bias_pairs,
    find_input_file,
    format_bias_tag,
    normalize_bias_pairs_arg,
    parse_config_path,
    parse_floats,
)


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
    if "kbias" in cfg:
        defaults["kbias"] = cfg["kbias"]
    if "kbiasangle" in cfg:
        defaults["kbiasangle"] = cfg["kbiasangle"]
    if "biaspairs" in cfg:
        defaults["biaspairs"] = cfg["biaspairs"]
    if "kdistx" in cfg:
        defaults["kdistx"] = cfg["kdistx"]
    if "kdisty" in cfg:
        defaults["kdisty"] = cfg["kdisty"]
    if "kdistz" in cfg:
        defaults["kdistz"] = cfg["kdistz"]
    if "knorm" in cfg:
        defaults["knorm"] = cfg["knorm"]
    if "kdihed" in cfg:
        defaults["kdihed"] = cfg["kdihed"]
    if "krot" in cfg:
        defaults["krot"] = cfg["krot"]
    if "bias" in cfg:
        defaults["bias"] = cfg["bias"]
    if "biasangle" in cfg:
        defaults["biasangle"] = cfg["biasangle"]
    if "biasdir" in cfg:
        defaults["biasdir"] = cfg["biasdir"]
    if "flip" in cfg:
        defaults["flip"] = parse_bool(cfg["flip"])

    if defaults:
        p.set_defaults(**defaults)


def _parse_args(
    cfg: dict[str, str],
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="tileumbrella_analysis.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--capdb",
        type=str,
        default="CA.pdb",
        help="Reference trajectory (CA only)",
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
        "--biaspairs",
        type=str,
        default=None,
        help=(
            "Explicit bias/biasangle pairs as "
            "'bias:biasangle=bias:biasangle', "
            "'bias_biasangle=bias_biasangle', or expanded "
            "forms like '5.0:{90,120}={5.4,5.6}:{90,120}'. "
            "Overrides --bias/--biasangle grid generation."
        ),
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
    cfg_path = parse_config_path()
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
        cfg["biasdir"] = format_value(args.biasdir)
        if args.biasangle is not None:
            cfg["biasangle"] = format_value(args.biasangle)
        cfg["k"] = format_value(args.k)
        if args.kbias is not None:
            cfg["kbias"] = format_value(args.kbias)
        if args.kbiasangle is not None:
            cfg["kbiasangle"] = format_value(args.kbiasangle)
        if args.biaspairs is not None:
            cfg["biaspairs"] = format_value(normalize_bias_pairs_arg(str(args.biaspairs)))
        if args.kdistx is not None:
            cfg["kdistx"] = format_value(args.kdistx)
        if args.kdisty is not None:
            cfg["kdisty"] = format_value(args.kdisty)
        if args.kdistz is not None:
            cfg["kdistz"] = format_value(args.kdistz)
        if args.knorm is not None:
            cfg["knorm"] = format_value(args.knorm)
        if args.kdihed is not None:
            cfg["kdihed"] = format_value(args.kdihed)
        if args.krot is not None:
            cfg["krot"] = format_value(args.krot)
        cfg["flip"] = format_value(bool(args.flip))

        write_config(cfg_path, cfg)

    pdb_path = find_input_file(".", str(args.capdb))
    s = PDBReader(str(pdb_path)).select_CA()

    dcd_path = find_input_file(".", str(args.cadcd))
    traj = load_dcd(str(dcd_path), s)

    refsel = str(args.refsel)
    othersel = str(args.othersel)
    anchor = str(args.anchor)

    biasdir = str(args.biasdir)

    refrot1, refrot2 = parse_floats(str(args.rot), [90.0, 90.0], n_out=2)
    kinit, kbias, kdist, kcent, kangle = parse_floats(
        str(args.k),
        [500.0, 200.0],
        n_out=5,
    )

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

    asel1, asel2, aselt, bsel1, bsel2, bselt, as11, as12 = build_anchor_selections(
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

    a1 = StructureSelector(as11 + ".CA").atom_indices(s)
    a2 = StructureSelector(as12 + ".CA").atom_indices(s)

    distance_vector = traj.distance_vector(aca, bca)

    if bool(args.flip):
        angle_norm = traj.angle_norm(aca, aca1, aca2, bca, bca2, bca1)
    else:
        angle_norm = traj.angle_norm(aca, aca1, aca2, bca, bca1, bca2)

    dihedral = traj.dihedral(acat1, aca, bca, bcat1)
    angle1 = traj.angle(aca, bca, bcat1)
    angle2 = traj.angle(acat1, aca, bca)

    biasdihedral = traj.dihedral(aca, a1, a2, bca)

    biasx0 = harmonic_energy_xyz(
        distance_vector, kdistx * kilojoule / mole / nanometer**2, 0.0 * nanometer, axis="x"
    )
    biasy0 = harmonic_energy_xyz(
        distance_vector, kdisty * kilojoule / mole / nanometer**2, 0.0 * nanometer, axis="y"
    )
    biasz0 = harmonic_energy_xyz(
        distance_vector, kdistz * kilojoule / mole / nanometer**2, 0.0 * nanometer, axis="z"
    )
    biasnorm = harmonic_energy_angle(angle_norm, knorm * kilojoule / mole / radian**2, 0.0 * radian)
    biastwist = harmonic_energy_dihedral(
        dihedral, kdihed * kilojoule / mole / radian**2, 0.0 * radian
    )
    biasangle1 = harmonic_energy_angle(
        angle1, krot * kilojoule / mole / radian**2, np.radians(refrot1) * radian
    )
    biasangle2 = harmonic_energy_angle(
        angle2, krot * kilojoule / mole / radian**2, np.radians(refrot2) * radian
    )

    for biasval, biasangleval in bias_pairs:
        tag = format_bias_tag(biasval, biasangleval)

        bdir = Path(f"run_{tag}")

        if biasdir == "x":
            biasx = harmonic_energy_xyz(
                distance_vector,
                kbias * kilojoule / mole / nanometer**2,
                biasval * nanometer,
                axis="x",
            )
            biasy = biasy0
            biasz = biasz0
        elif biasdir == "y":
            biasx = biasx0
            biasy = harmonic_energy_xyz(
                distance_vector,
                kbias * kilojoule / mole / nanometer**2,
                biasval * nanometer,
                axis="y",
            )
            biasz = biasz0
        elif biasdir == "z":
            biasx = biasx0
            biasy = biasy0
            biasz = harmonic_energy_xyz(
                distance_vector,
                kbias * kilojoule / mole / nanometer**2,
                biasval * nanometer,
                axis="z",
            )
        else:
            raise SystemExit(f"ERROR: invalid biasdir {biasdir}")

        if biasangleval is not None:
            biasdih = harmonic_energy_dihedral(
                biasdihedral,
                kbiasangle * kilojoule / mole / radian**2,
                np.radians(biasangleval) * radian,
            )

        with open(str(bdir / "bias.dat"), "w") as f:
            if biasangleval is None:
                f.write(
                    "Step\tx_dist_bias[kJ/mol]\ty_bias_bias[kJ/mol]\tz_bias_bias[kJ/mol]\t"
                    "angle_bias[kJ/mol]\ttorsion_bias[kJ/mol]\trot_angle_bias[kJ/mol]\n"
                )
            else:
                f.write(
                    "Step\tx_dist_bias[kJ/mol]\ty_bias_bias[kJ/mol]\tz_bias_bias[kJ/mol]\t"
                    "angle_bias[kJ/mol]\ttorsion_bias[kJ/mol]\trot_angle_bias[kJ/mol]\t"
                    "dih_bias[kJ/mol]\n"
                )

            n = len(biasx)
            for i in range(n):
                bx = biasx[i].value_in_unit(kilojoule / mole)
                by = biasy[i].value_in_unit(kilojoule / mole)
                bz = biasz[i].value_in_unit(kilojoule / mole)
                ba = biasnorm[i].value_in_unit(kilojoule / mole)
                bt = biastwist[i].value_in_unit(kilojoule / mole)
                bra = (biasangle1[i] + biasangle2[i]).value_in_unit(kilojoule / mole)

                if biasangleval is None:
                    f.write(f"{i}\t" f"{bx}\t" f"{by}\t" f"{bz}\t" f"{ba}\t" f"{bt}\t" f"{bra}\n")
                else:
                    bdh = biasdih[i].value_in_unit(kilojoule / mole)
                    f.write(
                        f"{i}\t"
                        f"{bx}\t"
                        f"{by}\t"
                        f"{bz}\t"
                        f"{ba}\t"
                        f"{bt}\t"
                        f"{bra}\t"
                        f"{bdh}\n"
                    )

        with open(str(bdir / "geometry.dat"), "w") as f:
            if biasangleval is None:
                f.write(
                    "Step\tX_Distance[nm]\tY_Distance[nm]\tZ_Distance[nm]\t"
                    "NormalAngle[deg]\tTorsionAngle[deg]\tRotAngle1[deg]\tRotAngle2[deg]\n"
                )
            else:
                f.write(
                    "Step\tX_Distance[nm]\tY_Distance[nm]\tZ_Distance[nm]\t"
                    "NormalAngle[deg]\tTorsionAngle[deg]\tRotAngle1[deg]\tRotAngle2[deg]\t"
                    "DihedralBias[deg]\n"
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

                if biasangleval is None:
                    f.write(
                        f"{i}\t"
                        f"{gx}\t"
                        f"{gy}\t"
                        f"{gz}\t"
                        f"{gn}\t"
                        f"{gd}\t"
                        f"{ga1}\t"
                        f"{ga2}\n"
                    )
                else:
                    gdh = biasdihedral[i].value_in_unit(degrees)
                    f.write(
                        f"{i}\t"
                        f"{gx}\t"
                        f"{gy}\t"
                        f"{gz}\t"
                        f"{gn}\t"
                        f"{gd}\t"
                        f"{ga1}\t"
                        f"{ga2}\t"
                        f"{gdh}\n"
                    )

        print(f"finished {tag}")


if __name__ == "__main__":
    main()
