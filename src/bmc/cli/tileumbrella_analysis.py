#!/usr/bin/env python3

from __future__ import annotations

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

from .tile_config import read_config
from .tileumbrella_shared import (
    build_anchor_selections,
    build_bias_pairs,
    find_input_file,
    format_bias_tag,
    parse_args,
    parse_config_path,
    parse_floats,
    write_args_config,
)

_HELP_OPTIONS = (
    "capdb",
    "cadcd",
    "refsel",
    "othersel",
    "anchor",
    "rot",
    "bias",
    "biasdir",
    "biasangle",
    "k",
    "kbias",
    "kbiasangle",
    "biaspairs",
    "kdistx",
    "kdisty",
    "kdistz",
    "kdist",
    "knorm",
    "kdihed",
    "krot",
    "flip",
    "normcap",
    "distcap",
    "config",
    "write_config",
)


def main() -> None:
    cfg_path = parse_config_path()
    cfg = read_config(cfg_path)

    args = parse_args(cfg, _HELP_OPTIONS, prog="tileumbrella_analysis.py")

    cfg_path = Path(args.config)
    if bool(args.write_config):
        write_args_config(cfg_path, cfg, args)

    pdb_path = find_input_file(".", str(args.capdb))
    s = PDBReader(str(pdb_path)).select_CA()

    dcd_path = find_input_file(".", str(args.cadcd))
    traj = load_dcd(str(dcd_path), s)

    refsel = str(args.refsel)
    othersel = str(args.othersel)
    anchor = str(args.anchor)

    biasdir = str(args.biasdir)

    refrot1, refrot2 = parse_floats(str(args.rot), [90.0, 90.0], n_out=2)
    _, kbias, kdist, _, kangle = parse_floats(
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

    # if args.normcap is not None:
    #    normcap = float(args.normcap)
    # else:
    #    normcap = None

    # if args.distcap is not None:
    #    distcap = float(args.distcap)
    #    if args.kdist is not None:
    #        kdist = float(args.kdist)
    # else:
    #    distcap = None

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

    asel1, asel2, aselt, bsel1, bsel2, bselt, as11, as12, bselc = build_anchor_selections(
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

    # bcac = StructureSelector(bselc + ".CA").atom_indices(s)

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
        distance_vector,
        kdistx * kilojoule / mole / nanometer**2,
        0.0 * nanometer,
        axis="x",
    )
    biasy0 = harmonic_energy_xyz(
        distance_vector,
        kdisty * kilojoule / mole / nanometer**2,
        0.0 * nanometer,
        axis="y",
    )
    biasz0 = harmonic_energy_xyz(
        distance_vector,
        kdistz * kilojoule / mole / nanometer**2,
        0.0 * nanometer,
        axis="z",
    )
    biasnorm = harmonic_energy_angle(
        angle_norm,
        knorm * kilojoule / mole / radian**2,
        0.0 * radian,
    )
    biastwist = harmonic_energy_dihedral(
        dihedral,
        kdihed * kilojoule / mole / radian**2,
        0.0 * radian,
    )
    biasangle1 = harmonic_energy_angle(
        angle1,
        krot * kilojoule / mole / radian**2,
        np.radians(refrot1) * radian,
    )
    biasangle2 = harmonic_energy_angle(
        angle2,
        krot * kilojoule / mole / radian**2,
        np.radians(refrot2) * radian,
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

        with open(str(bdir / "bias.dat"), "w", encoding="utf-8") as handle:
            if biasangleval is None:
                handle.write(
                    "Step\tx_dist_bias[kJ/mol]\ty_bias_bias[kJ/mol]\t"
                    "z_bias_bias[kJ/mol]\tangle_bias[kJ/mol]\t"
                    "torsion_bias[kJ/mol]\trot_angle_bias[kJ/mol]\n"
                )
            else:
                handle.write(
                    "Step\tx_dist_bias[kJ/mol]\ty_bias_bias[kJ/mol]\t"
                    "z_bias_bias[kJ/mol]\tangle_bias[kJ/mol]\t"
                    "torsion_bias[kJ/mol]\trot_angle_bias[kJ/mol]\t"
                    "dih_bias[kJ/mol]\n"
                )

            nframe = len(biasx)
            for i in range(nframe):
                bx = biasx[i].value_in_unit(kilojoule / mole)
                by = biasy[i].value_in_unit(kilojoule / mole)
                bz = biasz[i].value_in_unit(kilojoule / mole)
                ba = biasnorm[i].value_in_unit(kilojoule / mole)
                bt = biastwist[i].value_in_unit(kilojoule / mole)
                bra = (biasangle1[i] + biasangle2[i]).value_in_unit(kilojoule / mole)

                if biasangleval is None:
                    handle.write(f"{i}\t{bx}\t{by}\t{bz}\t{ba}\t{bt}\t{bra}\n")
                else:
                    bdh = biasdih[i].value_in_unit(kilojoule / mole)
                    handle.write(f"{i}\t{bx}\t{by}\t{bz}\t{ba}\t{bt}\t" f"{bra}\t{bdh}\n")

        with open(str(bdir / "geometry.dat"), "w", encoding="utf-8") as handle:
            if biasangleval is None:
                handle.write(
                    "Step\tX_Distance[nm]\tY_Distance[nm]\tZ_Distance[nm]\t"
                    "NormalAngle[deg]\tTorsionAngle[deg]\tRotAngle1[deg]\t"
                    "RotAngle2[deg]\n"
                )
            else:
                handle.write(
                    "Step\tX_Distance[nm]\tY_Distance[nm]\tZ_Distance[nm]\t"
                    "NormalAngle[deg]\tTorsionAngle[deg]\tRotAngle1[deg]\t"
                    "RotAngle2[deg]\tDihedralBias[deg]\n"
                )

            nframe = len(distance_vector)
            for i in range(nframe):
                gx = distance_vector[i][0].value_in_unit(nanometer)
                gy = distance_vector[i][1].value_in_unit(nanometer)
                gz = distance_vector[i][2].value_in_unit(nanometer)
                gn = angle_norm[i].value_in_unit(degrees)
                gd = dihedral[i].value_in_unit(degrees)
                ga1 = angle1[i].value_in_unit(degrees)
                ga2 = angle2[i].value_in_unit(degrees)

                if biasangleval is None:
                    handle.write(f"{i}\t{gx}\t{gy}\t{gz}\t{gn}\t{gd}\t" f"{ga1}\t{ga2}\n")
                else:
                    gdh = biasdihedral[i].value_in_unit(degrees)
                    handle.write(f"{i}\t{gx}\t{gy}\t{gz}\t{gn}\t{gd}\t" f"{ga1}\t{ga2}\t{gdh}\n")

        print(f"finished {tag}")


if __name__ == "__main__":
    main()
