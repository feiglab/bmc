#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from cocomo import COCOMO
from mdsim import MDSim


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="tileumbrella.py",
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

    p.add_argument(
        "--bias",
        dest="biasstr",
        type=str,
        default="6.00",
        help="One or two bias values (nm or nm:degree))",
    )

    p.add_argument(
        "--run",
        dest="nrun",
        type=int,
        default=1,
        help="Production run index to write (expects restart from run-1)",
    )
    p.add_argument(
        "--nstep",
        type=int,
        default=100000,
        help="Number of MD steps",
    )
    p.add_argument(
        "--tstep",
        type=float,
        default=0.004,
        help="Timestep",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Langevin friction (1/ps)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=298.0,
        help="Temperature (K)",
    )
    p.add_argument(
        "--nout",
        type=int,
        default=10000,
        help="Output/report interval (steps)",
    )
    p.add_argument(
        "--device",
        type=int,
        default=0,
        help="OpenMM resource device index",
    )
    p.add_argument(
        "--resources",
        type=str,
        default="CUDA",
        help="OpenMM platform/resources string",
    )
    p.add_argument(
        "--dir",
        dest="bdir",
        type=Path,
        default=None,
        help="Run directory (default: run_<bias formatted to 2 decimals>)",
    )

    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()

    mode = str(args.mode).lower()

    biasstr = args.biasstr.strip()

    parts = None
    for sep in (":", "_"):
        if sep in biasstr:
            parts = [p.strip() for p in biasstr.split(sep)]
            break

    if parts is None:
        biasval = float(biasstr)
        biasangleval = None
    elif len(parts) == 2:
        biasval = float(parts[0])
        biasangleval = float(parts[1])
    else:
        raise SystemExit(
            "ERROR: --biasstr must be 'bias' or 'bias:biasangle' " "or 'bias_biasangle'"
        )

    if not (1.0 <= biasval <= 20.0):
        raise SystemExit("ERROR: bias must be in [1.0, 20.0]")

    if biasangleval is not None:
        if not (-180.0 <= biasangleval <= 180.0):
            raise SystemExit("ERROR: biasangle must be in [-180.0, 180.0]")

        tag = f"{biasval:.2f}_{biasangleval:.0f}"
    else:
        tag = f"{biasval:.2f}"

    bdir = (Path(args.bdir) if args.bdir is not None else Path(f"run_{tag}")).resolve()

    if not bdir.is_dir():
        raise SystemExit(f"ERROR: directory does not exist: {bdir}")

    if args.nrun < 0:
        raise SystemExit("ERROR: --run must be >= 0")

    last = args.nrun - 1
    restart = bdir / f"biasprod_{last}.xml"
    if not restart.is_file():
        raise SystemExit(f"ERROR: restart file does not exist: {restart}")

    sysxml = bdir / f"bias_system_{tag}.xml"
    if not sysxml.is_file():
        raise SystemExit(f"ERROR: system xml does not exist: {sysxml}")

    if mode == "cocomo":
        sim = COCOMO(xml=str(sysxml), restart=str(restart))
    else:
        sim = MDSim(xml=str(sysxml), restart=str(restart))

    sim.setup_simulation(
        resources=str(args.resources),
        device=int(args.device),
        temperature=float(args.temperature),
        tstep=float(args.tstep),
        gamma=float(args.gamma),
    )

    if biasangleval is None:
        biaslist = [
            "Umbrella_x",
            "Umbrella_y",
            "Umbrella_z",
            "Umbrella_angle_norm",
            "Umbrella_dihedral",
            "Umbrella_angle",
            "Umbrella_COM",
        ]
    else:
        biaslist = [
            "Umbrella_x",
            "Umbrella_y",
            "Umbrella_z",
            "Umbrella_angle_norm",
            "Umbrella_dihedral",
            "Umbrella_angle",
            "Umbrella_COM",
            "Umbrella_biasdih",
        ]

    nrun = int(args.nrun)
    sim.simulate(
        nstep=int(args.nstep),
        nout=int(args.nout),
        logfile=str(bdir / f"biasprod_{nrun}.log"),
        dcdfile=str(bdir / f"biasprod_{nrun}.dcd"),
        elogfile=str(bdir / f"biasprod_{nrun}.dat"),
        forcelist=biaslist,
    )
    sim.write_state(str(bdir / f"biasprod_{nrun}.xml"))


if __name__ == "__main__":
    main()
