#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from cocomo import COCOMO
from mdsim import MDSim, PDBReader


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="equi_tileumbrella.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode: allow --mode plus shortcut flags --allatom/--cocomo.
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
        "--setup",
        type=Path,
        default=Path("setup"),
        help="Setup directory",
    )
    p.add_argument(
        "--equi",
        type=Path,
        default=Path("equi"),
        help="Equilibration output directory",
    )
    p.add_argument(
        "--pdb",
        type=Path,
        default=None,
        help=(
            "Reference PDB (used for restraints). Defaults depend on mode: "
            "allatom->dimer.solvated.pdb, cocomo->dimer.protein.pdb"
        ),
    )
    p.add_argument(
        "--device",
        type=int,
        default=0,
        help="Device index (OpenMM platform device id)",
    )
    p.add_argument(
        "--resources",
        type=str,
        default="CUDA",
        help="OpenMM platform/resources string",
    )

    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()

    mode = str(args.mode).lower()
    sdir = Path(args.setup).expanduser().resolve()
    edir = Path(args.equi).expanduser().resolve()
    resources = str(args.resources)
    device = int(args.device)

    if args.pdb is None:
        refpdb = Path("dimer.solvated.pdb") if mode == "allatom" else Path(
            "dimer.protein.pdb"
        )
    else:
        refpdb = Path(args.pdb)

    edir.mkdir(parents=True, exist_ok=True)

    if mode == "cocomo":
        sim = COCOMO(
            PDBReader(str(sdir / refpdb)).topology(),
            xml=str(sdir / "system.xml"),
            restart=str(sdir / "initial.xml"),
        )

        sim.set_position_restraint(selection="name CA", k=10.0)
        sim.setup_simulation(resources=resources, tstep=0.03)

        sim.minimize(nstep=1000)
        print(f"minimized energy: {sim.get_potentialEnergy()}")

        sim.set_velocities()
        sim.simulate(nstep=1000)
        print(f"energy after simulation: {sim.get_potentialEnergy()}")

    elif mode == "allatom":
        sim = MDSim(
            pdb=str(sdir / refpdb),
            xml=str(sdir / "system.xml"),
            restart=str(sdir / "initial.xml"),
        )

        sim.set_position_restraint(selection="protein and (name CA or name CB)")
        sim.setup_simulation(
            resources=resources, device=device, temperature=5, tstep=0.001
        )

        minsteps = 1000
        sim.minimize(nstep=minsteps)
        sim.write_state(str(edir / "equi_min.xml"))
        print(f"minimized for {minsteps} steps")

        equi_schedule = [
            [5, 10000],
            [10, 10000],
            [20, 10000],
            [50, 10000],
            [100, 10000],
            [200, 10000],
            [250, 10000],
            [298, 20000],
        ]

        for temp_k, nsteps in equi_schedule:
            pos = sim.get_positions()
            sim.setup_simulation(
                resources=resources,
                device=device,
                temperature=temp_k,
                gamma=1.0,
                tstep=0.001,
                positions=pos,
                resetvelocities=True,
            )
            sim.simulate(nstep=nsteps, logfile=str(edir / f"equi_{temp_k}.log"))
            sim.write_state(str(edir / f"equi_{temp_k}.xml"))
            print(f"{nsteps} steps at {temp_k}K")

        pos = sim.get_positions()
        vel = sim.get_velocities()
        sim.set_barostat(pressure=1, temperature=298)
        sim.setup_simulation(
            resources=resources,
            device=device,
            gamma=0.01,
            tstep=0.002,
            positions=pos,
            velocities=vel,
        )
        sim.simulate(
            nstep=10000,
            logfile=str(edir / "equi_298npt.log"),
            dcdfile=str(edir / "equi_298npt.dcd"),
        )
        print("10000 steps at 298K/1bar NPT")
    else:
        raise SystemExit(f"invalid mode {mode!r}")

    sim.write_state(str(edir / "equi_final.xml"))


if __name__ == "__main__":
    main()

