#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from cocomo import COCOMO
from mdsim import MDSim, PDBReader

from .tile_config import read_config
from .tileumbrella_shared import parse_args, parse_config_path, write_args_config

_HELP_OPTIONS = (
    "mode",
    "setup",
    "equi",
    "refpdb",
    "device",
    "resources",
    "config",
    "write_config",
)


def main() -> None:
    cfg_path = parse_config_path()
    cfg = read_config(cfg_path)

    args = parse_args(cfg, _HELP_OPTIONS, prog="equi_tileumbrella.py")

    mode = str(args.mode).lower()
    sdir = Path(args.setup).expanduser().resolve()
    edir = Path(args.equi).expanduser().resolve()
    resources = str(args.resources)
    device = int(args.device)

    if args.refpdb is None:
        if mode == "allatom":
            refpdb = Path("dimer.solvated.pdb")
        else:
            refpdb = Path("dimer.protein.pdb")
    else:
        refpdb = Path(args.refpdb)

    cfg_path = Path(args.config)
    if bool(args.write_config):
        write_args_config(
            cfg_path,
            cfg,
            args,
            overrides={
                "mode": mode,
                "refpdb": refpdb,
            },
        )

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
            resources=resources,
            device=device,
            temperature=5,
            tstep=0.001,
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
