#!/usr/bin/env python

# usage:
#
#  equi.tileumbrella.allatom.py [args]
#      args:   allatom      : mode 'allatom' or 'cocomo'
#              setup        : setup directory
#              equi         : equi directory
#              pdb          : needed for restraints
#              0            : device
#

import sys
from pathlib import Path

from mdsim import MDSim, PDBReader
from cocomo import COCOMO

def _argv(i: int, default: str) -> str:
    """Return argv[i] if present and non-empty; otherwise default."""
    return sys.argv[i] if len(sys.argv) > i and str(sys.argv[i]).strip() != "" else default

def _as_int(name: str, s: str) -> int:
    try:
        return int(s)
    except Exception as e:
        raise SystemExit(f"ERROR: {name} must be an int, got {s!r}") from e

def main() -> None:
    default_sdir = "setup"
    default_edir = "equi"
    default_device = "0"

    resources = 'CUDA'

    mode = _argv(1, "allatom")
    sdir = Path(_argv(2, default_sdir)).expanduser().resolve()
    edir = Path(_argv(3, default_edir)).expanduser().resolve()

    if mode.lower() == 'allatom':
        refpdb = Path(_argv(4, "dimer.solvated.pdb"))
    elif mode.lower() == 'cocomo':
        refpdb = Path(_argv(4, "dimer.protein.pdb"))
    else:
        raise SystemExit(f"invalid mode {mode}") 

    device = _as_int("device", _argv(5, default_device))

    edir.mkdir(parents=True, exist_ok=True)

    if mode.lower() == 'cocomo':
        sim=COCOMO(PDBReader(str(sdir / refpdb)).topology(),
                   xml = str(sdir / "system.xml"),
                   restart = str(sdir / "initial.xml"))
    
        sim.set_position_restraint(selection="name CA",k=10.0)
        sim.setup_simulation(resources=resources,tstep=0.03)

        sim.minimize(nstep=1000)
        print(f"minimized energy: {sim.get_potentialEnergy()}")

        sim.set_velocities()
        sim.simulate(nstep=1000)
        print(f"energy after simulation: {sim.get_potentialEnergy()}")

    elif mode.lower() == 'allatom':
        sim=MDSim(pdb = str(sdir / refpdb),
                  xml = str(sdir / "system.xml"),
                  restart = str(sdir / "initial.xml"))

        sim.set_position_restraint(selection="protein and (name CA or name CB)")

        sim.setup_simulation(resources=resources, device=device, temperature=5, tstep=0.001)

        minsteps=1000
        sim.minimize(nstep=minsteps)
        sim.write_state(str(edir / "equi_min.xml"))
        print(f"minimized for {minsteps} steps")

        equi_schedule=[ [5,10000], [10,10000], [20,10000], [50,10000], [100,10000], 
                        [200,10000], [250,10000], [298,20000] ] 

        for e in equi_schedule:
            pos=sim.get_positions()
            sim.setup_simulation(resources='CUDA', device=device, temperature=e[0], gamma=1.0, tstep=0.001, 
                                 positions=pos, resetvelocities=True)
            sim.simulate(nstep=e[1],logfile = str(edir / f"equi_{e[0]}.log"))
            sim.write_state(str(edir / f"equi_{e[0]}.xml"))
            print(f'{e[1]} steps at {e[0]}K')

        pos=sim.get_positions()
        vel=sim.get_velocities()
        sim.set_barostat(pressure=1,temperature=298)
        sim.setup_simulation(resources='CUDA', device=device, gamma=0.01, tstep=0.002, positions=pos, velocities=vel)
        sim.simulate(nstep=10000, logfile = str(edir / "equi_298npt.log"), dcdfile = str(edir / "equi_298npt.dcd"))
        print(f'10000 steps at 298K/1bar NPT')

    sim.write_state(str(edir / "equi_final.xml"))

if __name__ == "__main__":
    main()

