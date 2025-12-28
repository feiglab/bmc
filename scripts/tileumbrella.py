#!/usr/bin/env python

import sys
import numpy as np
from pathlib import Path

from mdsim import MDSim, PDBReader, StructureSelector
from cocomo import Assembly, COCOMO

def _argv(i: int, default: str) -> str:
    """Return argv[i] if present and non-empty; otherwise default."""
    return sys.argv[i] if len(sys.argv) > i and str(sys.argv[i]).strip() != "" else default

mode=_argv(1,"allatom")
biasval=_argv(2,"6.00")
nrun=int(_argv(3,"1"))
nstep=int(_argv(4,"100000"))
tstep=float(_argv(5,"0.004"))
gamma=float(_argv(6,"0.1"))

temperature=298

nout=10000

resources='CUDA'

if not biasval or float(biasval)<1.0 or float(biasval)>20.0:
    print("run with biasval (6.00) as argument")
    exit()

bdir=Path(f"run_{biasval}")

if not bdir.exists():
    print(f"directory {str(bdir)} does not exist")
    exit()

last=nrun-1 
restart=bdir / f"biasprod_{last}.xml"
if not restart.exists():
    print(f"restart file {str(restart)} does not exist")
    exit()

if simmode.lower() == 'cocomo':
    sim=COCOMO(xml=str(bdir / f"bias_system_{biasval}.xml"),restart=str(restart))
else:
    sim=MDSim(xml=str(bdir / f"bias_system_{biasval}.xml"),restart=str(restart))

sim.setup_simulation(resources=resources, temperature=temperature, tstep=tstep, gamma=gamma)

biaslist=['Umbrella_x', 'Umbrella_y', 'Umbrella_z', 'Umbrella_angle_norm', 
          'Umbrella_dihedral', 'Umbrella_angle', 'Umbrella_COM']
sim.simulate(nstep=nstep,nout=nout,
           logfile=str(bdir / f"biasprod_{nrun}.log"),
           dcdfile=str(bdir / f"biasprod_{nrun}.dcd"),
           elogfile=str(bdir / f"biasprod_{nrun}.dat"),forcelist=biaslist)

sim.write_state(str(bdir / f"biasprod_{nrun}.xml"))

