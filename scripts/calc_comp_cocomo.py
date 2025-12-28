#!/usr/bin/env python3

import os
import sys
from pathlib import Path

from cocomo import ComponentType

from openmm.unit import kilojoule, mole, kelvin
import numpy as np

def main() -> None:
    types=ComponentType.read_list('component_types', dir='.')
    types['hexamer'].writeout('sasa','hexamer.surface')
    types['hexamer'].writeout('enm','hexamer.enmpairs')
    types['pentamer'].writeout('sasa','pentamer.surface')
    types['pentamer'].writeout('enm','pentamer.enmpairs')
    types['trimer'].writeout('sasa','trimer.surface')
    types['trimer'].writeout('enm','trimer.enmpairs')

if __name__ == "__main__":
    main()

