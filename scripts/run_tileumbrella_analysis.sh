#!/bin/bash
set -euo pipefail

if [[ -n "${1:-}" && -d "$1" ]]; then
    cd -- "$1"
fi

for d in run_?.??; do
    n=`echo $d | cut -d_ -f2`
    tileumbrella_analysis.py CA.pdb ca.dcd $n &
done
wait


