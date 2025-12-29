#!/bin/bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ -n "${1:-}" && -d "$1" ]]; then
    cd -- "$1"
fi

for d in run_?.??; do
    n=`echo $d | cut -d_ -f2`
    $script_dir/analysis.py CA.pdb ca.dcd $n &
done
wait


