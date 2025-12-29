#!/bin/bash
set -euo pipefail

mode=${1:-allatom}

if [[ "${mode,,}" == "cocomo" ]]; then
   nsteps=${2:-10000000}
   maxrun=${3:-1}
   tag=CO
else
   nsteps=${2:-1250000}
   maxrun=${3:-10}
   tag=AA
fi

if [[ -n "${4:-}" && -d "$4" ]]; then
    cd -- "$4"
fi

[[ -r tag ]] && tag=$(<tag)

dir=$(pwd)

dir_esc=$(printf '%s' "$dir" | sed 's/[\/&|\\]/\\&/g')

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
#sdir_esc=$(printf '%s' "$script_dir" | sed 's/[\/&|\\]/\\&/g')

template=${script_dir}/job.tileumbrella.slurm.template

if [[ ! -r $template ]]; then
   echo "cannot find template"
   exit 1
fi


echo $maxrun > maxrun
for n in run_?.??; do
  biasval=${n#run_}
  echo 0 > "$n/last"

  sed \
    -e "s/BIAS/$biasval/g" \
    -e "s/MODE/$mode/g" \
    -e "s/BDIR/$n/g" \
    -e "s|DIR|$dir_esc|g" \
    -e "s/TAG/$tag/g" \
    -e "s/NSTEPS/$nsteps/g" \
    $template > "$n/job.prodbias.slurm"
done

