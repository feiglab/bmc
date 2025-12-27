#!/bin/bash
set -euo pipefail

nsteps=${1:-1250000}
maxrun=${2:-10}

if [[ -n "${3:-}" && -d "$3" ]]; then
    cd -- "$3"
fi


tag=US
[[ -r tag ]] && tag=$(<tag)

dir=$(pwd)

dir_esc=$(printf '%s' "$dir" | sed 's/[\/&|\\]/\\&/g')

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
sdir_esc=$(printf '%s' "$script_dir" | sed 's/[\/&|\\]/\\&/g')

template=${script_dir}/job.bmctileumbrella.slurm.template

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
    -e "s/BDIR/$n/g" \
    -e "s|DIR|$dir_esc|g" \
    -e "s/TAG/$tag/g" \
    -e "s/NSTEPS/$nsteps/g" \
    -e "s|SCRIPT|$sdir_esc|g" \
    $template > "$n/job.prodbias.slurm"
done

