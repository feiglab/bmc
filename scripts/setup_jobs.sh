#!/bin/bash
set -euo pipefail

mode_arg=${1:-}
nsteps_arg=${2:-}
maxrun_arg=${3:-}
workdir_arg=${4:-}

if [[ -n "$workdir_arg" && -d "$workdir_arg" ]]; then
    cd -- "$workdir_arg"
fi

default_mode=allatom
if [[ -r config ]]; then
    cfg_mode=$(
        awk '$1=="mode" && NF>=2 { print $2; exit }' config
    )
    if [[ -n "${cfg_mode:-}" ]]; then
        default_mode=$cfg_mode
    fi
fi

mode=${mode_arg:-$default_mode}

if [[ "${mode,,}" == "cocomo" ]]; then
   nsteps=${nsteps_arg:-10000000}
   maxrun=${maxrun_arg:-1}
   tag=CO
else
   nsteps=${nsteps_arg:-1250000}
   maxrun=${maxrun_arg:-10}
   tag=AA
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

