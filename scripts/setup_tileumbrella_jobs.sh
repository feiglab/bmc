#!/bin/bash
set -euo pipefail

queue_arg=${1:-}
mode_arg=${2:-}
nsteps_arg=${3:-}
maxrun_arg=${4:-}
workdir_arg=${5:-}

if [[ -n "$workdir_arg" && -d "$workdir_arg" ]]; then
    cd -- "$workdir_arg"
fi

# --- read defaults from config (if present) ---
default_mode=allatom
default_queue=all
cfg_nsteps=""
cfg_maxrun=""
cfg_queue=""

if [[ -r config ]]; then
    cfg_mode=$(
        awk '$1=="mode"   && NF>=2 { print $2; exit }' config
    )
    cfg_nsteps=$(
        awk '$1=="nsteps" && NF>=2 { print $2; exit }' config
    )
    cfg_maxrun=$(
        awk '$1=="maxrun" && NF>=2 { print $2; exit }' config
    )
    cfg_queue=$(
        awk '$1=="queue" && NF>=2 { print $2; exit }' config
    )
    if [[ -n "${cfg_mode:-}" ]]; then
        default_mode=$cfg_mode
    fi
fi

[[ -n "${cfg_queue:-}" ]] && default_queue=$cfg_queue
queue=${queue_arg:-$default_queue}

mode=${mode_arg:-$default_mode}

# --- mode-dependent defaults, overridden by config values if present ---
if [[ "${mode,,}" == "cocomo" ]]; then
   default_nsteps=10000000
   default_maxrun=1
   tag=CO
else
   default_nsteps=1250000
   default_maxrun=10
   tag=AA
fi

[[ -n "${cfg_nsteps:-}" ]] && default_nsteps=$cfg_nsteps
[[ -n "${cfg_maxrun:-}" ]] && default_maxrun=$cfg_maxrun

nsteps=${nsteps_arg:-$default_nsteps}
maxrun=${maxrun_arg:-$default_maxrun}

[[ -r tag ]] && tag=$(<tag)

dir=$(pwd)
dir_esc=$(printf '%s' "$dir" | sed 's/[\/&|\\]/\\&/g')

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
template=${script_dir}/job.tileumbrella.slurm.template

if [[ ! -r $template ]]; then
   echo "cannot find template"
   exit 1
fi

echo "$maxrun" > maxrun
for n in run_?.??; do
  biasval=${n#run_}
  echo 0 > "$n/last"

  sed \
    -e "s/BIAS/$biasval/g" \
    -e "s/MODE/$mode/g" \
    -e "s/BDIR/$n/g" \
    -e "s|DIR|$dir_esc|g" \
    -e "s/TAG/$tag/g" \
    -e "s/QUEUE/$queue/g" \
    -e "s/NSTEPS/$nsteps/g" \
    "$template" > "$n/job.prodbias.slurm"
done

