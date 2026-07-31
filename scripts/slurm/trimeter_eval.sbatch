#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH -c 32
#SBATCH -t 04:00:00
#SBATCH --job-name=trim-eval
#SBATCH -o logs/trim-eval-%j.out
#SBATCH -e logs/trim-eval-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was invoked from,
# immune to that copy (see scripts/slurm/README.md point 6).
CHARDIFF_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export CHARDIFF_ROOT
# Usage: sbatch scripts/slurm/trimeter_eval.sbatch <ckpt> <n_per_bucket> <out.json>
set -euo pipefail
CKPT="${1:?ckpt}"; N="${2:-50}"; OUT="${3:-}"
INS_ROOT=$CHARDIFF_ROOT

source "$INS_ROOT/env.sh"

srun apptainer exec --nv $APPTAINER_BINDS $SIF bash -lc "
    set -e
    export PYTHONPATH=$GCB_ROOT
    cd $INS_ROOT
    python trimeter/eval/restore.py --ckpt $CKPT --n_per_bucket $N ${OUT:+--out $OUT}
  "
echo EVAL_JOB_END
