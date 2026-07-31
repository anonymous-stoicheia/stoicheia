#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH -c 32
#SBATCH -t 08:00:00
#SBATCH --job-name=trim-ft2
#SBATCH -o logs/trim-ft-%j.out
#SBATCH -e logs/trim-ft-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was invoked from,
# immune to that copy (see scripts/slurm/README.md point 6).
CHARDIFF_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export CHARDIFF_ROOT
# Single-GPU trimeter restoration finetune -- the corpus (23.5k lines) is far smaller than
# the PHI/papyri corpus, so this doesn't need the 4-node torchrun setup insc_finetune_fold.sbatch
# uses; plain python, one GPU, matches insc_eval.sbatch's single-GPU launch pattern.
# Usage: sbatch scripts/slurm/trimeter_finetune.sbatch [config]
set -euo pipefail
CONFIG="${1:-configs/trimeter/finetune.json}"
INS_ROOT=$CHARDIFF_ROOT

source "$INS_ROOT/env.sh"

echo "config=$CONFIG $(date)"

srun apptainer exec --nv $APPTAINER_BINDS $SIF bash -lc "
    set -e
    export PYTHONPATH=$GCB_ROOT
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd $INS_ROOT
    python trimeter/train/finetune.py --config $INS_ROOT/$CONFIG
  "
echo FT_JOB_END
