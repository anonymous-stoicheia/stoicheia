#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 4
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 1
#SBATCH -c 256
#SBATCH -t 06:00:00
#SBATCH --job-name=insc-ftfold
#SBATCH -o logs/ftfold-%j.out
#SBATCH -e logs/ftfold-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
CHARDIFF_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export CHARDIFF_ROOT
# Rotating-fold restoration fine-tune: same recipe as insc_finetune_v2.sbatch, but the
# PHI/TM last-digit test/val split is parameterized instead of fixed at 3/4, so a full
# 10-fold rotation gives every inscription/papyrus in the corpus a model that provably
# never saw it during fine-tuning (not just the ones ending in digit 3).
# Usage: sbatch scripts/slurm/insc_finetune_fold.sbatch <config> <test_digit> <val_digit>
set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/slurm/insc_finetune_fold.sbatch <config> <test_digit> <val_digit>}"
TEST_DIGIT="${2:?test_digit required}"
VAL_DIGIT="${3:?val_digit required}"
INS_ROOT=$CHARDIFF_ROOT

# Sourced ONCE here, at the top-level sbatch shell -- not inside the per-node srun'd shell
# below. Re-sourcing env.sh inside a remote `bash -lc` on each freshly srun'd node depends on
# that login shell's own startup chain completing cleanly; on this cluster it doesn't (see
# the "/var/lib/slurm/env.sh: No such file" noise in a failed run's log) and $SIF/$APPTAINER_
# BINDS silently never get set, so apptainer misparses the following "bash" as the image path
# and dies in ~seconds -- a real failure mode hit repeatedly in practice. Resolving them here
# instead means every node gets the exact same, already-correct values baked into its own
# command line, no remote re-sourcing required.
source "$INS_ROOT/env.sh"

MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$MASTER MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))
NGPU=${SLURM_GPUS_PER_NODE:-4}
echo "nodes=$SLURM_NNODES master=$MASTER config=$CONFIG test_digit=$TEST_DIGIT val_digit=$VAL_DIGIT $(date)"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  apptainer exec --nv $APPTAINER_BINDS $SIF bash -lc "
    set -e
    export PYTHONPATH=$GCB_ROOT
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=16
    export INSC_TEST_DIGIT=$TEST_DIGIT INSC_VAL_DIGIT=$VAL_DIGIT
    cd $INS_ROOT
    torchrun \
      --nnodes=$SLURM_NNODES --nproc_per_node=$NGPU \
      --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      --node_rank=\$SLURM_PROCID \
      insc/train/finetune_v2.py --config $INS_ROOT/$CONFIG
  "
echo FT_JOB_END
