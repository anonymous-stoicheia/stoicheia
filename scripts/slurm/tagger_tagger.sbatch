#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 1
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 1
#SBATCH -c 64
#SBATCH -t 04:00:00
#SBATCH --job-name=gcb-tagger
#SBATCH -o logs/tagger-%j.out
#SBATCH -e logs/tagger-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
CHARDIFF_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export CHARDIFF_ROOT
# Usage: sbatch scripts/slurm/tagger_tagger.sbatch configs/tagger/tagger_fold0_pilot.json [--fold N] [--ckpt PATH]
set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/tagger.sbatch <config.json> [extra args]}"
shift || true
EXTRA="$*"
TAGGER_ROOT=$CHARDIFF_ROOT
source $TAGGER_ROOT/env.sh
mkdir -p "$TAGGER_ROOT/logs"
NGPU=${SLURM_GPUS_PER_NODE:-4}
export MASTER_ADDR=localhost MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))
echo "gpus=$NGPU config=$CONFIG extra='$EXTRA' $(date)"

apptainer exec --nv $APPTAINER_BINDS $SIF bash -lc "
  set -e
  export PYTHONPATH=$TAGGER_ROOT:$GCB_ROOT
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=16
  cd $TAGGER_ROOT
  torchrun --nproc_per_node=$NGPU --master_port=\$MASTER_PORT \
    -m tagger.train --config $CONFIG $EXTRA
"
echo "TAGGER RUN FINISHED $(date)"
