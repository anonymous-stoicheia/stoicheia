#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 32              # 32 nodes x 4 GH200 = 128 GPUs (same recipe as final.sbatch)
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 1
#SBATCH -c 256
#SBATCH -t 08:00:00
#SBATCH --job-name=gcb-fold
#SBATCH -o logs/fold-%j.out
#SBATCH -e logs/fold-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
CHARDIFF_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export CHARDIFF_ROOT
# One 10-fold replica training job. Usage: sbatch scripts/slurm/pretrain_fold.sbatch <fold 0-9>
# Identical to final.sbatch except the data root points at the fold's decontaminated
# shards (via GCB_DATA_ROOT -> stage_shards.sh) and out_dir at runs/gcb_fold_<k>.
# Submit as a chain via scripts/pretrain/launch_fold_chain.sh — resumes from out_dir/last.pt.
set -euo pipefail
FOLD="${1:?usage: sbatch scripts/fold.sbatch <fold 0-9>}"
source $CHARDIFF_ROOT/env.sh
mkdir -p "$GCB_ROOT/logs"

CONFIG=$GCB_ROOT/configs/pretrain/folds/fold_${FOLD}.json
FOLD_ROOT=$GCB_DATA/folds/fold_$FOLD
[ -f "$CONFIG" ] || { echo "missing $CONFIG"; exit 1; }
[ -f "$FOLD_ROOT/shards/v1_punct/index.parquet" ] || { echo "missing fold shards at $FOLD_ROOT"; exit 1; }

MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$MASTER MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))  # avoid fixed-port collisions
NGPU=${SLURM_GPUS_PER_NODE:-4}
echo "fold=$FOLD nodes=$SLURM_NNODES master=$MASTER gpus/node=$NGPU config=$CONFIG $(date)"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -lc "
  source $GCB_ROOT/env.sh
  export GCB_DATA_ROOT=$FOLD_ROOT
  source $GCB_ROOT/scripts/pretrain/stage_shards.sh
  apptainer exec --nv \$APPTAINER_BINDS \$GCB_STAGE_BIND \$SIF bash -lc '
    set -e
    export PYTHONPATH=$GCB_ROOT
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
    cd $GCB_ROOT
    torchrun \
      --nnodes=$SLURM_NNODES --nproc_per_node=$NGPU \
      --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      --node_rank=\$SLURM_PROCID \
      -m train.train --config $CONFIG
  '
"
echo "FOLD $FOLD JOB FINISHED (or checkpointed out at wall-time limit — chain resumes)"
