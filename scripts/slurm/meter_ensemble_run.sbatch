#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH --job-name=meter-ensemble
#SBATCH -o logs/ensemble-%j.out
#SBATCH -e logs/ensemble-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
CHARDIFF_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export CHARDIFF_ROOT
# Formal, logged run of the macronization probability-averaging ensemble
# (previously written but never actually executed/logged).
set -euo pipefail
MTR=$CHARDIFF_ROOT
GCB_ROOT=$CHARDIFF_ROOT
source $GCB_ROOT/env.sh
source $MTR/env.sh
R=$GCB_DATA/runs

apptainer exec --nv $APPTAINER_BINDS $SIF bash -c "
  set -e
  export PYTHONPATH=$MTR:$GCB_ROOT
  export MACRONIZER_SRC=$MACRONIZER_SRC
  cd $MTR
  python -m meter.ensemble \
    mac_pilot=$R/meter_mac_pilot/norma_pred_probs.json \
    mac_v2=$R/meter_mac_v2/norma_pred_probs.json \
    mac_v2_s1=$R/meter_mac_v2_s1/norma_pred_probs.json \
    mac_v2_s2=$R/meter_mac_v2_s2/norma_pred_probs.json \
    mac_cur=$R/meter_mac_cur/norma_pred_probs.json \
    joint_pilot=$R/meter_joint_pilot/norma_pred_probs.json
"
echo ENSEMBLE_DONE
