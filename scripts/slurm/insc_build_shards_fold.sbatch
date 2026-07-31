#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH --job-name=insc-buildfold
#SBATCH -o logs/buildfold-%j.out
#SBATCH -e logs/buildfold-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was invoked from,
# immune to that copy (see scripts/slurm/README.md point 6).
CHARDIFF_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export CHARDIFF_ROOT
# THE FIX for the fold-leakage bug: build_shards.py/build_pap_shards.py were run ONCE
# with the default (unset INSC_TEST_DIGIT/VAL_DIGIT -> "3"/"4") split and the resulting
# $INS_DATA/shards/iphi_punct + pap_punct were then reused, unmodified, by ALL 10 fold
# configs -- so every fold except t3v4/t4v5 fine-tuned directly on its own held-out test
# digit's real data (confirmed: the shared shard's index has ~36k records for EVERY digit
# except 3 and 4). This script builds each fold's OWN correctly-rotated shard pair by
# setting INSC_TEST_DIGIT/VAL_DIGIT before the build, not just before eval.
# Usage: sbatch scripts/slurm/insc_build_shards_fold.sbatch <test_digit> <val_digit>
set -euo pipefail
TEST_DIGIT="${1:?usage: sbatch scripts/slurm/insc_build_shards_fold.sbatch <test_digit> <val_digit>}"
VAL_DIGIT="${2:?val_digit required}"
INS_ROOT=$CHARDIFF_ROOT

source "$INS_ROOT/env.sh"

echo "test_digit=$TEST_DIGIT val_digit=$VAL_DIGIT $(date)"

srun apptainer exec --nv $APPTAINER_BINDS $SIF bash -lc "
    set -e
    export PYTHONPATH=$GCB_ROOT
    export INSC_TEST_DIGIT=$TEST_DIGIT INSC_VAL_DIGIT=$VAL_DIGIT
    cd $INS_ROOT/insc/data
    python build_shards.py     --out \$INS_DATA/shards/iphi_punct_fold${TEST_DIGIT}
    python build_pap_shards.py --out \$INS_DATA/shards/pap_punct_fold${TEST_DIGIT}
  "
echo BUILD_SHARDS_FOLD_DONE
