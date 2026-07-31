#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH --job-name=gcb-tagger-eval
#SBATCH -o logs/eval-%j.out
#SBATCH -e logs/eval-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
CHARDIFF_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export CHARDIFF_ROOT
# Usage: sbatch scripts/slurm/tagger_eval.sbatch <run_dir> <gold.conllu> [--ablations]
set -euo pipefail
RUN="${1:?usage: sbatch scripts/eval.sbatch <run_dir> <gold.conllu> [--ablations]}"
GOLD="${2:?gold conllu path}"
ABL="${3:-}"
TAGGER_ROOT=$CHARDIFF_ROOT
source $TAGGER_ROOT/env.sh

apptainer exec --nv $APPTAINER_BINDS $SIF bash -lc "
  set -e
  export PYTHONPATH=$TAGGER_ROOT:$GCB_ROOT
  cd $TAGGER_ROOT
  echo '=== default (lexicon + tag constraint) ==='
  python -m tagger.evaluate --run $RUN --gold $GOLD
  if [ -n '$ABL' ]; then
    echo '=== --no-lexicon ==='
    python -m tagger.evaluate --run $RUN --gold $GOLD --no-lexicon --out $RUN/ablation_nolex.conllu
    echo '=== --no-tag-constraint ==='
    python -m tagger.evaluate --run $RUN --gold $GOLD --no-tag-constraint --out $RUN/ablation_nocon.conllu
  fi
"
echo "EVAL FINISHED $(date)"
