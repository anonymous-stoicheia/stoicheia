# Stoicheia environment. Source this before running anything.
# Set the two roots for your system; everything else derives from them.

# Writable data root: corpora, shards, checkpoints, eval outputs.
export CHARDIFF_DATA="${CHARDIFF_DATA:?set CHARDIFF_DATA to a writable data root}"
# This repository's checkout.
export CHARDIFF_ROOT="${CHARDIFF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

export INS_DATA="$CHARDIFF_DATA/insc_data"          # documentary corpora + runs
export SYN_DATA="$CHARDIFF_DATA/parser_data"        # treebank shards + parser runs
export AGD_DATA="${AGD_DATA:-$CHARDIFF_DATA/agd}"   # normalized source corpora
export MACRONIZER_SRC="${MACRONIZER_SRC:-$CHARDIFF_DATA/macron_data}"  # meter silver data

export PYTHONPATH="$CHARDIFF_ROOT:${PYTHONPATH:-}"
# Optional: containerized execution. Leave unset to run in a plain venv.
#   export SIF=/path/to/container.sif
#   export APPTAINER_BINDS="-B $CHARDIFF_DATA"
