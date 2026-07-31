#!/bin/bash
# Stage the memmap shards to node-local storage, once per node per job (source this
# AFTER env.sh, on the GPU node, before apptainer). Prior investigation on this project
# found cross-node shared-disk reads — not the GPU — to be the multi-node throughput
# limiter, so training reads the planes from local disk when the node has room.
# Exports: GRC_DATA (loader data root), GCB_STAGE_BIND (extra apptainer bind, may be "").
# Set GCB_DATA_ROOT to stage a different data root (e.g. $GCB_DATA/folds/fold_3 for the
# 10-fold replicas); unset, it stages the flagship shards as before.
SRC_ROOT="${GCB_DATA_ROOT:-$GCB_DATA}"
# NODE_LOCAL_TMP: your cluster's per-job node-local scratch directory, if it has one
LOCAL=${NODE_LOCAL_TMP:-${TMPDIR:-/tmp}}
NEED_KB=20000000   # ~20 GB (shards are ~14 GB)
AVAIL_KB=$(df -P "$LOCAL" 2>/dev/null | awk 'NR==2{print $4}')
if [ -n "$AVAIL_KB" ] && [ "$AVAIL_KB" -gt "$NEED_KB" ]; then
  STAGE=$LOCAL/gcb-stage
  mkdir -p "$STAGE/shards"
  t0=$SECONDS
  rsync -a --delete "$SRC_ROOT/shards/v1_punct" "$SRC_ROOT/shards/bronze_punct" "$STAGE/shards/"
  echo "[stage] $(hostname): shards -> $STAGE/shards in $((SECONDS-t0))s"
  export GRC_DATA=$STAGE
  export GCB_STAGE_BIND="-B $STAGE"
else
  echo "[stage] $(hostname): no node-local space at $LOCAL (avail=${AVAIL_KB:-0}KB) — reading shared FS"
  export GRC_DATA=$SRC_ROOT
  export GCB_STAGE_BIND=""
fi
