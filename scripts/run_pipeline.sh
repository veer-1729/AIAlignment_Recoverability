#!/usr/bin/env bash
# Run the Stage-1 pipeline for one config across both arms.
#
# Parallelism is by PROCESS, never by thread: ALFWorld/TextWorld keep module-level
# global state, and constructing envs on multiple threads corrupts it -- measured
# at 36/36 branches single-threaded vs 3 branches + 11 errors at 4 threads. Each
# shard below is an independent process writing its own JSONL file.
#
# Usage:  scripts/run_pipeline.sh CONFIG [NUM_SHARDS]
set -euo pipefail

CONFIG="${1:?usage: run_pipeline.sh CONFIG [NUM_SHARDS]}"
SHARDS="${2:-4}"
ARMS="${ARMS:-A_replication B_montecarlo}"

run_sharded () {  # $1=script  $2=arm
  local pids=()
  for s in $(seq 0 $((SHARDS - 1))); do
    python "$1" --config "$CONFIG" --arm "$2" --shard "$s" --num-shards "$SHARDS" \
      > "/tmp/$(basename "$1" .py).$2.s$s.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  grep -h "^\[0" /tmp/$(basename "$1" .py)."$2".s*.log || true
  return $rc
}

for arm in $ARMS; do
  echo "########## ARM $arm ##########"
  run_sharded scripts/01_run_rollouts.py "$arm"
  python scripts/02_select_checkpoints.py --config "$CONFIG" --arm "$arm"
  run_sharded scripts/03_run_branches.py "$arm"
done

echo "########## VALIDATE ##########"
python scripts/04_validate.py --config "$CONFIG"
echo "########## ANALYZE ##########"
python scripts/05_analyze.py --config "$CONFIG"
