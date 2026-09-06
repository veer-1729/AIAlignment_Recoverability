#!/usr/bin/env bash
# Backfill missing checkpoints in bounded chunks, so an unidentified leak cannot
# accumulate far enough to stop the run.
#
# Two runs have now been killed by a filesystem filling up, and the cause is NOT
# identified. df reports ~170 GB used on a volume where du -x finds ~27 GB of
# files, and the gap grows at 1-2 GB/min while shards run. It is not
# unlinked-but-open files: lsof +L1 as root finds 0.1 GB system-wide and none on
# that device, and every shard process shows zero deleted descriptors.
#
# So this does not try to find the cause. It bounds the exposure instead. Each
# round runs every shard for a fixed number of checkpoints, waits for all of them
# to exit, and starts again; --only-missing means a restart re-reads what is
# already on disk and costs nothing but process startup. If the space is
# reclaimed when processes exit, an unbounded problem becomes a bounded one and
# the run completes in chunks.
#
# THAT CONDITION IS NOT ASSUMED. df is printed before each round and immediately
# after every shard has exited. If free space does not return between rounds,
# chunking is not the answer and the loop's no-progress check stops it rather
# than burning startup cost forever. Read those two numbers before trusting a
# long unattended run to this.
#
# Usage:  scripts/run_backfill_chunked.sh CONFIG [ARM] [NUM_SHARDS]
# Env:    CHUNK (checkpoints per shard per round, default 40)
#         MIN_FREE_GB (per-checkpoint guard, default 5)
#         MAX_ROUNDS (safety stop, default 40)
#         LOGDIR (default /data/cnc/logs -- keep logs OFF the root filesystem)

set -uo pipefail

CONFIG="${1:?usage: run_backfill_chunked.sh CONFIG [ARM] [NUM_SHARDS]}"
ARM="${2:-B_montecarlo}"
SHARDS="${3:-10}"
CHUNK="${CHUNK:-40}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
MAX_ROUNDS="${MAX_ROUNDS:-40}"
LOGDIR="${LOGDIR:-/data/cnc/logs}"

mkdir -p "$LOGDIR"
RUNDIR="$(python -c "
import sys; sys.path.insert(0, 'src')
from cnc import experiment
print(experiment.run_dir(experiment.load_config(sys.argv[1])).base)
" "$CONFIG")" || { echo "could not resolve the run dir from $CONFIG"; exit 1; }
echo "run dir: $RUNDIR"

prev_remaining=-1

for round in $(seq 1 "$MAX_ROUNDS"); do
  echo "########## ROUND $round  (chunk=$CHUNK, shards=$SHARDS) ##########"
  echo "-- df before --"; df -h "$RUNDIR" / | sed 's/^/   /'

  # Every shard starts together. Each computes its own missing list from disk at
  # startup, so a stagger longer than one checkpoint would let later shards see a
  # shorter list, shift the stride underneath them, and run some checkpoints
  # twice while running others not at all.
  pids=()
  for s in $(seq 0 $((SHARDS - 1))); do
    python scripts/03_run_branches.py \
      --config "$CONFIG" --arm "$ARM" \
      --shard "$s" --num-shards "$SHARDS" \
      --only-missing --tag "r${round}" --limit "$CHUNK" \
      --min-free-gb "$MIN_FREE_GB" \
      > "$LOGDIR/backfill.r${round}.s${s}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done

  # Every shard process has now exited, so anything they were holding open is
  # released here. This is the measurement that matters.
  echo "-- df after (all shard processes exited) --"; df -h "$RUNDIR" / | sed 's/^/   /'

  grep -h "^\[03\] resume:" "$LOGDIR/backfill.r${round}.s0.log" || true
  grep -lh "ABORTING" "$LOGDIR"/backfill.r${round}.s*.log 2>/dev/null \
    | sed 's/^/   aborted: /' || true

  remaining="$(sed -n 's/.*already complete, \([0-9]*\) still to run.*/\1/p' \
    "$LOGDIR/backfill.r${round}.s0.log" | head -1)"
  if [ -z "$remaining" ]; then
    echo "could not read the remaining count from shard 0's log; stopping so it can be checked by hand"
    exit 1
  fi
  echo "remaining before this round: $remaining"

  # `remaining` is read at the START of a round, so the round that finishes the
  # work reports a non-zero count. One more round confirms zero and exits.
  if [ "$remaining" -eq 0 ]; then
    echo "########## COMPLETE after $((round - 1)) rounds ##########"
    exit 0
  fi

  # No progress across a whole round means the guards are firing before any
  # checkpoint completes -- looping again would just burn startup cost forever.
  if [ "$remaining" -eq "$prev_remaining" ]; then
    echo "no progress this round ($remaining still outstanding, unchanged)."
    echo "the disk guard is stopping shards before they complete work; stopping."
    exit 2
  fi
  prev_remaining="$remaining"
done

echo "hit MAX_ROUNDS=$MAX_ROUNDS with work outstanding; rerun to continue"
exit 3
