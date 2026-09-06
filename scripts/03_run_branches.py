#!/usr/bin/env python
"""Stage 1, step 3: same-prefix counterfactual branching.

For every checkpoint: replay into the exact state the base rollout reached,
verify it step by step, then execute every candidate action -- ``continue``
(n_replicates suffixes), ``intervene`` (defer-to-expert, zero LLM calls), and
``quit`` (analytic).

Branches whose prefix fails verification are recorded with
``replay_verified=False`` and never executed. The rejection rate is reported
here, not hidden: the paper's 100% match rate is a statement about what survives
this filter.

Usage:
    python scripts/03_run_branches.py --config configs/pilot.yaml --arm A_replication --workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cnc import experiment
from cnc.branching.runner import branch_census, run_checkpoint_branches, usable_checkpoints
from cnc.storage import DiskExhausted, JsonlWriter, check_disk, disk_headroom, read_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True)
    # Process-level sharding only. Threading ALFWorld env construction corrupts
    # its module globals and silently drops branches -- see storage.RunDir.
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="cap checkpoints (smoke runs)")
    ap.add_argument("--only-missing", action="store_true",
                    help="skip checkpoints that already have a complete verified branch set")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output shard file; use when resuming so an "
                         "existing shard file is never reopened or overwritten")
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    ap.add_argument("--min-free-inodes", type=int, default=50000)
    args = ap.parse_args()

    cfg = experiment.load_config(args.config)
    rcfg = experiment.build_runner_config(cfg, args.arm)
    template = experiment.build_template(cfg)
    policy = experiment.build_policy(cfg, template)
    rd = experiment.run_dir(cfg)

    tasks = {t["task_id"]: t for t in read_jsonl(rd.tasks)}
    rollouts = {r["rollout_id"]: r for r in rd.read_all("rollouts") if r["arm"] == rcfg.arm}
    cps = sorted(
        (c for c in rd.read_all("checkpoints") if c["arm"] == rcfg.arm),
        key=lambda c: c["checkpoint_id"],
    )
    if args.limit:
        cps = cps[: args.limit]

    # Resume BEFORE sharding, not after.
    #
    # The authority on what is already done is the branch data itself, never the
    # shard logs -- a killed process leaves a log that stops mid-sentence. And the
    # order matters as much as the source: the previous run's failures were
    # concentrated in some shards and absent from others, so filtering a shard's
    # own stride would leave two shards with nothing to do and six with a full
    # load, and wall-clock is set by the slowest. Filtering first and striding the
    # remainder spreads the work evenly. Which shard runs a checkpoint carries no
    # scientific meaning -- branch seeds derive from checkpoint_id, not shard.
    if args.only_missing:
        census = branch_census(
            (b for b in rd.read_all("branches") if b["arm"] == rcfg.arm), rcfg.n_replicates)
        done = {cid for cid, r in census.items() if r["complete"]}
        before = len(cps)
        cps = [c for c in cps if c["checkpoint_id"] not in done]
        print("[03] resume: {}/{} checkpoints already complete, {} still to run".format(
            before - len(cps), before, len(cps)))

    if args.num_shards > 1:
        cps = [c for i, c in enumerate(cps) if i % args.num_shards == args.shard]

    print("[03] arm={} shard={}/{} checkpoints={} replicates={}".format(
        rcfg.arm, args.shard, args.num_shards, len(cps), rcfg.n_replicates))
    if not cps:
        print("[03] nothing to do")
        return 0

    # Write incrementally, not at completion.
    #
    # Buffering a whole shard in memory and flushing at the end means a process
    # that dies at 90% loses everything -- and a scaled run puts real memory
    # pressure on the box, which makes an OOM kill a live possibility rather
    # than a hypothetical. Streaming each checkpoint's branches to disk as they
    # finish caps the loss at one checkpoint and makes progress observable while
    # the shard is still running.
    errors = []
    branches = []
    prov = experiment.make_provenance(cfg, rcfg, template)
    out_path = rd.shard_path("branches", args.shard, args.num_shards, tag=args.tag)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0 and args.only_missing:
        # Appending to a file this run is also *reading* for its resume decision
        # invites confusion about what a record means. Pass a distinct --tag.
        print("[03] REFUSING to append to existing {} during a resume; pass a fresh --tag".format(
            out_path))
        return 2

    # Stop while there is still room to write, rather than discovering the disk
    # is full at the moment of writing. The previous run swallowed ENOSPC per
    # checkpoint and kept going: one shard burned hours failing 146 of 197.
    h = check_disk(rd.base, args.min_free_gb, args.min_free_inodes)
    print("[03] disk at start: {:.1f} GB free ({:.1f}%), {} inodes free ({:.1f}%)".format(
        h["free_gb"], h["pct_bytes_free"], h["free_inodes"], h["pct_inodes_free"]))
    t0 = time.time()
    aborted = None

    with JsonlWriter(out_path, prov) as w:
        for i, cp in enumerate(cps):
            try:
                # Both the guard and any ENOSPC from the writer raise
                # DiskExhausted, which is deliberately NOT an OSError so the
                # broad except below cannot swallow it.
                check_disk(rd.base, args.min_free_gb, args.min_free_inodes)
                rollout = rollouts[cp["rollout_id"]]
                task = tasks[cp["task_id"]]
                got = run_checkpoint_branches(
                    cp, rollout, task, policy, rcfg,
                    env_factory=experiment.make_env_factory(task["game_file"], rcfg),
                )
                w.write_many(got)
                w.sync()          # commit each checkpoint to the platter, not the page cache
                branches.extend(got)
            except DiskExhausted as exc:
                aborted = str(exc)
                print("[03] shard={} ABORTING at checkpoint {}/{}: {}".format(
                    args.shard, i + 1, len(cps), exc), flush=True)
                break
            except Exception as exc:
                errors.append((cp["checkpoint_id"], repr(exc)))
            if (i + 1) % 5 == 0 or i + 1 == len(cps):
                el, hh = time.time() - t0, disk_headroom(rd.base)
                print("[03] shard={} {}/{} checkpoints  {:.0f}s  ({:.1f}s/cp)  "
                      "disk {:.1f}GB/{} inodes free".format(
                          args.shard, i + 1, len(cps), el, el / (i + 1),
                          hh["free_gb"], hh["free_inodes"]), flush=True)

    executed = [b for b in branches if b["action"] != "quit"]
    verified = [b for b in executed if b["replay_verified"]]
    usable = usable_checkpoints(branches)
    print("[03] branches={} ({} failed checkpoints)".format(len(branches), len(errors)))
    print("[03] V1 replay match rate: {}/{} = {:.4f}".format(
        len(verified), len(executed), len(verified) / max(1, len(executed))))
    print("[03] usable checkpoints (all branches verified): {}/{}".format(
        len(usable), len(cps)))
    print("[03] terminal reasons: {}".format(
        dict(Counter(b["terminal_reason"] for b in executed))))
    for act in ("continue", "intervene"):
        rows = [b for b in verified if b["action"] == act]
        if rows:
            print("[03]   {:<10} success {}/{} = {:.3f}".format(
                act, sum(b["success"] for b in rows), len(rows),
                sum(b["success"] for b in rows) / len(rows)))
    for cid, err in errors[:5]:
        print("[03]   ERROR {}: {}".format(cid, err))
    print("[03] wrote {}".format(out_path))
    if aborted:
        print("[03] SHARD ABORTED (disk): {}".format(aborted))
        print("[03] everything above this point is committed; rerun with --only-missing")
        return 3
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
