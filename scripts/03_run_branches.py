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
from cnc.branching.runner import run_checkpoint_branches, usable_checkpoints
from cnc.storage import JsonlWriter, read_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True)
    # Process-level sharding only. Threading ALFWorld env construction corrupts
    # its module globals and silently drops branches -- see storage.RunDir.
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="cap checkpoints (smoke runs)")
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
    if args.num_shards > 1:
        cps = [c for i, c in enumerate(cps) if i % args.num_shards == args.shard]
    print("[03] arm={} shard={}/{} checkpoints={} replicates={}".format(
        rcfg.arm, args.shard, args.num_shards, len(cps), rcfg.n_replicates))

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
    out_path = rd.shard_path("branches", args.shard, args.num_shards)
    t0 = time.time()

    with JsonlWriter(out_path, prov) as w:
        for i, cp in enumerate(cps):
            try:
                rollout = rollouts[cp["rollout_id"]]
                task = tasks[cp["task_id"]]
                got = run_checkpoint_branches(
                    cp, rollout, task, policy, rcfg,
                    env_factory=experiment.make_env_factory(task["game_file"], rcfg),
                )
                w.write_many(got)
                branches.extend(got)
            except Exception as exc:
                errors.append((cp["checkpoint_id"], repr(exc)))
            if (i + 1) % 5 == 0 or i + 1 == len(cps):
                el = time.time() - t0
                print("[03] shard={} {}/{} checkpoints  {:.0f}s  ({:.1f}s/cp)".format(
                    args.shard, i + 1, len(cps), el, el / (i + 1)), flush=True)

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
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
