#!/usr/bin/env python
"""Stage 1, step 2: select decision prefixes and assign splits.

Costs no model calls: prompt rendering is a pure function of the prefix, so a
checkpoint's stored ``rendered_prompt`` -- the string Stage 2 will replay through
a forward pass -- is produced on CPU.

Two split schemes are computed and stored side by side, never pooled:
``split_paper`` reproduces App. B.3's fixed prefix counts with whole trajectories
kept together, and ``split_task_level`` holds out whole games as a stricter
generalization test.

Usage:
    python scripts/02_select_checkpoints.py --config configs/pilot.yaml --arm A_replication
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cnc import experiment, splits
from cnc.branching.runner import build_checkpoints
from cnc.storage import JsonlWriter, read_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True)
    args = ap.parse_args()

    cfg = experiment.load_config(args.config)
    rcfg = experiment.build_runner_config(cfg, args.arm)
    template = experiment.build_template(cfg)
    policy = experiment.build_policy(cfg, template)
    rd = experiment.run_dir(cfg)

    tasks = {t["task_id"]: t for t in read_jsonl(rd.tasks)}
    rollouts = sorted(
        (r for r in rd.read_all("rollouts") if r["arm"] == rcfg.arm),
        key=lambda r: r["rollout_id"],  # shard order must not affect splits
    )
    print("[02] arm={} rollouts={}".format(rcfg.arm, len(rollouts)))

    cps = []
    for rollout in rollouts:
        task = tasks[rollout["task_id"]]
        cps.extend(build_checkpoints(rollout, task, policy, rcfg))

    scfg = cfg.get("splits") or {}
    paper = splits.assign_paper_split(
        cps, counts=tuple(scfg.get("paper_counts", (8, 4, 8))), seed=int(scfg.get("seed", 0))
    )
    task_level = splits.assign_task_level_split(
        cps,
        fractions=tuple(scfg.get("task_level_fractions", (0.6, 0.2, 0.2))),
        seed=int(scfg.get("seed", 0)),
    )
    for cp, a, b in zip(cps, paper, task_level):
        cp["split_paper"] = a
        cp["split_task_level"] = b

    # Fail loudly rather than write a leaky dataset.
    for field, key in (("split_paper", "rollout_id"), ("split_task_level", "task_id")):
        bad = splits.check_no_leakage(cps, field, key)
        assert not bad, "leakage in {} across {}: {}".format(field, key, bad[:5])

    prov = experiment.make_provenance(cfg, rcfg, template)
    with JsonlWriter(rd.checkpoints, prov) as w:
        w.write_many(cps)

    print("[02] checkpoints: {}".format(len(cps)))
    print("[02] step-index distribution: {}".format(
        sorted(Counter(c["t"] for c in cps).items())[:12]))
    print("[02] split_paper: {}".format(dict(Counter(paper))))
    print("[02] split_task_level: {}".format(dict(Counter(task_level))))
    print("[02] wrote {}".format(rd.checkpoints))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
