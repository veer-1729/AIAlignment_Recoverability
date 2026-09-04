#!/usr/bin/env python
"""Stage 1, step 1: run base trajectories.

Each arm gets its **own** base rollouts over the **same game set**. That is not
an implementation shortcut -- Arm B uses a different (stochastic) policy, so its
trajectories are draws from a different distribution and reusing Arm A's greedy
prefixes would silently mix the two. The arms stay linkable through ``task_id``
and analytically separate through the ``arm`` field on every record.

Usage:
    python scripts/01_run_rollouts.py --config configs/pilot.yaml --arm A_replication
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cnc import experiment
from cnc.branching.runner import run_rollout
from cnc.env.factory import collect_game_files, sample_games_per_task_type
from cnc.storage import JsonlWriter, derive_seed, read_jsonl, stable_hash, task_record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True)
    # Parallelism is by process, not thread: ALFWorld/TextWorld globals are not
    # thread-safe (see storage.RunDir sharding notes). Run several shards
    # concurrently as separate processes.
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    cfg = experiment.load_config(args.config)
    rcfg = experiment.build_runner_config(cfg, args.arm)
    template = experiment.build_template(cfg)
    policy = experiment.build_policy(cfg, template)
    rd = experiment.run_dir(cfg)

    alf = cfg["alfworld"]
    games = collect_game_files(
        split=alf.get("split", "train"),
        data_dir=alf.get("data_dir"),
        require_solvable=True,
    )
    picks = sample_games_per_task_type(games, per_type=int(alf["games_per_task_type"]))
    print("[01] {} solvable games -> {} selected".format(len(games), len(picks)))

    # task_id is a deterministic function of the game file, so every shard can
    # derive the same ids independently; only shard 0 writes the manifest.
    by_file = {
        g["game_file"]: task_record(
            task_id="task_" + stable_hash(g["game_file"], 12),
            game_file=g["game_file"],
            task_type=g["task_type"],
            split_source=g["split_source"],
            solvable=g["solvable"],
        )
        for g in picks
    }
    prov = experiment.make_provenance(cfg, rcfg, template)
    if (args.shard in (None, 0)) and not os.path.exists(rd.tasks):
        with JsonlWriter(rd.tasks, prov) as w:
            w.write_many([by_file[g["game_file"]] for g in picks])

    per_game = int((cfg.get("rollouts") or {}).get("per_game", 1))
    jobs = [(by_file[g["game_file"]], i) for g in picks for i in range(per_game)]
    if args.num_shards > 1:
        jobs = [j for k, j in enumerate(jobs) if k % args.num_shards == args.shard]

    errors = []
    rollouts = []
    for task, i in jobs:
        rollout_id = "{}::{}::r{}".format(task["task_id"], rcfg.arm, i)
        try:
            env = experiment.make_env_factory(task["game_file"], rcfg)()
            try:
                rollouts.append(
                    run_rollout(env, task, policy, rcfg, rollout_id,
                                seed=derive_seed(rollout_id))
                )
            finally:
                env.close()
        except Exception as exc:
            errors.append((task["task_id"], repr(exc)))

    out_path = rd.shard_path("rollouts", args.shard, args.num_shards)
    with JsonlWriter(out_path, prov) as w:
        w.write_many(rollouts)

    n = len(rollouts)
    wins = sum(1 for r in rollouts if r["success"])
    steps = [r["n_steps"] for r in rollouts] or [0]
    print("[01] arm={} shard={}/{} rollouts={} ({} failed)".format(
        rcfg.arm, args.shard, args.num_shards, n, len(errors)))
    print("[01] base success rate: {}/{} = {:.3f}".format(wins, n, wins / max(1, n)))
    print("[01] steps: min={} mean={:.1f} max={}".format(
        min(steps), sum(steps) / len(steps), max(steps)))
    for tid, err in errors[:5]:
        print("[01]   ERROR {}: {}".format(tid, err))
    print("[01] wrote {}".format(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
