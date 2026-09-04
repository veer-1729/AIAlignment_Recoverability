#!/usr/bin/env python
"""Does re-running the SAME game under a different seed change the outcome?

Base success rate is the wrong question for Arm B. Arm B exists to estimate
Qhat(s,a) = E[U | s,a], and an expectation is only worth taking when the thing
being averaged actually varies. A policy can sit perfectly inside the 15-60%
success band and still be useless here, if every game is deterministically won
or deterministically lost and the "randomness" never changes an outcome.

The temperature sweep measured success across DIFFERENT games, which confounds
policy stochasticity with per-game difficulty -- and the near-identical per-shard
results at 0.2/0.3/0.5 suggest difficulty was doing most of the work.

This measures the thing that matters: within one game, across seeds, do outcomes
differ? That fraction is what decides whether Arm B's Monte-Carlo estimate has
anything to estimate.

    python scripts/check_seed_variance.py --config configs/seedvar_t050.yaml
"""
from __future__ import annotations
import argparse, os, statistics, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cnc import experiment
from cnc.storage import read_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="B_montecarlo")
    args = ap.parse_args()

    cfg = experiment.load_config(args.config)
    rd = experiment.run_dir(cfg)
    tasks = {t["task_id"]: t for t in read_jsonl(rd.tasks)}
    rs = [r for r in rd.read_all("rollouts") if r["arm"] == args.arm]

    by_game = defaultdict(list)
    for r in rs:
        by_game[r["task_id"]].append(r)

    temp = {a["name"]: a["temperature"] for a in cfg["arms"]}[args.arm]
    print("config={}  arm={}  T={}".format(cfg["run_id"], args.arm, temp))
    print("rollouts={}  games={}".format(len(rs), len(by_game)))

    varying = mixed_steps = 0
    print("\n{:<28} {:<10} {:>16} {:>10}".format("task_type", "outcomes", "steps", "varies?"))
    for gid in sorted(by_game):
        g = sorted(by_game[gid], key=lambda r: r["rollout_id"])
        outs = [int(r["success"]) for r in g]
        steps = [r["n_steps"] for r in g]
        v = len(set(outs)) > 1
        varying += int(v)
        mixed_steps += int(len(set(steps)) > 1)
        print("{:<28} {:<10} {:>16} {:>10}".format(
            tasks.get(gid, {}).get("task_type", "?")[:27],
            "".join(str(o) for o in outs),
            ",".join(str(s) for s in steps),
            "YES" if v else "no"))

    n = len(by_game)
    succ = sum(r["success"] for r in rs) / max(1, len(rs))
    print("\n  base success (all rollouts):        {:.3f}".format(succ))
    print("  games whose OUTCOME varies by seed: {}/{} = {:.3f}   <-- the number that matters".format(
        varying, n, varying / max(1, n)))
    print("  games whose STEP COUNT varies:      {}/{} = {:.3f}   (weaker evidence of stochasticity)".format(
        mixed_steps, n, mixed_steps / max(1, n)))
    if varying == 0 and mixed_steps > 0:
        print("\n  READ: the policy IS stochastic (trajectories differ) but outcomes are")
        print("  pinned by game difficulty. Qhat would average identical values.")
    elif varying == 0:
        print("\n  READ: no variation at all -- sampling is not changing behaviour.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
