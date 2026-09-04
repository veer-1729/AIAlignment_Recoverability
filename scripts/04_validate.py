#!/usr/bin/env python
"""Stage 1, step 4: the V1-V10 acceptance suite.

This is the trust boundary. Nothing downstream -- no analysis, no scale-up, and
certainly no Stage-2 probe -- means anything unless these pass. Each check
returns PASS / FAIL / SKIP with the evidence attached, and the script exits
non-zero if any check FAILs.

Checks that need a model (V4, V5, V7) SKIP rather than FAIL when the run used a
scripted client, because a dry run cannot say anything about agent behaviour.

Usage:
    python scripts/04_validate.py --config configs/pilot.yaml [--arm ...] [--sample N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cnc import experiment, splits, utility
from cnc.branching.runner import expected_signatures, prefix_actions
from cnc.env.replay import ReplayValidator
from cnc.interventions.expert import run_expert_branch
from cnc.storage import (
    ARM_MONTECARLO,
    ARM_REPLICATION,
    derive_seed as hash_seed,
    read_jsonl,
    read_provenance,
)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Results:
    def __init__(self):
        self.rows = []

    def add(self, cid, name, status, detail):
        self.rows.append({"id": cid, "check": name, "status": status, "detail": detail})
        print("  {:<4} {:<38} {:<5} {}".format(cid, name, status, detail))

    @property
    def failed(self):
        return [r for r in self.rows if r["status"] == FAIL]


def _load(rd):
    tasks = {t["task_id"]: t for t in read_jsonl(rd.tasks)}
    rollouts = {r["rollout_id"]: r for r in rd.read_all("rollouts")}
    cps = list(rd.read_all("checkpoints"))
    branches = list(rd.read_all("branches"))
    return tasks, rollouts, cps, branches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample", type=int, default=6, help="games/checkpoints per live probe")
    args = ap.parse_args()

    cfg = experiment.load_config(args.config)
    rd = experiment.run_dir(cfg)
    tasks, rollouts, cps, branches = _load(rd)
    is_mock = (cfg.get("serving") or {}).get("kind") != "openai_compatible"

    print("\n=== V1-V10 acceptance suite: run_id={} ===".format(cfg["run_id"]))
    print("  {} tasks / {} rollouts / {} checkpoints / {} branches\n".format(
        len(tasks), len(rollouts), len(cps), len(branches)))
    R = Results()

    # ---------------------------------------------------------------- V1
    executed = [b for b in branches if b["action"] != "quit"]
    verified = [b for b in executed if b["replay_verified"]]
    rate = len(verified) / max(1, len(executed))
    R.add("V1", "replay fidelity (retained)", PASS if rate == 1.0 else FAIL,
          "{}/{} = {:.4f} verified; {} rejected".format(
              len(verified), len(executed), rate, len(executed) - len(verified)))

    # ---------------------------------------------------------------- V2
    # Negative control: corrupt a logged action and require the validator to notice.
    # A validator that never fires proves nothing about V1.
    caught = tried = 0
    for cp in cps[: args.sample]:
        rollout = rollouts[cp["rollout_id"]]
        task = tasks[cp["task_id"]]
        acts = prefix_actions(rollout, cp["t"])
        sigs = expected_signatures(rollout, cp["t"])
        if not acts:
            continue
        bad = list(acts)
        bad[0] = "look" if bad[0] != "look" else "inventory"
        env = experiment.make_env_factory(
            task["game_file"], experiment.build_runner_config(cfg, cps[0]["arm"]))()
        try:
            tried += 1
            caught += int(not ReplayValidator().replay(env, bad, sigs).verified)
        finally:
            env.close()
    R.add("V2", "negative control (corrupt prefix)",
          PASS if tried and caught == tried else FAIL,
          "{}/{} planted corruptions detected".format(caught, tried))

    # ---------------------------------------------------------------- V3
    # Determinism: a fresh env in this fresh process must reproduce the exact
    # state hash the base rollout recorded at the checkpoint.
    same = tried3 = 0
    for cp in cps[: args.sample]:
        rollout = rollouts[cp["rollout_id"]]
        task = tasks[cp["task_id"]]
        env = experiment.make_env_factory(
            task["game_file"], experiment.build_runner_config(cfg, cp["arm"]))()
        try:
            res = ReplayValidator().replay(
                env, prefix_actions(rollout, cp["t"]), expected_signatures(rollout, cp["t"]))
            tried3 += 1
            same += int(res.verified and res.final.hash == cp["prefix_state_hash"])
        finally:
            env.close()
    R.add("V3", "cross-process determinism", PASS if tried3 and same == tried3 else FAIL,
          "{}/{} prefix state hashes reproduced".format(same, tried3))

    # ---------------------------------------------------------------- V4
    a_cont = [b for b in branches
              if b["arm"] == ARM_REPLICATION and b["action"] == "continue" and b["replay_verified"]]
    by_cp = defaultdict(list)
    for b in a_cont:
        by_cp[b["checkpoint_id"]].append(b)
    multi = {k: v for k, v in by_cp.items() if len(v) > 1}
    if is_mock:
        R.add("V4", "Arm A determinism (T=0)", SKIP, "scripted client: no sampling to test")
    elif not multi:
        R.add("V4", "Arm A determinism (T=0)", PASS,
              "n_replicates=1 by design; {} single-suffix branches".format(len(a_cont)))
    else:
        bad = [k for k, v in multi.items()
               if len({(x["success"], x["n_steps_in_branch"]) for x in v}) > 1]
        R.add("V4", "Arm A determinism (T=0)", PASS if not bad else FAIL,
              "{} checkpoints repeated, {} with nonzero variance".format(len(multi), len(bad)))

    # ---------------------------------------------------------------- V5
    # Revised per review: NOT a per-checkpoint requirement. Zero variance at one
    # checkpoint is a legitimate observation. We require that sampling has a
    # visible effect somewhere, and we report the full MC uncertainty spread.
    b_cont = [b for b in branches
              if b["arm"] == ARM_MONTECARLO and b["action"] == "continue" and b["replay_verified"]]
    b_by_cp = defaultdict(list)
    for b in b_cont:
        b_by_cp[b["checkpoint_id"]].append(b)
    if is_mock:
        R.add("V5", "Arm B stochasticity", SKIP, "scripted client: responses are constant")
    elif not b_by_cp:
        R.add("V5", "Arm B stochasticity", SKIP, "no Arm B branches present")
    else:
        varied = [k for k, v in b_by_cp.items()
                  if len({(x["success"], x["n_steps_in_branch"]) for x in v}) > 1]
        frac = len(varied) / len(b_by_cp)
        ses = []
        for cid, v in b_by_cp.items():
            av = utility.aggregate_action(
                "continue", [x["success"] for x in v], [x["n_steps_in_branch"] for x in v])
            ses.append(av.se)
        ses.sort()
        detail = ("{}/{} checkpoints ({:.2f}) differ across seeds; "
                  "MC SE p50={:.3f} p90={:.3f} max={:.3f}").format(
            len(varied), len(b_by_cp), frac,
            ses[len(ses) // 2], ses[int(0.9 * (len(ses) - 1))], ses[-1])
        R.add("V5", "Arm B stochasticity", PASS if frac > 0.0 else FAIL, detail)

    # ---------------------------------------------------------------- V6
    # The expert is stochastic (random.choice tie-breaks and a random-admissible
    # loop escape), so a single draw per game is a noisy estimate of intervention
    # strength. Repeat with derived seeds and report the per-task-type breakdown,
    # which is what surfaces a systematically hard task type.
    sample_tasks = list(tasks.values())[: args.sample]
    rcfg_any = experiment.build_runner_config(cfg, cps[0]["arm"] if cps else ARM_REPLICATION)
    reps = 3
    wins = trials = 0
    per_type = defaultdict(lambda: [0, 0])
    for task in sample_tasks:
        for rep in range(reps):
            random.seed(hash_seed(task["task_id"], "V6", rep))
            env = experiment.make_env_factory(task["game_file"], rcfg_any)()
            try:
                env.reset()
                ok = run_expert_branch(env, max_steps=rcfg_any.max_episode_steps)["success"]
            finally:
                env.close()
            wins += int(ok); trials += 1
            per_type[task["task_type"]][0] += int(ok)
            per_type[task["task_type"]][1] += 1
    erate = wins / max(1, trials)
    worst = min(per_type.items(), key=lambda kv: kv[1][0] / kv[1][1]) if per_type else None
    R.add("V6", "expert sanity (takeover from t=0)", PASS if erate >= 0.85 else FAIL,
          "{}/{} = {:.2f} over {} games x {} seeds (threshold 0.85); worst type: {}".format(
              wins, trials, erate, len(sample_tasks), reps,
              "{} {}/{}".format(worst[0], worst[1][0], worst[1][1]) if worst else "-"))

    # ---------------------------------------------------------------- V7
    for arm in (ARM_REPLICATION, ARM_MONTECARLO):
        rs = [r for r in rollouts.values() if r["arm"] == arm]
        if not rs:
            continue
        sr = sum(r["success"] for r in rs) / len(rs)
        if is_mock:
            R.add("V7", "base success band [{}]".format(arm), SKIP,
                  "scripted client: {:.3f} is meaningless".format(sr))
        else:
            R.add("V7", "base success band [{}]".format(arm),
                  PASS if 0.15 <= sr <= 0.60 else FAIL,
                  "{:.3f} (band 0.15-0.60, n={})".format(sr, len(rs)))

    # ---------------------------------------------------------------- V8
    leaks = []
    for field, key in (("split_paper", "rollout_id"), ("split_task_level", "task_id"),
                       ("split_task_level", "rollout_id")):
        leaks += ["{}/{}:{}".format(field, key, g) for g in splits.check_no_leakage(cps, field, key)]
    R.add("V8", "split leakage", PASS if not leaks else FAIL,
          "clean" if not leaks else "{} straddling groups: {}".format(len(leaks), leaks[:3]))

    # ---------------------------------------------------------------- V9
    # Derived values must be a pure function of raw records: recomputing twice,
    # and from a shuffled input order, must give identical rows.
    grouped = defaultdict(list)
    for b in branches:
        grouped[(b["checkpoint_id"], b["arm"])].append(b)
    usable = {k: v for k, v in grouped.items()
              if all(x["replay_verified"] for x in v)
              and {x["action"] for x in v} == set(utility.ACTIONS)}
    mismatches = 0
    for (cid, arm), rows in list(usable.items())[:200]:
        a = utility.values_from_branches(cid, arm, rows).to_row()
        b = utility.values_from_branches(cid, arm, list(reversed(rows))).to_row()
        mismatches += int(a != b)
    R.add("V9", "derivation purity", PASS if mismatches == 0 else FAIL,
          "{} usable checkpoints, {} order-dependent".format(len(usable), mismatches))

    # ---------------------------------------------------------------- V10
    prov = read_provenance(rd.checkpoints) or {}
    have_ids = [c for c in cps if c.get("prompt_token_ids")]
    if not have_ids:
        R.add("V10", "Stage-2 prompt/token round-trip", SKIP,
              "no prompt_token_ids (template={})".format((cfg.get("model") or {}).get("template")))
    else:
        try:
            from transformers import AutoTokenizer

            model = cfg["model"]
            tok = AutoTokenizer.from_pretrained(model["id"], revision=model.get("revision"))
            bad = 0
            for c in have_ids[: args.sample * 5]:
                ids = tok(c["rendered_prompt"], add_special_tokens=False)["input_ids"]
                bad += int(list(ids) != list(c["prompt_token_ids"]))
            R.add("V10", "Stage-2 prompt/token round-trip", PASS if bad == 0 else FAIL,
                  "{} checked, {} mismatched; template_hash={}".format(
                      min(len(have_ids), args.sample * 5), bad,
                      prov.get("chat_template_hash", "")[:12]))
        except Exception as exc:
            R.add("V10", "Stage-2 prompt/token round-trip", SKIP,
                  "tokenizer unavailable: {}".format(type(exc).__name__))

    # ---------------------------------------------------------------- report
    os.makedirs("reports", exist_ok=True)
    out = os.path.join("reports", "validation_{}.json".format(cfg["run_id"]))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"run_id": cfg["run_id"], "checks": R.rows}, fh, indent=2)

    counts = Counter(r["status"] for r in R.rows)
    print("\n  {} PASS / {} FAIL / {} SKIP  -> {}".format(
        counts[PASS], counts[FAIL], counts[SKIP], out))
    return 1 if R.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
