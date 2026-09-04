#!/usr/bin/env python
"""Variance diagnostics the headline numbers hide.

Four questions the summary statistics in 05_analyze.py cannot answer:

1. HOW MISLEADING IS A SINGLE EXPERT SAMPLE? The ALFWorld expert is stochastic
   (unseeded random.choice for tie-breaks and loop escapes). Arm A runs one
   suffix per action, so its Q(s,intervene) is ONE draw from that. A single
   failed expert branch therefore does not establish irrecoverability. Arm B
   runs five, so we can measure directly how often a single draw would have
   labelled the oracle action differently from the 5-sample estimate.

2. IS THE INTERVENTION ACTUALLY CONSTANT? A 0.76 mean success rate says nothing
   about spread. Report SD and conditional variance of Q_I, Q_C and tau.

3. HOW MUCH CONDITIONAL STRUCTURE SURVIVES the obvious covariates? If tau is
   fully explained by step index and task type, an activation probe has nothing
   left to find.

4. IS CONFLICT MASS INFLATED BY A COARSE SCALAR? Arm A's empirical
   continuation risk takes only two values, so binning on it is maximally
   coarse and conflict mass is trivially high. Recompute against Arm B's graded
   scalar and report per-bin purity.

    python scripts/variance_report.py --config configs/validation.yaml
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from cnc import experiment, utility
from cnc.storage import ARM_MONTECARLO, ARM_REPLICATION, read_jsonl


def sd(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1)) if len(x) > 1 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = experiment.load_config(args.config)
    rd = experiment.run_dir(cfg)
    tasks = {t["task_id"]: t for t in read_jsonl(rd.tasks)}
    cps = {c["checkpoint_id"]: c for c in rd.read_all("checkpoints")}
    branches = list(rd.read_all("branches"))

    by_cp = defaultdict(list)
    for b in branches:
        by_cp[(b["checkpoint_id"], b["arm"])].append(b)

    out = {}
    for arm in (ARM_REPLICATION, ARM_MONTECARLO):
        rows, per_cp = [], {}
        for (cid, a), bs in by_cp.items():
            if a != arm or not all(x["replay_verified"] for x in bs):
                continue
            if {x["action"] for x in bs} != set(utility.ACTIONS):
                continue
            cv = utility.values_from_branches(cid, arm, bs)
            r = cv.to_row()
            cp = cps.get(cid, {})
            r["t"] = cp.get("t")
            r["task_type"] = tasks.get(cp.get("task_id"), {}).get("task_type", "?")
            rows.append(r)
            per_cp[cid] = bs
        if not rows:
            continue

        print("\n" + "=" * 74)
        print("ARM {}   n={}".format(arm, len(rows)))
        print("=" * 74)

        qc = [r["value_continue"] for r in rows]
        qi = [r["value_intervene"] for r in rows]
        tau = [r["tau"] for r in rows]

        # -- Q2/Q3: is the intervention actually constant? -------------------
        print("\n[2] SPREAD -- is the intervention constant?")
        print("  {:<22} {:>9} {:>9} {:>9} {:>9}".format("", "mean", "SD", "min", "max"))
        for name, v in (("Q_continue", qc), ("Q_intervene", qi), ("tau", tau)):
            print("  {:<22} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}".format(
                name, float(np.mean(v)), sd(v), float(np.min(v)), float(np.max(v))))
        pi = [r["p_success_intervene"] for r in rows]
        print("  {:<22} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}".format(
            "p_success_intervene", float(np.mean(pi)), sd(pi), float(np.min(pi)), float(np.max(pi))))
        print("  intervene branch outcomes: {}".format(
            dict(Counter(round(p, 2) for p in pi))))

        # -- Q3: conditional variance after the obvious covariates ----------
        print("\n[3] CONDITIONAL VARIANCE of tau after controlling for covariates")
        tot = float(np.var(tau))
        print("  total var(tau)                    {:.4f}".format(tot))
        for key, label in (("task_type", "task type"), ("t", "step index")):
            groups = defaultdict(list)
            for r in rows:
                groups[r[key]].append(r["tau"])
            within = sum(len(g) * np.var(g) for g in groups.values()) / len(rows)
            print("  within-group var | {:<15} {:.4f}   ({:.0f}% of total retained)".format(
                label, within, 100 * within / tot if tot else 0))
        groups = defaultdict(list)
        for r in rows:
            groups[(r["task_type"], (r["t"] or 0) // 10)].append(r["tau"])
        within = sum(len(g) * np.var(g) for g in groups.values()) / len(rows)
        print("  within-group var | {:<15} {:.4f}   ({:.0f}% of total retained)".format(
            "type x step-decile", within, 100 * within / tot if tot else 0))

        # -- Q4: is conflict mass inflated by a coarse scalar? ---------------
        print("\n[4] CONFLICT MASS vs SCALAR GRANULARITY")
        g = np.array([r["p_fail_continue"] for r in rows])
        uniq = np.unique(g)
        print("  distinct values of the continuation-risk scalar: {}  {}".format(
            len(uniq), np.round(uniq, 2)))
        bins = np.searchsorted(uniq, g)
        conflicted = 0
        for b in np.unique(bins):
            m = bins == b
            acts = Counter(r["oracle_action"] for r, k in zip(rows, m) if k)
            pure = "PURE" if len(acts) == 1 else "mixed"
            conflicted += int(m.sum()) if len(acts) > 1 else 0
            print("    risk={:.2f}  n={:<4} {:<6} {}".format(
                float(uniq[b]), int(m.sum()), pure, dict(acts)))
        print("  conflict mass = {:.3f}".format(conflicted / len(rows)))
        print("  NOTE: with {} distinct scalar values this is a statement about the".format(len(uniq)))
        print("  insufficiency of continuation OUTCOME, not about individual states")
        print("  being intrinsically ambiguous.")

        out[arm] = {
            "n": len(rows),
            "sd": {"Q_continue": sd(qc), "Q_intervene": sd(qi), "tau": sd(tau)},
            "mean": {"Q_continue": float(np.mean(qc)), "Q_intervene": float(np.mean(qi)),
                     "tau": float(np.mean(tau))},
            "n_scalar_values": int(len(uniq)),
        }

        # -- Q1: how misleading is ONE expert sample? -----------------------
        if arm == ARM_MONTECARLO:
            print("\n[1] SINGLE EXPERT SAMPLE vs 5-SAMPLE ESTIMATE")
            flips = trials = 0
            varying = 0
            per_cp_sd = []
            for cid, bs in per_cp.items():
                iv = sorted([b for b in bs if b["action"] == utility.INTERVENE],
                            key=lambda b: b["replicate"])
                if len(iv) < 2:
                    continue
                succ = [bool(b["success"]) for b in iv]
                per_cp_sd.append(sd([float(s) for s in succ]))
                varying += int(len(set(succ)) > 1)
                full = utility.values_from_branches(cid, arm, bs).oracle_action
                others = [b for b in bs if b["action"] != utility.INTERVENE]
                for one in iv:  # pretend we only ran this single expert branch
                    single = utility.values_from_branches(
                        cid, arm, others + [dict(one, replicate=0)]).oracle_action
                    trials += 1
                    flips += int(single != full)
            print("  checkpoints where the expert outcome varies across seeds: {}/{} = {:.3f}".format(
                varying, len(per_cp_sd), varying / max(1, len(per_cp_sd))))
            print("  mean within-checkpoint SD of expert success:              {:.3f}".format(
                float(np.mean(per_cp_sd)) if per_cp_sd else 0.0))
            print("  ORACLE LABEL FLIPS if you had run only one expert branch: {}/{} = {:.3f}".format(
                flips, trials, flips / max(1, trials)))
            print("  -> this is the error rate Arm A's n=1 intervene branch carries.")
            out[arm]["single_sample_oracle_flip_rate"] = flips / max(1, trials)
            out[arm]["expert_varies_frac"] = varying / max(1, len(per_cp_sd))

    os.makedirs("reports", exist_ok=True)
    p = "reports/variance_{}.json".format(cfg["run_id"])
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote {}".format(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
