#!/usr/bin/env python
"""Are the rescued checkpoints exchangeable with the originals?

The scaled dataset was assembled in two executions. The first kept 928
checkpoints before a full disk stopped it; a backfill run, days later and against
a freshly started server, produced the rest. Pooling them is only legitimate if
the two cohorts are drawn from the same distribution -- and there are two
specific reasons to doubt it.

The split is not random. Shards walk a globally sorted checkpoint list in stride
and each died at a different point, so the surviving cohort is dense at the head
of that order and the rescued cohort dense at the tail. If checkpoint id ordering
correlates with anything -- and it sorts by task id, so it correlates with game
identity -- then cohort is confounded with task.

The servers were not the same process. Same model, same weights, same decoding
parameters, but a different vLLM instance under different load. vLLM is not
bitwise deterministic across batch compositions, which is fine for individual
samples and not fine if it shifts the distribution.

Why this matters beyond tidiness: the probe study splits by rollout and by game,
never by cohort, so a cohort effect would sit on both sides of every split. It
would inflate the apparent variance of every target, and any feature correlated
with cohort -- activations included, since they were extracted in one pass but
describe prompts from two -- could pick it up as signal.

The test is stratified because the confound is expected. An unstratified
difference between cohorts tells you almost nothing here: the cohorts genuinely
contain different games. What matters is whether the difference survives
comparing like with like.

    python scripts/cohort_check.py --config configs/scale.yaml --arm B_montecarlo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from cnc import experiment, utility
from cnc.branching.runner import complete_group
from cnc.storage import read_jsonl

N_BOOT = 2000
METRICS = ("value_continue", "value_intervene", "tau",
           "p_success_continue", "p_success_intervene")


def cohort_of(basename: str) -> str:
    """`branches.s003.jsonl` -> "original"; `branches.s003.rescue2.jsonl` -> "rescue2"."""
    parts = basename.split(".")
    return parts[2] if len(parts) >= 4 else "original"


def stratified_bootstrap(a, sa, b, sb, rng, n_boot=N_BOOT):
    """Cohort difference in means, comparing like with like, with a 95% CI.

    Each task type contributes its own within-stratum difference, weighted by how
    much data it holds; a stratum a cohort barely covers contributes nothing,
    because there is no like-for-like comparison to make there. Resampling
    happens WITHIN each cohort, so the interval reflects sampling noise and not
    the fixed, known, and expected difference in task composition.

    Returns ``(mean, lo, hi)``; the interval excluding zero is the finding.
    """
    strata = set(sa) | set(sb)
    sa, sb = np.asarray(sa), np.asarray(sb)
    boots = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(a), len(a))
        ib = rng.integers(0, len(b), len(b))
        aa, bb, ssa, ssb = a[ia], b[ib], sa[ia], sb[ib]
        num = den = 0.0
        for st in strata:
            ma, mb = ssa == st, ssb == st
            if ma.sum() < 2 or mb.sum() < 2:
                continue
            w = ma.sum() + mb.sum()
            num += w * (aa[ma].mean() - bb[mb].mean())
            den += w
        if den:
            boots.append(num / den)
    if not boots:
        return float("nan"), float("nan"), float("nan")
    boots = np.asarray(boots)
    return (float(boots.mean()),
            float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="B_montecarlo")
    args = ap.parse_args()

    cfg = experiment.load_config(args.config)
    rd = experiment.run_dir(cfg)
    n_rep = experiment.build_runner_config(cfg, args.arm).n_replicates

    tasks = {t["task_id"]: t for t in read_jsonl(rd.tasks)}
    cps = {c["checkpoint_id"]: c for c in rd.read_all("checkpoints") if c["arm"] == args.arm}
    sources = rd.branch_sources()

    by_cp = defaultdict(list)
    for b in rd.read_branches():
        if b["arm"] == args.arm:
            by_cp[b["checkpoint_id"]].append(b)

    rows = []
    for cid, bs in by_cp.items():
        if not complete_group(cid, bs, n_rep) or cid not in cps:
            continue
        r = utility.values_from_branches(cid, args.arm, bs).to_row()
        cp = cps[cid]
        r["cohort"] = cohort_of(sources.get((cid, args.arm), ""))
        r["task_type"] = tasks.get(cp.get("task_id"), {}).get("task_type", "?")
        r["t"] = cp.get("t")
        rows.append(r)

    if not rows:
        print("no usable checkpoints")
        return 1

    cohorts = sorted({r["cohort"] for r in rows})
    print("usable checkpoints: {}".format(len(rows)))
    print("cohorts: {}".format(
        {c: sum(1 for r in rows if r["cohort"] == c) for c in cohorts}))
    if len(cohorts) < 2:
        print("\nsingle cohort -- nothing to compare, the dataset came from one execution")
        return 0

    # The confound, quantified. If task-type composition is identical the
    # stratification below is a formality; if it is skewed, the unstratified
    # comparison underneath is the misleading one.
    print("\nTASK-TYPE COMPOSITION (the confound)")
    ttypes = sorted({r["task_type"] for r in rows})
    print("  {:<24} {}".format("task type", "  ".join("{:>10}".format(c) for c in cohorts)))
    for tt in ttypes:
        frac = []
        for c in cohorts:
            sub = [r for r in rows if r["cohort"] == c]
            frac.append(sum(1 for r in sub if r["task_type"] == tt) / max(1, len(sub)))
        print("  {:<24} {}".format(tt, "  ".join("{:>10.3f}".format(f) for f in frac)))

    mean_t = {c: np.mean([r["t"] for r in rows if r["cohort"] == c and r["t"] is not None])
              for c in cohorts}
    print("  {:<24} {}".format("mean step index t",
                               "  ".join("{:>10.2f}".format(mean_t[c]) for c in cohorts)))

    # Two cohorts only; with more, compare each against the largest.
    ref = max(cohorts, key=lambda c: sum(1 for r in rows if r["cohort"] == c))
    others = [c for c in cohorts if c != ref]
    rng = np.random.default_rng(0)
    out = {"n": len(rows), "reference": ref, "metrics": {}}

    print("\nPER-METRIC COMPARISON vs '{}'".format(ref))
    print("  {:<22} {:>9} {:>9} {:>10} {:>22} {}".format(
        "metric", ref, "other", "raw diff", "stratified diff [95% CI]", ""))
    verdicts = []
    for other in others:
        for m in METRICS:
            a = np.array([r[m] for r in rows if r["cohort"] == ref], dtype=float)
            b = np.array([r[m] for r in rows if r["cohort"] == other], dtype=float)
            sa = [r["task_type"] for r in rows if r["cohort"] == ref]
            sb = [r["task_type"] for r in rows if r["cohort"] == other]
            raw = a.mean() - b.mean()
            strat, lo, hi = stratified_bootstrap(a, sa, b, sb, rng)
            sig = lo > 0 or hi < 0
            verdicts.append(sig)
            print("  {:<22} {:>9.3f} {:>9.3f} {:>10.3f}   {:>+7.3f} [{:+.3f}, {:+.3f}] {}".format(
                m, a.mean(), b.mean(), raw, strat, lo, hi, "*" if sig else ""))
            out["metrics"]["{}|{}".format(other, m)] = {
                "ref_mean": float(a.mean()), "other_mean": float(b.mean()),
                "raw_diff": float(raw), "stratified_diff": strat, "ci95": [lo, hi],
                "differs": bool(sig)}

    print("\nVERDICT")
    if not any(verdicts):
        print("  No metric shows a cohort difference that survives stratification.")
        print("  The cohorts are exchangeable on these measures; pooling is sound,")
        print("  and any raw difference above is task composition, not execution.")
    else:
        n = sum(verdicts)
        print("  {} of {} metrics differ between cohorts AFTER stratifying on task type.".format(
            n, len(verdicts)))
        print("  Pooling is NOT clearly sound. Report cohort as a covariate, and check")
        print("  whether the probe results hold within the larger cohort alone before")
        print("  claiming them for the pooled dataset.")

    p = "reports/cohort_check_{}.json".format(cfg["run_id"])
    os.makedirs("reports", exist_ok=True)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote {}".format(p))
    print("* marks a bootstrap 95% CI on the stratified difference excluding zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
