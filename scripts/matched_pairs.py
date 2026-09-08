#!/usr/bin/env python
"""C: same-game matched-state test. The strongest remaining confound test.

THE CLAIM IT WOULD LICENSE. Within the same task instance, at the same maximal
continuation failure probability and similar trajectory progress, does the internal
state track changing recoverability? If so, the effect cannot be game identity, task
type, or position in the episode -- all three are held approximately fixed inside a
pair, and only the state differs.

CRITERIA, FIXED BEFORE ANY MODEL WAS FITTED (reports/PHASE3_PREREGISTRATION_REPORT.md,
committed before the census returned):

    same game, both at risk = 1.00
    |progress difference| <= the TIGHTEST tolerance yielding >= 50 opposite-oracle
        cross-rollout pairs; the census selected 0.15
    |Q_I difference| >= 0.50
    CROSS-ROLLOUT ONLY -- two checkpoints from one rollout share a trajectory prefix,
        and the earlier is a literal ancestor of the later, which is a stronger
        confound than sharing a game
    matched without replacement, greedy by ascending progress distance, so no
        checkpoint enters two pairs and a few dense games cannot dominate

The 928-cohort fails this rule at every permitted tolerance (29 opposite-oracle pairs
against a requirement of 50), so C does not replicate independently and that is
reported rather than engineered around by loosening the rule.

POWER. The pooled cohort qualifies at exactly 50 opposite-oracle pairs -- zero margin.
After a game-level split only the pairs whose game is held out are scorable, roughly
30% per split. This test is thin by construction and the output says so on every line.
It is run because the rule says to run it, and read as weak evidence either way.

    python scripts/matched_pairs.py --config configs/scale.yaml
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cnc import experiment
from cnc.features import prefix as prefix_features
from probe_study import derive, grouped_split, fit_reg
from conflict_set_analysis import cohort_of

LAYER = 24
DPROG_TOL = 0.15      # selected by the census under the pre-registered rule
DQI_MIN = 0.50


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="B_montecarlo")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cohort", default="all", choices=("all", "original"))
    ap.add_argument("--opposite-only", action="store_true",
                    help="restrict to pairs whose oracle actions differ; the rule counts "
                         "those, so this is the stricter reading of the same criteria")
    args = ap.parse_args()

    cfg = experiment.load_config(args.config)
    rd = experiment.run_dir(cfg)
    cps = {c["checkpoint_id"]: c for c in rd.read_all("checkpoints") if c["arm"] == args.arm}
    rows = derive(rd, args.arm, experiment.build_runner_config(cfg, args.arm).n_replicates)

    z = np.load(rd.path("activations_{}.npz".format(args.arm)), allow_pickle=True)
    A, aids = z["activations"], list(z["checkpoint_ids"])
    ai = {c: i for i, c in enumerate(aids)}
    keep = sorted(c for c in aids if c in rows and c in cps)
    if args.cohort != "all":
        src = rd.branch_sources()
        keep = [c for c in keep if cohort_of(src.get((c, args.arm), "")) == args.cohort]
    ACT = np.ascontiguousarray(A[[ai[c] for c in keep]][:, LAYER, :], dtype=np.float32)
    del A, z

    lens = {}
    for r in rd.read_all("rollouts"):
        if r.get("arm") == args.arm or "arm" not in r:
            lens[r["rollout_id"]] = r.get("n_steps") or 0

    F = np.array([prefix_features.to_vector(cps[c]["features_26d"]) for c in keep])
    task = np.array([cps[c]["task_id"] for c in keep])
    qi = np.array([rows[c]["value_intervene"] for c in keep])
    pfail = np.array([rows[c]["p_fail_continue"] for c in keep])
    oracle = np.array([rows[c]["oracle_action"] for c in keep])
    idx = {c: i for i, c in enumerate(keep)}
    prog = np.array([cps[c]["t"] / lens[cps[c]["rollout_id"]] if lens.get(cps[c]["rollout_id"])
                     else np.nan for c in keep])
    roll = np.array([cps[c]["rollout_id"] for c in keep])

    band = pfail.max()
    conf = [i for i in range(len(keep)) if pfail[i] == band and not np.isnan(prog[i])]

    # ---- deterministic matching -------------------------------------------------
    by_game = defaultdict(list)
    for i in conf:
        by_game[task[i]].append(i)
    cands = []
    for g, v in by_game.items():
        for a, b in combinations(v, 2):
            if roll[a] == roll[b]:
                continue                                   # cross-rollout only
            dp, dq = abs(prog[a] - prog[b]), abs(qi[a] - qi[b])
            if dp <= DPROG_TOL and dq >= DQI_MIN:
                cands.append((dp, dq, a, b))
    cands.sort(key=lambda x: (x[0], -x[1]))                # greedy: tightest progress first
    used, pairs = set(), []
    for dp, dq, a, b in cands:
        if a in used or b in used:
            continue
        used.add(a); used.add(b)
        pairs.append({"a": a, "b": b, "dprog": dp, "dqi": dq,
                      "opposite": bool(oracle[a] != oracle[b]), "game": str(task[a])})
    if args.opposite_only:
        pairs = [p for p in pairs if p["opposite"]]

    print("=" * 78)
    print("MATCHED-PAIR TEST   cohort={}   dprog<={}  dQI>={}  opposite_only={}".format(
        args.cohort, DPROG_TOL, DQI_MIN, args.opposite_only))
    print("=" * 78)
    print("  conflict-set checkpoints        : {}".format(len(conf)))
    print("  candidate pairs before matching : {}".format(len(cands)))
    print("  matched pairs (no replacement)  : {}".format(len(pairs)))
    print("  of which opposite oracle action : {}".format(sum(p["opposite"] for p in pairs)))
    print("  distinct games represented      : {}".format(len({p["game"] for p in pairs})))
    if not pairs:
        print("\n  NO PAIRS -- test unsupported for this cohort under the fixed criteria.")
        return 0
    dps = np.array([p["dprog"] for p in pairs]); dqs = np.array([p["dqi"] for p in pairs])
    print("  progress distance   : min {:.3f}  median {:.3f}  max {:.3f}".format(
        dps.min(), np.median(dps), dps.max()))
    print("  true |Q_I| gap      : min {:.3f}  median {:.3f}  max {:.3f}".format(
        dqs.min(), np.median(dqs), dqs.max()))

    blocks = {"observables": F, "activations": ACT, "obs+activations": np.hstack([F, ACT])}
    per = defaultdict(list)
    ntest = []

    for s in range(args.seeds):
        tr, te = grouped_split(task, frac=0.30, seed=s)
        # A pair is scorable only if BOTH members sit in held-out games. Since pairs
        # are within-game and the split is by game, that is automatic -- but it is
        # asserted rather than assumed.
        sp = [p for p in pairs if te[p["a"]] and te[p["b"]]]
        ntest.append(len(sp))
        if len(sp) < 5:
            for nm in blocks:
                per[nm].append(np.nan)
            continue
        for nm, X in blocks.items():
            pred, _ = fit_reg(X[tr], qi[tr], task[tr], X[te])
            pos = {g: j for j, g in enumerate(np.where(te)[0])}
            ok = [int((pred[pos[p["a"]]] > pred[pos[p["b"]]]) == (qi[p["a"]] > qi[p["b"]]))
                  for p in sp]
            per[nm].append(float(np.mean(ok)))

    print("\n  scorable pairs per split : {}   (chance = 0.500)".format(ntest))
    print("\n  {:<20} {:>8} {:>7}  {:<6} {}".format("features", "mean", "sd", "sign", "per-split"))
    out = {"cohort": args.cohort, "dprog_tol": DPROG_TOL, "dqi_min": DQI_MIN,
           "opposite_only": bool(args.opposite_only), "n_pairs": len(pairs),
           "n_opposite": int(sum(p["opposite"] for p in pairs)),
           "n_games": len({p["game"] for p in pairs}),
           "scorable_per_split": ntest, "ranking_accuracy": {}}
    for nm in blocks:
        v = [x for x in per[nm] if not np.isnan(x)]
        if not v:
            print("    {:<20} {:>8}".format(nm, "--"))
            out["ranking_accuracy"][nm] = {"mean": None, "per_seed": per[nm]}
            continue
        mu = float(np.mean(v)); sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        above = all(x > 0.5 for x in v)
        out["ranking_accuracy"][nm] = {"mean": mu, "sd": sd, "above_chance_all_splits": above,
                                       "per_seed": [None if np.isnan(x) else float(x) for x in per[nm]]}
        print("    {:<20} {:>8.3f} {:>7.3f}  {:<6} [{}]".format(
            nm, mu, sd, "all>.5" if above else "mixed",
            " ".join("  n/a" if np.isnan(x) else "{:.3f}".format(x) for x in per[nm])))

    a = out["ranking_accuracy"]["observables"]["mean"]
    b = out["ranking_accuracy"]["activations"]["mean"]
    if a is not None and b is not None:
        print("\n  activations minus observables : {:+.3f}".format(b - a))
        out["delta_act_minus_obs"] = float(b - a)

    print("\n  POWER: with {} matched pairs split by game, roughly {} are scorable per".format(
        len(pairs), int(np.mean([x for x in ntest if x]))))
    print("  split. The binomial standard error on an accuracy at that n is about {:.2f},".format(
        0.5 / max(1.0, np.sqrt(np.mean([x for x in ntest if x] or [1])))))
    print("  so only a large separation is detectable. This test is thin by construction;")
    print("  it is reported because the pre-registered rule admitted it, and it is weak")
    print("  evidence in either direction.")

    os.makedirs("reports", exist_ok=True)
    tag = "opp" if args.opposite_only else "all"
    p = "reports/matched_pairs_{}_{}_{}.json".format(cfg["run_id"], args.cohort, tag)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote {}".format(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
