#!/usr/bin/env python
"""Feasibility census. Measures what the substrate can support, before anything is fitted.

Two questions the exploratory battery depends on, and both are cheap:

1. DOES THE ACTIVATION STORE HOLD EVERY LAYER? The layerwise analysis needs all of
   them. If they are already on disk it costs nothing; if only {8,16,24} were kept
   it means re-running forward passes over saved prompts. Print the array shape and
   settle it rather than assuming either way.

2. HOW MUCH MATCHED-PAIR DATA EXISTS? The same-game matched-state test needs two
   risk=1 checkpoints from the SAME game at similar trajectory progress with
   different Q_I. That is a demanding conjunction and it may simply not be there.
   This counts the candidates across a grid of thresholds so the criteria can be
   fixed against real availability -- BEFORE any model is fitted, so that choosing
   them cannot be tuning on the answer.

Reports counts only. Fits nothing, scores nothing, and touches no activation values
beyond the array's shape.

    python scripts/census.py --config configs/scale.yaml --cohort all
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cnc import experiment, utility
from probe_study import derive
from conflict_set_analysis import cohort_of

PROG_TOL = (0.05, 0.10, 0.15, 0.25)     # max |progress difference| within a pair
QI_SEP = (0.25, 0.50, 0.75, 1.00)       # min |Q_I difference| within a pair


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="B_montecarlo")
    ap.add_argument("--cohort", default="all", choices=("all", "original"))
    args = ap.parse_args()

    cfg = experiment.load_config(args.config)
    rd = experiment.run_dir(cfg)
    cps = {c["checkpoint_id"]: c for c in rd.read_all("checkpoints") if c["arm"] == args.arm}
    rows = derive(rd, args.arm, experiment.build_runner_config(cfg, args.arm).n_replicates)

    # --- 1. the activation store -------------------------------------------------
    path = rd.path("activations_{}.npz".format(args.arm))
    z = np.load(path, allow_pickle=True)
    A = z["activations"]
    print("=" * 78)
    print("ACTIVATION STORE")
    print("=" * 78)
    print("  path        : {}".format(path))
    print("  keys        : {}".format(list(z.keys())))
    print("  shape       : {}   dtype {}".format(A.shape, A.dtype))
    print("  on-disk MB  : {:.0f}".format(os.path.getsize(path) / 1e6))
    if A.ndim == 3:
        print("  -> {} checkpoints x {} LAYERS x {} hidden".format(*A.shape))
        print("  -> layerwise analysis needs NO forward passes" if A.shape[1] > 3
              else "  -> only a few layers stored; layerwise analysis needs forward passes")
    aids = list(z["checkpoint_ids"])
    del A, z

    # --- 2. matched-pair feasibility ---------------------------------------------
    keep = sorted(c for c in aids if c in rows and c in cps)
    if args.cohort != "all":
        src = rd.branch_sources()
        keep = [c for c in keep if cohort_of(src.get((c, args.arm), "")) == args.cohort]

    # Normalised progress. The base rollout's own length is the denominator; a
    # checkpoint two thirds of the way through a short episode is not at the same
    # place as one two thirds through a long one, but it is the closest thing to a
    # task-independent position that this substrate defines.
    lens = {}
    for r in rd.read_all("rollouts"):
        if r.get("arm") == args.arm or "arm" not in r:
            lens[r["rollout_id"]] = r.get("n_steps") or 0
    missing_len = sum(1 for c in keep if not lens.get(cps[c]["rollout_id"]))

    recs = []
    for c in keep:
        cp, rw = cps[c], rows[c]
        L = lens.get(cp["rollout_id"]) or 0
        recs.append({"cid": c, "game": cp.get("task_id"), "rollout": cp["rollout_id"],
                     "t": cp.get("t"), "prog": (cp["t"] / L) if L else None,
                     "pfail": rw["p_fail_continue"], "qi": rw["value_intervene"],
                     "qc": rw["value_continue"], "tau": rw["tau"],
                     "oracle": rw["oracle_action"]})

    band = max(r["pfail"] for r in recs)
    conf = [r for r in recs if r["pfail"] == band and r["prog"] is not None]

    print("\n" + "=" * 78)
    print("MATCHED-PAIR FEASIBILITY   cohort={}".format(args.cohort))
    print("=" * 78)
    print("  checkpoints            : {}".format(len(keep)))
    print("  rollout length missing : {}".format(missing_len))
    print("  risk band              : {:.2f}".format(band))
    print("  in conflict set        : {}".format(len(conf)))

    by_game = defaultdict(list)
    for r in conf:
        by_game[r["game"]].append(r)
    multi = {g: v for g, v in by_game.items() if len(v) >= 2}
    print("  games with >=1 conflict checkpoint : {}".format(len(by_game)))
    print("  games with >=2 (pairs possible)    : {}".format(len(multi)))
    print("  checkpoints inside those games     : {}".format(sum(len(v) for v in multi.values())))

    ps = np.array([r["prog"] for r in conf])
    print("  progress within conflict set       : "
          "min {:.2f}  p25 {:.2f}  median {:.2f}  p75 {:.2f}  max {:.2f}".format(
              ps.min(), np.percentile(ps, 25), np.median(ps),
              np.percentile(ps, 75), ps.max()))

    # All within-game candidate pairs, then the grid. Counting every pair rather
    # than a greedy matching, because the question here is how much the substrate
    # could support at all; the actual test will match without replacement.
    allp = []
    for g, v in multi.items():
        for a, b in combinations(v, 2):
            if a["rollout"] == b["rollout"]:
                same_roll = True
            else:
                same_roll = False
            allp.append({"dprog": abs(a["prog"] - b["prog"]),
                         "dqi": abs(a["qi"] - b["qi"]),
                         "opp": a["oracle"] != b["oracle"],
                         "same_rollout": same_roll})
    print("  candidate within-game pairs (all)  : {}".format(len(allp)))
    print("    of which same rollout            : {}".format(sum(p["same_rollout"] for p in allp)))
    print("    of which opposite oracle action  : {}".format(sum(p["opp"] for p in allp)))

    print("\n  pairs surviving both thresholds   [rows = max progress gap, cols = min |dQ_I|]")
    print("  {:>12}".format("") + "".join("{:>12}".format("dQI>={:.2f}".format(q)) for q in QI_SEP))
    grid = {}
    for tol in PROG_TOL:
        cells, cellso = [], []
        for q in QI_SEP:
            sel = [p for p in allp if p["dprog"] <= tol and p["dqi"] >= q]
            cells.append(len(sel))
            cellso.append(sum(p["opp"] for p in sel))
            grid["{:.2f}|{:.2f}".format(tol, q)] = {"n": len(sel),
                                                    "n_opposite_oracle": int(sum(p["opp"] for p in sel))}
        print("  dprog<={:.2f}".format(tol) + "".join("{:>12}".format(c) for c in cells))
        print("  {:>12}".format("(opposite)") + "".join("{:>12}".format(c) for c in cellso))

    out = {"cohort": args.cohort, "n_checkpoints": len(keep), "band": float(band),
           "n_conflict": len(conf), "n_games_with_pairs": len(multi),
           "n_candidate_pairs": len(allp),
           "n_opposite_oracle": int(sum(p["opp"] for p in allp)),
           "grid": grid}
    os.makedirs("reports", exist_ok=True)
    p = "reports/census_{}_{}.json".format(cfg["run_id"], args.cohort)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote {}".format(p))
    print("\nNo model was fitted and no activation value was read. These counts exist so")
    print("the matching criteria can be fixed against real availability before anything")
    print("is scored, rather than loosened afterwards to reach a number that works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
