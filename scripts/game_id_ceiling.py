#!/usr/bin/env python
"""POST-HOC DIAGNOSTIC -- not part of the pre-registered Stage 2 study.

How much of scheme 1's activation advantage could be game identification?

The scaled dataset holds 252 games and 504 rollouts -- exactly two rollouts per
game. The paper's leakage rule groups by base trajectory, so `grouped_by_rollout`
puts ~70% of test rollouts in the same split as a SIBLING ROLLOUT OF THE SAME GAME.
Q_C, Q_I and tau are strongly game-determined, and game identity is richly
recoverable from activations (task text, receptacle layout) while the 26
observables carry only a task-TYPE one-hot. So the two feature blocks are not
competing symmetrically: activations have a channel to the answer that
observables structurally lack.

This does not accuse the study of a bug. Rollout-level grouping IS the paper's
protocol, and scheme 1 is the faithful replication of it. The question is only
what scheme 1's numbers mean, and that cannot be answered from scheme 1 alone.

The instrument: a game-identity one-hot. Ridge on it learns a per-game mean, so
under scheme 1 it predicts each test row from its sibling rollout in train. That
block is PURE channel and nothing else -- it contains no state information at all,
not even the step index. Its R2 is therefore a ceiling on what the channel could
explain, measured rather than argued.

Three comparisons, in increasing sharpness:

    game_onehot          vs  obs+act_L24     can pure leakage match activations?
    obs+game - obs       vs  obs+act - obs   is the channel worth as much as the
                                             activations, added to the same base?
    obs+game+act_L24     vs  obs+game        THE SHARP ONE: does L24 still add
                                             anything once game identity is
                                             ALREADY given to the model? If not,
                                             scheme 1's advantage was channel.

Both split schemes are run, and scheme 2 is the negative control: with no game
shared between train and test, the one-hot can only ever predict the intercept,
so its R2 must come out at or below zero. If it does not, this script is wired
wrong and nothing else it prints should be believed.

Built-in consistency check: under scheme 1 / seed 0, `obs+act_L24` here must
reproduce probe_study.py's own numbers (Q_continue 0.732, Q_intervene 0.706,
tau 0.561). The script asserts nothing -- it prints them next to each other and
you check. If they disagree, the data prep has drifted and the rest is void.

    python scripts/game_id_ceiling.py --config configs/scale.yaml
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cnc import experiment
from cnc.features import prefix as prefix_features

# Reuse the study's own estimators rather than re-implementing them, so alpha
# selection, scaling and the split rule are identical by construction and not by
# inspection. Any divergence here would make the comparison meaningless.
from probe_study import derive, grouped_split, fit_reg

REFERENCE = {  # probe_study.py, scheme 1, seed 0, obs+act_L24 -- the wiring check
    "Q_continue": 0.732, "Q_intervene": 0.706, "tau": 0.561,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="B_montecarlo")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--layer", type=int, default=24,
                    help="single layer, to keep this cheap; 24 is where the "
                         "pre-registered study's advantage was largest")
    args = ap.parse_args()

    from sklearn.metrics import r2_score

    cfg = experiment.load_config(args.config)
    rd = experiment.run_dir(cfg)
    cps = {c["checkpoint_id"]: c for c in rd.read_all("checkpoints") if c["arm"] == args.arm}
    rows = derive(rd, args.arm, experiment.build_runner_config(cfg, args.arm).n_replicates)

    z = np.load(rd.path("activations_{}.npz".format(args.arm)), allow_pickle=True)
    A, aids = z["activations"], list(z["checkpoint_ids"])
    ai = {c: i for i, c in enumerate(aids)}
    keep = sorted(c for c in aids if c in rows and c in cps)
    n = len(keep)

    # One layer only -- this is a diagnostic, not a sweep.
    ACT = np.ascontiguousarray(A[[ai[c] for c in keep]][:, args.layer, :], dtype=np.float32)
    del A, z

    F = np.array([prefix_features.to_vector(cps[c]["features_26d"]) for c in keep])
    roll = np.array([cps[c]["rollout_id"] for c in keep])
    task = np.array([cps[c]["task_id"] for c in keep])

    # The instrument. Column order is fixed by sorted game id so the block is
    # reproducible; a game absent from train contributes an all-zero column there
    # and the model falls back to the intercept for it, which is the correct and
    # honest behaviour rather than something to patch around.
    games = sorted(set(task.tolist()))
    gidx = {g: i for i, g in enumerate(games)}
    G = np.zeros((n, len(games)), dtype=np.float32)
    G[np.arange(n), [gidx[t] for t in task]] = 1.0

    targets = {"Q_continue": np.array([rows[c]["value_continue"] for c in keep]),
               "Q_intervene": np.array([rows[c]["value_intervene"] for c in keep]),
               "tau": np.array([rows[c]["tau"] for c in keep])}

    blocks = {
        "observables":          F,
        "game_onehot":          G,
        "obs+game":             np.hstack([F, G]),
        "obs+act_L{}".format(args.layer):      np.hstack([F, ACT]),
        "obs+game+act_L{}".format(args.layer): np.hstack([F, G, ACT]),
    }

    print("POST-HOC DIAGNOSTIC -- outside the pre-registered study")
    print("checkpoints: {}   games: {}   rollouts: {}   layer: {}".format(
        n, len(games), len(set(roll.tolist())), args.layer))
    print("rollouts per game: {:.2f}".format(len(set(roll.tolist())) / max(1, len(games))))

    out = {"n": n, "n_games": len(games), "layer": args.layer, "schemes": {}}

    for scheme, groups in (("grouped_by_rollout", roll), ("held_out_game", task)):
        control = scheme == "held_out_game"
        print("\n" + "#" * 78)
        print("# {}{}".format(scheme,
              "   [NEGATIVE CONTROL: game_onehot must be <= 0 here]" if control else
              "   [the paper's split -- where the channel is open]"))
        print("#" * 78)

        # Fraction of test rows whose game is also represented in train. This is
        # the channel's width, and it is the number that makes the two schemes
        # different; printing it means the reader does not have to take the
        # premise of this script on faith.
        tr0, te0 = grouped_split(groups, frac=0.30, seed=0)
        shared = float(np.mean([t in set(task[tr0].tolist()) for t in task[te0]]))
        print("  seed 0: test rows whose game also appears in train: {:.1%}".format(shared))
        out["schemes"][scheme] = {"train_game_overlap_seed0": shared, "targets": {}}

        for tname, y in targets.items():
            print("\n  TARGET {}".format(tname))
            per_block = {}
            for bname, X in blocks.items():
                r2s = []
                for s in range(args.seeds):
                    tr, te = grouped_split(groups, frac=0.30, seed=s)
                    pred, _ = fit_reg(X[tr], y[tr], groups[tr], X[te])
                    r2s.append(r2_score(y[te], pred))
                mu, sd = float(np.mean(r2s)), float(np.std(r2s, ddof=1))
                per_block[bname] = {"mean_r2": mu, "sd": sd,
                                    "seed0_r2": float(r2s[0]), "per_seed": [float(v) for v in r2s]}
                note = ""
                if bname.startswith("obs+act") and not control:
                    ref = REFERENCE.get(tname)
                    if ref is not None:
                        note = "   <- wiring check vs probe_study seed0 {:.3f}".format(ref)
                if bname == "game_onehot" and control:
                    note = "   <- control: must be <= 0" + ("  OK" if mu <= 0.02 else "  !! WIRED WRONG")
                print("    {:<22} R2 {:>7.3f}  sd {:.3f}   (seed0 {:>7.3f}){}".format(
                    bname, mu, sd, r2s[0], note))

            base = per_block["observables"]["mean_r2"]
            act = per_block["obs+act_L{}".format(args.layer)]["mean_r2"]
            gam = per_block["obs+game"]["mean_r2"]
            both = per_block["obs+game+act_L{}".format(args.layer)]["mean_r2"]
            print("    {:-<74}".format(""))
            print("    activations add over observables      {:+.3f}".format(act - base))
            print("    game identity adds over observables   {:+.3f}".format(gam - base))
            print("    activations add over observables+game {:+.3f}   <- THE SHARP ONE".format(both - gam))
            out["schemes"][scheme]["targets"][tname] = {
                "blocks": per_block,
                "act_over_obs": act - base,
                "game_over_obs": gam - base,
                "act_over_obs_plus_game": both - gam,
            }

    os.makedirs("reports", exist_ok=True)
    p = "reports/game_id_ceiling_{}.json".format(cfg["run_id"])
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote {}".format(p))
    print("\nHOW TO READ THIS")
    print("  'activations add over observables+game' is the quantity that matters.")
    print("  If it is close to 'activations add over observables', the activations")
    print("  carry state-specific signal that game identity does not supply, and")
    print("  scheme 1's advantage is not explained by the sibling-rollout channel.")
    print("  If it collapses toward zero, scheme 1's advantage was largely channel")
    print("  and only the held_out_game scheme should be quoted.")
    print("  Either way this bounds the channel; it does not replace scheme 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
