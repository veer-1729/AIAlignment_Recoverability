#!/usr/bin/env python
"""Stage 2, step 2: do activations predict control quantities beyond the 26 features?

The question is comparative, not absolute. A probe that reads tau off the
residual stream is only interesting if it beats what you can already get from
the paper's 26 observable prefix features -- step index, history length,
admissible-command count, action-type indicators, self-reported confidence and
so on. So every target is fit three ways: features alone, activations alone, and
both, and the number that matters is the delta.

Targets, all derived from Arm B, whose Q(s,intervene) averages five expert seeds.
That matters: the ALFWorld expert is stochastic, so a single expert branch is one
draw rather than an expected value, and probes should not be trained to predict
a coin flip.

Evaluation is on HELD-OUT GAMES (split_task_level), not held-out checkpoints.
Checkpoints from one trajectory share nearly all their prefix, so a
checkpoint-level split would let the probe memorise a game and report a score
that will not survive contact with a new one.

p >> n here (3584 activation dims against a few hundred checkpoints), so both
probes are linear with the regularisation strength chosen on validation.

    python scripts/probe_pilot.py --config configs/validation.yaml
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from cnc import experiment, utility
from cnc.features import prefix as prefix_features
from cnc.storage import read_jsonl

ALPHAS = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def derive_rows(rd, arm):
    by_cp = defaultdict(list)
    for b in rd.read_branches():
        if b["arm"] == arm:
            by_cp[b["checkpoint_id"]].append(b)
    out = {}
    for cid, bs in by_cp.items():
        if not all(x["replay_verified"] for x in bs):
            continue
        if {x["action"] for x in bs} != set(utility.ACTIONS):
            continue
        out[cid] = utility.values_from_branches(cid, arm, bs).to_row()
    return out


def fit_reg(Xtr, ytr, Xva, yva, Xte, yte):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    best, best_a = -np.inf, None
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        s = r2_score(yva, m.predict(Xva))
        if s > best:
            best, best_a = s, a
    m = Ridge(alpha=best_a).fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    return r2_score(yte, m.predict(Xte)), best_a


def fit_clf(Xtr, ytr, Xva, yva, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    best, best_c = -np.inf, None
    for c in CS:
        m = LogisticRegression(C=c, max_iter=2000).fit(Xtr, ytr)
        s = f1_score(yva, m.predict(Xva), average="macro")
        if s > best:
            best, best_c = s, c
    m = LogisticRegression(C=best_c, max_iter=2000).fit(
        np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    p = m.predict(Xte)
    return accuracy_score(yte, p), f1_score(yte, p, average="macro"), best_c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="B_montecarlo")
    ap.add_argument("--layers", default="", help="comma list; default = every 4th + last")
    args = ap.parse_args()

    from sklearn.preprocessing import StandardScaler

    cfg = experiment.load_config(args.config)
    rd = experiment.run_dir(cfg)
    cps = {c["checkpoint_id"]: c for c in rd.read_all("checkpoints") if c["arm"] == args.arm}
    rows = derive_rows(rd, args.arm)

    z = np.load(rd.path("activations_{}.npz".format(args.arm)), allow_pickle=True)
    A, aids = z["activations"], list(z["checkpoint_ids"])
    idx = {cid: i for i, cid in enumerate(aids)}

    keep = [c for c in aids if c in rows and c in cps]
    print("checkpoints with activations + derived values: {}".format(len(keep)))

    split = np.array([cps[c].get("split_task_level", "unused") for c in keep])
    F = np.array([prefix_features.to_vector(cps[c]["features_26d"]) for c in keep])
    Aa = np.array([A[idx[c]] for c in keep], dtype=np.float32)
    print("split sizes: {}".format({s: int((split == s).sum()) for s in set(split)}))

    tr, va, te = split == "train", split == "val", split == "test"
    if te.sum() < 20:
        print("WARNING: test split has only {} checkpoints".format(int(te.sum())))

    targets = {
        "continue_success":  ("clf", np.array([rows[c]["p_success_continue"] > 0.5 for c in keep], int)),
        "Q_continue":        ("reg", np.array([rows[c]["value_continue"] for c in keep])),
        "Q_intervene":       ("reg", np.array([rows[c]["value_intervene"] for c in keep])),
        "tau":               ("reg", np.array([rows[c]["tau"] for c in keep])),
        "oracle_action":     ("clf", np.array([utility.ACTIONS.index(rows[c]["oracle_action"]) for c in keep])),
    }

    n_layers = A.shape[1]
    layers = ([int(x) for x in args.layers.split(",") if x]
              or list(range(0, n_layers, 4)) + [n_layers - 1])
    layers = sorted(set(l for l in layers if 0 <= l < n_layers))
    print("layers probed: {}\n".format(layers))

    results = {}
    for name, (kind, y) in targets.items():
        print("=" * 74)
        print("TARGET  {}   ({})".format(name, kind))
        if kind == "clf":
            print("  class balance: {}".format(dict(zip(*np.unique(y, return_counts=True)))))
        print("=" * 74)

        def run(X, label):
            sc = StandardScaler().fit(X[tr])
            Xs = sc.transform(X)
            if kind == "reg":
                r2, a = fit_reg(Xs[tr], y[tr], Xs[va], y[va], Xs[te], y[te])
                print("  {:<34} R2 = {:>7.3f}   (alpha {:g})".format(label, r2, a))
                return {"r2": float(r2)}
            acc, f1, c = fit_clf(Xs[tr], y[tr], Xs[va], y[va], Xs[te], y[te])
            print("  {:<34} acc = {:>6.3f}   macroF1 = {:>6.3f}   (C {:g})".format(label, acc, f1, c))
            return {"acc": float(acc), "macro_f1": float(f1)}

        results[name] = {"features_26d": run(F, "26-D features (baseline)")}
        best = None
        for L in layers:
            r = run(Aa[:, L, :], "activations L{}".format(L))
            key = "r2" if kind == "reg" else "macro_f1"
            if best is None or r[key] > best[1][key]:
                best = (L, r)
            results[name]["act_L{}".format(L)] = r
        results[name]["best_layer"] = best[0]
        results[name]["combined"] = run(
            np.hstack([F, Aa[:, best[0], :]]), "26-D + activations L{}".format(best[0]))

        key = "r2" if kind == "reg" else "macro_f1"
        b, a_, c_ = results[name]["features_26d"][key], best[1][key], results[name]["combined"][key]
        print("  -> baseline {:.3f} | best activations {:.3f} (L{}) | combined {:.3f}".format(
            b, a_, best[0], c_))
        print("  -> DELTA over baseline: activations {:+.3f}   combined {:+.3f}\n".format(a_ - b, c_ - b))

    os.makedirs("reports", exist_ok=True)
    p = "reports/probe_pilot_{}.json".format(cfg["run_id"])
    with open(p, "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote {}".format(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
