#!/usr/bin/env python
"""E: where in the network does control-relevant information appear?

The pre-registered probe used layers {8, 16, 24} and every result rose with depth.
Three points make a line, so that pattern is suggestive and nothing more. The store
holds all 29 layers, so the full curve costs compute and no new data.

FOUR CURVES, each the gain over the 26-observable baseline under held-out games:

    Q_C          the actor's own continuation prospects
    Q_I          the expert's value -- what recoverability actually is
    tau          the control-relevant difference
    Q_I | risk=1 the same question inside the conflict set, where Q_C is pinned and
                 the signal cannot be a better picture of continuation state

The questions are ordinal, not just "is it decodable": does the expert's value emerge
LATER than the actor's own prospects? Does tau have a profile of its own, or is it
just whichever of its two components is weaker at that depth?

EVERY LAYER IS REPORTED. No best layer is chosen and no others hidden -- selecting on
the test set would inflate the estimate by exactly the size of the search, which is
the mistake the whole programme is built to avoid.

ORACLE-ACTION CLASSIFICATION IS NOT RUN, and that is a cost decision rather than a
result decision. At 29 layers x 5 seeds x 26 logistic fits it is ~3,700 additional
multinomial fits over 3,584 features, comparable to the entire scaled study. It is
also not run on a subset of layers, because choosing which layers to classify would be
the search this avoids.

    python scripts/layerwise.py --config configs/scale.yaml --cohort all
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cnc import experiment
from cnc.features import prefix as prefix_features
from probe_study import derive, grouped_split, fit_reg
from conflict_set_analysis import cohort_of


def curve_row(name, deltas, sds, consistent):
    """One line of the layer curve, drawn as text so the shape is visible without a
    plotting dependency. Scale is fixed across every curve so they are comparable."""
    lo, hi = -0.05, 0.30
    cells = []
    for d in deltas:
        if d is None or np.isnan(d):
            cells.append(" ")
            continue
        f = (d - lo) / (hi - lo)
        cells.append(" .:-=+*#@"[min(8, max(0, int(f * 8)))])
    return "".join(cells)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="B_montecarlo")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cohort", default="all", choices=("all", "original"))
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
    if args.cohort != "all":
        src = rd.branch_sources()
        keep = [c for c in keep if cohort_of(src.get((c, args.arm), "")) == args.cohort]
    sel = [ai[c] for c in keep]
    # float16 on disk; kept that way in memory and upcast one layer at a time, so the
    # peak is ~30 MB per layer rather than 800 MB for the whole store.
    Aall = A[sel]
    del A, z
    n_layers = Aall.shape[1]

    F = np.array([prefix_features.to_vector(cps[c]["features_26d"]) for c in keep])
    task = np.array([cps[c]["task_id"] for c in keep])
    qc = np.array([rows[c]["value_continue"] for c in keep])
    qi = np.array([rows[c]["value_intervene"] for c in keep])
    tau = np.array([rows[c]["tau"] for c in keep])
    pfail = np.array([rows[c]["p_fail_continue"] for c in keep])
    band = pfail.max()
    conflict = pfail == band

    print("=" * 78)
    print("LAYERWISE EMERGENCE   cohort={}   layers={}   n={}".format(
        args.cohort, n_layers, len(keep)))
    print("=" * 78)
    print("  held-out games, {} splits, gain over the 26-observable baseline".format(args.seeds))
    print("  conflict set n = {}".format(int(conflict.sum())))

    curves = {"Q_continue": (qc, None), "Q_intervene": (qi, None),
              "tau": (tau, None), "Q_intervene|risk=1": (qi, conflict)}

    # Baselines first: one per curve, independent of layer.
    print("\n  observable baselines (mean R2 over splits)")
    base = {}
    for cname, (y, mask) in curves.items():
        vals = []
        for s in range(args.seeds):
            tr, te = grouped_split(task, frac=0.30, seed=s)
            if mask is not None:
                tr, te = tr & mask, te & mask
            p, _ = fit_reg(F[tr], y[tr], task[tr], F[te])
            vals.append(r2_score(y[te], p))
        base[cname] = vals
        print("    {:<22} {:.3f}".format(cname, float(np.mean(vals))))

    out = {"cohort": args.cohort, "n": len(keep), "n_layers": int(n_layers),
           "n_conflict": int(conflict.sum()),
           "baselines": {k: {"mean": float(np.mean(v)), "per_seed": [float(x) for x in v]}
                         for k, v in base.items()},
           "curves": {}}

    for cname, (y, mask) in curves.items():
        print("\n" + "#" * 78)
        print("# {}   (baseline R2 {:.3f})".format(cname, float(np.mean(base[cname]))))
        print("#" * 78)
        print("  {:>5} {:>8} {:>7} {:>7}  {}".format("layer", "delta", "sd", "sign", "per-split"))
        rec, means = {}, []
        for L in range(n_layers):
            XL = np.ascontiguousarray(Aall[:, L, :], dtype=np.float32)
            X = np.hstack([F, XL])
            ds = []
            for s in range(args.seeds):
                tr, te = grouped_split(task, frac=0.30, seed=s)
                if mask is not None:
                    tr, te = tr & mask, te & mask
                p, _ = fit_reg(X[tr], y[tr], task[tr], X[te])
                ds.append(r2_score(y[te], p) - base[cname][s])
            del XL, X
            mu = float(np.mean(ds)); sd = float(np.std(ds, ddof=1))
            same = all(v > 0 for v in ds) or all(v < 0 for v in ds)
            rec[str(L)] = {"delta": mu, "sd": sd, "consistent": bool(same),
                           "per_seed": [float(v) for v in ds]}
            means.append(mu)
            print("  {:>5} {:>+8.3f} {:>7.3f} {:>7}  [{}]".format(
                L, mu, sd, "same" if same else "FLIPS",
                " ".join("{:+.3f}".format(v) for v in ds)))

        arr = np.array(means)
        first_pos = next((L for L in range(n_layers)
                          if rec[str(L)]["delta"] > 0.02 and rec[str(L)]["consistent"]), None)
        diffs = np.diff(arr)
        out["curves"][cname] = {
            "per_layer": rec,
            "argmax_layer": int(np.argmax(arr)), "max_delta": float(arr.max()),
            "first_layer_delta_gt_0.02_and_consistent": first_pos,
            "monotone_nondecreasing": bool((diffs >= -0.005).all()),
            "frac_layers_consistent": float(np.mean([rec[str(L)]["consistent"]
                                                     for L in range(n_layers)])),
            "shape": curve_row(cname, means, None, None)}
        print("\n  shape (scale -0.05 to +0.30, layer 0 leftmost):")
        print("    |{}|".format(curve_row(cname, means, None, None)))
        print("    peak at layer {} ({:+.3f});  first consistent delta > 0.02 at layer {};"
              "  monotone: {}".format(int(np.argmax(arr)), float(arr.max()),
                                      first_pos if first_pos is not None else "never",
                                      out["curves"][cname]["monotone_nondecreasing"]))

    print("\n" + "#" * 78)
    print("# ORDINAL COMPARISON -- does the expert's value emerge later than the actor's?")
    print("#" * 78)
    for cname in curves:
        c = out["curves"][cname]
        print("  {:<22} first consistent >0.02 at layer {:<6} peak layer {:<4} peak {:+.3f}".format(
            cname,
            str(c["first_layer_delta_gt_0.02_and_consistent"]),
            c["argmax_layer"], c["max_delta"]))

    os.makedirs("reports", exist_ok=True)
    p = "reports/layerwise_{}_{}.json".format(cfg["run_id"], args.cohort)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote {}".format(p))
    print("\nNULL CANDIDATES: layers whose delta is sign-inconsistent across splits")
    for cname in curves:
        bad = [L for L in range(n_layers) if not out["curves"][cname]["per_layer"][str(L)]["consistent"]]
        print("  {:<22} {} of {} layers FLIP  {}".format(
            cname, len(bad), n_layers, bad[:12] + (["..."] if len(bad) > 12 else [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
