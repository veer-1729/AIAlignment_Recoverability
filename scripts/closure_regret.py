#!/usr/bin/env python
"""Q2: why does better Q_I prediction not reduce mean control regret?

Phase 3 found that activations improve Q_I prediction substantially and move mean
control regret by 0.013 with an inconsistent sign. Those two facts are compatible in
several ways and the memo could not distinguish them. The hypothesis this tests:

    the extra information sits where it cannot change the decision.

An estimate only changes an action when it crosses the boundary between the best and
second-best true action. If activations sharpen Q_I mostly on states whose best action
wins by a wide margin, the estimate improves and nothing happens. That is a specific,
measurable claim about WHERE the SSE reduction lands, not a story about why.

CONTROLLERS ARE THE EXISTING ONES, unchanged, so this is a decomposition of a result
rather than a new result:

    observable   Qhat_C from F ; Qhat_I from [F, oof Qhat_C]          (= R4_stacked)
    activation   Qhat_C from [F, ACT] ; Qhat_I from [F, ACT, oof Qhat_C]  (= R5_whitebox)

LAYER 24 IS PRIMARY. It was fixed before the full layer sweep. Layers 8 and 16 are
reported afterwards and descriptively; no conclusion depends on a layer chosen after
seeing results, and the 29-layer sweep is deliberately not re-run and re-selected from.

MARGIN DECILES ARE FIXED OVER ALL COHORT ROWS BEFORE ANY SPLIT. Binning on a quantity
computed from the test rows a model happened to get right would let the bins absorb the
effect being measured.

NO NEW DECISION-FOCUSED MODEL IS FITTED. The question here is only whether
decision-relevant signal exists to exploit; fitting something to exploit it would
answer a different question and would need its own pre-registration.

    python scripts/closure_regret.py --config configs/scale.yaml --cohort all
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cnc import experiment, utility
from cnc.features import prefix as prefix_features
from probe_study import derive, grouped_split, fit_reg, select_alpha
from conflict_set_analysis import cohort_of
from controllers import oof_predict, ACTS

PRIMARY_LAYER = 24
LAYERS = (8, 16, 24)
N_BOOT = 2000
CATS = ("unchanged", "obs_wrong_act_right", "obs_right_act_wrong", "wrong_to_wrong")


def boot_games(vals, games, fn, n=N_BOOT, seed=0):
    """Bootstrap clustered on game. Checkpoints inside one game share a task, a layout
    and often a trajectory prefix, so resampling rows would treat correlated
    observations as independent and produce an interval that is too tight."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(games)
    idx_by = {g: np.where(games == g)[0] for g in uniq}
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by[uniq[i]] for i in pick])
        try:
            out.append(fn(sel))
        except Exception:
            pass
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="B_montecarlo")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cohort", default="all", choices=("all", "original"))
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
    ACT = {L: np.ascontiguousarray(A[[ai[c] for c in keep]][:, L, :], dtype=np.float32)
           for L in LAYERS}
    del A, z

    F = np.array([prefix_features.to_vector(cps[c]["features_26d"]) for c in keep])
    task = np.array([cps[c]["task_id"] for c in keep])
    qc = np.array([rows[c]["value_continue"] for c in keep])
    qi = np.array([rows[c]["value_intervene"] for c in keep])
    pfail = np.array([rows[c]["p_fail_continue"] for c in keep])
    oracle = np.array([ACTS.index(rows[c]["oracle_action"]) for c in keep])
    Q = np.stack([qc, qi, np.zeros_like(qc)], axis=1)
    band = pfail.max()

    # Margin over ALL rows, before any split. Fixed edges.
    srt = np.sort(Q, axis=1)
    margin_all = srt[:, -1] - srt[:, -2]
    edges = np.percentile(margin_all, np.arange(10, 100, 10))

    print("=" * 78)
    print("Q2  DECISION RELEVANCE OF THE ACTIVATION Q_I ADVANTAGE   cohort={}".format(args.cohort))
    print("=" * 78)
    print("  n {}   games {}   conflict set {}   primary layer {}".format(
        len(keep), len(np.unique(task)), int((pfail == band).sum()), PRIMARY_LAYER))
    print("  true action margin over all rows: median {:.3f}  p10 {:.3f}  p90 {:.3f}".format(
        np.median(margin_all), np.percentile(margin_all, 10), np.percentile(margin_all, 90)))
    print("  decile edges (fixed before any split): {}".format(
        " ".join("{:.3f}".format(e) for e in edges)))

    out = {"cohort": args.cohort, "n": len(keep), "primary_layer": PRIMARY_LAYER,
           "margin_decile_edges": [float(e) for e in edges], "layers": {}}

    for L in LAYERS:
        tag = "L{}{}".format(L, "  [PRIMARY]" if L == PRIMARY_LAYER else "  (descriptive)")
        print("\n" + "#" * 78)
        print("# {}".format(tag))
        print("#" * 78)

        rec = []          # one entry per (split, test state)
        per_split_dreg = []
        for s in range(args.seeds):
            tr, te = grouped_split(task, frac=0.30, seed=s)
            gtr = task[tr]

            a_c = select_alpha(F[tr], qc[tr], gtr)
            qc_o, _ = fit_reg(F[tr], qc[tr], gtr, F[te], alpha=a_c)
            qc_o_oof = oof_predict(F[tr], qc[tr], gtr, a_c)
            qi_o, _ = fit_reg(np.hstack([F[tr], qc_o_oof[:, None]]), qi[tr], gtr,
                              np.hstack([F[te], qc_o[:, None]]))

            XA = np.hstack([F, ACT[L]])
            a_ca = select_alpha(XA[tr], qc[tr], gtr)
            qc_a, _ = fit_reg(XA[tr], qc[tr], gtr, XA[te], alpha=a_ca)
            qc_a_oof = oof_predict(XA[tr], qc[tr], gtr, a_ca)
            qi_a, _ = fit_reg(np.hstack([XA[tr], qc_a_oof[:, None]]), qi[tr], gtr,
                              np.hstack([XA[te], qc_a[:, None]]))

            zc = np.zeros(int(te.sum()))
            ch_o = np.stack([qc_o, qi_o, zc], 1).argmax(1)
            ch_a = np.stack([qc_a, qi_a, zc], 1).argmax(1)
            Qte, orte = Q[te], oracle[te]
            best = Qte.max(axis=1)
            reg_o = best - Qte[np.arange(len(ch_o)), ch_o]
            reg_a = best - Qte[np.arange(len(ch_a)), ch_a]
            per_split_dreg.append(float(reg_a.mean() - reg_o.mean()))

            gi = np.where(te)[0]
            for j in range(len(gi)):
                k = gi[j]
                ok_o, ok_a = ch_o[j] == orte[j], ch_a[j] == orte[j]
                if ch_o[j] == ch_a[j]:
                    cat = "unchanged"
                elif ok_a and not ok_o:
                    cat = "obs_wrong_act_right"
                elif ok_o and not ok_a:
                    cat = "obs_right_act_wrong"
                else:
                    cat = "wrong_to_wrong"
                rec.append({
                    "seed": s, "game": task[k], "margin": float(margin_all[k]),
                    "cat": cat, "dreg": float(reg_a[j] - reg_o[j]),
                    "flip": int(ch_o[j] != ch_a[j]),
                    "err_o": abs(qi[k] - qi_o[j]), "err_a": abs(qi[k] - qi_a[j]),
                    "sse_o": (qi[k] - qi_o[j]) ** 2, "sse_a": (qi[k] - qi_a[j]) ** 2,
                    "obs_correct": int(ok_o), "tau": float(qi[k] - qc[k]),
                    "conflict": int(pfail[k] == band)})

        n = len(rec)
        g = np.array([r["game"] for r in rec])
        dreg = np.array([r["dreg"] for r in rec])
        dsse = np.array([r["sse_o"] - r["sse_a"] for r in rec])
        derr = np.array([r["err_o"] - r["err_a"] for r in rec])
        mrg = np.array([r["margin"] for r in rec])
        cat = np.array([r["cat"] for r in rec])
        flip = np.array([r["flip"] for r in rec])
        tot_sse = dsse.sum()

        lo, hi = boot_games(dreg, g, lambda idx: float(dreg[idx].mean()))
        print("\n  mean regret difference (activation - observable): {:+.4f}  "
              "game-clustered 95% CI [{:+.4f}, {:+.4f}]".format(dreg.mean(), lo, hi))
        print("  per-split: [{}]   {}".format(
            " ".join("{:+.4f}".format(v) for v in per_split_dreg),
            "same" if (all(v > 0 for v in per_split_dreg) or all(v < 0 for v in per_split_dreg))
            else "FLIPS"))
        print("  total Q_I SSE reduction from activations: {:.1f}".format(tot_sse))
        print("  action flip rate: {:.3f}".format(flip.mean()))

        Lrec = {"n_state_rows": n, "mean_dregret": float(dreg.mean()),
                "dregret_ci95": [lo, hi], "per_split_dregret": per_split_dreg,
                "total_sse_reduction": float(tot_sse), "flip_rate": float(flip.mean()),
                "categories": {}, "margin_deciles": {}, "subsets": {}}

        # ---- the four exhaustive categories ------------------------------------
        print("\n  {:<24} {:>7} {:>8} {:>12} {:>10} {:>10} {:>10}".format(
            "category", "n", "frac", "reg contrib", "mean derr", "margin med", "SSE share"))
        for c in CATS:
            m = cat == c
            if not m.sum():
                print("  {:<24} {:>7}".format(c, 0)); continue
            contrib = float(dreg[m].sum() / n)
            share = float(dsse[m].sum() / tot_sse) if tot_sse else float("nan")
            Lrec["categories"][c] = {
                "n": int(m.sum()), "frac": float(m.mean()),
                "regret_contribution": contrib, "mean_derr": float(derr[m].mean()),
                "margin_median": float(np.median(mrg[m])),
                "margin_p10": float(np.percentile(mrg[m], 10)),
                "margin_p90": float(np.percentile(mrg[m], 90)),
                "sse_share": share}
            print("  {:<24} {:>7} {:>8.3f} {:>+12.4f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
                c, int(m.sum()), float(m.mean()), contrib, float(derr[m].mean()),
                float(np.median(mrg[m])), share))
        print("  (regret contributions sum to the mean difference above)")

        # ---- THE CRITICAL DECOMPOSITION ----------------------------------------
        same_act = cat == "unchanged"
        diff_act = ~same_act
        obs_wrong = np.array([not r["obs_correct"] for r in rec])
        print("\n  WHERE THE Q_I SSE REDUCTION LANDS")
        for label, m in (("states where both controllers agree", same_act),
                         ("states where they choose differently", diff_act),
                         ("states the observable controller gets WRONG", obs_wrong),
                         ("states it gets right", ~obs_wrong)):
            sh = float(dsse[m].sum() / tot_sse) if tot_sse else float("nan")
            Lrec["subsets"][label] = {"n": int(m.sum()), "frac_rows": float(m.mean()),
                                      "sse_share": sh,
                                      "mean_dregret": float(dreg[m].mean()) if m.sum() else None}
            print("    {:<42} {:>6} rows ({:>5.1%})   {:>6.1%} of SSE reduction".format(
                label, int(m.sum()), float(m.mean()), sh))

        # ---- margin deciles ------------------------------------------------------
        print("\n  BY TRUE ACTION-MARGIN DECILE (edges fixed before any split)")
        print("    {:>4} {:>7} {:>10} {:>11} {:>10} {:>10}".format(
            "dec", "n", "mean derr", "d regret", "flip rate", "SSE share"))
        db = np.searchsorted(edges, mrg)
        for d in range(10):
            m = db == d
            if not m.sum():
                continue
            sh = float(dsse[m].sum() / tot_sse) if tot_sse else float("nan")
            Lrec["margin_deciles"][str(d)] = {
                "n": int(m.sum()), "mean_derr": float(derr[m].mean()),
                "mean_dregret": float(dreg[m].mean()), "flip_rate": float(flip[m].mean()),
                "sse_share": sh, "margin_lo": float(mrg[m].min()), "margin_hi": float(mrg[m].max())}
            print("    {:>4} {:>7} {:>10.3f} {:>+11.4f} {:>10.3f} {:>10.3f}".format(
                d, int(m.sum()), float(derr[m].mean()), float(dreg[m].mean()),
                float(flip[m].mean()), sh))

        # ---- pre-registered subsets ---------------------------------------------
        print("\n  PRE-REGISTERED SUBSETS")
        tau_arr = np.array([r["tau"] for r in rec])
        conf = np.array([r["conflict"] for r in rec], dtype=bool)
        for label, m in (("tau <= 0", tau_arr <= 0), ("tau > 0", tau_arr > 0),
                         ("risk = 1 conflict set", conf)):
            if not m.sum():
                continue
            l2, h2 = boot_games(dreg[m], g[m], lambda idx: float(dreg[m][idx].mean()))
            sh = float(dsse[m].sum() / tot_sse) if tot_sse else float("nan")
            Lrec["subsets"][label] = {
                "n": int(m.sum()), "mean_derr": float(derr[m].mean()),
                "mean_dregret": float(dreg[m].mean()), "dregret_ci95": [l2, h2],
                "flip_rate": float(flip[m].mean()), "sse_share": sh}
            print("    {:<24} n={:<6} derr {:+.3f}   dregret {:+.4f} [{:+.4f}, {:+.4f}]"
                  "   flip {:.3f}   SSE share {:.3f}".format(
                      label, int(m.sum()), float(derr[m].mean()), float(dreg[m].mean()),
                      l2, h2, float(flip[m].mean()), sh))

        out["layers"]["L{}".format(L)] = Lrec

    os.makedirs("reports", exist_ok=True)
    p = "reports/closure_regret_{}_{}.json".format(cfg["run_id"], args.cohort)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote {}".format(p))
    print("\nHOW TO READ THIS")
    print("  The hypothesis is that the extra Q_I information sits where it cannot change")
    print("  an action. That predicts a large SSE share on states where both controllers")
    print("  agree, and a rising 'mean derr' with margin decile while 'd regret' stays flat")
    print("  near zero. The contrary pattern -- SSE concentrated at LOW margin and on")
    print("  baseline mistakes, with regret still not improving -- would mean the signal is")
    print("  decision-relevant but the MSE-trained probe is not exploiting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
