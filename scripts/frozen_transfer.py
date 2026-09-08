#!/usr/bin/env python
"""D: does the SAME linear direction transfer across independently collected data?

Refitting a probe in two datasets and getting a result twice proves decodability
twice. It does not show the two probes found the same thing. This freezes scaler,
coefficients, intercept and alpha on one cohort and applies them, untouched, to the
other. No destination-side fitting, scaling or calibration of any kind.

A CORRECTION TO THE OBVIOUS DESIGN. The brief says train on the 1,971 cohort and
evaluate on the 928. Those are NOT disjoint -- the 928 is a subset of the 1,971 --
so that direction would train on data containing its own test set and report a
meaningless number. The disjoint decomposition is used instead:

    original  = the pre-disk-failure execution            (n = 928)
    backfill  = everything collected afterwards            (n = 1,971 - 928)

Both directions are run between those two.

A SECOND CONFOUND, MEASURED RATHER THAN ASSUMED AWAY. The cohorts were split by
execution, not by game, and shards walked a checkpoint list that sorts by task id --
so the same game can appear in both. A probe could then transfer by recognising games
it saw in training. Game overlap is reported, and every transfer number is given twice:
over the whole destination cohort, and over destination checkpoints whose game does
NOT appear in the source. The second is the one that answers the question.

Cosine similarity between standardised coefficient vectors is reported, and
deliberately not led with: two probes can point in similar directions and still
transfer badly, and transfer is what was asked.

    python scripts/frozen_transfer.py --config configs/scale.yaml
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cnc import experiment
from cnc.features import prefix as prefix_features
from probe_study import derive, grouped_split, fit_reg, select_alpha, LAYERS
from conflict_set_analysis import cohort_of


def fit_reg_frozen(Xtr, ytr, gtr):
    """probe_study.fit_reg, but the fitted scaler and model are returned rather than
    consumed. Identical alpha selection and identical Ridge -- the wiring check below
    is that it reproduces fit_reg's in-cohort R2 to three decimals."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    a = select_alpha(Xtr, ytr, gtr)
    sc = StandardScaler().fit(Xtr)
    model = Ridge(alpha=a).fit(sc.transform(Xtr), ytr)
    return sc, model, a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="B_montecarlo")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    from sklearn.metrics import r2_score

    cfg = experiment.load_config(args.config)
    rd = experiment.run_dir(cfg)
    cps = {c["checkpoint_id"]: c for c in rd.read_all("checkpoints") if c["arm"] == args.arm}
    rows = derive(rd, args.arm, experiment.build_runner_config(cfg, args.arm).n_replicates)
    src_map = rd.branch_sources()

    z = np.load(rd.path("activations_{}.npz".format(args.arm)), allow_pickle=True)
    A, aids = z["activations"], list(z["checkpoint_ids"])
    ai = {c: i for i, c in enumerate(aids)}
    keep = sorted(c for c in aids if c in rows and c in cps)
    ACT = {L: np.ascontiguousarray(A[[ai[c] for c in keep]][:, L, :], dtype=np.float32)
           for L in LAYERS}
    del A, z

    F = np.array([prefix_features.to_vector(cps[c]["features_26d"]) for c in keep])
    task = np.array([cps[c]["task_id"] for c in keep])
    qc = np.array([rows[c]["value_continue"] for c in keep])
    qi = np.array([rows[c]["value_intervene"] for c in keep])
    tau = np.array([rows[c]["tau"] for c in keep])
    pfail = np.array([rows[c]["p_fail_continue"] for c in keep])
    coh = np.array(["original" if cohort_of(src_map.get((c, args.arm), "")) == "original"
                    else "backfill" for c in keep])

    is_o, is_b = coh == "original", coh == "backfill"
    games_o, games_b = set(task[is_o].tolist()), set(task[is_b].tolist())
    shared = games_o & games_b

    print("=" * 78)
    print("FROZEN CROSS-COHORT TRANSFER")
    print("=" * 78)
    print("  original : {:>5} checkpoints, {:>4} games".format(int(is_o.sum()), len(games_o)))
    print("  backfill : {:>5} checkpoints, {:>4} games".format(int(is_b.sum()), len(games_b)))
    print("  games in BOTH cohorts        : {}".format(len(shared)))
    print("  original rows in shared games: {} ({:.1%})".format(
        int(np.isin(task[is_o], list(shared)).sum()),
        float(np.isin(task[is_o], list(shared)).mean())))
    print("  backfill rows in shared games: {} ({:.1%})".format(
        int(np.isin(task[is_b], list(shared)).sum()),
        float(np.isin(task[is_b], list(shared)).mean())))

    targets = {"Q_continue": qc, "Q_intervene": qi, "tau": tau}
    band = pfail.max()
    out = {"n_original": int(is_o.sum()), "n_backfill": int(is_b.sum()),
           "n_shared_games": len(shared), "wiring_check": {}, "transfer": {}}

    # ---- wiring check ------------------------------------------------------------
    print("\n  WIRING CHECK  fit_reg_frozen vs fit_reg, in-cohort, held-out games, seed 0")
    ok_all = True
    for L in LAYERS:
        X = np.hstack([F, ACT[L]])
        tr, te = grouped_split(task, frac=0.30, seed=0)
        p1, _ = fit_reg(X[tr], qi[tr], task[tr], X[te])
        sc, m, _ = fit_reg_frozen(X[tr], qi[tr], task[tr])
        p2 = m.predict(sc.transform(X[te]))
        r1, r2 = r2_score(qi[te], p1), r2_score(qi[te], p2)
        ok = abs(r1 - r2) < 5e-4
        ok_all &= ok
        out["wiring_check"]["L{}".format(L)] = {"fit_reg": float(r1), "frozen": float(r2),
                                                "match": bool(ok)}
        print("    L{:<3} fit_reg {:.4f}   frozen {:.4f}   {}".format(
            L, r1, r2, "OK" if ok else "!! MISMATCH -- transfer numbers are void"))
    if not ok_all:
        print("\n  ABORTING: the frozen fitter does not reproduce the study's own estimator.")
        return 1

    # ---- transfer ----------------------------------------------------------------
    for direction, sm, dm in (("original -> backfill", is_o, is_b),
                              ("backfill -> original", is_b, is_o)):
        print("\n" + "#" * 78)
        print("# {}".format(direction))
        print("#" * 78)
        dm_unseen = dm & ~np.isin(task, list(games_o & games_b))
        print("  destination rows: {} total, {} in games unseen by the source".format(
            int(dm.sum()), int(dm_unseen.sum())))
        dres = {}
        for tname, y in targets.items():
            print("\n  TARGET {}".format(tname))
            print("    {:<26} {:>9} {:>9} {:>9} {:>8} {:>7}".format(
                "block", "within", "frozen", "unseen", "degrade", "cos"))
            for L in LAYERS:
                X = np.hstack([F, ACT[L]])
                # within-source: the study's own protocol, grouped by game inside source
                w = []
                for s in range(args.seeds):
                    st = np.where(sm)[0]
                    tr_l, te_l = grouped_split(task[sm], frac=0.30, seed=s)
                    p, _ = fit_reg(X[st[tr_l]], y[st[tr_l]], task[st[tr_l]], X[st[te_l]])
                    w.append(r2_score(y[st[te_l]], p))
                within = float(np.mean(w))

                sc, m, alpha = fit_reg_frozen(X[sm], y[sm], task[sm])
                froz = float(r2_score(y[dm], m.predict(sc.transform(X[dm]))))
                unseen = (float(r2_score(y[dm_unseen], m.predict(sc.transform(X[dm_unseen]))))
                          if dm_unseen.sum() > 30 else float("nan"))

                sc2, m2, _ = fit_reg_frozen(X[dm], y[dm], task[dm])
                cos = float(np.dot(m.coef_, m2.coef_) /
                            (np.linalg.norm(m.coef_) * np.linalg.norm(m2.coef_) + 1e-12))
                dres["{}|L{}".format(tname, L)] = {
                    "within_source": within, "frozen_dest": froz, "frozen_dest_unseen": unseen,
                    "degradation": float(within - froz), "cosine": cos, "alpha": alpha,
                    "within_per_seed": [float(x) for x in w]}
                print("    obs+act_L{:<17} {:>9.3f} {:>9.3f} {:>9.3f} {:>8.3f} {:>7.3f}".format(
                    L, within, froz, unseen, within - froz, cos))

            # risk = 1 subset, same freeze, evaluated only inside the conflict set
            cm = dm & (pfail == band)
            if cm.sum() > 50:
                for L in LAYERS:
                    X = np.hstack([F, ACT[L]])
                    sc, m, _ = fit_reg_frozen(X[sm], y[sm], task[sm])
                    rr = float(r2_score(y[cm], m.predict(sc.transform(X[cm]))))
                    dres["{}|L{}|risk1".format(tname, L)] = {"frozen_dest_risk1": rr,
                                                             "n": int(cm.sum())}
                print("    risk=1 subset (n={}):  ".format(int(cm.sum())) + "  ".join(
                    "L{} {:.3f}".format(L, dres["{}|L{}|risk1".format(tname, L)]["frozen_dest_risk1"])
                    for L in LAYERS))
        out["transfer"][direction] = dres

    os.makedirs("reports", exist_ok=True)
    p = "reports/frozen_transfer_{}.json".format(cfg["run_id"])
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote {}".format(p))
    print("\nHOW TO READ THIS")
    print("  'frozen' is the destination R2 of a probe that never saw the destination.")
    print("  'unseen' repeats it over destination games absent from the source, which is")
    print("  the number that cannot be explained by game recognition. 'within' is the")
    print("  source's own held-out-game performance, so degradation is the cost of moving")
    print("  to independently collected data. A frozen R2 near zero would mean the earlier")
    print("  results are dataset-specific linear correlates rather than a stable direction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
