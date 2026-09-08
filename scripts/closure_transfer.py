#!/usr/bin/env python
"""Q1: does Q_I have a stable DIRECTION, or just better-behaved calibration?

Phase 3 compared frozen destination R2 against the SOURCE probe's own held-out-game
R2. Those are different tasks on different rows, so a gap between them confounds three
things: direction instability, calibration/base-rate shift, and the destination simply
being harder. The claim "the tau direction does not transfer" rested on that gap and
cannot be supported by it.

This fixes the comparison. The frozen source probe and a DESTINATION-REFIT probe are
scored on IDENTICAL destination rows, so difficulty cancels; and calibration-free
statistics (Spearman, raw-space coefficient cosine) are reported beside R2, so scale
and offset are separable from ranking.

THE EVALUATION SET IS NOT REDEFINED. Exactly the existing A->B unseen set:
`is_original & ~isin(task, games_original & games_backfill)` -- 445 rows, 57 games. The
script asserts that count and aborts otherwise, because a different count would mean a
different test and every number would be incomparable to Phase 3.

THE COMPARATOR is trained on the 483 `original` rows whose game IS shared with
backfill, and tested on the 445. Game-disjoint by construction, identical test rows.

COEFFICIENTS ARE COMPARED IN RAW COORDINATES. A ridge on standardised inputs predicts
b0 + sum c_j (x_j - mu_j)/s_j, so the raw-space weight is c_j/s_j. Two probes with
independently fitted scalers have different s, so comparing c directly compares scaler
artifacts -- which is what Phase 3's cosine did. Both are reported, the standardised
one marked superseded.

RECALIBRATION IS A DIAGNOSTIC AND IS FITTED ON DESTINATION TRAINING ROWS ONLY.

    python scripts/closure_transfer.py --config configs/scale.yaml
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
from frozen_transfer import fit_reg_frozen

EXPECT_UNSEEN_ROWS = 445
EXPECT_UNSEEN_GAMES = 57
N_SIZE_MATCH = 5      # draws for the size-matched frozen comparator


def raw_coef(scaler, model):
    """Ridge weights mapped back to raw input units: c_j / sigma_j."""
    s = np.asarray(scaler.scale_, dtype=np.float64).copy()
    s[s == 0] = 1.0
    return np.asarray(model.coef_, dtype=np.float64) / s


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="B_montecarlo")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr, spearmanr

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
    coh = np.array(["original" if cohort_of(src_map.get((c, args.arm), "")) == "original"
                    else "backfill" for c in keep])

    is_o, is_b = coh == "original", coh == "backfill"
    shared = set(task[is_o].tolist()) & set(task[is_b].tolist())
    unseen = is_o & ~np.isin(task, list(shared))       # THE fixed evaluation set
    destrain = is_o & np.isin(task, list(shared))      # comparator training rows

    n_u, n_g = int(unseen.sum()), len(np.unique(task[unseen]))
    print("=" * 78)
    print("Q1  DIRECTION vs CALIBRATION IN FROZEN TRANSFER   (backfill -> original)")
    print("=" * 78)
    print("  source (backfill)      : {} rows".format(int(is_b.sum())))
    print("  destination unseen set : {} rows, {} games   [expect {} / {}]".format(
        n_u, n_g, EXPECT_UNSEEN_ROWS, EXPECT_UNSEEN_GAMES))
    print("  comparator train rows  : {} (destination rows in shared games)".format(
        int(destrain.sum())))
    if n_u != EXPECT_UNSEEN_ROWS or n_g != EXPECT_UNSEEN_GAMES:
        print("\n  ABORTING: the evaluation set is not the one Phase 3 used. Every number")
        print("  here would be incomparable to the existing result.")
        return 1
    print("  evaluation set matches Phase 3 exactly.")

    targets = {"Q_intervene": qi, "Q_continue": qc, "tau": tau}
    out = {"n_unseen": n_u, "n_unseen_games": n_g, "n_comparator_train": int(destrain.sum()),
           "targets": {}}

    frozen_pred = {}      # (target, layer) -> predictions on the 445, for part (d)

    for tname, y in targets.items():
        print("\n" + "#" * 78)
        print("# TARGET {}".format(tname))
        print("#" * 78)
        print("  {:<7} {:>8} {:>8} {:>8} {:>9} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8}"
              " {:>9} {:>9}".format(
                  "layer", "srcHO", "frozen", "refit", "froz-ref", "RMSE", "pears", "spear",
                  "cal_a", "cal_b", "recal_R2", "sizematch", "sm-refit"))
        trec = {}
        for L in LAYERS:
            X = np.hstack([F, ACT[L]])

            # source held-out-game performance, the study's own protocol inside backfill
            w = []
            st = np.where(is_b)[0]
            for s in range(args.seeds):
                tr_l, te_l = grouped_split(task[is_b], frac=0.30, seed=s)
                p, _ = fit_reg(X[st[tr_l]], y[st[tr_l]], task[st[tr_l]], X[st[te_l]])
                w.append(r2_score(y[st[te_l]], p))
            src_ho = float(np.mean(w))

            # frozen source probe
            sc_s, m_s, a_s = fit_reg_frozen(X[is_b], y[is_b], task[is_b])
            pf = m_s.predict(sc_s.transform(X[unseen]))
            frozen_pred[(tname, L)] = pf
            r_froz = float(r2_score(y[unseen], pf))

            # destination-refit comparator: identical test rows, game-disjoint training
            sc_d, m_d, a_d = fit_reg_frozen(X[destrain], y[destrain], task[destrain])
            pr = m_d.predict(sc_d.transform(X[unseen]))
            r_ref = float(r2_score(y[unseen], pr))

            rmse = float(np.sqrt(np.mean((y[unseen] - pf) ** 2)))
            pe = float(pearsonr(y[unseen], pf)[0])
            sp = float(spearmanr(y[unseen], pf)[0])
            # descriptive calibration: true ~ a*pred + b
            a_cal, b_cal = np.polyfit(pf, y[unseen], 1)

            # (e) affine recalibration fitted on DESTINATION TRAINING rows only
            pf_tr = m_s.predict(sc_s.transform(X[destrain]))
            ra, rb = np.polyfit(pf_tr, y[destrain], 1)
            r_recal = float(r2_score(y[unseen], ra * pf + rb))

            # SIZE-MATCHED FROZEN COMPARATOR. `frozen` trains on all 1043 source
            # rows and `refit` on 483, so frozen-minus-refit confounds a transfer
            # advantage with 2.2x the training data. This refits the frozen probe on
            # random source subsamples of exactly the comparator's size, so the only
            # remaining difference is which cohort the rows came from.
            rng = np.random.default_rng(0)
            src_idx = np.where(is_b)[0]
            n_match = int(destrain.sum())
            sm_scores = []
            for _ in range(N_SIZE_MATCH):
                pick = rng.choice(src_idx, size=min(n_match, len(src_idx)), replace=False)
                sc_m, m_m, _ = fit_reg_frozen(X[pick], y[pick], task[pick])
                sm_scores.append(float(r2_score(y[unseen],
                                                m_m.predict(sc_m.transform(X[unseen])))))
            sm_mean = float(np.mean(sm_scores))

            cr_s, cr_d = raw_coef(sc_s, m_s), raw_coef(sc_d, m_d)
            trec["L{}".format(L)] = {
                "source_heldout_r2": src_ho, "frozen_r2": r_froz,
                "dest_refit_r2": r_ref, "frozen_minus_refit": float(r_froz - r_ref),
                "rmse": rmse, "pearson": pe, "spearman": sp,
                "calib_slope": float(a_cal), "calib_intercept": float(b_cal),
                "recalibrated_r2": r_recal,
                "recal_recovers": float(r_recal - r_froz),
                "frozen_size_matched_r2": sm_mean,
                "frozen_size_matched_per_draw": sm_scores,
                "size_matched_minus_refit": float(sm_mean - r_ref),
                "cosine_raw": cos(cr_s, cr_d),
                "cosine_standardized_SUPERSEDED": cos(m_s.coef_, m_d.coef_),
                "alpha_source": a_s, "alpha_dest": a_d}
            print("  {:<7} {:>8.3f} {:>8.3f} {:>8.3f} {:>9.3f} {:>7.3f} {:>7.3f} {:>7.3f}"
                  " {:>8.3f} {:>8.3f} {:>8.3f} {:>9.3f} {:>9.3f}".format(
                      "L{}".format(L), src_ho, r_froz, r_ref, r_froz - r_ref,
                      rmse, pe, sp, float(a_cal), float(b_cal), r_recal,
                      sm_mean, sm_mean - r_ref))
        print("\n  cosine, raw activation coordinates (c/sigma) vs standardized (superseded)")
        for L in LAYERS:
            t = trec["L{}".format(L)]
            print("    L{:<3} raw {:>7.3f}    standardized {:>7.3f}".format(
                L, t["cosine_raw"], t["cosine_standardized_SUPERSEDED"]))
        out["targets"][tname] = trec

    # ---- (d) direct tau probe vs the difference of frozen Q_I and Q_C probes ------
    print("\n" + "#" * 78)
    print("# (d) DIRECT tau PROBE vs FROZEN Q_I MINUS FROZEN Q_C, identical rows")
    print("#" * 78)
    print("  {:<7} {:>12} {:>14} {:>10}".format("layer", "direct tau", "Q_I - Q_C", "diff"))
    drec = {}
    for L in LAYERS:
        rd_ = float(r2_score(tau[unseen], frozen_pred[("tau", L)]))
        rc_ = float(r2_score(tau[unseen],
                             frozen_pred[("Q_intervene", L)] - frozen_pred[("Q_continue", L)]))
        drec["L{}".format(L)] = {"direct_tau_r2": rd_, "difference_of_probes_r2": rc_,
                                 "delta": float(rc_ - rd_)}
        print("  {:<7} {:>12.3f} {:>14.3f} {:>10.3f}".format("L{}".format(L), rd_, rc_, rc_ - rd_))
    out["tau_direct_vs_difference"] = drec

    os.makedirs("reports", exist_ok=True)
    p = "reports/closure_transfer_{}.json".format(cfg["run_id"])
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote {}".format(p))
    print("\nTHE DECISION RULE, FIXED IN ADVANCE")
    print("  If tau keeps a strong Spearman and raw-space cosine AND affine recalibration")
    print("  restores most of its R2, then 'the tau direction does not transfer' is WITHDRAWN")
    print("  in favour of 'its frozen calibration is less stable'. Only calibration-free")
    print("  evidence -- Spearman and raw cosine -- can license the direction claim.")
    print("  'frozen - refit' is the comparison Phase 3 lacked: identical rows, so a gap")
    print("  there is about the probe and not about the destination being harder. But")
    print("  frozen trains on 1043 rows and refit on 483, so read 'sm-refit' instead --")
    print("  the frozen probe refitted on source subsamples of exactly the comparator's")
    print("  size, leaving cohort of origin as the only difference.")
    print("\n  PART (d) IS AN ALGEBRAIC IDENTITY, NOT EVIDENCE. Ridge is linear, so when the")
    print("  three fits share an alpha, fit(y1-y2) = fit(y1) - fit(y2) and the difference is")
    print("  exactly zero by construction. It is a wiring check that the alphas do match,")
    print("  and nothing at all about whether tau carries information beyond its components.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
