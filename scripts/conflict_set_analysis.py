#!/usr/bin/env python
"""Stage 2, targeted: is the residual stream carrying intervention-specific
information, or only a better representation of continuation state?

THE AMBIGUITY THIS RESOLVES. The scaled study found activations beat 26 observable
features on tau. But tau = Q_I - Q_C, so a probe that merely represents Q_C better
than the observables do will improve tau mechanically, without knowing anything
about the expert. Every headline in that study is compatible with both readings.

THE ISOLATION. Restrict to the risk = 1.00 conflict set: states where all five
measured continuations failed, so continuation prospects are pinned at their worst
and the live decision is intervene-versus-quit. Then predict Q_I directly -- the
expert's value, not a difference involving Q_C -- having already handed the model
Q_C as a feature. Anything activations add there is about how well the EXPERT will
do, which is the quantity the whole programme is about.

NESTED MODELS, held-out games primary:

    1  observables + Q_C                 the strong baseline: everything the
                                         scaled study's features know, plus the
                                         measured continuation value itself
    2  activations                       alone, per layer
    3  observables + Q_C + activations   the headline: delta R2 of 3 over 1

Same question again as a control decision: intervene vs quit, AUROC and macro-F1.

WHY A FOURTH MODEL IS REPORTED. Model 3 fits ONE ridge alpha over 27 observable
columns and 3,584 activation dimensions. The scaled study pre-registered that this
structure is confounded -- the activation block drags alpha up and shrinks the
standardised coefficients of the small block, so model 3 can lose for reasons that
have nothing to do with activation content. That prediction was borne out: C1 came
out negative at every layer and the artifact grew when n halved, exactly as the
mechanism says. Model 3 is the statistic that was asked for and is reported as
primary, but a residualised version with independent alphas is reported beside it,
because on the previous run those two disagreed and the residualised one was right.

THE SPLIT IS COMPUTED ON THE FULL COHORT AND THEN INTERSECTED with the conflict
set, exactly as `run_pass`'s C3 does. Recomputing a split over conflict-set groups
alone would be a different partition and not comparable to anything in the study.

Estimators, alpha and C grids, scaler, fold count and the train-only selection
protocol are IMPORTED from probe_study.py rather than restated, so they are
identical by construction.

    python scripts/conflict_set_analysis.py --config configs/scale.yaml --cohort all
    python scripts/conflict_set_analysis.py --config configs/scale.yaml --cohort original
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cnc import experiment, utility
from cnc.features import prefix as prefix_features

from probe_study import derive, grouped_split, fit_reg, select_alpha, LAYERS, CS

MIN_TEST = 40          # below this a per-split AUROC is not worth reading
MIN_TRAIN = 100


def cohort_of(basename: str) -> str:
    """`branches.s003.jsonl` -> "original"; `branches.s003.rescue2.jsonl` -> "rescue2".

    Identical to cohort_check.py, so "original" here means exactly what it meant in
    the pooling analysis and in the n=928 robustness run.
    """
    parts = basename.split(".")
    return parts[2] if len(parts) >= 4 else "original"


def fit_clf_proba(Xtr, ytr, gtr, Xte):
    """probe_study.fit_clf, with probabilities returned as well.

    Selection is unchanged and deliberately still on macro-F1, not AUROC: switching
    the criterion to the metric being reported would tune on the thing under test.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    n_sp = min(5, len(np.unique(gtr)))
    best, best_c = -np.inf, CS[len(CS) // 2]
    if n_sp >= 2:
        folds = list(GroupKFold(n_splits=n_sp).split(Xtr_s, ytr, gtr))
        for c in CS:
            scores = []
            for tr, va in folds:
                if len(np.unique(ytr[tr])) < 2:
                    continue
                m = LogisticRegression(C=c, max_iter=3000).fit(Xtr_s[tr], ytr[tr])
                scores.append(f1_score(ytr[va], m.predict(Xtr_s[va]), average="macro"))
            if scores and np.mean(scores) > best:
                best, best_c = np.mean(scores), c
    model = LogisticRegression(C=best_c, max_iter=3000).fit(Xtr_s, ytr)
    return model.predict(Xte_s), model.predict_proba(Xte_s)[:, 1], best_c


def summarise(name, vals):
    """Per-split values first, then mean/sd/sign. The sign column uses the study's
    own standard: a delta whose sign is not consistent across the five independent
    group splits is not established, whatever any single split's interval says."""
    # null rather than NaN: NaN is not valid JSON and silently poisons any reader
    # that parses strictly. A split that could not be scored stays visibly absent.
    js = [None if (x is None or np.isnan(x)) else float(x) for x in vals]
    v = [x for x in js if x is not None]
    if not v:
        print("    {:<34} {:>8} {:>7}  {:<5}  (no split scorable)".format(name, "--", "--", "--"))
        return {"per_seed": js, "mean": None, "sd": None, "consistent": False, "n_scored": 0}
    mu = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    same = all(x > 0 for x in v) or all(x < 0 for x in v)
    print("    {:<34} {:>+8.3f} {:>7.3f}  {:<5}  {}".format(
        name, mu, sd, "same" if same else "FLIPS",
        "[" + " ".join("  n/a " if x is None else "{:+.3f}".format(x) for x in js) + "]"))
    return {"per_seed": js, "mean": mu, "sd": sd, "consistent": bool(same), "n_scored": len(v)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="B_montecarlo")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cohort", default="all", choices=("all", "original"),
                    help="'original' reproduces the n=928 robustness cohort by branch "
                         "source file, so it does not depend on a separate config")
    args = ap.parse_args()

    from sklearn.metrics import r2_score, f1_score, roc_auc_score, accuracy_score

    cfg = experiment.load_config(args.config)
    rd = experiment.run_dir(cfg)
    cps = {c["checkpoint_id"]: c for c in rd.read_all("checkpoints") if c["arm"] == args.arm}
    rows = derive(rd, args.arm, experiment.build_runner_config(cfg, args.arm).n_replicates)

    z = np.load(rd.path("activations_{}.npz".format(args.arm)), allow_pickle=True)
    A, aids = z["activations"], list(z["checkpoint_ids"])
    ai = {c: i for i, c in enumerate(aids)}
    keep = sorted(c for c in aids if c in rows and c in cps)

    if args.cohort != "all":
        sources = rd.branch_sources()
        keep = [c for c in keep if cohort_of(sources.get((c, args.arm), "")) == args.cohort]
    n = len(keep)

    ACT = {L: np.ascontiguousarray(A[[ai[c] for c in keep]][:, L, :], dtype=np.float32)
           for L in LAYERS}
    del A, z

    F = np.array([prefix_features.to_vector(cps[c]["features_26d"]) for c in keep])
    roll = np.array([cps[c]["rollout_id"] for c in keep])
    task = np.array([cps[c]["task_id"] for c in keep])
    qc = np.array([rows[c]["value_continue"] for c in keep])
    qi = np.array([rows[c]["value_intervene"] for c in keep])
    pfail = np.array([rows[c]["p_fail_continue"] for c in keep])
    oracle = np.array([rows[c]["oracle_action"] for c in keep])

    # Pick the band by exact equality against the observed unique value, the way
    # run_pass does, rather than trusting a float literal.
    bands = np.unique(pfail)
    band = bands[int(np.argmin(np.abs(bands - 1.0)))]
    conflict = pfail == band

    # The control decision. Within a fully-failing continuation, "continue" should
    # never be oracle-optimal; if any appear they are excluded from the binary task
    # and counted, rather than silently folded into one of the two classes.
    is_quit = oracle == utility.QUIT
    is_int = oracle == utility.INTERVENE
    binary = conflict & (is_quit | is_int)
    n_cont_in_band = int((conflict & ~(is_quit | is_int)).sum())

    print("=" * 78)
    print("CONFLICT-SET ANALYSIS   cohort={}   arm={}".format(args.cohort, args.arm))
    print("=" * 78)
    print("checkpoints in cohort        : {}".format(n))
    print("risk band selected           : {:.4f}".format(float(band)))
    print("conflict-set size            : {}".format(int(conflict.sum())))
    print("  oracle intervene           : {}".format(int((conflict & is_int).sum())))
    print("  oracle quit                : {}".format(int((conflict & is_quit).sum())))
    print("  oracle continue (excluded) : {}".format(n_cont_in_band))
    print("Q_I within conflict set      : mean {:.3f}  sd {:.3f}  [{:.2f}, {:.2f}]".format(
        qi[conflict].mean(), qi[conflict].std(ddof=1), qi[conflict].min(), qi[conflict].max()))
    print("Q_C within conflict set      : mean {:.3f}  sd {:.3f}  (pinned by construction)".format(
        qc[conflict].mean(), qc[conflict].std(ddof=1)))

    base_all = np.hstack([F, qc[:, None]])
    out = {"cohort": args.cohort, "n_cohort": n, "band": float(band),
           "n_conflict": int(conflict.sum()),
           "n_intervene": int((conflict & is_int).sum()),
           "n_quit": int((conflict & is_quit).sum()),
           "n_continue_excluded": n_cont_in_band,
           "qi_sd_in_band": float(qi[conflict].std(ddof=1)),
           "qc_sd_in_band": float(qc[conflict].std(ddof=1)),
           "schemes": {}}

    for scheme, groups in (("held_out_game", task), ("grouped_by_rollout", roll)):
        print("\n" + "#" * 78)
        print("# {}   {}".format(scheme, "[PRIMARY]" if scheme == "held_out_game" else "[secondary]"))
        print("#" * 78)

        # ---- per-split sizes, printed before any result so sufficiency is visible
        sizes = []
        for s in range(args.seeds):
            tr, te = grouped_split(groups, frac=0.30, seed=s)
            sizes.append((int((conflict & tr).sum()), int((conflict & te).sum()),
                          int((binary & tr).sum()), int((binary & te).sum())))
        print("  per-split sizes  (regression train/test, binary train/test):")
        for s, (a, b, c, d) in enumerate(sizes):
            thin = "  << THIN" if (b < MIN_TEST or a < MIN_TRAIN) else ""
            print("    seed {}   {:>5}/{:<5}   {:>5}/{:<5}{}".format(s, a, b, c, d, thin))
        usable = all(b >= MIN_TEST and a >= MIN_TRAIN for a, b, _, _ in sizes)
        print("  sufficiency: {}".format(
            "OK -- every split clears {} train / {} test".format(MIN_TRAIN, MIN_TEST) if usable
            else "MARGINAL -- at least one split is under the floor; read per-split values, not the mean"))

        sch = {"per_split_sizes": sizes, "sufficient": bool(usable), "regression": {}, "classification": {}}

        # ================= PRIMARY REGRESSION: Q_I =================
        print("\n  [R] TARGET Q_intervene   (Q_C already supplied to the baseline)")
        r_base, r_act, r_both, r_resid = [], defaultdict(list), defaultdict(list), defaultdict(list)
        for s in range(args.seeds):
            tr, te = grouped_split(groups, frac=0.30, seed=s)
            mt, me = conflict & tr, conflict & te
            gtr = groups[mt]

            p0, a0 = fit_reg(base_all[mt], qi[mt], gtr, base_all[me])
            r0 = r2_score(qi[me], p0)
            r_base.append(r0)

            # Residualised companion: baseline gets its own alpha, activations get
            # theirs, so neither penalises the other. Out-of-fold on train.
            from sklearn.model_selection import GroupKFold
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            Btr, ytr_ = base_all[mt], qi[mt]
            n_sp = min(5, len(np.unique(gtr)))
            oof = np.zeros(len(ytr_))
            if n_sp >= 2:
                for a_, b_ in GroupKFold(n_splits=n_sp).split(Btr, ytr_, gtr):
                    scl = StandardScaler().fit(Btr[a_])
                    oof[b_] = Ridge(alpha=a0).fit(scl.transform(Btr[a_]), ytr_[a_]).predict(
                        scl.transform(Btr[b_]))
            resid_tr, resid_te = ytr_ - oof, qi[me] - p0

            for L in LAYERS:
                pa, _ = fit_reg(ACT[L][mt], qi[mt], gtr, ACT[L][me])
                r_act[L].append(r2_score(qi[me], pa))
                X = np.hstack([base_all, ACT[L]])
                pb, _ = fit_reg(X[mt], qi[mt], gtr, X[me])
                r_both[L].append(r2_score(qi[me], pb))
                pr, _ = fit_reg(ACT[L][mt], resid_tr, gtr, ACT[L][me])
                r_resid[L].append(r2_score(resid_te, pr))

        print("    {:<34} {:>8} {:>7}  {:<5}  {}".format("", "mean", "sd", "sign", "per-split"))
        sch["regression"]["baseline_obs_qc"] = summarise("[1] observables + Q_C  (R2)", r_base)
        for L in LAYERS:
            sch["regression"]["act_L{}".format(L)] = summarise(
                "[2] activations L{} alone  (R2)".format(L), r_act[L])
        for L in LAYERS:
            sch["regression"]["both_L{}".format(L)] = summarise(
                "[3] obs+Q_C+act L{}  (R2)".format(L), r_both[L])
        print("    " + "-" * 74)
        for L in LAYERS:
            d = [b - a for a, b in zip(r_base, r_both[L])]
            sch["regression"]["delta_L{}".format(L)] = summarise(
                "*** dR2 of [3] over [1], L{}".format(L), d)
        print("    " + "-" * 74)
        for L in LAYERS:
            sch["regression"]["resid_L{}".format(L)] = summarise(
                "(companion) act L{} -> residual Q_I".format(L), r_resid[L])

        # ================= PRIMARY DECISION: intervene vs quit =================
        print("\n  [C] TARGET oracle decision within conflict set: intervene vs quit")
        y_all = is_quit.astype(int)          # 1 = quit, 0 = intervene
        c_base_f, c_base_a = [], []
        c_act_f, c_act_a = defaultdict(list), defaultdict(list)
        c_both_f, c_both_a = defaultdict(list), defaultdict(list)
        for s in range(args.seeds):
            tr, te = grouped_split(groups, frac=0.30, seed=s)
            mt, me = binary & tr, binary & te
            gtr = groups[mt]
            if len(np.unique(y_all[mt])) < 2 or len(np.unique(y_all[me])) < 2:
                c_base_f.append(np.nan); c_base_a.append(np.nan)
                for L in LAYERS:
                    c_act_f[L].append(np.nan); c_act_a[L].append(np.nan)
                    c_both_f[L].append(np.nan); c_both_a[L].append(np.nan)
                continue

            pr0, pp0, _ = fit_clf_proba(base_all[mt], y_all[mt], gtr, base_all[me])
            c_base_f.append(f1_score(y_all[me], pr0, average="macro"))
            c_base_a.append(roc_auc_score(y_all[me], pp0))
            for L in LAYERS:
                pr1, pp1, _ = fit_clf_proba(ACT[L][mt], y_all[mt], gtr, ACT[L][me])
                c_act_f[L].append(f1_score(y_all[me], pr1, average="macro"))
                c_act_a[L].append(roc_auc_score(y_all[me], pp1))
                X = np.hstack([base_all, ACT[L]])
                pr2, pp2, _ = fit_clf_proba(X[mt], y_all[mt], gtr, X[me])
                c_both_f[L].append(f1_score(y_all[me], pr2, average="macro"))
                c_both_a[L].append(roc_auc_score(y_all[me], pp2))

        print("    {:<34} {:>8} {:>7}  {:<5}  {}".format("", "mean", "sd", "sign", "per-split"))
        sch["classification"]["baseline_macro_f1"] = summarise("[1] observables + Q_C  macroF1", c_base_f)
        sch["classification"]["baseline_auroc"] = summarise("[1] observables + Q_C  AUROC", c_base_a)
        for L in LAYERS:
            sch["classification"]["act_L{}_macro_f1".format(L)] = summarise(
                "[2] activations L{} alone  macroF1".format(L), c_act_f[L])
            sch["classification"]["act_L{}_auroc".format(L)] = summarise(
                "[2] activations L{} alone  AUROC".format(L), c_act_a[L])
        print("    " + "-" * 74)
        for L in LAYERS:
            df = [b - a for a, b in zip(c_base_f, c_both_f[L])]
            da = [b - a for a, b in zip(c_base_a, c_both_a[L])]
            sch["classification"]["delta_L{}_macro_f1".format(L)] = summarise(
                "*** d-macroF1 of [3] over [1], L{}".format(L), df)
            sch["classification"]["delta_L{}_auroc".format(L)] = summarise(
                "*** d-AUROC   of [3] over [1], L{}".format(L), da)

        out["schemes"][scheme] = sch

    os.makedirs("reports", exist_ok=True)
    p = "reports/conflict_set_{}_{}.json".format(cfg["run_id"], args.cohort)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote {}".format(p))
    print("\nHOW TO READ THIS")
    print("  The lines marked *** are the statistics that were asked for: what")
    print("  activations add once observables AND the measured continuation value")
    print("  are already in the model, inside the band where continuation prospects")
    print("  are pinned. A 'same' sign column means every one of the five independent")
    print("  group splits agreed; 'FLIPS' means it did not, which is a null by the")
    print("  standard used throughout the scaled study regardless of any mean.")
    print("  The (companion) residual lines exist because model [3] shares one ridge")
    print("  alpha across 27 observable and 3,584 activation columns, which the scaled")
    print("  study pre-registered as confounded and then confirmed. Where the two")
    print("  disagree, the residual line is the one carrying information about")
    print("  activation content.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
