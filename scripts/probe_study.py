#!/usr/bin/env python
"""Stage 2: does the activation advantage survive scale, conditioning, and a
held-out-game split?

This answers four questions, and the output is labelled with which is which:

  Q1  Does the activation advantage survive at scale?
        -> the main battery, delta vs the 26-feature observable baseline.
  Q2  Does it survive conditioning on continuation value?
        -> C1 and C2. See the alpha-shrinkage note below: C2 is the arbiter.
  Q3  Does it generalise to held-out games?
        -> the `held_out_game` split scheme, which shares no game with training.
  Q4  Is the dataset statistically sufficient?
        -> two independent uncertainty sources, reported side by side: a paired
           bootstrap over the test set (seed 0), and the spread of each delta
           across independent group-level splits (--seeds).

Method commitments, all of them made before looking at the scaled data.

HYPERPARAMETERS NEVER SEE HELD-OUT DATA. Selection happens by grouped
cross-validation strictly inside the training split; test is touched once.

LAYERS ARE PRE-REGISTERED, NOT SEARCHED. {8, 16, 24}, fixed in advance from the
pilot, and every one is reported everywhere -- including inside the per-band
conditional test, where taking the best of three on the test set would inflate
the estimate by exactly the amount of the search.

TWO SPLIT SCHEMES. Grouped by rollout (the paper's own leakage rule) and by game
(strictly harder: a game seen in training never appears in test).

THE HYPOTHESIS IS TESTED SEPARATELY FROM OVERALL PREDICTION. "Among states with
similar continuation prospects, do activations carry extra information about
intervention value?" is not answered by an unconditional R2 on tau -- a probe
could score well there purely by predicting Q_C. Three conditional tests isolate
it, and a prediction worth recording in advance: since tau = Q_I - Q_C, holding
Q_C fixed makes predicting tau equivalent to predicting Q_I, and in the pilot
activations LOST to observables on Q_I. So the conditional test may well come out
negative even though the unconditional one was strongly positive. That is the
point of running it, and a clean negative is a result rather than a failure.

WHY C2 IS THE ARBITER FOR Q2. C1 concatenates 27 observable columns with 3584
activation dims and fits ONE ridge alpha over the lot. The activation block drags
the selected alpha up, which shrinks the standardised Q_C coefficient -- the single
most informative column for tau -- well below its unpenalised value. So C1 can lose
for a reason that has nothing to do with activation content. C2 removes that
coupling: it residualises tau on [observables, Q_C] with its own alpha, then asks
whether activations predict what is left. If C1 and C2 disagree, believe C2.

    python scripts/probe_study.py --config configs/scale.yaml --seeds 5
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from cnc import experiment, utility
from cnc.features import prefix as prefix_features

LAYERS = (8, 16, 24)                 # pre-registered, not searched
ALPHAS = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
CS = (1e-4, 1e-3, 1e-2, 1e-1, 1e0)
N_BOOT = 2000
MIN_BAND_N = 40                      # band sizes for C3, fixed in advance
MIN_BAND_TR, MIN_BAND_TE = 25, 10


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def derive(rd, arm):
    by = defaultdict(list)
    for b in rd.read_all("branches"):
        if b["arm"] == arm:
            by[b["checkpoint_id"]].append(b)
    out = {}
    for cid, bs in by.items():
        if all(x["replay_verified"] for x in bs) and {x["action"] for x in bs} == set(utility.ACTIONS):
            out[cid] = utility.values_from_branches(cid, arm, bs).to_row()
    return out


def grouped_split(groups, frac=0.30, seed=0):
    """Hold out whole groups. Returns boolean train/test masks."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_te = max(1, int(round(frac * len(uniq))))
    te_groups = set(uniq[:n_te])
    te = np.array([g in te_groups for g in groups])
    return ~te, te


# --------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------
def boot_ci(fn, *arrays, n=N_BOOT, seed=0):
    """Percentile CI over paired resamples of the test set.

    Seeded per call rather than from a module-level generator, so a number in the
    report does not depend on how many bootstraps happened to run before it.
    """
    rng = np.random.default_rng(seed)
    m = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        try:
            vals.append(fn(*[a[idx] for a in arrays]))
        except Exception:
            pass
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def select_alpha(Xtr, ytr, gtr):
    """Ridge alpha by grouped CV inside train only. Returns alpha, and the
    fitted scaler is deliberately not reused -- each caller refits its own."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    n_sp = min(5, len(np.unique(gtr)))
    if n_sp < 2:
        return ALPHAS[len(ALPHAS) // 2]
    Xs = StandardScaler().fit_transform(Xtr)
    gkf = GroupKFold(n_splits=n_sp)
    folds = list(gkf.split(Xs, ytr, gtr))
    best, best_a = -np.inf, ALPHAS[len(ALPHAS) // 2]
    for a in ALPHAS:
        sc = [r2_score(ytr[va], Ridge(alpha=a).fit(Xs[tr], ytr[tr]).predict(Xs[va]))
              for tr, va in folds]
        if np.mean(sc) > best:
            best, best_a = np.mean(sc), a
    return best_a


def fit_reg(Xtr, ytr, gtr, Xte, alpha=None):
    """Ridge; alpha chosen by grouped CV inside train only unless pinned."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    a = select_alpha(Xtr, ytr, gtr) if alpha is None else alpha
    sc = StandardScaler().fit(Xtr)
    model = Ridge(alpha=a).fit(sc.transform(Xtr), ytr)
    return model.predict(sc.transform(Xte)), a


def fit_clf(Xtr, ytr, gtr, Xte):
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
    return model.predict(Xte_s), best_c


# --------------------------------------------------------------------------
# one (scheme, seed) pass
# --------------------------------------------------------------------------
def run_pass(D, groups, seed, verbose):
    """Full battery + conditional tests for one split seed.

    `verbose` controls printing only; the numbers are identical either way.
    Bootstrap CIs are computed on the verbose (seed 0) pass only -- across-seed
    spread is the uncertainty estimate for the other seeds, and running 2000
    resamples five times over would report the same interval five times.
    """
    from sklearn.metrics import accuracy_score, f1_score, r2_score

    F, ACT, feats, targets = D["F"], D["ACT"], D["feats"], D["targets"]
    tau, qc, pfail = D["tau"], D["qc"], D["pfail"]
    tr, te = grouped_split(groups, frac=0.30, seed=seed)
    out = {"seed": seed, "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "n_train_groups": len(np.unique(groups[tr])),
           "n_test_groups": len(np.unique(groups[te]))}

    if verbose:
        print("  train={}  test={}  ({} train groups, {} test groups)".format(
            out["n_train"], out["n_test"], out["n_train_groups"], out["n_test_groups"]))

    # ---- main battery: Q1 (and Q3, via the scheme) ------------------------
    for tname, (kind, y) in targets.items():
        if verbose:
            print("\n  TARGET {} [{}]".format(
                tname, "regression" if kind == "reg" else "classification"))
        base_pred, res = None, {"kind": kind}
        for fname, X in feats.items():
            if kind == "reg":
                pred, hp = fit_reg(X[tr], y[tr], groups[tr], X[te])
                score = r2_score(y[te], pred)
                entry = {"r2": float(score), "alpha": hp}
                line = "R2 {:>6.3f}".format(score)
                if verbose:
                    lo, hi = boot_ci(r2_score, y[te], pred, seed=seed)
                    entry["ci95"] = [lo, hi]
                    line += "  [{:>6.3f}, {:>6.3f}]".format(lo, hi)
            else:
                pred, hp = fit_clf(X[tr], y[tr], groups[tr], X[te])
                score = f1_score(y[te], pred, average="macro")
                entry = {"acc": float(accuracy_score(y[te], pred)),
                         "macro_f1": float(score), "C": hp}
                line = "acc {:>6.3f}  macroF1 {:>6.3f}".format(entry["acc"], score)
                if verbose:
                    lo, hi = boot_ci(lambda a, b: f1_score(a, b, average="macro"),
                                     y[te], pred, seed=seed)
                    entry["ci95"] = [lo, hi]
                    line += "  [{:>6.3f}, {:>6.3f}]".format(lo, hi)

            if fname == "observables":
                base_pred, base_score = pred, score
            else:
                d = score - base_score
                entry["delta_vs_observables"] = float(d)
                line += "   D {:+.3f}".format(d)
                if verbose:
                    # paired bootstrap on the DIFFERENCE -- the quantity we care about
                    if kind == "reg":
                        f = lambda a, b, c: r2_score(a, b) - r2_score(a, c)
                    else:
                        f = (lambda a, b, c: f1_score(a, b, average="macro")
                             - f1_score(a, c, average="macro"))
                    dlo, dhi = boot_ci(f, y[te], pred, base_pred, seed=seed)
                    entry["delta_ci95"] = [dlo, dhi]
                    line += " [{:+.3f}, {:+.3f}] {}".format(
                        dlo, dhi, "*" if (dlo > 0 or dhi < 0) else " ")
            res[fname] = entry
            if verbose:
                print("    {:<20} {}".format(fname, line))
        out[tname] = res

    # ---- conditional tests: Q2 -------------------------------------------
    if verbose:
        print("\n  " + "-" * 70)
        print("  CONDITIONAL TESTS -- tau given continuation prospects  [Q2]")
        print("  " + "-" * 70)
    cond = {}

    # C1: observables + Q_C  vs  + activations.  Susceptible to the shared-alpha
    # shrinkage described in the module docstring; reported, but not the arbiter.
    base = np.hstack([F, qc[:, None]])
    p0, a0 = fit_reg(base[tr], tau[tr], groups[tr], base[te])
    r0 = r2_score(tau[te], p0)
    cond["C1_baseline_r2"] = float(r0)
    cond["C1_baseline_alpha"] = a0
    if verbose:
        print("    [C1] baseline = observables + Q_C          R2 {:>6.3f}  (alpha {:g})".format(r0, a0))
    for L in LAYERS:
        X = np.hstack([base, ACT[L]])
        p1, a1 = fit_reg(X[tr], tau[tr], groups[tr], X[te])
        r1 = r2_score(tau[te], p1)
        e = {"r2": float(r1), "delta": float(r1 - r0), "alpha": a1}
        line = "R2 {:>6.3f}   D {:+.3f}".format(r1, r1 - r0)
        if verbose:
            dlo, dhi = boot_ci(lambda a, b, c: r2_score(a, b) - r2_score(a, c),
                               tau[te], p1, p0, seed=seed)
            e["ci95"] = [dlo, dhi]
            line += " [{:+.3f}, {:+.3f}] {}".format(dlo, dhi, "*" if (dlo > 0 or dhi < 0) else " ")
        cond["C1_L{}".format(L)] = e
        if verbose:
            print("    [C1]   + activations L{:<2}                  {}  (alpha {:g})".format(L, line, a1))

    # C2: can activations predict the RESIDUAL tau left after conditioning on
    # observables AND Q_C?  Each block keeps its own alpha, so the baseline is
    # not penalised by the activation block's regularisation.
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    Btr, ttr, gtr = base[tr], tau[tr], groups[tr]
    a_base = a0                                  # identical data to C1's baseline
    oof = np.zeros(len(ttr))
    n_sp = min(5, len(np.unique(gtr)))
    for a_, b_ in GroupKFold(n_splits=n_sp).split(Btr, ttr, gtr):
        sc = StandardScaler().fit(Btr[a_])
        oof[b_] = Ridge(alpha=a_base).fit(sc.transform(Btr[a_]), ttr[a_]).predict(sc.transform(Btr[b_]))
    resid_tr = ttr - oof
    resid_te = tau[te] - p0                      # p0 already uses the C1 baseline
    cond["C2_resid_var"] = float(resid_te.var())
    cond["C2_total_var"] = float(tau[te].var())
    if verbose:
        print("    [C2] residual tau after observables + Q_C  "
              "(resid var {:.4f} of total {:.4f})".format(resid_te.var(), tau[te].var()))
    for L in LAYERS:
        pr, _ = fit_reg(ACT[L][tr], resid_tr, groups[tr], ACT[L][te])
        r = r2_score(resid_te, pr)
        e = {"r2": float(r)}
        line = "R2 {:>6.3f}".format(r)
        if verbose:
            lo, hi = boot_ci(r2_score, resid_te, pr, seed=seed)
            e["ci95"] = [lo, hi]
            line += "  [{:>6.3f}, {:>6.3f}] {}".format(lo, hi, "*" if lo > 0 else " ")
        cond["C2_L{}".format(L)] = e
        if verbose:
            print("    [C2]   activations L{:<2} -> residual       {}".format(L, line))

    # C3: within continuation-risk bands. Every pre-registered layer reported --
    # taking the best of three on the test set is the search this study avoids.
    if verbose:
        print("    [C3] within continuation-risk bands (n>={} only)".format(MIN_BAND_N))
    for band in np.unique(pfail):
        m = pfail == band
        if m.sum() < MIN_BAND_N or (m & tr).sum() < MIN_BAND_TR or (m & te).sum() < MIN_BAND_TE:
            continue
        mt, me = m & tr, m & te
        pb, _ = fit_reg(F[mt], tau[mt], groups[mt], F[me])
        rb = r2_score(tau[me], pb)
        rec = {"n": int(m.sum()), "n_test": int(me.sum()), "obs_r2": float(rb)}
        if verbose:
            print("      risk={:.2f}  n={:<4} (te {:<3})  observables R2 {:>6.3f}".format(
                float(band), int(m.sum()), int(me.sum()), rb))
        for L in LAYERS:
            X = np.hstack([F, ACT[L]])
            pa, _ = fit_reg(X[mt], tau[mt], groups[mt], X[me])
            ra = r2_score(tau[me], pa)
            e = {"r2": float(ra), "delta": float(ra - rb)}
            line = "R2 {:>6.3f}   D {:+.3f}".format(ra, ra - rb)
            if verbose:
                dlo, dhi = boot_ci(lambda a, b, c: r2_score(a, b) - r2_score(a, c),
                                   tau[me], pa, pb, seed=seed)
                e["ci95"] = [dlo, dhi]
                line += " [{:+.3f}, {:+.3f}] {}".format(dlo, dhi, "*" if (dlo > 0 or dhi < 0) else " ")
            rec["L{}".format(L)] = e
            if verbose:
                print("                    + activations L{:<2}       {}".format(L, line))
        cond["C3_band_{:.2f}".format(float(band))] = rec

    out["conditional"] = cond
    return out


def seed_summary(passes, targets):
    """Across-seed mean/SD of every delta. This is the Q4 evidence: a delta whose
    sign flips across independent group splits is not established, however tight
    the single-split bootstrap looked."""
    print("\n  " + "=" * 70)
    print("  SEED STABILITY over {} splits -- delta vs observables  [Q4]".format(len(passes)))
    print("  {:<24} {:<16} {:>8} {:>8} {:>9}".format("target", "features", "mean", "sd", "sign"))
    print("  " + "-" * 70)
    rows = {}
    for tname in targets:
        for fname in passes[0][tname]:
            if fname == "kind" or fname == "observables":
                continue
            ds = [p[tname][fname]["delta_vs_observables"] for p in passes]
            mu, sd = float(np.mean(ds)), float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0
            same = all(d > 0 for d in ds) or all(d < 0 for d in ds)
            rows["{}|{}".format(tname, fname)] = {"mean": mu, "sd": sd, "consistent": bool(same),
                                                  "per_seed": [float(d) for d in ds]}
            print("  {:<24} {:<16} {:>+8.3f} {:>8.3f} {:>9}".format(
                tname, fname, mu, sd, "same" if same else "FLIPS"))
    print("  " + "-" * 70)
    for key, lab in ([("C1_L{}".format(L), "C1 +act L{}".format(L)) for L in LAYERS]
                     + [("C2_L{}".format(L), "C2 resid L{}".format(L)) for L in LAYERS]):
        ds = [p["conditional"][key]["delta" if key.startswith("C1") else "r2"] for p in passes]
        mu, sd = float(np.mean(ds)), float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0
        same = all(d > 0 for d in ds) or all(d < 0 for d in ds)
        rows[key] = {"mean": mu, "sd": sd, "consistent": bool(same),
                     "per_seed": [float(d) for d in ds]}
        print("  {:<24} {:<16} {:>+8.3f} {:>8.3f} {:>9}".format(
            "tau | conditioned", lab, mu, sd, "same" if same else "FLIPS"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="B_montecarlo")
    ap.add_argument("--seeds", type=int, default=5,
                    help="independent group-level splits; seed 0 carries the bootstrap CIs")
    args = ap.parse_args()

    cfg = experiment.load_config(args.config)
    rd = experiment.run_dir(cfg)
    cps = {c["checkpoint_id"]: c for c in rd.read_all("checkpoints") if c["arm"] == args.arm}
    rows = derive(rd, args.arm)

    z = np.load(rd.path("activations_{}.npz".format(args.arm)), allow_pickle=True)
    A, aids = z["activations"], list(z["checkpoint_ids"])
    ai = {c: i for i, c in enumerate(aids)}
    keep = sorted(c for c in aids if c in rows and c in cps)
    n = len(keep)
    print("checkpoints: {}   (activations x derived values)".format(n))

    # Materialise ONLY the pre-registered layers. Keeping all 29 in float32 is
    # ~830 MB at this n, on a box where memory pressure has already cost a run.
    ACT = {L: np.ascontiguousarray(A[[ai[c] for c in keep]][:, L, :], dtype=np.float32)
           for L in LAYERS}
    del A, z

    F = np.array([prefix_features.to_vector(cps[c]["features_26d"]) for c in keep])
    roll = np.array([cps[c]["rollout_id"] for c in keep])
    task = np.array([cps[c]["task_id"] for c in keep])

    qc = np.array([rows[c]["value_continue"] for c in keep])
    qi = np.array([rows[c]["value_intervene"] for c in keep])
    tau = np.array([rows[c]["tau"] for c in keep])
    pfail = np.array([rows[c]["p_fail_continue"] for c in keep])
    succ = (np.array([rows[c]["p_success_continue"] for c in keep]) > 0.5).astype(int)
    oracle = np.array([utility.ACTIONS.index(rows[c]["oracle_action"]) for c in keep])

    print("\n[VARIANCE]  mean / sd / range")
    var_report = {}
    for nm, v in (("Q_continue", qc), ("Q_intervene", qi), ("tau", tau)):
        print("  {:<14} {:>7.3f} {:>7.3f}  [{:.2f}, {:.2f}]".format(
            nm, v.mean(), v.std(ddof=1), v.min(), v.max()))
        var_report[nm] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                          "min": float(v.min()), "max": float(v.max())}

    feats = {"observables": F}
    for L in LAYERS:
        feats["act_L{}".format(L)] = ACT[L]
        feats["obs+act_L{}".format(L)] = np.hstack([F, ACT[L]])

    targets = {
        "Q_continue": ("reg", qc), "Q_intervene": ("reg", qi), "tau": ("reg", tau),
        "continue_success": ("clf", succ), "oracle_action": ("clf", oracle),
    }
    D = {"F": F, "ACT": ACT, "feats": feats, "targets": targets,
         "tau": tau, "qc": qc, "pfail": pfail}

    results = {"n": n, "layers": list(LAYERS), "seeds": args.seeds, "variance": var_report}
    for scheme, groups in (("grouped_by_rollout", roll), ("held_out_game", task)):
        print("\n" + "#" * 74)
        print("# SPLIT SCHEME: {}   ({} groups)  {}".format(
            scheme, len(np.unique(groups)),
            "[Q3: no game shared with training]" if scheme == "held_out_game" else ""))
        print("#" * 74)
        passes = []
        for s in range(args.seeds):
            if s == 0:
                print("\n--- seed 0 (detailed, with paired bootstrap CIs) ---")
            elif s == 1:
                print("\n--- seeds 1..{} (silent; feeding the stability table) ---".format(args.seeds - 1))
            passes.append(run_pass(D, groups, s, verbose=(s == 0)))
        stab = seed_summary(passes, targets) if args.seeds > 1 else {}
        results[scheme] = {"seed0": passes[0], "all_seeds": passes, "seed_stability": stab}

    os.makedirs("reports", exist_ok=True)
    p = "reports/probe_study_{}.json".format(cfg["run_id"])
    with open(p, "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote {}".format(p))
    print("\n* marks a seed-0 bootstrap 95% CI excluding zero.")
    print("FLIPS marks a delta whose sign is not consistent across split seeds -- "
          "that is a null result regardless of any single split's CI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
