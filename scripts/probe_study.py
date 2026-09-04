#!/usr/bin/env python
"""Stage 2: does the activation advantage survive scale, conditioning, and a
held-out-game split?

Four things this does that the pilot did not.

HYPERPARAMETERS NEVER SEE HELD-OUT DATA. The pilot picked regularisation on a
validation split and refit on train+val. Here selection happens by grouped
cross-validation strictly inside the training split; test is touched once.

LAYERS ARE PRE-REGISTERED, NOT SEARCHED. {8, 16, 24}, fixed in advance from the
pilot, and every one is reported. Searching all 29 and reporting the best inflates
the estimate by exactly the amount of the search.

TWO SPLIT SCHEMES. Grouped by rollout (the paper's own leakage rule) and by game
(strictly harder: a game seen in training never appears in test).

THE ACTUAL HYPOTHESIS IS TESTED SEPARATELY FROM OVERALL PREDICTION. "Among states
with similar continuation prospects, do activations carry extra information about
intervention value?" is not answered by an unconditional R2 on tau -- a probe
could score well there purely by predicting Q_C. Three conditional tests isolate
it, and a prediction worth recording in advance: since tau = Q_I - Q_C, holding
Q_C fixed makes predicting tau equivalent to predicting Q_I, and in the pilot
activations LOST to observables on Q_I. So the conditional test may well come out
negative even though the unconditional one was strongly positive. That is the
point of running it.

    python scripts/probe_study.py --config configs/scale.yaml
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
RNG = np.random.default_rng(0)


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


def boot_ci(fn, *arrays, n=N_BOOT):
    """Percentile CI over paired resamples of the test set."""
    m = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = RNG.integers(0, m, m)
        try:
            vals.append(fn(*[a[idx] for a in arrays]))
        except Exception:
            pass
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def fit_reg(Xtr, ytr, gtr, Xte):
    """Ridge; alpha chosen by grouped CV inside train only."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    n_sp = min(5, len(np.unique(gtr)))
    best, best_a = -np.inf, ALPHAS[len(ALPHAS) // 2]
    if n_sp >= 2:
        gkf = GroupKFold(n_splits=n_sp)
        for a in ALPHAS:
            sc_ = []
            for tr, va in gkf.split(Xtr, ytr, gtr):
                m = Ridge(alpha=a).fit(Xtr[tr], ytr[tr])
                sc_.append(r2_score(ytr[va], m.predict(Xtr[va])))
            if np.mean(sc_) > best:
                best, best_a = np.mean(sc_), a
    model = Ridge(alpha=best_a).fit(Xtr, ytr)
    return model.predict(Xte), best_a


def fit_clf(Xtr, ytr, gtr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    n_sp = min(5, len(np.unique(gtr)))
    best, best_c = -np.inf, CS[len(CS) // 2]
    if n_sp >= 2:
        gkf = GroupKFold(n_splits=n_sp)
        for c in CS:
            sc_ = []
            for tr, va in gkf.split(Xtr, ytr, gtr):
                if len(np.unique(ytr[tr])) < 2:
                    continue
                m = LogisticRegression(C=c, max_iter=3000).fit(Xtr[tr], ytr[tr])
                sc_.append(f1_score(ytr[va], m.predict(Xtr[va]), average="macro"))
            if sc_ and np.mean(sc_) > best:
                best, best_c = np.mean(sc_), c
    model = LogisticRegression(C=best_c, max_iter=3000).fit(Xtr, ytr)
    return model.predict(Xte), best_c


def main() -> int:
    from sklearn.metrics import accuracy_score, f1_score, r2_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="B_montecarlo")
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

    F = np.array([prefix_features.to_vector(cps[c]["features_26d"]) for c in keep])
    Aa = np.array([A[ai[c]] for c in keep], dtype=np.float32)
    roll = np.array([cps[c]["rollout_id"] for c in keep])
    task = np.array([cps[c]["task_id"] for c in keep])

    qc = np.array([rows[c]["value_continue"] for c in keep])
    qi = np.array([rows[c]["value_intervene"] for c in keep])
    tau = np.array([rows[c]["tau"] for c in keep])
    pfail = np.array([rows[c]["p_fail_continue"] for c in keep])
    succ = (np.array([rows[c]["p_success_continue"] for c in keep]) > 0.5).astype(int)
    oracle = np.array([utility.ACTIONS.index(rows[c]["oracle_action"]) for c in keep])

    print("\n[VARIANCE]  mean / sd / range")
    for nm, v in (("Q_continue", qc), ("Q_intervene", qi), ("tau", tau)):
        print("  {:<14} {:>7.3f} {:>7.3f}  [{:.2f}, {:.2f}]".format(
            nm, v.mean(), v.std(ddof=1), v.min(), v.max()))

    feats = {"observables": F}
    for L in LAYERS:
        feats["act_L{}".format(L)] = Aa[:, L, :]
        feats["obs+act_L{}".format(L)] = np.hstack([F, Aa[:, L, :]])

    targets = {
        "Q_continue": ("reg", qc), "Q_intervene": ("reg", qi), "tau": ("reg", tau),
        "continue_success": ("clf", succ), "oracle_action": ("clf", oracle),
    }

    results = {"n": n, "layers": list(LAYERS)}
    for scheme, groups in (("grouped_by_rollout", roll), ("held_out_game", task)):
        tr, te = grouped_split(groups, frac=0.30, seed=0)
        print("\n" + "#" * 74)
        print("# SPLIT: {}   train={}  test={}  ({} train groups, {} test groups)".format(
            scheme, int(tr.sum()), int(te.sum()),
            len(np.unique(groups[tr])), len(np.unique(groups[te]))))
        print("#" * 74)
        res = {}

        for tname, (kind, y) in targets.items():
            print("\n  TARGET {} [{}]".format(tname, "regression" if kind == "reg" else "classification"))
            base_pred = None
            res[tname] = {"kind": kind}
            for fname, X in feats.items():
                if kind == "reg":
                    pred, hp = fit_reg(X[tr], y[tr], groups[tr], X[te])
                    r2 = r2_score(y[te], pred)
                    lo, hi = boot_ci(lambda a, b: r2_score(a, b), y[te], pred)
                    entry = {"r2": float(r2), "ci95": [lo, hi], "alpha": hp}
                    line = "R2 {:>6.3f}  [{:>6.3f}, {:>6.3f}]".format(r2, lo, hi)
                else:
                    pred, hp = fit_clf(X[tr], y[tr], groups[tr], X[te])
                    acc, f1 = accuracy_score(y[te], pred), f1_score(y[te], pred, average="macro")
                    lo, hi = boot_ci(lambda a, b: f1_score(a, b, average="macro"), y[te], pred)
                    entry = {"acc": float(acc), "macro_f1": float(f1), "ci95": [lo, hi], "C": hp}
                    line = "acc {:>6.3f}  macroF1 {:>6.3f}  [{:>6.3f}, {:>6.3f}]".format(acc, f1, lo, hi)
                if fname == "observables":
                    base_pred = pred
                else:
                    # paired bootstrap on the DIFFERENCE -- the quantity we care about
                    if kind == "reg":
                        d = r2_score(y[te], pred) - r2_score(y[te], base_pred)
                        dlo, dhi = boot_ci(
                            lambda a, b, c: r2_score(a, b) - r2_score(a, c), y[te], pred, base_pred)
                    else:
                        d = (f1_score(y[te], pred, average="macro")
                             - f1_score(y[te], base_pred, average="macro"))
                        dlo, dhi = boot_ci(
                            lambda a, b, c: f1_score(a, b, average="macro")
                            - f1_score(a, c, average="macro"), y[te], pred, base_pred)
                    entry["delta_vs_observables"] = float(d)
                    entry["delta_ci95"] = [dlo, dhi]
                    sig = "*" if (dlo > 0 or dhi < 0) else " "
                    line += "   D {:+.3f} [{:+.3f}, {:+.3f}] {}".format(d, dlo, dhi, sig)
                res[tname][fname] = entry
                print("    {:<20} {}".format(fname, line))

        # ---- THE HYPOTHESIS: extra information about tau given continuation prospects
        print("\n  {}".format("-" * 70))
        print("  CONDITIONAL TESTS -- tau given continuation prospects")
        print("  {}".format("-" * 70))
        cond = {}

        # C1: observables + Q_C  vs  + activations
        base = np.hstack([F, qc[:, None]])
        p0, _ = fit_reg(base[tr], tau[tr], groups[tr], base[te])
        r0 = r2_score(tau[te], p0)
        print("    [C1] baseline = observables + Q_C            R2 {:>6.3f}".format(r0))
        cond["C1_baseline_r2"] = float(r0)
        for L in LAYERS:
            X = np.hstack([base, Aa[:, L, :]])
            p1, _ = fit_reg(X[tr], tau[tr], groups[tr], X[te])
            r1 = r2_score(tau[te], p1)
            d = r1 - r0
            dlo, dhi = boot_ci(lambda a, b, c: r2_score(a, b) - r2_score(a, c), tau[te], p1, p0)
            sig = "*" if (dlo > 0 or dhi < 0) else " "
            print("    [C1]   + activations L{:<2}                    R2 {:>6.3f}   D {:+.3f} [{:+.3f}, {:+.3f}] {}".format(
                L, r1, d, dlo, dhi, sig))
            cond["C1_L{}".format(L)] = {"r2": float(r1), "delta": float(d), "ci95": [dlo, dhi]}

        # C2: can activations predict the RESIDUAL tau left by the observable baseline?
        from sklearn.model_selection import GroupKFold
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        oof = np.zeros(int(tr.sum()))
        Ftr, ttr, gtr = F[tr], tau[tr], groups[tr]
        gkf = GroupKFold(n_splits=min(5, len(np.unique(gtr))))
        for a_, b_ in gkf.split(Ftr, ttr, gtr):
            sc = StandardScaler().fit(Ftr[a_])
            oof[b_] = Ridge(alpha=1e2).fit(sc.transform(Ftr[a_]), ttr[a_]).predict(sc.transform(Ftr[b_]))
        resid_tr = ttr - oof
        pbase_te, _ = fit_reg(F[tr], tau[tr], groups[tr], F[te])
        resid_te = tau[te] - pbase_te
        print("    [C2] residual tau after observables   (var {:.3f} of total {:.3f})".format(
            float(resid_te.var()), float(tau[te].var())))
        for L in LAYERS:
            pr, _ = fit_reg(Aa[tr][:, L, :], resid_tr, groups[tr], Aa[te][:, L, :])
            r = r2_score(resid_te, pr)
            lo, hi = boot_ci(lambda a, b: r2_score(a, b), resid_te, pr)
            sig = "*" if lo > 0 else " "
            print("    [C2]   activations L{:<2} -> residual        R2 {:>6.3f}  [{:>6.3f}, {:>6.3f}] {}".format(
                L, r, lo, hi, sig))
            cond["C2_L{}".format(L)] = {"r2": float(r), "ci95": [lo, hi]}

        # C3: within continuation-risk bands
        print("    [C3] within continuation-risk bands (n>=40 only)")
        for band in np.unique(pfail):
            m = pfail == band
            if m.sum() < 40 or (m & tr).sum() < 25 or (m & te).sum() < 10:
                continue
            mt, me = m & tr, m & te
            pb, _ = fit_reg(F[mt], tau[mt], groups[mt], F[me])
            rb = r2_score(tau[me], pb)
            best = None
            for L in LAYERS:
                X = np.hstack([F, Aa[:, L, :]])
                pa, _ = fit_reg(X[mt], tau[mt], groups[mt], X[me])
                ra = r2_score(tau[me], pa)
                if best is None or ra > best[1]:
                    best = (L, ra)
            print("      risk={:.2f}  n={:<4} (te {:<3})  obs R2 {:>6.3f}   +act L{} R2 {:>6.3f}   D {:+.3f}".format(
                float(band), int(m.sum()), int(me.sum()), rb, best[0], best[1], best[1] - rb))
            cond["C3_band_{:.2f}".format(float(band))] = {
                "n": int(m.sum()), "obs_r2": float(rb), "best_layer": best[0],
                "act_r2": float(best[1]), "delta": float(best[1] - rb)}

        res["conditional"] = cond
        results[scheme] = res

    os.makedirs("reports", exist_ok=True)
    p = "reports/probe_study_{}.json".format(cfg["run_id"])
    with open(p, "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote {}".format(p))
    print("\n* marks a bootstrap 95% CI excluding zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
