#!/usr/bin/env python
"""Decision-focused linear controller: does aligning the LOSS with the decision help?

Closure established that the activation signal is decision-relevant but that its regret
effects cancel -- Delta-regret -0.0898 [-0.1437,-0.0432] where tau <= 0 against +0.0593
[+0.0188,+0.1068] where tau > 0, non-overlapping intervals, replicated. An MSE-trained
probe spends capacity uniformly and is provably damaging in one regime while provably
helping in the other. So the question is not whether the representation carries
control-relevant information -- it does -- but whether an objective that cares about the
induced decision can keep the gain without the damage.

THE LOSS. Elmachtoub & Grigas define SPO+ for minimisation over a feasible region S:

    l(chat, c) = max_{w in S} {(c - 2*chat)' w} + 2*chat'w*(c) - z*(c)

Our controller MAXIMISES over three fixed actions, so substitute c = -Q, chat = -Qhat.
S is the three unit vectors, w* selects an action, z*(c) = -max_a Q_a, and the whole
thing collapses to

    l(Qhat, Q) = max_a {2*Qhat_a - Q_a} + max_a Q_a - 2*Qhat_{a*}     a* = argmax_a Q_a

with subgradient 2(e_atilde - e_astar), atilde = argmax_a {2*Qhat_a - Q_a}. Convex in
Qhat and therefore in the linear parameters.

WHY THIS LOSS AND NOT A WEIGHTING. The closure report proposed weighting Ridge by
1/true_action_margin. That is withdrawn: it is misaligned with regret, since a
small-margin mistake can cost almost nothing while a large-margin mistake is expensive,
and it would downweight exactly the highest-margin decile where closure found 93.9% of
the activation improvement. SPO+ instead has the property that its underlying SPO loss
IS our reported regret, so the surrogate targets the metric rather than a proxy.

QUIT IS A KNOWN CONSTANT (exactly 0), not a prediction. That constrains the hypothesis
class -- only Qhat_C and Qhat_I are parameterised -- and leaves the loss convex; the
subgradient's quit component is simply unused.

THE GATES. Every synthetic property is checked before any research data is loaded, and
the script ABORTS if one fails. A decision loss that is subtly wrong would produce a
plausible-looking number, which is the worst possible failure mode here.

BOTH ARMS SHARE objective, optimiser, lambda grid and selection rule, so representation
and loss cannot be confounded. Layer 24 is primary because it was fixed before the
all-layer sweep; 8 and 16 are secondary diagnostics and no conclusion may rest on
choosing among them.

    python scripts/decision_focused.py --config configs/scale.yaml --cohort all
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cnc import experiment
from cnc.features import prefix as prefix_features
from probe_study import derive, grouped_split, fit_reg, select_alpha
from conflict_set_analysis import cohort_of
from controllers import oof_predict, ACTS
from closure_baselines import fit_tfidf

PRIMARY_LAYER = 24
LAYERS = (8, 16, 24)
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2)   # covers ALPHAS mapped to mean form
ADAM_LR, ADAM_T = 0.01, 500
N_BOOT = 2000


# ---------------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------------
def spo_plus(Qhat, Q):
    """Row-wise SPO+ loss. Qhat, Q both (n, 3) with column 2 the fixed quit value."""
    astar = np.argmax(Q, axis=1)
    return (np.max(2.0 * Qhat - Q, axis=1) + np.max(Q, axis=1)
            - 2.0 * Qhat[np.arange(len(Q)), astar])


def spo_plus_subgrad(Qhat, Q):
    """d l / d Qhat, shape (n, 3). 2(e_atilde - e_astar)."""
    n = len(Q)
    g = np.zeros_like(Qhat)
    g[np.arange(n), np.argmax(2.0 * Qhat - Q, axis=1)] += 2.0
    g[np.arange(n), np.argmax(Q, axis=1)] -= 2.0
    return g


def regret_of(Qhat, Q):
    return Q.max(axis=1) - Q[np.arange(len(Q)), np.argmax(Qhat, axis=1)]


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
def fit_spo(X, Q, lam, T=ADAM_T, lr=ADAM_LR, trace=False):
    """Linear Qhat_C, Qhat_I (quit fixed at 0), trained on SPO+ with full-batch Adam.

    Adam rather than a convex-rate subgradient schedule because the piecewise-linear
    objective plus a wide lambda grid makes a 1/(lambda t) step unusable at the small
    end. Convergence is not assumed: the objective at t = 0, T/2, T is returned so it
    can be audited, and a synthetic recovery test gates the whole script.
    """
    n, d = X.shape
    W = np.zeros((d, 2)); b = np.zeros(2)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    zero = np.zeros((n, 1))
    traj = []
    for t in range(1, T + 1):
        Qh = np.hstack([X @ W + b, zero])
        if trace and (t == 1 or t == T // 2 or t == T):
            traj.append((t, float(spo_plus(Qh, Q).mean() + 0.5 * lam * float((W ** 2).sum()))))
        G = spo_plus_subgrad(Qh, Q)[:, :2] / n
        gW = X.T @ G + lam * W
        gb = G.sum(axis=0)
        mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * gW ** 2
        mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb ** 2
        W -= lr * (mW / (1 - b1 ** t)) / (np.sqrt(vW / (1 - b2 ** t)) + eps)
        b -= lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)
    return W, b, traj


def predict_spo(X, W, b):
    return np.hstack([X @ W + b, np.zeros((len(X), 1))])


def fit_spo_cv(Xtr, Qtr, gtr, Xte):
    """Lambda by GroupKFold(5) inside train on validation SPO+ loss, then refit on all
    of train. Same fold structure and same train-only discipline as select_alpha."""
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    n_sp = min(5, len(np.unique(gtr)))
    best_lam, best = LAMBDAS[len(LAMBDAS) // 2], np.inf
    if n_sp >= 2:
        folds = list(GroupKFold(n_splits=n_sp).split(Xtr, Qtr[:, 0], gtr))
        for lam in LAMBDAS:
            sc = []
            for a_, b_ in folds:
                s = StandardScaler().fit(Xtr[a_])
                W, bb, _ = fit_spo(s.transform(Xtr[a_]), Qtr[a_], lam)
                sc.append(float(spo_plus(predict_spo(s.transform(Xtr[b_]), W, bb),
                                         Qtr[b_]).mean()))
            if np.mean(sc) < best:
                best, best_lam = float(np.mean(sc)), lam
    s = StandardScaler().fit(Xtr)
    W, bb, traj = fit_spo(s.transform(Xtr), Qtr, best_lam, trace=True)
    return predict_spo(s.transform(Xte), W, bb), best_lam, traj


# ---------------------------------------------------------------------------
# gates -- synthetic only, run before any research data is loaded
# ---------------------------------------------------------------------------
def gates():
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print("    {:<52} {}{}".format(name, "PASS" if cond else "FAIL",
                                       ("   " + detail) if detail else ""))

    Q = np.column_stack([rng.normal(0, 1, 20000), rng.normal(0, 1, 20000), np.zeros(20000)])
    Qh = np.column_stack([rng.normal(0, 1, 20000), rng.normal(0, 1, 20000), np.zeros(20000)])

    # brute force in the ORIGINAL minimisation form, enumerating the three actions
    brute = []
    for i in range(2000):
        c, ch = -Q[i], -Qh[i]
        a_ = int(np.argmin(c))
        brute.append(max((c - 2 * ch)[j] for j in range(3)) + 2 * ch[a_] - float(c[a_]))
    check("brute-force min-form == simplified max-form (2000)",
          np.allclose(brute, spo_plus(Qh[:2000], Q[:2000]), atol=1e-9))
    check("perfect prediction -> exactly 0", np.allclose(spo_plus(Q, Q), 0.0, atol=1e-12))
    v = int((spo_plus(Qh, Q) < regret_of(Qh, Q) - 1e-12).sum())
    check("SPO+ >= regret on every row (20000)", v == 0, "violations {}".format(v))

    # monotone in the true cost of the wrongly chosen action
    losses = [spo_plus(np.array([[0.4, 0.6, 0.0]]), np.array([[1.0, x, 0.0]]))[0]
              for x in (0.5, 0.0, -0.5, -1.0)]
    check("loss increases as the wrong action's true value falls",
          all(losses[i] < losses[i + 1] for i in range(3)),
          " ".join("{:.2f}".format(x) for x in losses))
    check("oracle-choosing prediction beats non-oracle",
          spo_plus(np.array([[1.0, .5, 0.]]), np.array([[1.0, .5, 0.]]))[0]
          < spo_plus(np.array([[0.4, .6, 0.]]), np.array([[1.0, .5, 0.]]))[0])

    # analytic subgradient vs central differences
    Qa = np.array([[0.7, -0.2, 0.0]]); Qha = np.array([[0.1, 0.9, 0.0]])
    num = []
    for j in range(3):
        e = np.zeros((1, 3)); e[0, j] = 1e-6
        num.append(float(((spo_plus(Qha + e, Qa) - spo_plus(Qha - e, Qa)) / 2e-6)[0]))
    check("analytic subgradient == finite difference",
          np.allclose(spo_plus_subgrad(Qha, Qa)[0], num, atol=1e-5))

    # optimiser recovery: Q exactly linear in x, so near-zero loss is attainable
    n, d = 800, 20
    Xs = rng.normal(size=(n, d))
    Wt = rng.normal(size=(d, 2))
    Qs = np.hstack([Xs @ Wt, np.zeros((n, 1))])
    W, bb, _ = fit_spo(Xs, Qs, lam=1e-6, T=1500)
    r = float(regret_of(predict_spo(Xs, W, bb), Qs).mean())
    l = float(spo_plus(predict_spo(Xs, W, bb), Qs).mean())
    base = float(regret_of(np.zeros_like(Qs), Qs).mean())
    check("optimiser recovers a realisable linear solution", r < 0.05 * base,
          "regret {:.4f} vs {:.4f} at zero-init, SPO+ {:.4f}".format(r, base, l))
    return ok


# ---------------------------------------------------------------------------
def boot_games(vals, games, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(games)
    by = {g: np.where(games == g)[0] for g in uniq}
    out = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([by[uniq[i]] for i in pick])
        out.append(float(vals[sel].mean()))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="B_montecarlo")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cohort", default="all", choices=("all", "original"))
    args = ap.parse_args()

    from sklearn.metrics import r2_score

    print("=" * 78)
    print("GATES -- synthetic only, no research data loaded yet")
    print("=" * 78)
    if not gates():
        print("\n  ABORTING: a synthetic gate failed. A subtly wrong decision loss would")
        print("  produce a plausible number, which is the worst failure mode available here.")
        return 1
    print("  all gates passed\n")

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
    txt = np.array([cps[c].get("rendered_prompt") or "" for c in keep], dtype=object)
    qc = np.array([rows[c]["value_continue"] for c in keep])
    qi = np.array([rows[c]["value_intervene"] for c in keep])
    pfail = np.array([rows[c]["p_fail_continue"] for c in keep])
    oracle = np.array([ACTS.index(rows[c]["oracle_action"]) for c in keep])
    Q = np.stack([qc, qi, np.zeros_like(qc)], axis=1)
    tau = qi - qc
    band = pfail.max()

    print("=" * 78)
    print("DECISION-FOCUSED CONTROLLER   cohort={}   n={}   primary layer {}".format(
        args.cohort, len(keep), PRIMARY_LAYER))
    print("=" * 78)
    print("  lambda grid {}   Adam lr {} T {}".format(LAMBDAS, ADAM_LR, ADAM_T))

    out = {"cohort": args.cohort, "n": len(keep), "primary_layer": PRIMARY_LAYER,
           "lambdas": list(LAMBDAS), "adam": {"lr": ADAM_LR, "T": ADAM_T}, "layers": {}}

    for L in LAYERS:
        XA = np.hstack([F, ACT[L]])
        tag = "[PRIMARY]" if L == PRIMARY_LAYER else "(secondary diagnostic)"
        print("\n" + "#" * 78)
        print("# LAYER {}  {}".format(L, tag))
        print("#" * 78)

        rec = defaultdict(lambda: {"regret": [], "p90": [], "agree": [], "freq": [],
                                   "r2": [], "lam": []})
        rows_state = []
        for s in range(args.seeds):
            tr, te = grouped_split(task, frac=0.30, seed=s)
            gtr = task[tr]
            Qte, orte = Q[te], oracle[te]
            zc = np.zeros(int(te.sum()))
            preds = {}

            # --- existing MSE controllers, exactly as run in Phase 3 -------------
            a_c = select_alpha(F[tr], qc[tr], gtr)
            qc_o, _ = fit_reg(F[tr], qc[tr], gtr, F[te], alpha=a_c)
            qc_o_oof = oof_predict(F[tr], qc[tr], gtr, a_c)
            qi_o, _ = fit_reg(np.hstack([F[tr], qc_o_oof[:, None]]), qi[tr], gtr,
                              np.hstack([F[te], qc_o[:, None]]))
            preds["MSE_obs_stacked(existing)"] = np.stack([qc_o, qi_o, zc], 1)

            a_ca = select_alpha(XA[tr], qc[tr], gtr)
            qc_a, _ = fit_reg(XA[tr], qc[tr], gtr, XA[te], alpha=a_ca)
            qc_a_oof = oof_predict(XA[tr], qc[tr], gtr, a_ca)
            qi_a, _ = fit_reg(np.hstack([XA[tr], qc_a_oof[:, None]]), qi[tr], gtr,
                              np.hstack([XA[te], qc_a[:, None]]))
            preds["MSE_obs+act_stacked(existing)"] = np.stack([qc_a, qi_a, zc], 1)

            # --- plain MSE, the architectural match to the SPO+ arms -------------
            for nm, X in (("MSE_obs_plain", F), ("MSE_obs+act_plain", XA)):
                preds[nm] = np.stack([fit_reg(X[tr], qc[tr], gtr, X[te])[0],
                                      fit_reg(X[tr], qi[tr], gtr, X[te])[0], zc], 1)

            # --- the experiment: identical loss, optimiser and lambda rule -------
            for nm, X in (("SPO_obs", F), ("SPO_obs+act", XA)):
                p, lam, traj = fit_spo_cv(X[tr], Q[tr], gtr, X[te])
                preds[nm] = p
                rec[nm]["lam"].append(lam)
                if s == 0:
                    print("  {:<16} seed0 lambda {:<8g} objective t=1/{}/{}: {}".format(
                        nm, lam, ADAM_T // 2, ADAM_T,
                        "  ".join("{:.4f}".format(v) for _, v in traj)))

            # --- contextual reference, unmodified --------------------------------
            if all(len(t) for t in txt):
                preds["TFIDF(reference)"] = np.stack(
                    [fit_tfidf(txt[tr], qc[tr], gtr, txt[te])[0],
                     fit_tfidf(txt[tr], qi[tr], gtr, txt[te])[0], zc], 1)

            for nm, P in preds.items():
                ch = np.argmax(P, axis=1)
                reg = Qte.max(axis=1) - Qte[np.arange(len(ch)), ch]
                rec[nm]["regret"].append(float(reg.mean()))
                rec[nm]["p90"].append(float(np.percentile(reg, 90)))
                rec[nm]["agree"].append(float((ch == orte).mean()))
                rec[nm]["freq"].append([float((ch == i).mean()) for i in range(3)])
                rec[nm]["r2"].append(float(r2_score(qi[te], P[:, 1])))

            gi = np.where(te)[0]
            A_, B_ = preds["SPO_obs"], preds["SPO_obs+act"]
            cA, cB = np.argmax(A_, axis=1), np.argmax(B_, axis=1)
            rA = Qte.max(axis=1) - Qte[np.arange(len(cA)), cA]
            rB = Qte.max(axis=1) - Qte[np.arange(len(cB)), cB]
            oA = np.argmax(preds["MSE_obs_stacked(existing)"], axis=1)
            oB = np.argmax(preds["MSE_obs+act_stacked(existing)"], axis=1)
            mrA = Qte.max(axis=1) - Qte[np.arange(len(oA)), oA]
            mrB = Qte.max(axis=1) - Qte[np.arange(len(oB)), oB]
            for j in range(len(gi)):
                k = gi[j]
                ok_a, ok_b = cA[j] == orte[j], cB[j] == orte[j]
                cat = ("unchanged" if cA[j] == cB[j] else
                       "obs_wrong_act_right" if ok_b and not ok_a else
                       "obs_right_act_wrong" if ok_a and not ok_b else "wrong_to_wrong")
                rows_state.append({"game": task[k], "tau": float(tau[k]), "cat": cat,
                                   "dreg_spo": float(rB[j] - rA[j]),
                                   "dreg_mse": float(mrB[j] - mrA[j]),
                                   "conflict": int(pfail[k] == band)})

        # ---- table ----------------------------------------------------------
        order = ["MSE_obs_stacked(existing)", "MSE_obs+act_stacked(existing)",
                 "MSE_obs_plain", "MSE_obs+act_plain", "SPO_obs", "SPO_obs+act"]
        if "TFIDF(reference)" in rec:
            order.append("TFIDF(reference)")
        print("\n  {:<30} {:>8} {:>8} {:>8} {:>7} {:>7}  {}".format(
            "controller", "regret", "sd", "p90", "agree", "Q_I R2", "per-split regret"))
        Lrec = {}
        for nm in order:
            r = rec[nm]
            Lrec[nm] = {"regret_mean": float(np.mean(r["regret"])),
                        "regret_sd": float(np.std(r["regret"], ddof=1)),
                        "regret_per_seed": r["regret"], "p90": float(np.mean(r["p90"])),
                        "oracle_agreement": float(np.mean(r["agree"])),
                        "qi_r2": float(np.mean(r["r2"])),
                        "action_freq": {a: float(np.mean([f[i] for f in r["freq"]]))
                                        for i, a in enumerate(ACTS)},
                        "lambda_per_seed": r["lam"] or None}
            print("  {:<30} {:>8.3f} {:>8.3f} {:>8.3f} {:>7.3f} {:>7.3f}  [{}]".format(
                nm, Lrec[nm]["regret_mean"], Lrec[nm]["regret_sd"], Lrec[nm]["p90"],
                Lrec[nm]["oracle_agreement"], Lrec[nm]["qi_r2"],
                " ".join("{:.3f}".format(v) for v in r["regret"])))

        print("\n  action frequencies")
        for nm in order:
            f = Lrec[nm]["action_freq"]
            print("    {:<30} ".format(nm) + "  ".join(
                "{} {:.3f}".format(a, f[a]) for a in ACTS))

        # ---- PRIMARY ENDPOINT ------------------------------------------------
        g = np.array([r["game"] for r in rows_state])
        d_spo = np.array([r["dreg_spo"] for r in rows_state])
        d_mse = np.array([r["dreg_mse"] for r in rows_state])
        ta = np.array([r["tau"] for r in rows_state])
        cat = np.array([r["cat"] for r in rows_state])
        cf = np.array([r["conflict"] for r in rows_state], dtype=bool)

        per = [b - a for a, b in zip(rec["SPO_obs"]["regret"], rec["SPO_obs+act"]["regret"])]
        same = all(v > 0 for v in per) or all(v < 0 for v in per)
        lo, hi = boot_games(d_spo, g)
        print("\n  " + "=" * 74)
        print("  PRIMARY ENDPOINT   regret(SPO obs+act) - regret(SPO obs)")
        print("  " + "=" * 74)
        print("    split means : {:+.4f}  sd {:.4f}  {}   [{}]".format(
            float(np.mean(per)), float(np.std(per, ddof=1)),
            "same" if same else "FLIPS", " ".join("{:+.4f}".format(v) for v in per)))
        print("    per-state   : {:+.4f}  game-clustered 95% CI [{:+.4f}, {:+.4f}]".format(
            d_spo.mean(), lo, hi))
        mlo, mhi = boot_games(d_mse, g)
        print("    (MSE arms, same rows, for reference: {:+.4f} [{:+.4f}, {:+.4f}])".format(
            d_mse.mean(), mlo, mhi))
        Lrec["primary"] = {"delta_regret_split_means": float(np.mean(per)),
                           "sd": float(np.std(per, ddof=1)), "per_split": per,
                           "consistent": bool(same),
                           "per_state_mean": float(d_spo.mean()), "ci95": [lo, hi],
                           "mse_reference_mean": float(d_mse.mean()), "mse_ci95": [mlo, mhi]}

        # ---- tau decomposition: did the cancellation break? -------------------
        print("\n  TAU DECOMPOSITION -- the cancellation closure found")
        print("    {:<14} {:>6} {:>12} {:>24} {:>12}".format(
            "subset", "n", "SPO dreg", "95% CI", "MSE dreg"))
        Lrec["tau"] = {}
        for lab, m in (("tau <= 0", ta <= 0), ("tau > 0", ta > 0),
                       ("risk=1 set", cf)):
            if not m.sum():
                continue
            l2, h2 = boot_games(d_spo[m], g[m])
            Lrec["tau"][lab] = {"n": int(m.sum()), "spo_dregret": float(d_spo[m].mean()),
                                "ci95": [l2, h2], "mse_dregret": float(d_mse[m].mean())}
            print("    {:<14} {:>6} {:>+12.4f} {:>13}{:+.4f}, {:+.4f}] {:>+12.4f}".format(
                lab, int(m.sum()), float(d_spo[m].mean()), "[", l2, h2,
                float(d_mse[m].mean())))

        # ---- transition categories -------------------------------------------
        print("\n  TRANSITION CATEGORIES (SPO obs -> SPO obs+act)")
        print("    {:<24} {:>7} {:>8} {:>14}".format("category", "n", "frac", "reg contrib"))
        Lrec["categories"] = {}
        for c in ("unchanged", "obs_wrong_act_right", "obs_right_act_wrong", "wrong_to_wrong"):
            m = cat == c
            contrib = float(d_spo[m].sum() / len(d_spo)) if m.sum() else 0.0
            Lrec["categories"][c] = {"n": int(m.sum()), "frac": float(m.mean()),
                                     "regret_contribution": contrib}
            print("    {:<24} {:>7} {:>8.3f} {:>+14.4f}".format(
                c, int(m.sum()), float(m.mean()), contrib))
        print("    (contributions sum to the per-state mean above)")

        out["layers"]["L{}".format(L)] = Lrec

    os.makedirs("reports", exist_ok=True)
    p = "reports/decision_focused_{}_{}.json".format(cfg["run_id"], args.cohort)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote {}".format(p))
    print("\nDECISION RULE, FIXED BEFORE THE RUN")
    print("  Strong success needs ALL of: SPO obs+act below SPO obs on mean regret;")
    print("  direction consistent across splits AND both cohorts; the tau>0 damage")
    print("  materially shrunk; and no catastrophic p90 or subgroup degradation.")
    print("  Otherwise the white-box control line STOPS -- no second weighting, no second")
    print("  loss, no nonlinear probe, no other layer, no RL, no steering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
