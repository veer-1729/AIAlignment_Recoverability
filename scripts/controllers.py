#!/usr/bin/env python
"""A + B: how much more useful is recoverability than ordinary future prediction?

Everything so far measures whether quantities are DECODABLE. This measures whether
decoding them buys better CONTROL, which is a different question and the one the
programme is actually about. A probe can win on R2 and change no decision.

THE LADDER. Each controller picks an action by `argmax_a Qhat_a(s)` under the study's
utility definition, with Qhat_quit fixed at 0. They differ only in what they know:

    floors        always-continue / always-intervene / always-quit
    R1  risk      the frozen argmax action per continuation-risk level (A1)
    R2  predict   Qhat_C from observables; Qhat_I is the TRAIN MEAN, a constant
    R3  obs       Qhat_C and Qhat_I both from observables
    R4  stacked   Qhat_C from observables; Qhat_I from [observables, Qhat_C]
    R5  white-box Qhat_C and Qhat_I from [observables, activations L24]
    ceiling       argmax over TRUE Q -- regret zero by construction, the denominator

R2 is the load-bearing comparison. It is a controller that forecasts what happens if
the agent continues and has no model of the expert at all -- ordinary future
prediction. R3 and R4 add recoverability from observables; R5 adds it from the
residual stream. R2 -> R3 isolates "does modelling intervention value help control at
all", and R4 -> R5 isolates "do activations help control beyond observable features".

A NOTE ON READING THE USER'S SPEC. "observables + predicted Q_C" and "observables +
predicted Q_C, Q_I" are ambiguous under an argmax rule, because a controller needs
some Qhat for every action before it can choose. The ladder above resolves it by
varying what the controller can estimate rather than what it is handed, which is the
reading that makes R2 vs R3 answer the stated scientific question. The alternative
reading -- stacking a predicted Q_C as an extra column -- is R4, so both are present.

LEAKAGE. Qhat_C enters R4/R5 as a feature. On train it is an OUT-OF-FOLD prediction,
never an in-sample one; an in-sample Qhat_C is overfitted and would make the stacked
model look better than it can be at test time. On test it is the frozen model's
output. Every prediction model is fitted on training games only.

B (error characterisation) rides along: the per-checkpoint held-out predictions that
R4 and R5 produce inside risk=1 are dumped with their metadata, and the improvement
from activations is stratified by task type, progress, baseline error and oracle
action. No new fitting.

Held-out games only -- the primary regime. Layer 24 only, which is the pre-registered
deepest layer and the strongest everywhere so far; sweeping layers here would be a
search, and E does the layer question properly.

    python scripts/controllers.py --config configs/scale.yaml --cohort all
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cnc import experiment, utility
from cnc.features import prefix as prefix_features
from cnc.storage import read_jsonl
from probe_study import derive, grouped_split, fit_reg, select_alpha
from conflict_set_analysis import cohort_of

LAYER = 24
HIGH_REGRET = 0.5        # fixed in advance: about half of tau's sd
ACTS = ("continue", "intervene", "quit")


def oof_predict(X, y, g, alpha):
    """Out-of-fold predictions on train, at a pinned alpha.

    Used to build Qhat_C as a feature for the stacked models. In-sample predictions
    here would be overfitted, and a downstream model would learn to trust a column
    that is far worse at test time than it looks in training.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    n_sp = min(5, len(np.unique(g)))
    oof = np.zeros(len(y))
    if n_sp < 2:
        return oof
    for a_, b_ in GroupKFold(n_splits=n_sp).split(X, y, g):
        sc = StandardScaler().fit(X[a_])
        oof[b_] = Ridge(alpha=alpha).fit(sc.transform(X[a_]), y[a_]).predict(sc.transform(X[b_]))
    return oof


def evaluate(choice, Q, oracle_idx):
    """Utility, regret and agreement for a chosen-action vector.

    Regret is against TRUE Q from the branches, so it is the real cost of the
    decision and not the model's opinion of it.
    """
    util = Q[np.arange(len(choice)), choice]
    best = Q.max(axis=1)
    reg = best - util
    return {"utility": float(util.mean()),
            "regret_mean": float(reg.mean()),
            "regret_median": float(np.median(reg)),
            "regret_p90": float(np.percentile(reg, 90)),
            "frac_high_regret": float((reg > HIGH_REGRET).mean()),
            "oracle_agreement": float((choice == oracle_idx).mean()),
            "action_freq": {a: float((choice == i).mean()) for i, a in enumerate(ACTS)},
            "_regret": reg}


def summarise(name, per_seed, key):
    v = [p[key] for p in per_seed]
    mu, sd = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return {"mean": mu, "sd": sd, "per_seed": [float(x) for x in v]}


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
    tasks = {t["task_id"]: t for t in read_jsonl(rd.tasks)}

    z = np.load(rd.path("activations_{}.npz".format(args.arm)), allow_pickle=True)
    A, aids = z["activations"], list(z["checkpoint_ids"])
    ai = {c: i for i, c in enumerate(aids)}
    keep = sorted(c for c in aids if c in rows and c in cps)
    if args.cohort != "all":
        src = rd.branch_sources()
        keep = [c for c in keep if cohort_of(src.get((c, args.arm), "")) == args.cohort]
    ACT = np.ascontiguousarray(A[[ai[c] for c in keep]][:, LAYER, :], dtype=np.float32)
    del A, z

    lens = {}
    for r in rd.read_all("rollouts"):
        if r.get("arm") == args.arm or "arm" not in r:
            lens[r["rollout_id"]] = r.get("n_steps") or 0

    F = np.array([prefix_features.to_vector(cps[c]["features_26d"]) for c in keep])
    task = np.array([cps[c]["task_id"] for c in keep])
    qc = np.array([rows[c]["value_continue"] for c in keep])
    qi = np.array([rows[c]["value_intervene"] for c in keep])
    pfail = np.array([rows[c]["p_fail_continue"] for c in keep])
    oracle_idx = np.array([ACTS.index(rows[c]["oracle_action"]) for c in keep])
    prog = np.array([cps[c]["t"] / lens[cps[c]["rollout_id"]] if lens.get(cps[c]["rollout_id"])
                     else np.nan for c in keep])
    ttype = np.array([tasks.get(cps[c].get("task_id"), {}).get("task_type", "?") for c in keep])

    # True Q for all three actions. Quit is analytic and exactly 0.
    Q = np.stack([qc, qi, np.zeros_like(qc)], axis=1)
    levels = np.unique(pfail)
    n = len(keep)

    print("=" * 78)
    print("CONTROLLERS  (A) + ERROR CHARACTERISATION (B)   cohort={}".format(args.cohort))
    print("=" * 78)
    print("  checkpoints {}   risk levels {}   layer {}".format(n, list(np.round(levels, 2)), LAYER))
    print("  oracle mix  " + "  ".join("{} {:.3f}".format(a, (oracle_idx == i).mean())
                                       for i, a in enumerate(ACTS)))

    per_seed = defaultdict(list)
    per_seed_by_level = defaultdict(lambda: defaultdict(list))
    dump = []

    for s in range(args.seeds):
        tr, te = grouped_split(task, frac=0.30, seed=s)
        gtr = task[tr]

        # ---- prediction models, all fitted on training games only ---------------
        a_c = select_alpha(F[tr], qc[tr], gtr)
        qc_obs_te, _ = fit_reg(F[tr], qc[tr], gtr, F[te], alpha=a_c)
        qc_obs_tr_oof = oof_predict(F[tr], qc[tr], gtr, a_c)

        qi_obs_te, _ = fit_reg(F[tr], qi[tr], gtr, F[te])

        Fq_tr = np.hstack([F[tr], qc_obs_tr_oof[:, None]])
        Fq_te = np.hstack([F[te], qc_obs_te[:, None]])
        qi_stack_te, _ = fit_reg(Fq_tr, qi[tr], gtr, Fq_te)

        XA = np.hstack([F, ACT])
        a_ca = select_alpha(XA[tr], qc[tr], gtr)
        qc_act_te, _ = fit_reg(XA[tr], qc[tr], gtr, XA[te], alpha=a_ca)
        qc_act_tr_oof = oof_predict(XA[tr], qc[tr], gtr, a_ca)
        XAq_tr = np.hstack([XA[tr], qc_act_tr_oof[:, None]])
        XAq_te = np.hstack([XA[te], qc_act_te[:, None]])
        qi_act_te, _ = fit_reg(XAq_tr, qi[tr], gtr, XAq_te)

        zeros = np.zeros(int(te.sum()))
        qi_train_mean = float(qi[tr].mean())

        # ---- A1: the frozen risk -> action map, computed inside train only ------
        risk_map = {}
        for lv in levels:
            m = tr & (pfail == lv)
            risk_map[float(lv)] = int(np.argmax(Q[m].mean(axis=0))) if m.sum() else 0
        r1_choice = np.array([risk_map[float(v)] for v in pfail[te]])

        cand = {
            "floor_continue": np.zeros(int(te.sum()), dtype=int),
            "floor_intervene": np.ones(int(te.sum()), dtype=int),
            "floor_quit": np.full(int(te.sum()), 2, dtype=int),
            "R1_risk_only": r1_choice,
            "R2_predict_only": np.stack(
                [qc_obs_te, np.full(int(te.sum()), qi_train_mean), zeros], 1).argmax(1),
            # R2 pins Qhat_I to a constant above zero, so quit can never win its
            # argmax and it degenerates toward always-intervene. R2b removes the
            # intervene option entirely instead: a controller that forecasts what
            # happens if the agent continues and can only keep going or give up.
            # Reporting both separates "R3 can quit and R2 cannot" from "R3 models
            # intervention value".
            "R2b_no_intervene": np.stack(
                [qc_obs_te, np.full(int(te.sum()), -1e9), zeros], 1).argmax(1),
            "R3_obs": np.stack([qc_obs_te, qi_obs_te, zeros], 1).argmax(1),
            "R4_stacked": np.stack([qc_obs_te, qi_stack_te, zeros], 1).argmax(1),
            "R5_whitebox": np.stack([qc_act_te, qi_act_te, zeros], 1).argmax(1),
            "ceiling_true_Q": Q[te].argmax(1),
        }

        Qte, orte, pfte = Q[te], oracle_idx[te], pfail[te]
        for nm, ch in cand.items():
            r = evaluate(ch, Qte, orte)
            reg = r.pop("_regret")
            per_seed[nm].append(r)
            for lv in levels:
                m = pfte == lv
                if m.sum():
                    per_seed_by_level[nm][float(lv)].append(
                        {"n": int(m.sum()), "regret_mean": float(reg[m].mean()),
                         "utility": float(Qte[np.arange(len(ch)), ch][m].mean()),
                         "oracle_agreement": float((ch[m] == orte[m]).mean())})

        # ---- B: per-checkpoint dump inside the conflict set ---------------------
        te_idx = np.where(te)[0]
        for j, gi in enumerate(te_idx):
            if pfail[gi] != levels[-1]:
                continue
            dump.append({"seed": s, "cid": keep[gi], "game": str(task[gi]),
                         "task_type": str(ttype[gi]), "progress": float(prog[gi]),
                         "qc": float(qc[gi]), "qi": float(qi[gi]), "tau": float(qi[gi] - qc[gi]),
                         "oracle": ACTS[oracle_idx[gi]],
                         "pred_qi_stacked": float(qi_stack_te[j]),
                         "pred_qi_whitebox": float(qi_act_te[j]),
                         "err_stacked": float(abs(qi[gi] - qi_stack_te[j])),
                         "err_whitebox": float(abs(qi[gi] - qi_act_te[j]))})

    # ================= A2 report =================
    print("\n" + "#" * 78)
    print("# A2  CONTROLLERS, held-out games, {} splits".format(args.seeds))
    print("#" * 78)
    print("  {:<20} {:>8} {:>9} {:>9} {:>9} {:>8} {:>7}".format(
        "controller", "utility", "regret", "reg_med", "reg_p90", "hi-reg", "agree"))
    out = {"cohort": args.cohort, "n": n, "layer": LAYER, "controllers": {}}
    order = ["floor_continue", "floor_intervene", "floor_quit", "R1_risk_only",
             "R2_predict_only", "R2b_no_intervene", "R3_obs", "R4_stacked",
             "R5_whitebox", "ceiling_true_Q"]
    for nm in order:
        rec = {k: summarise(nm, per_seed[nm], k) for k in
               ("utility", "regret_mean", "regret_median", "regret_p90",
                "frac_high_regret", "oracle_agreement")}
        rec["action_freq"] = {a: float(np.mean([p["action_freq"][a] for p in per_seed[nm]]))
                              for a in ACTS}
        out["controllers"][nm] = rec
        print("  {:<20} {:>8.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>8.3f} {:>7.3f}".format(
            nm, rec["utility"]["mean"], rec["regret_mean"]["mean"],
            rec["regret_median"]["mean"], rec["regret_p90"]["mean"],
            rec["frac_high_regret"]["mean"], rec["oracle_agreement"]["mean"]))
    print("\n  action frequencies")
    for nm in order:
        f = out["controllers"][nm]["action_freq"]
        print("    {:<20} " .format(nm) + "  ".join("{} {:.3f}".format(a, f[a]) for a in ACTS))

    print("\n  KEY CONTRASTS (mean regret; negative = the richer controller is better)")
    pairs = [("R2_predict_only", "R3_obs", "does modelling Q_I help at all?"),
             ("R2b_no_intervene", "R3_obs",
              "value of an intervention option, vs forecasting alone"),
             ("floor_intervene", "R3_obs", "does the best learned controller beat the best floor?"),
             ("R3_obs", "R4_stacked", "does stacking Qhat_C help?"),
             ("R4_stacked", "R5_whitebox", "do activations help CONTROL?"),
             ("R1_risk_only", "R5_whitebox", "risk-only vs white-box")]
    out["contrasts"] = {}
    for a, b, why in pairs:
        d = [x["regret_mean"] - y["regret_mean"] for x, y in zip(per_seed[b], per_seed[a])]
        same = all(v > 0 for v in d) or all(v < 0 for v in d)
        out["contrasts"]["{}->{}".format(a, b)] = {
            "delta_regret_mean": float(np.mean(d)), "sd": float(np.std(d, ddof=1)),
            "consistent": bool(same), "per_seed": [float(v) for v in d]}
        print("    {:<34} {:>+7.4f}  sd {:.4f}  {:<5}  [{}]".format(
            "{} -> {}".format(a, b), float(np.mean(d)), float(np.std(d, ddof=1)),
            "same" if same else "FLIPS", " ".join("{:+.3f}".format(v) for v in d)))
        print("      ({})".format(why))

    # ================= A3 report =================
    print("\n" + "#" * 78)
    print("# A3  REGRET BY CONTINUATION-RISK LEVEL")
    print("#" * 78)
    print("  {:<20}".format("controller") + "".join("{:>12}".format("risk {:.2f}".format(v))
                                                    for v in levels))
    out["by_level"] = {}
    for nm in order:
        cells, rec = [], {}
        for lv in levels:
            vals = [d["regret_mean"] for d in per_seed_by_level[nm][float(lv)]]
            ns = [d["n"] for d in per_seed_by_level[nm][float(lv)]]
            if vals:
                rec[str(round(float(lv), 2))] = {"regret_mean": float(np.mean(vals)),
                                                 "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                                 "mean_n_test": float(np.mean(ns))}
                cells.append("{:>12.3f}".format(np.mean(vals)))
            else:
                cells.append("{:>12}".format("--"))
        out["by_level"][nm] = rec
        print("  {:<20}".format(nm) + "".join(cells))

    # Three contrasts per risk level, because they answer different questions and
    # only the second is clean.
    #
    #   R2b -> R4  is confounded. R2b cannot select intervene AT ALL (its Qhat_I is
    #              pinned to -1e9), so its regret is dominated by a missing action
    #              rather than by missing information, at every level.
    #   R1  -> R4  IS the recoverability question. Both controllers have all three
    #              actions; they differ only in that R1 knows the continuation-risk
    #              level and R4 knows observables plus predicted Q_C and Q_I.
    #   R4  -> R5  is the same question for activations specifically.
    out["value_by_level"] = {}
    for lo, hi, label, note in (
            ("R2b_no_intervene", "R4_stacked", "R2b->R4",
             "CONFOUNDED: R2b has no intervene action; reported for completeness"),
            ("R1_risk_only", "R4_stacked", "R1->R4",
             "CLEAN: same action set, risk level vs full observable model"),
            ("R4_stacked", "R5_whitebox", "R4->R5",
             "CLEAN: same action set, observables vs observables+activations")):
        print("\n  MARGINAL VALUE BY RISK LEVEL: {}   (positive = the richer one is better)".format(label))
        print("  {}".format(note))
        rec = {}
        for lv in levels:
            a = [d["regret_mean"] for d in per_seed_by_level[lo][float(lv)]]
            b = [d["regret_mean"] for d in per_seed_by_level[hi][float(lv)]]
            if not a or not b or len(a) != len(b):
                print("    risk {:.2f}   -- not scorable in every split "
                      "(a={} splits, b={})".format(float(lv), len(a), len(b)))
                continue
            d = [x - y for x, y in zip(a, b)]
            same = all(v > 0 for v in d) or all(v < 0 for v in d)
            rec[str(round(float(lv), 2))] = {
                "delta_regret": float(np.mean(d)),
                "sd": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                "consistent": bool(same), "n_splits": len(d),
                "per_seed": [float(v) for v in d]}
            print("    risk {:.2f}   {:>+8.3f}  sd {:.3f}  {:<5}  n_splits={}  [{}]".format(
                float(lv), float(np.mean(d)),
                float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                "same" if same else "FLIPS", len(d),
                " ".join("{:+.3f}".format(v) for v in d)))
        out["value_by_level"][label] = rec
    out["recoverability_value_by_level"] = out["value_by_level"].get("R1->R4", {})

    # ================= B report =================
    print("\n" + "#" * 78)
    print("# B  WHERE DOES THE ACTIVATION ADVANTAGE ON Q_I LIVE?  (risk = 1 only)")
    print("#" * 78)
    imp = np.array([d["err_stacked"] - d["err_whitebox"] for d in dump])
    eb = np.array([d["err_stacked"] for d in dump])
    print("  held-out conflict-set predictions pooled over splits : {}".format(len(dump)))
    print("  mean |error| observables+Qhat_C : {:.3f}".format(eb.mean()))
    print("  mean |error| + activations      : {:.3f}".format(
        np.array([d["err_whitebox"] for d in dump]).mean()))
    print("  mean improvement                : {:+.3f}   ({:.1%} of rows improved)".format(
        imp.mean(), float((imp > 0).mean())))

    out["B"] = {"n_rows": len(dump), "mean_err_baseline": float(eb.mean()),
                "mean_err_whitebox": float(np.array([d["err_whitebox"] for d in dump]).mean()),
                "mean_improvement": float(imp.mean()),
                "frac_improved": float((imp > 0).mean()), "strata": {}}

    def stratify(label, keyfn, bins=None):
        print("\n  improvement by {}".format(label))
        groups = defaultdict(list)
        for d, i in zip(dump, imp):
            groups[keyfn(d)].append(i)
        rec = {}
        for k in sorted(groups, key=lambda x: str(x)):
            v = np.array(groups[k])
            rec[str(k)] = {"n": len(v), "mean_improvement": float(v.mean()),
                           "frac_improved": float((v > 0).mean())}
            print("    {:<28} n={:<5} mean {:+.3f}   improved {:.1%}".format(
                str(k), len(v), v.mean(), float((v > 0).mean())))
        out["B"]["strata"][label] = rec

    stratify("task type", lambda d: d["task_type"])
    stratify("progress quartile", lambda d: "Q{}".format(
        min(4, int(d["progress"] * 4) + 1)))
    qs = np.percentile(eb, [25, 50, 75])
    stratify("baseline |error| quartile", lambda d: "E{}".format(
        1 + int(np.searchsorted(qs, d["err_stacked"]))))
    stratify("oracle action", lambda d: d["oracle"])
    stratify("tau sign", lambda d: "tau>0 (intervene better)" if d["tau"] > 0 else "tau<=0")

    os.makedirs("reports", exist_ok=True)
    p = "reports/controllers_{}_{}.json".format(cfg["run_id"], args.cohort)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    dp = "reports/controllers_dump_{}_{}.json".format(cfg["run_id"], args.cohort)
    with open(dp, "w") as fh:
        json.dump(dump, fh)
    print("\nwrote {}\nwrote {}".format(p, dp))

    # ---- nulls, emitted explicitly so the memo is built from output -------------
    nulls = [k for k, v in out["contrasts"].items() if not v["consistent"]
             or abs(v["delta_regret_mean"]) < 0.005]
    print("\nNULL CANDIDATES (sign-inconsistent across splits, or |effect| < 0.005):")
    for k in nulls:
        v = out["contrasts"][k]
        print("  {:<40} {:+.4f}  {}".format(k, v["delta_regret_mean"],
                                            "consistent" if v["consistent"] else "FLIPS"))
    if not nulls:
        print("  none among the controller contrasts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
