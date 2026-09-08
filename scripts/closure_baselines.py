#!/usr/bin/env python
"""Q3: does the white-box Q_I advantage survive stronger black-box baselines?

Every comparison so far pitted activations against a LINEAR model on 26 hand-built
features. The parent paper's own controller was a random forest, and the full prefix
text exists on disk. So the honest question is whether the advantage is over
"observables" or merely over "a linear model of 26 observables".

TWO BASELINES, BOTH FIXED IN ADVANCE, NEITHER TUNED. No model family is searched and
no winner is selected -- that would answer "can I find a black-box model that wins",
which is a different and much easier question.

    A  RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=2,
                             random_state=0)          -- mirrors the paper's RF family
    B  TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_features=20000,
                       min_df=3, sublinear_tf=True) -> Ridge
                                                     -- the raw rendered prompt

`rendered_prompt` is the byte-exact string sent to the model and was verified to
round-trip under the pinned tokenizer in V10, so baseline B uses real inputs rather
than a reconstruction. VOCABULARY AND IDF ARE FIT ON TRAINING GAMES ONLY; fitting them
on all rows would leak held-out game vocabulary into the representation.

Five models compared on identical splits: linear 26D, RF 26D, TF-IDF prefix, activations,
observables+activations. Then every estimate is pushed through the existing controller,
because a prediction improvement only counts as an oversight improvement if it reduces
regret.

    python scripts/closure_baselines.py --config configs/scale.yaml --cohort all
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cnc import experiment
from cnc.features import prefix as prefix_features
from probe_study import derive, grouped_split, fit_reg, select_alpha
from conflict_set_analysis import cohort_of
from controllers import ACTS

LAYER = 24
RF_KW = dict(n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=0, n_jobs=-1)
TFIDF_KW = dict(lowercase=True, ngram_range=(1, 2), max_features=20000, min_df=3,
                sublinear_tf=True)


def fit_rf(Xtr, ytr, Xte):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(**RF_KW).fit(Xtr, ytr)
    return m.predict(Xte)


def fit_tfidf(txt_tr, ytr, gtr, txt_te):
    """TF-IDF fitted on TRAINING text only, then the study's own ridge protocol."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(**TFIDF_KW)
    Xtr = v.fit_transform(txt_tr).toarray().astype(np.float32)
    Xte = v.transform(txt_te).toarray().astype(np.float32)
    return fit_reg(Xtr, ytr, gtr, Xte)[0], Xtr.shape[1]


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
    ACT = np.ascontiguousarray(A[[ai[c] for c in keep]][:, LAYER, :], dtype=np.float32)
    del A, z

    F = np.array([prefix_features.to_vector(cps[c]["features_26d"]) for c in keep])
    task = np.array([cps[c]["task_id"] for c in keep])
    qc = np.array([rows[c]["value_continue"] for c in keep])
    qi = np.array([rows[c]["value_intervene"] for c in keep])
    pfail = np.array([rows[c]["p_fail_continue"] for c in keep])
    oracle = np.array([ACTS.index(rows[c]["oracle_action"]) for c in keep])
    Q = np.stack([qc, qi, np.zeros_like(qc)], axis=1)
    band = pfail.max()
    conflict = pfail == band

    txt = [cps[c].get("rendered_prompt") or "" for c in keep]
    have_text = sum(1 for t in txt if t)
    txt = np.array(txt, dtype=object)

    print("=" * 78)
    print("Q3  DOES THE ADVANTAGE SURVIVE STRONGER BLACK-BOX BASELINES?  cohort={}".format(
        args.cohort))
    print("=" * 78)
    print("  n {}   conflict set {}   layer {}".format(len(keep), int(conflict.sum()), LAYER))
    print("  rendered_prompt present on {}/{} checkpoints".format(have_text, len(keep)))
    print("  prompt length chars: median {:.0f}  min {}  max {}".format(
        np.median([len(t) for t in txt]), min(len(t) for t in txt), max(len(t) for t in txt)))
    use_text = have_text == len(keep)
    if not use_text:
        print("  !! prefix text incomplete -- baseline B is SKIPPED and reported as unavailable")
    print("  RF: {}".format(RF_KW))
    print("  TFIDF: {}".format(TFIDF_KW))

    out = {"cohort": args.cohort, "n": len(keep), "layer": LAYER,
           "text_available": bool(use_text), "rf_params": {k: v for k, v in RF_KW.items()},
           "tfidf_params": TFIDF_KW, "prediction": {}, "control": {}}

    models = ["linear_26D", "rf_26D", "activations", "obs+activations"]
    if use_text:
        models.insert(2, "tfidf_prefix")

    # ================= Q_I prediction, overall and in the conflict set ============
    for scope, mask in (("overall", None), ("risk=1 conflict set", conflict)):
        print("\n" + "#" * 78)
        print("# Q_I PREDICTION -- {}".format(scope))
        if mask is not None:
            print("# (measured Q_C supplied to every model, as in the conflict-set analysis)")
        print("#" * 78)
        per = {m: [] for m in models}
        vocab_sizes = []
        for s in range(args.seeds):
            tr, te = grouped_split(task, frac=0.30, seed=s)
            if mask is not None:
                tr, te = tr & mask, te & mask
            gtr = task[tr]
            base = np.hstack([F, qc[:, None]]) if mask is not None else F
            act = np.hstack([ACT, qc[:, None]]) if mask is not None else ACT
            both = np.hstack([base, ACT])

            per["linear_26D"].append(r2_score(qi[te], fit_reg(base[tr], qi[tr], gtr, base[te])[0]))
            per["rf_26D"].append(r2_score(qi[te], fit_rf(base[tr], qi[tr], base[te])))
            per["activations"].append(r2_score(qi[te], fit_reg(act[tr], qi[tr], gtr, act[te])[0]))
            per["obs+activations"].append(
                r2_score(qi[te], fit_reg(both[tr], qi[tr], gtr, both[te])[0]))
            if use_text:
                p_t, nv = fit_tfidf(txt[tr], qi[tr], gtr, txt[te])
                vocab_sizes.append(nv)
                per["tfidf_prefix"].append(r2_score(qi[te], p_t))

        print("  {:<20} {:>8} {:>7}  {:<6} {}".format("model", "mean R2", "sd", "sign", "per-split"))
        rec = {}
        for m in models:
            v = per[m]
            mu, sd = float(np.mean(v)), float(np.std(v, ddof=1))
            rec[m] = {"mean_r2": mu, "sd": sd, "per_seed": [float(x) for x in v]}
            print("  {:<20} {:>8.3f} {:>7.3f}  {:<6} [{}]".format(
                m, mu, sd, "", " ".join("{:.3f}".format(x) for x in v)))
        base_mu = rec["linear_26D"]["mean_r2"]
        print("\n  gain over the existing linear 26D baseline")
        for m in models[1:]:
            d = [a - b for a, b in zip(per[m], per["linear_26D"])]
            same = all(x > 0 for x in d) or all(x < 0 for x in d)
            rec[m]["gain_over_linear26D"] = {"mean": float(np.mean(d)),
                                             "sd": float(np.std(d, ddof=1)),
                                             "consistent": bool(same),
                                             "per_seed": [float(x) for x in d]}
            print("    {:<20} {:>+8.3f}  sd {:.3f}  {:<5}  [{}]".format(
                m, float(np.mean(d)), float(np.std(d, ddof=1)),
                "same" if same else "FLIPS", " ".join("{:+.3f}".format(x) for x in d)))
        if vocab_sizes:
            print("  tfidf vocabulary per split: {}".format(vocab_sizes))
            rec["tfidf_vocab_sizes"] = vocab_sizes
        out["prediction"][scope] = rec

    # ================= propagate through the controller ===========================
    print("\n" + "#" * 78)
    print("# CONTROL -- every estimate pushed through the existing controller")
    print("#" * 78)
    ctl = {m: {"regret": [], "p90": [], "agree": [], "freq": []} for m in models}
    for s in range(args.seeds):
        tr, te = grouped_split(task, frac=0.30, seed=s)
        gtr = task[tr]
        zc = np.zeros(int(te.sum()))
        Qte, orte = Q[te], oracle[te]
        best = Qte.max(axis=1)
        both = np.hstack([F, ACT])
        preds = {
            "linear_26D": (fit_reg(F[tr], qc[tr], gtr, F[te])[0],
                           fit_reg(F[tr], qi[tr], gtr, F[te])[0]),
            "rf_26D": (fit_rf(F[tr], qc[tr], F[te]), fit_rf(F[tr], qi[tr], F[te])),
            "activations": (fit_reg(ACT[tr], qc[tr], gtr, ACT[te])[0],
                            fit_reg(ACT[tr], qi[tr], gtr, ACT[te])[0]),
            "obs+activations": (fit_reg(both[tr], qc[tr], gtr, both[te])[0],
                                fit_reg(both[tr], qi[tr], gtr, both[te])[0]),
        }
        if use_text:
            preds["tfidf_prefix"] = (fit_tfidf(txt[tr], qc[tr], gtr, txt[te])[0],
                                     fit_tfidf(txt[tr], qi[tr], gtr, txt[te])[0])
        for m in models:
            pc, pi = preds[m]
            ch = np.stack([pc, pi, zc], 1).argmax(1)
            reg = best - Qte[np.arange(len(ch)), ch]
            ctl[m]["regret"].append(float(reg.mean()))
            ctl[m]["p90"].append(float(np.percentile(reg, 90)))
            ctl[m]["agree"].append(float((ch == orte).mean()))
            ctl[m]["freq"].append([float((ch == i).mean()) for i in range(3)])

    print("  {:<20} {:>9} {:>9} {:>8}   {}".format("model", "regret", "p90", "agree", "actions"))
    for m in models:
        f = np.mean(np.array(ctl[m]["freq"]), axis=0)
        out["control"][m] = {
            "regret_mean": float(np.mean(ctl[m]["regret"])),
            "regret_sd": float(np.std(ctl[m]["regret"], ddof=1)),
            "regret_per_seed": ctl[m]["regret"],
            "p90": float(np.mean(ctl[m]["p90"])),
            "oracle_agreement": float(np.mean(ctl[m]["agree"])),
            "action_freq": {a: float(f[i]) for i, a in enumerate(ACTS)}}
        print("  {:<20} {:>9.3f} {:>9.3f} {:>8.3f}   {}".format(
            m, float(np.mean(ctl[m]["regret"])), float(np.mean(ctl[m]["p90"])),
            float(np.mean(ctl[m]["agree"])),
            "  ".join("{} {:.3f}".format(a, f[i]) for i, a in enumerate(ACTS))))

    os.makedirs("reports", exist_ok=True)
    p = "reports/closure_baselines_{}_{}.json".format(cfg["run_id"], args.cohort)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote {}".format(p))
    print("\nTHE INTERPRETATION RULE, FIXED IN ADVANCE")
    print("  If activations stay substantially ahead: stronger evidence that the actor's")
    print("  own representation makes intervention value unusually accessible.")
    print("  If the RF or the prefix text closes the gap: the claim NARROWS to 'Q_I is")
    print("  linearly accessible in the actor representation beyond the original")
    print("  26-feature LINEAR baseline', and 'information absent from observables' is")
    print("  withdrawn. Either way the control column stands separately -- a prediction")
    print("  gain counts as an oversight gain only if it reduces regret.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
