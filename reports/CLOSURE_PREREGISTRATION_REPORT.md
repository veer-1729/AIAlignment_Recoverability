# Pre-registration — closure battery (Q1–Q3)

Committed **before** any closure script was written or run. Frozen predecessors: `stage2-frozen`
(`9d7a48d`), `f1e78cb` (Phase 3 pre-registration), `phase3-battery` (`84f0bd8`). No Phase 3
artifact is modified, renamed or deleted.

Standing rules carried forward unchanged: held-out whole games is primary; the five existing
split seeds are reused; all preprocessing, alpha grids, fold counts and train-only
hyperparameter selection are imported from `probe_study.py`; sign consistency across the five
splits is the standard; every script emits nulls explicitly; no test-set target is ever a
feature except in a labelled ceiling.

---

## Q1 — does Q_I have a stable direction while τ does not?

**The evaluation set is fixed and is not redefined.** Exactly the existing A→B unseen set from
`frozen_transfer.py`: destination = `original`, and

    unseen = is_original & ~isin(task_id, games_original ∩ games_backfill)

which is 445 rows across 57 games. Any script that reports a different count is wrong and its
numbers are void.

**The destination-refit comparator.** Trained on `original` rows whose game *is* in the shared
set (483 rows), evaluated on exactly the 445 unseen rows. This is game-disjoint by construction —
no training game appears in the test set — and it holds the test rows identical between the
frozen and refit conditions, which is the comparison that was missing.

**(a)** Per target `{Q_I, Q_C, τ}` × layer `{8, 16, 24}`: source held-out-game R²; frozen A→B R²
on the 445; destination-refit R² on the same 445; **frozen − refit**.

**(b)** For the frozen destination predictions: RMSE, Pearson r, Spearman ρ, and the slope and
intercept of `true ~ prediction`. The calibration figures are **descriptive only** and are not
used to license any claim.

**(c) Coefficients are compared in raw activation coordinates, not standardized ones.** A ridge
fit on standardized inputs predicts `b₀ + Σ cⱼ(xⱼ − μⱼ)/σⱼ`, so the raw-space weight is
`cⱼ/σⱼ`. Two probes with independently fitted scalers have different σ, so comparing their `c`
directly compares scaler artifacts. Cosine is computed on `c/σ` for both probes. The
standardized-space cosine from Phase 3 is reported alongside, marked as the superseded quantity.

**(d)** For τ: the directly trained frozen τ probe versus `frozen_Q_I_pred − frozen_Q_C_pred`
from the independently frozen source probes, on identical rows.

**(e) Diagnostic only.** A two-parameter affine map `a·pred + b` fitted on **destination
training rows (the 483 shared-game rows)** and applied to the 445 test rows. Never fitted on
test.

**Decision rule, fixed now.**
- If τ retains strong correlation and raw-space cosine, and affine recalibration restores most
  of its R², then **the claim "the τ direction does not transfer" is withdrawn** and replaced
  with "its frozen calibration is less stable."
- "Stable Q_I direction, unstable τ direction" is licensed **only** if the calibration-free
  evidence (Spearman, raw-space cosine) and the post-recalibration R² both support the
  distinction.
- If the evidence demands the narrower reading, the narrower reading is taken.

---

## Q2 — why does better Q_I prediction not reduce mean control regret?

**Layer 24 is primary and is not substituted.** It was fixed before any layer search. Layers 8
and 16 are reported descriptively afterwards; no conclusion depends on a layer chosen after
seeing results. The full 29-layer sweep is *not* re-run here — E already reported it, and
re-selecting from it would be the search this programme avoids.

Per held-out test state, at every risk level: true `Q_continue`, `Q_intervene`, `Q_quit = 0`;
observable-model and activation-model predicted values; each model's chosen action; the true
oracle action; each model's regret; the per-state regret difference; the **true action margin**
(best true action value minus second-best); and the absolute and squared Q_I prediction-error
improvement.

**Four exhaustive, mutually exclusive categories:** action unchanged; observable wrong →
activation correct; observable correct → activation wrong; wrong → different wrong. For each:
count and fraction, contribution to the total mean-regret difference, mean Q_I prediction-error
improvement, and the true action-margin distribution.

**Margin bins are fixed now:** deciles of the true action margin computed **once over all rows
of the cohort**, before any split, so the bin edges cannot depend on which states a model got
right. Within each bin: activation improvement in Q_I prediction error, activation minus
observable regret, and action-flip rate.

**The critical decomposition.** Total Q_I sum-of-squared-error reduction attributed to: states
where both controllers choose the same action; states where they differ; states where the
observable controller is wrong; and each margin decile. This answers whether the extra
information sits where it cannot change a decision.

Also reported for `τ ≤ 0` vs `τ > 0`, and for the risk = 1 conflict set.

Uncertainty: bootstrap clustered on **game**, plus the existing five-split sign checks.

**No new decision-focused model is fitted here.** The question is only whether decision-relevant
signal exists.

---

## Q3 — does the advantage survive stronger black-box baselines?

**One nonlinear model, one text model, both fixed in advance. No model family is tuned and no
winner is selected.**

**Baseline A — nonlinear 26D.** `RandomForestRegressor(n_estimators=200, max_depth=6,
min_samples_leaf=2, random_state=0)`. No hyperparameter search of any kind.

**Baseline B — raw prefix text.** `rendered_prompt` is stored byte-exact on every checkpoint and
was validated to round-trip in V10, so this baseline is available. Settings fixed now:
`TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_features=20000, min_df=3,
sublinear_tf=True)` → Ridge, alpha chosen by the same grouped-CV-inside-train protocol as every
other model. **Vocabulary and IDF are fitted on training games only.** No embedding models, no
alternatives tried.

Evaluated on the same held-out-game splits and both cohorts. Targets: Q_I overall, and Q_I
inside risk = 1 with measured Q_C supplied to the baseline, mirroring the conflict-set analysis.

Estimates are propagated through the existing controller and scored on mean regret, p90 regret,
oracle agreement and action distribution.

Five models compared without selection: existing linear 26D; fixed nonlinear 26D; TF-IDF prefix;
existing activation probe; existing observable+activation probe.

**Interpretation rules, fixed now.**
- If activations remain substantially better: stronger evidence that the actor's own
  representation makes intervention value unusually accessible.
- If either baseline closes the gap: **the claim narrows explicitly** to "Q_I is linearly
  accessible in the actor's representation beyond the original 26-feature *linear* baseline,"
  and the statement that the hidden state contains information absent from observables is
  withdrawn.
- Either way the control result stands separately: a prediction improvement counts as an
  oversight improvement only if it reduces regret.

---

## What would change the recommendation

- Q1 showing τ's degradation is calibration rather than direction → story B narrows and the
  value/advantage split weakens.
- Q2 showing the extra Q_I signal concentrated at large action margins → story A strengthens
  and becomes the headline.
- Q3 showing a nonlinear or text baseline matching the activation probe → story C becomes the
  honest recommendation and the white-box framing is dropped.

Any of these outcomes is reported as found. None is a failure of the battery.
