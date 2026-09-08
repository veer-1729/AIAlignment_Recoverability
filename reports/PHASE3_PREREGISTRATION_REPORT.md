# Pre-registration — exploratory battery (A–E)

Written **before** the feasibility census returned and before any analysis in this
battery was run, and committed so the timestamp is checkable. Frozen baseline is
tag `stage2-frozen` (commit `9d7a48d`).

The battery is exploratory by design. That makes it *more* important, not less, to
fix the decision rules in advance — an exploratory battery with post-hoc thresholds
is a search, and a search reports whichever cell worked.

---

## Standing rules for every analysis

1. **Held-out whole games is the primary regime.** Grouped-by-rollout is reported
   as secondary and never as the headline.
2. **The five existing split seeds are reused unchanged.** No new split scheme.
3. **All preprocessing, regularization, alpha/C grids, fold counts and train-only
   hyperparameter selection are imported from `probe_study.py`,** so the protocol is
   identical by construction rather than by inspection.
4. **Sign consistency across the five splits is the standard.** A mean whose sign is
   not consistent is not established, whatever any single split's interval says.
5. **Every script emits `null_candidates` alongside `positive_candidates`** — every
   FLIPS, every interval containing zero, every effect below 0.01 — so the memo is
   assembled from output rather than from recollection.
6. **No test-set Q values are used as features** except in an explicitly labelled
   oracle-information ceiling.

---

## C — same-game matched-state test: the matching rule, fixed in advance

The census reports candidate pair counts across a grid of progress-gap and Q_I
separation thresholds. **The grid selects the cell; I do not.**

**Rule.** Use `dQ_I ≥ 0.50` and the **tightest** progress tolerance in
`{0.05, 0.10, 0.15, 0.25}` that yields **≥ 50 opposite-oracle pairs**. Tightest, not
the one with the most pairs — the test is about matched states, and a loose progress
gap dilutes exactly what it is supposed to control for.

**If no tolerance reaches 50 opposite-oracle pairs, C is reported as unsupported by
this substrate.** It is not re-run with `dQ_I ≥ 0.25`, and the pair count is not
padded by loosening progress beyond 0.25.

**Same-rollout pairs are excluded.** Two checkpoints from one rollout at different
`t` share a trajectory prefix, which is a strictly stronger confound than sharing a
game: the earlier state is a literal ancestor of the later one. The census counts
them separately so the cost of excluding them is visible.

**Matching is without replacement,** greedy by ascending progress distance, so no
checkpoint appears in two pairs and a few well-connected games cannot dominate.

**Primary statistic:** pairwise ranking accuracy — does the model put the higher true
Q_I state above the other — for observables, activations, and both, trained on other
games and evaluated only on pairs drawn from held-out games.

---

## A — controllers

**A1 (risk-only).** The mapping `g(Q_C) = argmax_a E[U(a) | Q_C level]` is computed
**strictly inside the training mask** and frozen before touching test. `U(a)` is the
realized branch utility — a label, never a feature. Pooled means over all data would
leak held-out outcomes into the mapping and are not used.

**A2 (nested controllers).** Reported together, with floors and a ceiling, because a
controller comparison without both is uninterpretable:

- floors: always-continue, always-intervene, always-quit
- 1 continuation risk / Q_C only
- 2 observables
- 3 observables + predicted Q_C
- 4 observables + predicted Q_C, Q_I
- 5 observables + activation-based predicted Q_C, Q_I
- ceiling: argmax over **true** Q values — regret zero by definition, reported only
  as the denominator and labelled as oracle information

Every prediction model is fit on training games only. At test time
`â = argmax_a Q̂_a(s)`, then regret is `max_a Q_a(s) − Q_â(s)` against true Q.

**Regret is reported per continuation-risk level, not only pooled.** Pooled mean
regret is dominated by the risk = 1.00 states, where τ's range is widest; a pooled
number would hide the risk = 0.00 states where every controller is near-perfect.
A3's stratification is emitted by A2 rather than recomputed.

**Prediction of the A3 hypothesis, recorded before measurement.** The stated
hypothesis is that ordinary forecasting suffices when continuation prospects are
clearly good and fails when failure probability saturates. §6.2 of the frozen report
already shows risk = 0.00 is 561/598 oracle-continue and risk = 1.00 splits
686/534 — so the *opportunity* for recoverability to matter is concentrated at the
top of the risk range by construction. **A3 confirming this is therefore weak
evidence, close to circular; A3 failing to confirm it would be strong evidence and
the more informative outcome.** This is stated now so the confirmation is not later
presented as a discovery.

---

## D — frozen cross-cohort transfer

The source scaler and coefficients are frozen together. `probe_study.fit_reg` refits
its scaler inside each call and does not return it, so a `fit_reg_frozen` variant is
used that returns `(scaler, model, alpha)` with identical `select_alpha` and Ridge.

**Wiring check:** `fit_reg_frozen` evaluated within its source cohort must reproduce
`fit_reg`'s in-cohort R² to three decimals. If it does not, the transfer numbers are
void and are reported as such.

No destination-cohort fitting, refitting, or recalibration of any kind. Transfer
performance is the statistic; coefficient cosine similarity is reported beside it and
explicitly **not** led with, since two probes can point similarly and still transfer
badly.

---

## E — layerwise

All layers the store holds, **held-out games only**, targets Q_C / Q_I / τ, five
splits, both cohorts. Layer 0 included as a floor if present.

**Oracle-action classification is dropped from E and this is a cost decision, not a
result decision.** At ~29 layers × 5 seeds × 26 logistic fits per cell it is roughly
3,700 additional multinomial fits on 3,584 features — comparable to the entire scaled
study. It is not run on a layer subset either, because choosing which layers to
classify on would be exactly the search this programme avoids.

**Every layer is plotted.** No best layer is selected and no others hidden.

---

## What would make me abandon the current headline

Recorded so the memo cannot quietly preserve it:

- If D shows frozen probes transfer near zero in both directions, the "stable
  direction" reading dies and the correct claim shrinks to "independently decodable
  in each dataset."
- If A2 shows controller 5 does not beat controller 4, activations add nothing
  *for control* beyond what predicted Q values already supply, and the white-box
  story becomes a statement about estimation rather than about oversight.
- If C is unsupported for sample-size reasons, the within-game claim is not made at
  all rather than made weakly.
