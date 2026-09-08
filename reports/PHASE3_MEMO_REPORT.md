# Research memo — exploratory battery A–E

**Substrate:** frozen at tag `stage2-frozen` (`9d7a48d`). 1,971 checkpoints across 252
ALFWorld games, plus an independent 928-checkpoint single-execution cohort. Qwen2.5-7B-Instruct,
29 stored layers, linear probes only. Held-out whole games is the primary regime throughout.
Pre-registration committed at `f1e78cb`, before the feasibility census returned.

**Provenance:** nine runs, all exit 0, every script hash-verified against a value fixed before
execution. Raw outputs and sha256 sums in `PROBE_RAW_CAPTURE_REPORT.md`. Two cases where a
hardcoded output path would have destroyed a preserved result were caught and backed up
before overwriting.

---

## A — Controllers: does knowing recoverability improve control?

Nine controllers choosing by `argmax_a Q̂_a(s)`, differing only in what they know. Regret is
against true Q from the branches. The true-Q ceiling returns exactly 0.000 regret and 1.000
oracle agreement in both cohorts, so the machinery is sound.

**Strongest positive — modelling intervention value transforms control.** Mean regret falls
from 0.305 (best constant action) and 0.313 (risk-level lookup) to **0.137** once a controller
estimates Q_I at all. The `floor_intervene → R3_obs` contrast is **−0.167** pooled and
**−0.204** on the independent cohort, sign-consistent across all five splits in both.

**Strongest null — activations do not reliably improve control regret.** `R4_stacked →
R5_whitebox` is **−0.013** pooled and **−0.015** independent, **sign-inconsistent across
splits in both cohorts**. This despite R5 having better tail regret (p90 0.510 vs 0.588) and
better oracle agreement (0.654 vs 0.601).

**A second null worth stating: a constant action beats a learned controller.**
`always-intervene` (0.305) beats the risk-only controller (0.313) on the pooled cohort.

**Where scalar control actually fails.** R1 (risk level, all three actions) versus R4
(observables + predicted Q values, all three actions) — the only contrast where both
controllers have the same action set:

| risk | 0.00 | 0.20 | 0.40 | 0.60 | 0.80 | **1.00** |
|---|---|---|---|---|---|---|
| R1 risk-only | 0.003 | 0.048 | 0.066 | 0.004 | 0.056 | **0.509** |
| R4 full model | 0.079 | 0.017 | 0.115 | 0.035 | 0.028 | **0.180** |

R1 is *better* at risk 0.00, 0.40 and 0.60. The richer controller wins decisively in exactly
one place: **saturated failure probability, by 0.329 pooled and 0.239 independent.**

*Uncertainty:* per-split signs were computed for the pooled contrasts but **not** for this
per-level R1/R4 comparison — the script emits per-level means only. The effect is large and
replicates across cohorts, but it does not carry the sign-consistency check the other numbers do.

*Most plausible alternative:* the mid-risk bands hold 30–45 checkpoints total (≈10 test rows
per split), so R5's apparent degradation there — and much of the per-level structure — could
be noise on tiny cells.

**Licensed:** *estimating intervention value, from any source, is what converts a poor
controller into a good one; the marginal control value of estimating it from activations
rather than from 26 observable features is not established.*

**Not licensed:** *white-box probes improve oversight decisions.* They improve the estimate
and do not reliably improve the decision.

---

## B — Where does the activation advantage on Q_I live?

Held-out conflict-set predictions, 1,794 pooled rows / 856 independent.

**Strongest positive — the advantage is concentrated where the baseline fails.** By baseline
error quartile, mean improvement runs **−0.116 → +0.033 → +0.071 → +0.399** (29.8% → 75.1%
of rows improved). Nearly identical independently: −0.129 → +0.049 → +0.128 → +0.389.

**And concentrated in irrecoverable states.** Where τ ≤ 0 — intervening is *not* better —
improvement is **+0.245** against **+0.028** where τ > 0. Independent cohort: +0.210 vs +0.050.
By oracle action, quit **+0.185** vs intervene +0.029.

**By task type**, the appliance-interaction tasks dominate: pick_cool +0.221/+0.283,
pick_clean +0.212/+0.267, pick_heat +0.120/+0.083 — against look_at +0.001/+0.023,
pick_two_obj +0.014/+0.019, pick_and_place +0.015/**−0.128**.

**Strongest null — no trajectory-position effect.** Progress quartiles give
+0.080 / +0.096 / +0.127 / +0.080. Flat.

*Most plausible alternative, and it is serious:* **regression to the mean.** Rows are
stratified *by the baseline's own error*, so the top quartile is partly rows where the
baseline was unluckily wrong, and any second model would improve on them. The E1 → E4 gradient
cannot be cleanly separated from this. The τ-sign and task-type strata do not have this defect,
because neither is selected on model error.

**Licensed:** *the activation advantage on Q_I is concentrated in irrecoverable states and in
appliance-interaction task types, and is absent or negative in simple pick-and-place.*

**Not licensed:** *activations detect irrecoverability* as a mechanism, or any reading of the
baseline-error gradient as evidence rather than as partly regression to the mean.

---

## C — Same-game matched-state test

Pairs from the same game, both at maximal failure probability, similar progress, different Q_I,
cross-rollout only, matched without replacement. Criteria fixed before the census returned.

**Strongest positive — the pre-registered cell is positive in both cohorts.** At
`dprog ≤ 0.15`: observables 0.637, activations **0.756**, both 0.789 — **+0.120**. Independent
cohort **+0.094**. Every split above chance for every block.

**Strongest null — at the best-powered tolerance, both cohorts are negative.** At
`dprog ≤ 0.25`, pooled **−0.016** and independent **−0.058**. The independent cohort has
exactly **one** cell where all five splits scored, and it is that negative one.

**The trend.** Across tolerances, observables degrade monotonically as matching tightens
(0.725 → 0.637 → 0.605 pooled; 0.842 → 0.738 → 0.688 independent) and `act − obs` rises
monotonically in both cohorts and both settings. **Activations themselves rise only in the
pooled cohort** — flat in the independent one. That sub-claim is one-cohort.

*Uncertainty:* binomial se **0.12 to 0.26**. The effect is roughly one standard error.

*Most plausible alternative:* tight matching selects for pairs where two same-game states sit
close in progress but far apart in Q_I — i.e. pairs where position *cannot* explain the answer.
The 26 observables are largely position and task type, so their degradation is **partly built
into the selection**, not purely a finding.

**Licensed:** *in the pre-registered cell, activations rank within-game progress-matched states
better than observables in both cohorts, underpowered in both.*

**Not licensed:** *within the same task instance, the internal state tracks changing
recoverability.* The test cannot carry that at this power.

---

## D — Frozen cross-cohort transfer

Scaler, coefficients, intercept and alpha frozen on one execution and applied untouched to the
other. Wiring check passed at all three layers (`fit_reg` and the frozen fitter agree to four
decimals). Only `backfill → original` is interpretable: the reverse direction has 99.2% game
overlap leaving 8 unseen rows, below the guard.

**Strongest positive — Q_I transfers with no degradation to unseen games.** Evaluated only on
445 rows from 57 games the source never saw, `unseen − within-source` is
**+0.016 / +0.073 / +0.014** across layers. A probe frozen on one execution predicts a
*different* execution's unseen games at least as well as the source predicts its own held-out
games — on the target that is recoverability.

**Strongest null — τ does not transfer.** `unseen − within` is
**−0.142 / −0.090 / −0.116**, a ~30% relative drop at layer 24 (0.271 against 0.387), and
non-monotone in depth. Q_C is intermediate (−0.061 / −0.052 / +0.003).

**Game recognition, measured directly** as `frozen − unseen` (same probe, two subsets of the
same cohort): Q_C **+0.087–0.096**, Q_I **+0.037–0.059**, τ **+0.130–0.169**.

**Void:** `Q_continue` inside risk = 1 returns R² of −19.2 and −20.6. Q_C has sd 0.141 there
against Q_I's 0.996, so R²'s denominator collapses and any frozen offset produces an
arbitrarily large negative number. **Metric failure, not a transfer result.** It repeats in
both directions, which confirms the diagnosis.

*Most plausible alternative:* the unseen set is 445 rows from 57 specific games; if those games
are systematically easier the number is inflated. Nothing in the design rules this out.

*A withdrawn argument, recorded:* I claimed the ordering of degradation across targets
confirmed a seen-games mechanism. It does not — across three targets, game-determinedness,
observable R², within-source R² and degradation are perfectly collinear, a headroom effect
predicts the same ordering, and three targets admit six orderings, so a monotone match is one
draw from six.

**Licensed:** *a linear probe for intervention value, frozen entirely, transfers across
independently collected executions to unseen games without degradation.*

**Not licensed:** *the model carries a stable intervention-advantage direction.* τ specifically
fails to transfer.

---

## E — Layerwise emergence

All 29 layers, four curves, held-out games, both cohorts. Every layer reported.

**Strongest positive — the pre-registered layers understate every result.** Peaks sit at
layers 27/28/28/19 (pooled) and 21/21/23/19 (independent). Layer 24 falls short of peak in
**all eight curves**, by 0.012 to 0.059. The standard worry about fixing layers in advance is
that the choice flattered the result; **the data shows the opposite.**

**Strongest null — the registered ordinal hypothesis is refuted.** Q_I does *not* emerge later
than Q_C. On sustained onset — the first layer from which every deeper layer also clears
delta > 0.02 with a consistent sign — Q_I is at layer **4 vs Q_C's 10** (pooled) and **10 vs 11**
(independent).

**But the gap magnitude is not robust.** Under first-onset the independent cohort shows the
large separation (7 layers) and pooled almost none (1); under sustained onset, exactly the
reverse (6 and 1). **The sign is stable across both statistics and both cohorts; the magnitude
is stable across neither.**

**No curve is monotone** in either cohort. The clean L8 → L16 → L24 rise reported in the frozen
study was three points sampled from a non-monotone curve — a direct correction to §7.1.

**Peak location moves 5–7 layers between cohorts on every curve except one:** the conflict-set
curve peaks at layer **19 in both**. That same curve is also the least trustworthy in the
independent cohort — largest peak (+0.209), 11 of 29 layers sign-inconsistent, sustained onset
jumping from 1 to 15, lowest pass count (18/29). Three independent robustness measures agree,
and it carries the biggest headline number in the battery.

**A tension recorded rather than resolved:** τ is the *most* robust curve in E (first =
sustained = 2, 27/29 layers passing, in both cohorts) and the *only* target that fails to
transfer in D. These are different properties — within-dataset layerwise stability and
cross-execution transfer — so there is no contradiction, but both belong in any writeup or a
reader will assume one is an error.

**Licensed:** *the predicted depth ordering is absent; intervention value is decodable no later
than continuation value.*

**Not licensed:** *the expert's value emerges earlier than the actor's.* Supported at a
meaningful size in one cohort only.

---

## Ranking

| finding | surprise | evidence | novelty vs CINC | oversight value | headline |
|---|---|---|---|---|---|
| Q_I transfers frozen to unseen games; τ does not | high | **strong** | high | high | strong |
| Estimating Q_I helps control; doing it from activations does not | high | **strong** | medium | **highest** | strong |
| Scalar control fails only at saturated risk, and knife-edge there | medium | strong | high | high | strong |
| Advantage concentrated in irrecoverable states | medium | medium | high | medium | medium |
| Pre-registered layers understated every result | low | strong | low | low | weak |
| Depth ordering absent | medium | medium | medium | low | weak |
| Within-game matched states | high | **weak** | high | high | strong-but-unsupported |

---

## The strongest coherent stories

### 1. The white-box direction is for intervention *value*, not intervention *advantage*

A probe for Q_I, frozen completely, transfers across independently collected executions to
unseen games with **no degradation** (+0.016/+0.073/+0.014). A probe for τ, on the same rows
by the same procedure, **loses ~30%**. Inside the conflict set — where Q_C is pinned at sd 0.141
against Q_I's 0.996 — activations add to Q_I prediction with Q_C already supplied, so this is
not a better picture of continuation state.

The component is stable; the difference is not. τ's instability looks inherited from combining
two estimates rather than from being represented directly. This sharpens the frozen study's
claim rather than repeating it, and it is the best-evidenced finding in the battery.

### 2. Recoverability estimation is what buys control — the white-box part does not, yet

Mean regret: 0.305 for the best constant action, 0.313 for a risk-level controller, **0.137**
once anything estimates Q_I, 0.123 with activations. The step that matters is having an
intervention-value estimate at all. Adding activations moves regret by 0.013 and **flips sign
across splits in both cohorts**.

This is a negative result about white-box oversight specifically, and it is well-evidenced.
It reframes the programme's own headline: the residual stream carries genuine
intervention-specific information, and that information does not yet change enough decisions
to matter. Anyone proposing white-box runtime oversight should have to clear this bar.

### 3. Scalar control fails in one narrow place, and fails there catastrophically

A risk-only controller matches or beats a full observable model at four of six risk levels.
At saturated failure probability it loses by **0.329**. And its policy there is decided by
whether mean Q_I sits above or below zero — **+0.009 pooled, −0.080 independent** — so the same
rule intervenes 57% of the time in one sample and 8% in the other, flipping the action for 62%
of the data.

That is *Calibration Is Not Control*'s target-error argument given a precise empirical location
and a demonstration that the failure is knife-edge rather than gradual.

---

## Limitations, including my own

**Three reporting traps in my own instrumentation**, all found during the battery rather than
by design: the sign column marks *consistency*, not success, so a reliably **negative** effect
prints "same"; `sd` prints 0.000 when only one split scored, reading as perfect stability when
it means the opposite; and `obs+activations` is identical to `activations` in every
matched-pair cell, because 26 columns cannot flip a ranking among 3,584 at these sample sizes.

**A tendency, stated because it recurred.** Twice in this battery I read a trend off too few
points and had to withdraw it — the matched-pair "mechanistic signature" across three
tolerances, and the transfer degradation ordering across three targets. Both times the shape
was real; both times I treated shape as confirmation without asking what else produces that
shape. Both withdrawals are in the record above.

**One pre-registration ambiguity.** C's rule said "≥ 50 opposite-oracle pairs," which reads as
either candidates (50 exactly, selecting `dprog ≤ 0.15`) or post-matching (46, which would have
selected 0.25). I applied it to candidates because that is what the census measured, and ran
the alternative as a robustness check rather than choosing between them after the fact.

**Not run, and why.** Oracle-action classification across 29 layers was dropped from E on cost
(~3,700 additional multinomial fits over 3,584 features, comparable to the whole scaled study),
and deliberately not run on a subset, because choosing which layers to classify would be exactly
the search this programme avoids.
