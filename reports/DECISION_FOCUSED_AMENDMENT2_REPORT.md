# Amendment 2 — the primary comparison is void, and why

Amends `reports/DECISION_FOCUSED_PREREGISTRATION_REPORT.md` (`534d153`) and
`reports/DECISION_FOCUSED_AMENDMENT_REPORT.md` (`7851b64`). **Neither is edited.** This
amendment reverses a determination made in the first one, which is recorded in §3 rather than
quietly dropped.

Written while still blind to every outcome quantity in both cohorts. The runs completed
(exit 0 each) and their outputs are on disk unread.

---

## 1. Determination: the primary comparison is VOID, both cohorts

Amendment 1 §2 defined void as **failure to reach** — trajectory minimum not plateaued, or the
objective still falling substantially at `T` — and explicitly excluded oscillation around a
resolved optimum. Seed-0 traces of the penalised training objective at `t = 1 / T/2 / T`:

| cohort | layer | arm | lambda | t=1 | T/2 | T | T/2 -> T |
|---|---|---|---|---|---|---|---|
| all | 8 | obs | 0.01 | 1.4941 | 0.5653 | 0.5631 | −0.4% |
| all | 8 | **obs+act** | 100 | 1.4941 | 1.4423 | 1.5349 | **+6.4% rising** |
| all | 16 | obs | 0.01 | 1.4941 | 0.5653 | 0.5631 | −0.4% |
| all | 16 | **obs+act** | 1 | 1.4941 | 0.5776 | 0.8508 | **+47.3% rising** |
| all | **24** | obs | 0.01 | 1.4941 | 0.5653 | 0.5631 | −0.4% |
| all | **24** | **obs+act** | 10 | 1.4941 | 0.9155 | 0.7399 | **−19.2% still falling** |
| original | 8 | obs | 0.1 | 1.4539 | 0.6018 | 0.6018 | 0.0% |
| original | 8 | **obs+act** | 100 | 1.4539 | 1.1505 | 1.0218 | **−11.2% still falling** |
| original | 16 | obs | 0.1 | 1.4539 | 0.6018 | 0.6018 | 0.0% |
| original | 16 | **obs+act** | 1 | 1.4539 | 0.3901 | 0.4871 | **+24.9% rising** |
| original | **24** | obs | 0.1 | 1.4539 | 0.6018 | 0.6018 | 0.0% |
| original | **24** | **obs+act** | 100 | 1.4539 | 1.3563 | 1.4979 | **+10.4% rising** |

**Six wide-arm fits, six failures**, in both directions, at boundary and interior lambdas.
**Six narrow-arm fits, zero failures** — −0.4% in `all`, exactly 0.0% in `original`.

`cohort=all` layer 24 is the decisive cell: a monotone 19.2% **fall** across the final 250
iterations is the "still falling at `T`" branch verbatim. It is not a plateau under any reading.
`cohort=original` layer 24 fails differently — rising 10.4% and ending **above** its `t=1`
value — at a lambda four orders of magnitude away. Same layer, same script, two cohorts, two
distinct failure modes.

**This leg rests on real data under a rule committed at `7851b64` before these traces existed.
It does not depend on any synthetic work.**

---

## 2. What the experiment can and cannot now report

The regret numbers exist and are preserved. They cannot be read as a test of the hypothesis,
because the two arms were not fit to comparable quality: the `d = 26` arm converged everywhere
and the `d = 3610` arm converged nowhere. Any difference between them confounds representation
with optimiser adequacy, and **the confound runs against the hypothesis under test** — the
handicap falls on the arm whose benefit is the question.

Under `534d153`'s null rule this is **not** a null. The rule presupposes both arms were fit. The
honest description is *the comparison did not run*, which is a different result and triggers
nothing.

---

## 3. Reversal: Amendment 1 §3 was wrong

Amendment 1 §3 determined that layer 8 was **NOT VOID**, reasoning that at `lambda = 100` the
true minimiser is approximately zero, so a near-zero solution meant the arm was fitting rather
than failing. **That reading is withdrawn.** It rested on behaviour I later found holds only for
uncorrelated features. With correlated features — which real activations certainly are — the
excursion does not vanish at high lambda. Layer 8 is void with the rest.

The executing session's original flag was closer to right than my correction to it. The
correction was right that subgradient methods are not descent methods, and wrong to conclude the
layer-8 fit was therefore sound. Both stand together.

**The trigger invoked is not the trigger written.** Amendment 1 §4 placed the burden on
demonstrating non-convergence at the **low end of the lambda grid**, on the theory that a
corrupted lambda *selection* was the only route to invalidity. That was too narrow. What
actually happened is failure at the selected lambda across the whole grid, including interior
values, in both directions. I did not anticipate that branch. Disclosed rather than
retrofitted.

---

## 4. Mechanism: partly corroborated, partly contested, one error of mine

Synthetic only, `fit_spo` imported from `4e2d124` rather than retyped. My instrumented loop
reproduces `fit_spo` **exactly** (identical traced values to 1e-12 at three lambdas); the
executing session independently confirmed bit-for-bit equality of `W`, `b` and the trace.

**Corroborated — and this is the load-bearing part.** With rank-50 correlated features at
`d = 3610`, the excursion **does not vanish at high lambda**: 11.3 / 12.1 / 12.7% at
`lambda = 0.01 / 1 / 100`, against ~0.4% for iid features at the same `d` and lambda. The
executing session reproduced this independently (7.9 / 2.0 / 9.6%): the high-lambda survival
holds on a different machine. Real data agrees — `cohort=original` L8 and L24 both sit at
`lambda = 100` with 11.2% and 10.4% excursions, inside the predicted band, where the iid version
would have predicted ~0.4%.

**Contested, and my error.** I reported the dimension sweep as a *percentage* rise,
`(final − min)/min`, and led with 118.9% at `d = 3610`. That denominator is 0.0288. **This is a
ratio against a near-zero denominator — the exact failure mode recorded in the closure report's
limitations, repeated here by me.** Where `d > n` the model interpolates and the objective
approaches zero, so the percentage explodes for a reason unrelated to instability.

In **absolute** objective units the trend is orderly and monotone in `d` at every lambda:

| d | lam=0.01 | lam=1 | lam=100 |
|---|---|---|---|
| 26 | 0.0001 | 0.0001 | 0.0000 |
| 200 | 0.0014 | 0.0017 | 0.0001 |
| 1000 | 0.0039 | 0.0048 | 0.0011 |
| 3610 | 0.0342 | 0.0186 | 0.0060 |

The executing session did not reproduce my percentage table (4.6% against my 118.9% at
`d = 3610`) and proposed my generative scale as the culprit. **I tested that and it is not the
cause** — rerunning under their noise specification gives 120.0% against my 118.9%. The
divergence is therefore **unexplained**, and the percentage-scale d-trend is recorded as
**contested and unresolved**. The absolute-units trend is what I claim; the percentage headline
is withdrawn.

---

## 5. The repair, validated as named

Amendment 1 §6 named it before any of this: same loss, same `lr`, same grid, same splits, `T`
raised **and** back-half iterate averaging, applied identically to both arms. Tested on
correlated `d = 3610`:

| lambda | T | final regret | averaged regret |
|---|---|---|---|
| 0.01 | 500 | 0.0364 | 0.0198 |
| 0.01 | 2000 | 0.0285 | **0.0130** |
| 1 | 500 | 0.0916 | 0.0665 |
| 1 | 2000 | 0.0832 | **0.0650** |
| 100 | 500 | 0.1321 | 0.0800 |
| 100 | 2000 | 0.1462 | **0.0799** |

Averaging helps at every lambda and every `T`. **Longer `T` alone does not**: at `lambda = 100`
the final iterate gets *worse* from 0.1321 to 0.1462 as `T` rises, confirming the prediction that
a steady-state oscillation is not cured by more iterations. The executing session reproduced the
averaging benefit independently (28–57% regret reduction).

**Added to the spec, and it matters:** averaging must be used **inside the inner-CV fits as well
as the refit**. Otherwise lambda is selected on the wandering estimator and deployed on the
averaged one.

**Measured cost.** Both cohorts, `T = 500`: **28 min 46 s** on 8 cores. Cost is linear in `T`
(averaging is a running mean and free), so `T = 2000` is approximately **2 hours**.

---

## 6. Not authorised by this document

No repair run is authorised here. No outcome quantity has been inspected. The user decides
whether a re-run happens; Amendment 1 §6 committed to that and it holds. Still prohibited
regardless: a second loss, a second weighting, a nonlinear probe, a new layer, new data.
