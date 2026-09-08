# Amendment 1 to the decision-focused (SPO+) pre-registration

Amends `reports/DECISION_FOCUSED_PREREGISTRATION_REPORT.md` (`534d153`). The original document
is **not edited**; this amendment is a separate object, as required by the rule that an executed
artifact and a corrected artifact are different objects.

The executed script is unchanged and remains `scripts/decision_focused.py` at `4e2d124`,
sha256 `0fa3d34b9c33f176d31f0ed345ecd17cd9f1824bfb99999d59c99c463858c0c9`. Nothing in this
amendment alters the running experiment.

---

## 1. Disclosure — what had been seen when this was written

**Seen:** the layer-8, `cohort=all` seed-0 convergence traces and selected lambdas, for both
arms. Those are reproduced in §3.

**Not seen:** any regret value, the primary endpoint, the tau decomposition, the transition
categories, p90, oracle agreement, or the sign of any of them, at any layer or cohort. A
blinding instruction is in force with the executing session, which has confirmed it and is
withholding all outcome quantities.

**The honest caveat, stated rather than buried.** The layer-8 traces are not outcome data, but
they are not perfectly uninformative about the layer-8 outcome either: an arm shrunk to a
near-constant controller will almost certainly show worse regret at that layer. Layer 8 is a
pre-registered *secondary diagnostic*; the primary endpoint is layer 24 and remains unseen.
This amendment is therefore written with partial knowledge of a secondary layer's likely
direction and no knowledge of the primary. That is a weaker freeze than the original and it
is recorded as such.

---

## 2. The pre-registered convergence rule was written for the wrong optimiser class

The original document asked that the objective at `T/2` and `T` be close, and I instructed the
executing session to treat a violation as voiding the comparison.

**That rule is wrong as stated.** SPO+ is a maximum of affine functions and therefore piecewise
linear. Its subgradient method is **not a descent method**. Under a constant step size the
iterates converge to a neighbourhood of the optimum whose radius scales with the step and then
oscillate inside it; a non-monotone final iterate is the expected behaviour, not evidence of
failure. I imported an intuition from smooth gradient descent into a nonsmooth problem.

**Replacement rule, fixed now.**

- **VOID — failure to reach.** The trajectory minimum has not plateaued, or the objective is
  still falling substantially at `T`. The optimiser had not arrived anywhere when it stopped.
- **NOT VOID — oscillation around a resolved optimum.** The trajectory minimum has plateaued
  and the bounce amplitude is small relative to the between-arm gap being interpreted.

Only the first voids a comparison. The second is reported as a limitation with the amplitude
quoted, and the comparison stands.

---

## 3. Layer 8 under the replacement rule: NOT VOID

```
LAYER 8, cohort=all
  SPO_obs      seed0 lambda 0.01   objective t=1/250/500: 1.4941  0.5653  0.5631
  SPO_obs+act  seed0 lambda 100    objective t=1/250/500: 1.4941  1.4423  1.5349
```

The traced quantity is the **penalised** objective (`decision_focused.py:114`). `W` initialises
at zeros, so the identical `1.4941` at `t=1` in both arms is the pure SPO+ loss at zero
prediction — a useful consistency check that the two arms start from the same point.

At `lambda = 100` with 7,220 coordinates the penalty is `50*||W||^2`, so the total of `1.4423`
bounds `||W||^2 <= 0.029`, and the small drop from the zero-weight value implies it is nearer
`0.001`. RMS weight per coordinate is therefore between about `4e-4` and `2e-3`.

**A correction to arithmetic I asserted, and which the executing session then amplified.** I
compared that to an Adam step of `lr = 0.01` and called the solution "smaller than one step";
the executing session tightened it to "more than an order of magnitude below one step". **Both
are wrong.** Adam is self-damping under alternating-sign gradients: the steady-state step is
approximately `lr*(1-b1)/(1+b1) = 0.01 * 0.1/1.9 = 5.3e-4`, not `0.01`. The RMS weight is thus
of the *same order* as one steady-state step, not far below it.

The conclusion survives the correction, with a smaller margin than either of us claimed: the
oscillation amplitude is comparable to the magnitude of the solution itself. Near-zero `W` is
approximately the true `lambda = 100` minimiser, so **the arm is fitting that lambda roughly
correctly rather than failing to train.** With unregularised intercepts, what it delivers is a
near-constant-action controller.

Two consequences that follow from the corrected number and that I would otherwise have got
wrong:
- **Longer `T` alone cannot cure this.** The oscillation is a steady-state property of a
  constant step, not a symptom of stopping early. Only iterate averaging removes it.
- **"Constant-action controller" slightly overstates the collapse.** On standardised features
  with `d = 3610`, `||x||` is of order 60, so even an RMS weight of `2e-3` produces prediction
  variation of order `+/-0.1` across rows, with oscillation inside that. That is
  decision-relevant, so the final-versus-averaged iterate distinction can move test regret.

---

## 4. Where the real threat is, and who carries the burden of proof

The trace at the *selected* lambda cannot tell us whether the inner cross-validation was
choosing among sound fits. That is the only mechanism by which the lambda selection could have
been corrupted, and it is the question that matters.

**Burden of proof, fixed now.** The null rule in `534d153` **fires unless optimiser failure is
positively demonstrated at the LOW end of the lambda grid.** Specifically:

- If the low-lambda fits plateaued and CV still selected a boundary lambda, **that is
  cross-validation working correctly** — activations overfit and shrinkage validates better.
  That is a null, in full, with no repair and no re-run.
- Only non-plateaued low-lambda fits can void the lambda selection.

This is written in this direction deliberately. "The optimiser failed" is a convenient escape
from an unwelcome null, and the burden is therefore placed on the escape, not on the null.

**The high-end boundary selection is the safe direction.** Extending the grid upward can only
shrink obs+act further toward the constant controller and cannot manufacture an activation
benefit. A high-boundary selection cannot flatter the hypothesis under test.

---

## 5. A gate weakness, disclosed

The optimiser-recovery gate that I singled out as the one protecting against exactly this
failure tested `d = 20`, `T = 1500`, `lambda = 1e-6` — and passed cleanly. The hard regime is
`d = 3610`, `T = 500`, `lambda` up to `1e2`. **The gate did not cover the regime it was meant to
protect.** It established that the loss, subgradient and optimiser are jointly capable of
recovering a realisable solution; it did not establish that they do so at the dimensionality and
regularisation actually used.

---

## 6. The single repair, named now, executed only on the user's instruction

If and only if §4's condition is met — non-plateaued fits at the low end of the grid — the
repair is:

> Identical loss, identical `lr`, identical lambda grid, identical splits, identical selection
> rule. Two changes, applied **identically to both arms**: `T` raised, and the **iterate averaged
> over the back half of the trajectory** (Polyak-Ruppert) in place of the final iterate.

Averaging is the standard remedy with convergence guarantees for nonsmooth subgradient methods,
where the final iterate has none. It is named here, before any regret number has been seen, so
that it cannot be tuned to the outcome. It is **not** a second loss, a second weighting, a
nonlinear probe, or a new layer, all of which remain prohibited.

**It will not be executed on my own initiative.** The user's instruction was that this is the
last experiment; a re-run is theirs to authorise, and the decision goes to them with the
diagnostic in hand rather than being taken here.

---

## 7. Limitations recorded now, regardless of outcome

1. **The SPO+ arms are optimiser-limited in a way the MSE comparators are not.** Ridge is solved
   in closed form; SPO+ is solved by 500 constant-step subgradient iterations. Any SPO+ shortfall
   against an MSE controller is therefore **partly confounded with optimiser adequacy** and
   cannot be read as a pure statement about the objective. This applies to the whole
   SPO-versus-MSE comparison and is not specific to the layer-8 anomaly.
2. **A coverage claim in the original document is not quite true.** It said the grid
   `1e-4 ... 1e2` covers `ALPHAS = (1e0 ... 1e6)` mapped to mean form, which at `n ~ 1380` is
   `7.2e-4 ... 7.2e2`. The bottom is covered; **the top is not** — `1e2 < 7.2e2`. Harmless in
   direction, per §4, but the claim was overstated.
3. **Traces exist for seed 0 only.** `trace=True` is passed on the refit for the reported seed
   alone, so seeds 1-4 produce no trajectory at any layer or cohort. Convergence is audited on
   one of five splits per arm. The script is not being changed to fix this mid-run.
4. **This amendment was written after seeing a secondary layer's traces**, per §1.

---

## 8. Unchanged

The primary endpoint, the layer-24 primacy, both cohorts, the five split seeds, the four
strong-success conditions, and the null rule of `534d153` are unchanged. No outcome quantity has
been inspected. No new experiment is authorised by this document.
