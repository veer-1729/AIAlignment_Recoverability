# Pre-registration — decision-focused linear controller (SPO+)

Committed **before** the experiment script was written and before any result exists. Frozen
predecessors: `stage2-frozen` (`9d7a48d`), `phase3-battery` (`84f0bd8`), `closure-battery`
(`280c80a`). No prior artifact is modified, renamed or deleted.

This is the **last new experiment before synthesis**. The decision rules below include the
condition under which the white-box control line stops entirely.

---

## The question

Closure established that the activation signal is decision-relevant but that its regret effects
cancel: Δregret **−0.0898** [−0.1437, −0.0432] where τ ≤ 0 and **+0.0593** [+0.0188, +0.1068]
where τ > 0, non-overlapping. An MSE-trained probe spends capacity uniformly and is provably
damaging in one regime while provably helping in the other.

> Can the same linear representation yield lower control regret when trained with an objective
> that cares about the induced decision rather than about squared prediction error?

This tests **loss/decision alignment**, not representation capacity.

**The closure report's own suggestion is withdrawn.** It proposed weighting Ridge by
`1 / true_action_margin`. That is misaligned with regret — a small-margin mistake can cost
almost nothing while a large-margin mistake is expensive — and it would downweight precisely the
highest-margin decile where closure found 93.9% of the activation improvement. It is not
implemented.

---

## The objective: SPO+, derived for this exact problem

**Setup.** Three actions with true values `Q = (Q_C, Q_I, 0)`; quit is analytic and exactly 0.
The controller chooses `â = argmax_a Q̂_a`. Regret is `max_a Q_a − Q_â`.

**Elmachtoub & Grigas define SPO+ for minimisation** over a feasible region `S`:

```
ℓ_SPO+(ĉ, c) = max_{w∈S} {(c − 2ĉ)ᵀw} + 2ĉᵀw*(c) − z*(c)
```

with `w*(c) ∈ argmin_{w∈S} cᵀw` and `z*(c) = cᵀw*(c)`.

**Our problem maximises**, so substitute `c = −Q`, `ĉ = −Q̂`. Here `S` is the three unit vectors,
so `w*` selects an action and `z*(c) = −max_a Q_a`. Substituting and simplifying:

```
ℓ_SPO+(Q̂, Q) = max_a {2Q̂_a − Q_a}  +  max_a Q_a  −  2Q̂_{a*}       where a* = argmax_a Q_a
```

**Subgradient** with respect to the prediction vector, writing `ã = argmax_a {2Q̂_a − Q_a}`:

```
∂ℓ/∂Q̂ = 2(e_ã − e_{a*})
```

Convex in `Q̂` (a max of affine functions plus an affine term), hence convex in the linear
parameters. The `quit` component is a known constant, so its subgradient component is simply
unused — the hypothesis class fixes `Q̂_quit ≡ 0` and only `Q̂_C, Q̂_I` are parameterised.

**Is SPO+ appropriate here?** Yes, and the checks are: the feasible set is finite, fixed across
samples, and identical to the controller's actual action set; the objective is linear in the
value vector; the induced `w*` is exactly the existing `argmax` controller; and `ℓ_SPO` under
this substitution *is* our regret, so the surrogate targets the reported metric rather than a
proxy for it. Fixing one component to its known value is a constrained hypothesis class, not a
violation. **No substitute loss is used, and none will be introduced silently.**

**Verified before writing this document**, on synthetic data only:

- brute-force enumeration in the minimisation form agrees with the simplified maximisation form
  on 20,000 random cases, zero mismatches;
- perfect prediction gives exactly 0;
- **`ℓ_SPO+ ≥ regret` with 0 violations in 200,000 draws**, at scales 0.01, 1 and 100;
- the analytic subgradient matches a central finite difference.

The script repeats all of these as gates and **aborts before touching research data if any fail**.

---

## Model, regularisation, optimiser

**Model.** `Q̂_C = xᵀθ_C + b_C`, `Q̂_I = xᵀθ_I + b_I`, `Q̂_quit ≡ 0`. Linear, as required.
Features standardised with the same `StandardScaler` used everywhere in this programme.

**Objective.** `(1/n) Σ_i ℓ_SPO+(Q̂(x_i), Q_i) + (λ/2)‖Θ‖²_F`, intercepts unregularised.

**Regularisation rule, fixed now and identical for both feature sets.** λ is selected by
`GroupKFold(5)` **inside the training split only**, minimising mean validation SPO+ loss — the
same fold structure and the same train-only discipline as `probe_study.select_alpha`, with the
decision loss as the criterion, which is the point of the experiment. Grid:

```
λ ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2}
```

Chosen to cover the existing Ridge grid mapped into mean form: `Ridge` minimises
`‖y−Xw‖² + α‖w‖²`, i.e. `(1/n)‖·‖² + (α/n)‖w‖²`, so `ALPHAS = (1e0 … 1e6)` at n ≈ 1,380
corresponds to `λ_eff ≈ 7e-4 … 7e2`. Seven values, mirroring the seven in `ALPHAS`.
**No other hyperparameter is searched.**

**Optimiser.** Full-batch Adam, `lr = 0.01`, `β = (0.9, 0.999)`, **T = 500** iterations, final
iterate, identical for every model. The training objective at `t = 0, T/2, T` is printed so
convergence is auditable rather than assumed, and a synthetic recovery test — data generated
with `Q` exactly linear in `x` — must drive both SPO+ loss and regret to approximately zero
before any research data is touched.

---

## Design

**Both arms use the identical objective, optimiser and λ rule**, so representation and loss are
never confounded:

- **A** observable features only (the 26-D block)
- **B** observable features + activations

**Layer 24 is primary**, fixed before the all-layer search. Layers 8 and 16 are reported as
pre-registered secondary diagnostics; **success may not depend on choosing among them.**

Both cohorts: pooled 1,971 and independent 928. Held-out whole games, the five existing split
seeds, unchanged.

**Primary endpoint:** held-out-game mean regret, specifically

```
regret(obs + activations, decision-focused) − regret(obs only, decision-focused)
```

Prediction R² is secondary and is explicitly not the objective.

**Reported on identical test rows:** the existing MSE observable controller, the existing MSE
observable+activation controller, decision-focused A, decision-focused B, and the existing
TF-IDF controller as contextual reference — **unmodified and untuned.** Metrics: mean regret,
game-clustered bootstrap CI, per-split values and signs, p90 regret, oracle agreement, action
distribution. Plus the τ ≤ 0 / τ > 0 decomposition and closure's four transition categories with
their contributions to the total regret difference.

---

## Decision rules, fixed now

**Strong success** — all four must hold:
1. decision-focused B has lower mean regret than decision-focused A;
2. the direction is consistent across held-out-game splits **and** across both cohorts, not
   driven by one split;
3. the τ > 0 damage materially shrinks or disappears;
4. the gain does not come from catastrophic degradation of another subgroup or of p90 regret.

Then: *intervention-value information in activations was control-relevant, but ordinary
prediction loss used it incorrectly; aligning the objective with the decision converts some of
it into better control.*

**Null** — if decision-focused training gives no robust activation-over-observable regret
benefit, **the white-box control line stops.** No other weighting function, no other decision
loss, no nonlinear probe, no other layer, no RL, no steering, no subgroup-specific model. The
result stands as: *decodable intervention-value information does not automatically translate
into improved runtime oversight, even after one principled attempt to align training with
downstream regret.*

**Observable-only improvement** — if the objective helps both arms but activations add nothing
reliable: *the bottleneck was objective alignment, not white-box information.*

**Activation-specific improvement** — the strongest possible outcome, and therefore audited
hardest before any headline: leakage, regularisation imbalance between arms, and split
instability all checked explicitly, and the claim is not made if any is unresolved.

---

## What is not permitted in this experiment

No new environment trajectories or branch outcomes. No change to the frozen datasets, values,
utilities, splits or activation extraction. No post-hoc layer selection. No test-set-based
selection of anything. No second loss if the first disappoints.
