# Closure report — Q1, Q2, Q3

**Substrate:** unchanged. 1,971 checkpoints / 252 games, plus the independent 928-checkpoint
single-execution cohort. Existing trajectories, branches and activations only; no new
environment interaction. Held-out whole games primary throughout.

**Provenance.** Pre-registration at `b21f8f6`, pushed and externally verified. Every closure
script hash-verified by the executing session against a value published before execution; all
runs exit 0; Phase 3 artifacts preserved intact.

**How the two registrations should be described**, stated precisely because they differ:

- **Closure battery.** The pre-registration was published before *I* inspected any output, and
  that is the sense in which it is externally frozen. It did **not** demonstrably precede
  inspection by the executing session, which had already run and seen Q2 when the push landed.
  The supportable wording is *frozen before the analyst who reads it inspected the output*.
- **Phase 3 battery.** Committed locally before execution and **subsequently** published. The
  executed scripts were independently hash-verified before each run — a genuinely strong chain,
  and a different guarantee from a timestamped registration. **The later push does not prove the
  historical ordering** and is not offered as doing so. One Phase 3 claim was in fact withdrawn
  on inspection: its document said it was written before the feasibility census returned, and the
  commit postdates the census logs by 27–38 seconds.

---

## The three questions

| | result | strongest licensed claim | withdrawn or narrowed | changes the story? |
|---|---|---|---|---|
| **Q1** Q_I stable direction, τ not? | **No such distinction.** On the same 445 unseen rows at layer 24, τ scores 0.271 frozen / 0.245 size-matched / 0.243 destination-refit; Q_C 0.569 / 0.534 / 0.558. Cohort of origin does almost no work for any target. | Frozen linear probes transfer across independently collected executions to unseen games about as well as probes fitted **on** those games. τ is harder on these rows than Q_I is, and that is a property of the rows. | **"The τ direction does not transfer" is withdrawn.** So is "a stable Q_I direction versus an unstable τ direction." Phase 3 compared frozen performance against the source's own held-out rows — a different row set — which cannot separate a failing probe from a hard destination. | **Yes — kills story B.** |
| **Q2** why doesn't better Q_I help control? | **It does help — and hurts equally.** τ ≤ 0: Δregret **−0.0898** [−0.1437, −0.0432]. τ > 0: **+0.0593** [+0.0188, +0.1068]. Both exclude zero and **exclude each other**. | The activation signal **is decision-relevant**. The pooled null is two established, opposite effects cancelling, not an absence of effect. 73.6% of the Q_I SSE reduction lands where the two controllers choose differently; 105.5% on states the observable controller gets wrong. | **My own hypothesis is refuted.** "The extra information sits where it cannot change a decision" predicted a large SSE share on agreeing states (it is 26.4%) and flat regret (decile 9 moves −0.472). Phase 3's "activations do not improve control" is narrowed to "their gains and damages cancel." | **Yes — kills story A's central clause.** |
| **Q3** does it survive stronger black-box baselines? | **Split by cohort.** TF-IDF over the raw prompt gains **+0.123** (consistent) on pooled Q_I against activations' **+0.120**, and is the best controller there (regret 0.102 vs 0.125). On the independent cohort it gains **+0.025 FLIPS** against activations' **+0.103** consistent, and its control is worse than the linear baseline. | Q_I is **linearly accessible in the actor's representation beyond the original 26-feature *linear* baseline**. Activations are the only representation sign-consistent in all four prediction blocks across both cohorts, and lead in the conflict set in both (+0.143 / +0.168 vs TF-IDF's +0.124 / +0.093). | **"The hidden state contains information absent from observables" is withdrawn**, exactly as pre-registered. A bag of words over the prompt the model itself received matches the activation probe on the primary dataset. | **Yes — narrows the white-box framing.** |

**A clean null worth its own line.** The fixed random forest on the same 26 features flips in
three of four prediction blocks and has the **worst control regret of all five models in both
cohorts** (0.153 vs the linear model's 0.137; 0.165 vs 0.141). "The 26-feature baseline was
merely underfit" is refuted — the linear model was not the weak link, and nonlinearity on those
features actively hurts decisions.

---

## Recommendation: **D**

None of A, B or C survives contact with all three results.

- **A ("Decodability Is Not Control")** fails on its central clause. Q2 shows the extra signal is
  *not* largely decision-irrelevant; it is decision-relevant in both directions at once.
- **B ("Stable Intervention-Value Readout")** fails outright. Q1 shows no stable-Q_I /
  unstable-τ distinction, because cohort of origin does essentially nothing for either.
- **C ("Without a White-Box Advantage")** overstates. The prefix baseline erases the advantage
  on the pooled cohort only, fails to replicate on the independent one, and loses in the
  conflict set in both.

**The claim the evidence forces:**

> **Estimating intervention value is what buys control. Where you read it from matters far less
> than claimed — and reading it *better* currently helps and hurts in equal measure.**

Four supporting legs, each replicated across both cohorts:

1. **Estimating Q_I at all is the step that matters.** Mean regret falls from 0.305 (best
   constant action) and 0.313 (risk-level lookup) to 0.137. The contrast is −0.167 pooled and
   −0.204 independent, sign-consistent across every split in both.
2. **The source of the estimate is not decisive.** A TF-IDF bag of words over the prompt matches
   the activation probe on the primary dataset and beats it on control there. Activations'
   distinguishing property is **stability**, not accuracy — the only representation that never
   flips sign, in either cohort, on either subset.
3. **Better estimation does not yet mean better control, because the errors are asymmetric.**
   Activations reduce regret where τ ≤ 0 and increase it where τ > 0, with non-overlapping
   confidence intervals. They also make Q_I *worse* on states the observable controller already
   gets right (−5.5% SSE share — a sign change).
4. **Cohort of origin does almost no work.** Frozen, size-matched and destination-refit probes
   land within 0.035 of each other on identical rows. That is a real and slightly deflationary
   finding about probe transfer: it transfers because the task is stable, not because a
   particular direction was found.

**And the paper's own argument gets a sharper location, unchanged by this battery:** a risk-only
controller matches or beats a full observable model at four of six risk levels and loses by
0.329 at saturated risk — where its policy is decided by whether mean Q_I sits above or below
zero (+0.009 pooled, −0.080 independent), flipping the action for 62% of the data.

---

## Limitations

**Two withdrawals inside the closure battery itself**, both from external checking rather than
my own review: "the frozen probe beats a locally refitted one" did not survive size matching
(the comparator trained on 483 rows against frozen's 1,043, and two-thirds to three-quarters of
the gap was that); and part (d) of Q1 was an **algebraic identity, not a test** — ridge is
linear, so with a shared alpha `fit(y1−y2) = fit(y1) − fit(y2)` and the difference is zero by
construction.

**One dissenting layer.** τ's size-matched advantage is +0.090 at L16 against +0.008 and +0.002
at L8 and L24. The Q1 reading rests on L8 and L24 agreeing while L16 dissents, and is stated
that way rather than averaged.

**"Harder rows" is measured, not explained.** Every probe scoring 0.24–0.27 on the 445 unseen
rows is consistent with those rows being intrinsically harder *and* with all four probes sharing
a limitation those rows expose. This design cannot separate them.

**Recalibration repairs nothing.** `recal_R2` sits *below* frozen for τ at every layer, so the
"less stable calibration" branch of the Q1 decision rule is not satisfied either. τ is the most
slope-miscalibrated target (0.70–0.79 against Q_I's 0.99–1.04) and correcting the slope recovers
no R².

**Ratios against near-zero denominators** recur: the independent cohort's L8 SSE shares reach
±2.5 on a denominator of 20.2, the same failure mode as the void Q_C risk=1 R² cells. Not quoted
as percentages anywhere.

**Cosine correction was right and immaterial.** Comparing standardized coefficients from
independently fitted scalers was wrong in principle; in raw coordinates the values agree to
within 0.005 almost everywhere. Which does usefully establish that the middling 0.39–0.53 values
are real rather than a standardization artifact.

---

## The conditional question

> *If activations contain substantial prediction improvement specifically on low-margin /
> baseline-wrong states but the ordinary MSE-trained probe still fails to reduce regret, is
> there enough evidence to justify exactly one subsequent decision-focused linear-probe
> experiment?*

**YES — but not on the premise as stated, and the distinction matters.**

The premise is only half met. The improvement **is** concentrated on baseline-wrong states
(105.5% of SSE reduction, pooled; 98.1% independent) and on states where the two controllers
choose differently (73.6% / 71.0%). It is **not** concentrated at low margin — 93.9% sits in the
*highest* margin decile.

What justifies the experiment is a different and stronger finding: **the probe's decision
effects have opposite, individually established signs.** Δregret is −0.0898 [−0.1437, −0.0432]
where τ ≤ 0 and +0.0593 [+0.0188, +0.1068] where τ > 0, with non-overlapping intervals, and the
same split appears in the independent cohort. That is exploitable structure, not noise: an
MSE-trained probe spends capacity uniformly and is provably damaging in one regime while
provably helping in the other.

**The minimal experiment, conceptually — not implemented here.** One linear probe family, one
pre-registered variant, no search: train the *same* Ridge probe on the *same* features and
splits, but weight training states by decision relevance rather than uniformly — inversely by
true action margin, so error where an estimate can flip an action costs more than error where it
cannot. Evaluate on the identical held-out-game splits with **regret** as the primary metric and
Q_I R² as secondary, and pre-register that a rise in regret, or a loss of the τ ≤ 0 gain, ends
the line.

It is one fit family and one weighting rule. If decision-weighting does not convert a
demonstrated, sign-split regret effect into a net regret reduction, the white-box control story
should be dropped rather than pursued further.
