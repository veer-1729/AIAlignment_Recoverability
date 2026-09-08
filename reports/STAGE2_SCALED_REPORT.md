# Runtime control of long-horizon LLM agents — scaled Stage 1 substrate and Stage 2 probe study

**Run:** `scale` · ALFWorld · Qwen2.5-7B-Instruct · Arm B (T=0.3, N=5)
**Dataset:** 252 games · 504 rollouts · **1,971 checkpoints** · 21,681 branch records
**Status:** Complete. Stage 1 validated; Stage 2 run on two independent datasets under two split schemes, plus a leak-ceiling diagnostic and a conflict-set test isolating intervention-specific signal (§7.8).

---

## 1. What the paper claims, and what we set out to test

*Calibration Is Not Control: Why LLM-Agent Oversight Needs Intervention*
(arXiv:2606.21399) argues that runtime oversight is usually posed as scalar risk
prediction — estimate `p_fail(h)`, threshold it, intervene — and that this targets
the wrong object. The control-relevant quantity is the **intervention advantage**

```
τ(h) = EU_intervene(h) − EU_continue(h)
```

A scalar `g` supports lossless routing **iff** the sign of τ is recoverable from
`g`. A perfectly calibrated `p_fail` can still merge a *recoverable* high-risk
state with an *irrecoverable* one, and no threshold on it separates them. The
paper calls this **target error** and quantifies it as **scalar abstraction loss**
`Gap(g) = V* − V_g`.

Their empirical result: Platt scaling drops ECE on the confidence scalar from
0.463 to 0.006 while control regret stays pinned at 0.318. Calibration fixes
prediction, not control.

**No code was released.** Everything here is reconstructed from the paper text
plus the upstream ALFWorld and ReAct sources. Section 4 records every place we had
to choose.

The question this run exists to answer is the natural next one: *if a scalar is
insufficient, is the missing information present in the model's own activations?*

---

## 2. Setup

| Dimension | Choice | Source |
|---|---|---|
| Benchmark | ALFWorld `AlfredTWEnv`, text-only, 6 task types | paper-stated |
| Base agent | Qwen2.5-7B-Instruct, pinned revision | paper-stated |
| Intervention | defer-to-expert (`AlfredExpert`, HANDCODED), zero LLM calls | paper-stated |
| Action set | `continue` / `intervene` / `quit` | paper-stated |
| Utility | `U_a = u − c_a − s·m`, `U_quit = 0`; w=1.0, c_int=0.05, c_cont=0, s=0.01 | paper Table 3 |
| Action-token cap | 24 | paper-stated |
| Decoding (Arm B) | T = 0.3, **N = 5** replicates per action | **ours** (§4.1) |
| Scaffold | act-only, 2 exemplars per task type | **ambiguous** (§4.2) |
| Episode cap | 50 steps; branches inherit remaining budget | **ambiguous** (§4.2) |
| Checkpoint selection | K=4 stratified quantile bins per rollout | **ours** (§4.2) |

**Scale.** 252 games (42 per task type) × 2 rollouts × 4 checkpoints. For
reference, the paper's entire four-benchmark main suite is 3,784 branched rows;
this is 1,971 ALFWorld checkpoints carrying 10 executed branches each.

---

## 3. Architecture

The design rule that everything else follows from: **raw outcomes are stored,
values are always re-derived.** A branch record holds `success` and
`n_steps_in_branch` and never a utility. Every number in this report is recomputed
from raw records by `cnc.utility` under a named parameter set, which is what makes
the 5×5 cost sweep a pure re-derivation rather than a second experiment.

```
task → rollout → checkpoint → branch
```

- `src/cnc/env/factory.py` — the only module importing ALFWorld. `SafeAlfredExpert`
  catches `HandCodedAgentTimeout`/`HandCodedAgentFailed` as *outcomes* rather than
  crashes, raises the expert step budget from the hard-coded 200 to 1,000 (replayed
  prefix steps spend it), and falls back to `look` on a non-admissible proposal —
  matching upstream. That last fix took expert success from 3/6 to 18/18.
- `src/cnc/env/replay.py` — compares the full state signature (observation, sorted
  admissible commands, reward/done/won, sorted PDDL facts) at **every** prefix step.
- `src/cnc/branching/runner.py` — seeds global `random` *before* replay, because the
  ALFWorld expert draws from it during replay as well as during the branch.
- `src/cnc/storage.py` — schemas, provenance, sharding, and `read_branches()` (§6.3).
- Parallelism is **process-level only**. Threading ALFWorld env construction corrupts
  its module globals and silently drops branches: a measured 12-checkpoint job yields
  36/36 branches single-threaded and 3 branches plus 11 errors at 4 threads. Those
  failures are caught and counted rather than raised, so threading would have
  quietly shrunk the dataset instead of crashing.

---

## 4. Deviations and ambiguities

### 4.1 Deliberate deviations

1. **Arm B exists.** The paper branches once per action, yielding a realized
   `U(s,a)`. Its own App. B.7 records zero within-action variance on repeats, which
   makes greedy single-rollout `U` degenerate as a regression target. Arm B uses
   T=0.3 and N=5 to estimate `Q̂(s,a)`. This changes the policy and therefore the
   data-generating distribution, so it is labelled an extension throughout and never
   presented as the replication.
2. **ALFWorld only.** The paper's largest control gap by a wide margin
   (Δ regret 0.396 vs 0.076 for the next benchmark), and the open-weight model makes
   activations accessible.
3. **Expert-only intervention.** The paper's stronger-model and self-repair ladder is
   out of scope here.
4. **An additional held-out-game split** beyond the paper's prefix-level split.

### 4.2 Replication ambiguities

The paper fixes only the generation budgets (24 action tokens, 8 confidence
tokens). It does **not** state the prompt format, shot count, whether admissible
commands are shown, the episode cap, or the prefix-selection rule, and there is no
author code to recover them from. We adopt the upstream ReAct reference convention
and make each a config knob recorded in provenance. **These are inferences, not
paper-matched settings, and are labelled as such.**

---

## 5. Validation (V1–V10) — 10 PASS / 0 FAIL / 0 SKIP

| # | Check | Result |
|---|---|---|
| V1 | Replay fidelity | **19,710 / 19,710 = 1.0000** verified; 0 rejected |
| V2 | Negative control (corrupted prefix) | 6/6 planted corruptions detected |
| V3 | Cross-process determinism | 6/6 prefix state hashes reproduced |
| V4 | Arm A determinism | n/a by design (Arm B only) |
| V5 | Arm B stochasticity | 297/1,971 (0.15) differ across seeds; MC SE p50 0.000, p90 0.010, max 0.598 |
| V6 | Expert sanity (takeover from t=0) | 18/18 = 1.00 |
| V7 | Base success band | 0.377 (band 0.15–0.60, n=504) |
| V8 | Split leakage | clean under both split schemes |
| V9 | Derivation purity | 1,971 usable, 0 order-dependent |
| V10 | Prompt/token round-trip | 30 sampled, 0 mismatched; **re-verified on all 1,971 at extraction, 0 mismatches** |

V2 matters as much as V1: a 100% match rate is only meaningful if the validator
can fail, and it detects every planted corruption.

**Dataset integrity.** 1,971/1,971 checkpoints complete, 0 partial, 0 absent, 0
malformed lines across 31 branch files, 0 games without usable checkpoints.

---

## 6. Descriptive results (training-free)

The scalar `g` throughout is the **measured** continuation-failure rate, not a
learned score. In Arm B this is deliberately the strongest possible
continuation-risk scalar — an oracle one — so abstraction loss against it is a
conservative lower bound on what a learned score would incur.

### 6.1 The replication holds at scale

| | n = 332 (Gate B) | **n = 1,971** |
|---|---|---|
| `Gap(g)` | 0.303 | **0.298** |
| Conflict-set mass | 91.3–100% | **100%** |

A sixfold increase in sample moved the headline number by 0.005.

**Oracle action balance:** continue 562 (28.5%) · intervene 866 (43.9%) · quit 543 (27.5%).
All three actions are well represented; no class is vestigial.

**Intervention advantage:** mean 0.836, sd 1.039, range [−2.056, 2.410], **50.6% positive**.
Close to the most informative balance a binary control decision can have.

### 6.2 The core claim, at full scale

Binning on measured continuation risk:

| risk | n | mean τ | oracle actions |
|---|---|---|---|
| 0.00 | 598 | −0.078 | continue 561, intervene 37 |
| 0.20 | 45 | +0.400 | intervene 44, continue 1 |
| 0.40 | 30 | +0.690 | intervene 26, quit 4 |
| 0.60 | 37 | +1.374 | intervene 36, quit 1 |
| 0.80 | 41 | +1.672 | intervene 37, quit 4 |
| **1.00** | **1,220** | +1.259 | **intervene 686, quit 534** |

The last row is the paper's claim at scale. 1,220 checkpoints share an *identical*
and *maximal* continuation risk — the agent fails every time if left alone — and
split almost evenly between "hand to the expert" and "abandon". No threshold on
continuation risk can separate them, however well calibrated. Five of six bands
have τ straddling zero.

**Caveat on 100% conflict mass.** With N=5, measured risk takes only six values, so
bins are coarse and a coarse bin is more likely to be mixed. The number to trust is
not "100%" but the composition of bin 5, where 1,220 states at one risk value split
686/534 — that is not a binning artifact.

### 6.3 Where τ's variance lives

| Conditioning | var(τ) | retained |
|---|---|---|
| total | 1.0789 | — |
| within task type | 0.9242 | 86% |
| within step index | 0.9143 | 85% |
| within task type × step decile | 0.7534 | **70%** |

**This is the headroom for Stage 2.** Seventy percent of the variance in
intervention advantage survives conditioning on the obvious observable covariates.
If activations carry control-relevant information, this is the space it lives in.

### 6.4 Is the expert too constant to be a target?

No. `Q_intervene` has **sd 0.895** over [−1.540, 0.940]. Per-checkpoint expert
success is bimodal — 410 checkpoints where the expert failed all five times, 1,376
where it succeeded all five — but 185 checkpoints (9.4%) vary across seeds, and
that is enough to make it a real random variable rather than a constant.

**Consequence for the paper's design.** Estimating `Q(s,I)` from a *single* expert
draw, as the paper does, flips the oracle action on **279 of 9,855 expert branches
= 2.8%**. That is a label-noise floor built into single-rollout branching, and it
is the strongest argument for Arm B's Monte-Carlo extension.

### 6.5 Are the Monte-Carlo targets usable?

τ spread divided by mean MC standard error is **12.59**. Signal dominates
estimation noise by an order of magnitude, so N=5 is adequate for regression.
Median MC SE on τ is 0.000; p90 is 0.464.

### 6.6 Pooling is sound

The dataset was assembled across two server processes after an infrastructure
failure (§8), and the split between cohorts is not random — it follows how far each
shard got, which tracks position in a task-sorted list. Under **game-level**
stratification, all ten cohort comparisons have confidence intervals containing
zero. The differences visible under task-type stratification were *which games each
cohort held*, not how they were executed; the two most significant collapsed from
+0.174 [+0.050, +0.302] to +0.071 [−0.119, +0.273] and from −0.085 to −0.004.

Independent support: `original` ran on the pre-reboot server and both `rescue2` and
`r1` on the post-reboot one. A server effect would group those two against
`original`; instead they differ from it in opposite directions on τ.

---

## 7. Stage 2: does the activation advantage survive?

> **Partial.** Split scheme 1 (`grouped_by_rollout`) is complete and reported below.
> Split scheme 2 (`held_out_game`) is still running, and **it is the arbiter** — see
> §7.2 for why scheme 1 cannot answer the question on its own. The single-cohort
> robustness run (n = 928) and the leak-ceiling diagnostic follow it.

**Pre-registered commitments, fixed before the scaled data was seen:**

- Layers **{8, 16, 24}**, chosen from the pilot and reported in full everywhere.
  Searching all 29 and reporting the best inflates the estimate by the size of the search.
- Hyperparameters selected by grouped cross-validation **strictly inside the training
  split**; test touched once.
- Two split schemes: grouped by rollout (the paper's leakage rule) and **held-out
  game** (no game shared with training).
- Linear/regularized probes only — Ridge and logistic regression. Probe capacity is
  deliberately not the variable under test.
- Uncertainty from two independent sources: paired bootstrap over the test set, and
  the spread of each delta across five independent group-level splits.

**Pre-registered prediction.** Since τ = Q_I − Q_C, holding Q_C fixed makes
predicting τ equivalent to predicting Q_I — and in the pilot, activations *lost* to
the 26 observable features on Q_I (0.574 vs 0.658). **The conditional test may
therefore come out negative even though the unconditional one was strongly
positive.** That is the point of running it, and a clean negative is a result.

**C2 is the arbiter for Q2, not C1.** C1 concatenates 27 observable columns with 3,584
activation dimensions under a single ridge alpha; the activation block drags alpha
up, shrinking the standardized Q_C coefficient — the most informative column for τ.
C1 can therefore lose for reasons unrelated to activation content. C2 residualizes
τ on `[observables, Q_C]` with its own alpha and asks whether activations predict
what remains. If they disagree, C2 is the one to believe.

**Pre-registered arbiter for the study as a whole,** recorded before the number was
seen: **scheme 2's C3 `risk = 1.00` band.** Inside that band Q_C is pinned at its
maximum, so τ variation is Q_I variation, and no game is shared with training. It is
the only cell that tests the hypothesis with the channel of §7.2 closed.

### 7.1 Scheme 1 — the paper's own split

Grouped by rollout, five independent splits, n = 1,971. Delta is against the 26
observable features; sign column is consistency across all five splits.

| target | L8 | L16 | L24 | sd (L24) | sign |
|---|---|---|---|---|---|
| Q_continue (R²) | +0.079 | +0.174 | **+0.222** | 0.030 | consistent |
| Q_intervene (R²) | +0.084 | +0.116 | **+0.146** | 0.056 | consistent |
| τ (R²) | +0.125 | +0.202 | **+0.223** | 0.049 | consistent |
| continue_success (macro-F1) | −0.000 | +0.022 | +0.038 | 0.013 | consistent |
| oracle_action (macro-F1) | +0.026 | +0.059 | **+0.067** | 0.011 | consistent |

28 of 30 target × block rows are sign-consistent across all five splits. The two
exceptions are `continue_success` at layer 8 (−0.000 and +0.003) — a clean null, not
a contradiction.

**The pilot's one negative result reversed.** `oracle_action` was −0.086 *against*
activations at n = 332; at n = 1,971 it is **+0.067**, sd 0.011, consistent across
every split. The likely explanation is unremarkable: a 3,584-dimensional probe on
~230 training rows is hopeless, and ~1,380 rows is merely hard.

`Q_intervene` reversed the same way (pilot −0.084, here +0.146), which matters
because it was the stated mechanism behind the pre-registered prediction above. The
prediction was a hedge with a mechanism attached; the mechanism evaporated at scale.

### 7.2 The sibling-rollout channel — why scheme 1 cannot answer the question

The dataset holds 252 games and 504 rollouts: **exactly two rollouts per game.**
Scheme 1 groups by rollout, so for roughly 70% of test rollouts, the *other rollout
of the same game* sits in training.

Q_C, Q_I and τ are strongly game-determined. Game identity is richly recoverable
from activations — task text and receptacle layout are in the prompt — while the 26
observables carry only a task-**type** one-hot, not game identity. The two feature
blocks are therefore not competing symmetrically: activations have a channel to the
answer that observables structurally lack, and §7.1's deltas are inflated by an
unknown amount.

This is not a defect in the run. Rollout-level grouping is the paper's own leakage
rule (App. B.3), so scheme 1 **is** the faithful replication and this channel is a
property of the paper's protocol rather than of this implementation. It does mean
§7.1 answers "can activations predict these targets under the paper's split?" and
not "do activations carry state-specific control information?"

Two things address it, neither yet in hand: scheme 2, which closes the channel by
construction, and `scripts/game_id_ceiling.py`, which measures the channel's width
with a game-identity one-hot — a block that is pure channel and contains no state
information at all, not even the step index.

One argument deliberately **not** made: monotonicity in depth (L8 < L16 < L24 on
every target in §7.1) is not evidence against the channel, since a game-identity
representation would also sharpen with depth. C1 in §7.4 already breaks the pattern.

### 7.3 Scheme 2 — held-out games

No game appears on both sides. Seed 0: 1,383 train / 588 test rows, 176 train games /
76 test games. Five independent splits.

| target (L24) | scheme 1 | **scheme 2** | sd | sign | change |
|---|---|---|---|---|---|
| Q_continue | +0.222 | **+0.174** | 0.034 | consistent | −22% |
| Q_intervene | +0.146 | **+0.120** | 0.036 | consistent | −18% |
| τ | +0.223 | **+0.160** | 0.045 | consistent | −28% |
| oracle_action | +0.067 | **+0.046** | 0.026 | consistent | −31% |
| continue_success | +0.038 | +0.013 | 0.021 | **FLIPS** | dies |

**The advantage survives the game-level split.** Every regression target keeps a
consistent sign across all five splits, monotone in depth, shrinking by roughly a
fifth to a third rather than collapsing. That is Q3 answered affirmatively for the
unconditional targets.

**Two things die.** `continue_success` flips at all six blocks (it was stable at L16
and L24 under scheme 1) — a complete null once games are held out, and a warning that
seed 0's `+0.039 *` was a lucky split. `oracle_action` survives at L24 only; L8 and
L16 both flip.

**Closing the channel hurt the observable baseline more than it hurt the
activations.** On τ the observable R² fell 0.318 → 0.167 (−47%), the activation R²
fell 0.557 → 0.363 (−35%), and the *delta* fell only 0.223 → 0.160 (−28%). The 26
features were themselves partly exploiting game identity through task type and step
index; the difference between the two blocks is the most robust of the three
quantities, which is the opposite of what the leakage objection would predict if the
channel were the whole story.

One oddity, recorded and not treated as a finding: the observable baseline on
`Q_intervene` *rose* under the harder split (0.515 → 0.595). R² is relative to the
test set's own variance, and the held-out-game test sets carry more between-game
variance in Q_I, which the task-type features capture. It is a property of the metric
and the split, not evidence about the features.

### 7.4 Conditional tests — τ given continuation prospects (Q2)

This is the actual hypothesis: *among states with similar continuation prospects, do
activations carry additional information about intervention value?* Three instruments,
pre-registered, and they do not all agree.

**C1 is uninformative, exactly as pre-registered.** Under scheme 2 it is negative at
every layer (−0.056, −0.078, −0.067, all with CIs excluding zero at seed 0). That is
the artifact named in advance: one ridge alpha over 27 observable columns plus 3,584
activation dimensions is dragged up by the activation block, shrinking the
standardized Q_C coefficient — the single most informative column for τ. **C1 going
negative is not evidence that activations hurt.** It is C1 behaving as predicted, and
the pre-registration says to believe C2 where they disagree.

**C2 is not established under the game-level split.** The per-seed values matter here,
because a mean and a sign flag hide the shape:

| C2 (absolute R² on residual τ) | scheme 1, five splits | scheme 2, five splits |
|---|---|---|
| L8 | .123 .114 .088 .193 .174 — consistent | .057 .025 .013 .013 **−.017** — FLIPS |
| L16 | .208 .176 .113 .218 .244 — consistent | .114 .090 .010 .017 **−.036** — FLIPS |
| L24 | .323 .232 .194 .266 .365 — consistent | .279 .236 .238 .242 **−.012** — FLIPS |

All three are stable under the paper's split and none survive game-level splitting by
the binary standard the study prints. **That verdict stands as the Q2 result.**

Two things qualify how much it says. First, what the standard measures: C2 stores an
*absolute* residual R², so FLIPS means one split predicted the residual worse than its
own mean — a harsh bar for a 3,584-dimensional ridge on ~900 held-out training rows.
Second, C2 is layer-dependent rather than uniformly null. L24 at
[.279 .236 .238 .242 −.012] is one fold away from stable; L16 at
[.114 .090 .010 .017 −.036] is genuinely weak and trending to zero; L8 is marginal
everywhere. Neither "a clean negative" nor "C2 answers Q2 affirmatively" is accurate.

> **Post-hoc observation, and a hypothesis that failed.** One split — seed index 4 —
> is the sole negative on all three C2 layers under `held_out_game` and the most
> negative on C1 L8 and L24, while being unremarkable or better than average on every
> unconditional target (`Q_continue` +0.213, its *highest* of five; τ +0.168 against a
> range of .093–.204) and positive on the arbiter band (+0.078). C1 and C2 are the
> only two tests that condition on Q_C, which is where that pointed.
>
> The obvious mechanism was that seed 4's baseline fit unusually well, leaving little
> residual variance for activations to explain and making R² on a near-empty target
> noisy. **It was stated in advance, tested, and falsified.** Under `held_out_game`:
>
> | seed | residual var | C1 baseline R² | C2 L24 |
> |---|---|---|---|
> | 0 | **0.2845** (smallest) | **0.724** (highest) | **+0.279** (best) |
> | 1 | 0.3930 | 0.633 | +0.236 |
> | 2 | 0.4419 (largest) | 0.574 (lowest) | +0.238 |
> | 3 | 0.4019 | 0.603 | +0.242 |
> | 4 | 0.3332 | 0.650 | **−0.012** (only negative) |
>
> Seed 0 and seed 4 share a profile — small residual, well-fitting baseline — and sit
> at opposite extremes of the outcome. The predicted direction is not merely absent,
> it is reversed at the end that matters. Seed 4 is second of five on both diagnostics
> and unremarkable on neither. Split sizes rule nothing in either: every seed has
> exactly 176 training and 76 test games, 1,372–1,383 training rows, and `C1_alpha` is
> 10 for all five in both schemes.
>
> **No mechanism was identified, and none is offered.** Proposing a second hypothesis
> after the first was falsified is the same manoeuvre as discounting the fold, one
> iteration later. The supportable statement is that one split behaved differently for
> reasons we could not determine — not that one split "failed".
>
> Seed 4 is not dropped, no four-seed mean is quoted, and the Q2 headline is unchanged.
> (A note on a comparison not made: `grouped_split` shuffles `np.unique(groups)`, which
> is 504 rollouts under scheme 1 and 252 games under scheme 2, so "seed 4" names two
> entirely different partitions. Its behaviour in one scheme is not evidence about the
> other. The falsification above rests only on the within-scheme comparison, which is
> the valid one.)

**C3, within continuation-risk bands, across all five splits.** Computed for every
seed and recovered from the JSON (`seed_summary` prints only the target and C1/C2 rows).

| band | scheme | seeds present | n_test | L8 | L16 | L24 | sign |
|---|---|---|---|---|---|---|---|
| 0.00 | 1 | 5/5 | 154–213 | −0.151 | −0.045 | −0.092 | FLIPS |
| 0.00 | 2 | 5/5 | 149–220 | +0.375 | +0.117 | +0.084 | FLIPS |
| 0.20 | 1 | 4/5 | 11–18 | +0.413 | +0.781 | +0.687 | FLIPS |
| 0.20 | 2 | 5/5 | 11–20 | +0.114 | +0.012 | −0.162 | FLIPS |
| 0.80 | 1 | 5/5 | 10–15 | −0.086 | −0.094 | −0.128 | FLIPS |
| 0.80 | 2 | 4/5 | 11–16 | −0.012 | +0.094 | −0.302 | FLIPS |
| **1.00** | **1** | **5/5** | **326–384** | **+0.087** | **+0.117** | **+0.158** | **consistent** |
| **1.00** | **2** | **5/5** | **324–387** | **+0.058** | **+0.095** | **+0.122** | **consistent** |

**The pre-registered arbiter holds.** Scheme 2's `risk = 1.00` band is present in all
five splits and sign-consistent at all three layers — every one of the fifteen
per-seed deltas is positive, monotone in depth, on a healthy cell (324–387 test rows,
observable R² 0.404–0.577). Mean to mean the reduction from scheme 1 at L24 is
+0.158 → +0.122, about 23%, in line with the unconditional targets.

That band is the paper's conflict set: 1,220 checkpoints where continuation risk is
maximal and identical, splitting 686 intervene / 534 quit, where no threshold on a
continuation-risk scalar can route correctly by construction. Activations predict
inside it, on games never seen in training.

**It is also the only stable C3 band in either scheme,** and the reason is sample
size, not regime. Bands 0.20 and 0.80 carry 10–20 test rows and observable R² values
of −4.26, −3.35, −2.24 and −4.42; band 0.20 qualifies in only 4 of 5 splits under
scheme 1 and band 0.80 in only 4 of 5 under scheme 2, because the filter is applied
per split. **`MIN_BAND_TE = 10` was set too permissively** — every band under roughly
20 test rows is noise. That threshold was fixed in advance, so these bands are
reported rather than dropped; the lesson is for the next study, not a licence to edit
this one.

**The two arbiters disagree, and that is reported rather than resolved by preference.**
C2 asks whether activations beat observables + Q_C on residual τ across the whole
distribution; C3's `risk = 1.00` band asks it where continuation risk is pinned at
maximum. The second is stable and the first is not. One qualification on reading them
as the same test: the band is defined on `p_fail`, and with all five continue branches
failing Q_C is still `−w − s·m`, so it varies slightly with branch length — and C3's
baseline is the 26 observables, which exclude Q_C, whereas C2's includes it. They
differ in baseline as well as in regime. The careful claim is **"activations add
information about intervention value in the high-risk regime, on unseen games"**, not
"activations add information conditional on Q_C across the board."

### 7.5 Independent replication on the single-cohort subset

The scaled dataset was assembled in two executions (§6.6). The robustness run repeats
the entire pre-registered study on the **`original` cohort alone** — n = 928, one
execution, one server process, no cross-execution question of any kind. 417 rollout
groups, 250 game groups. Runtime 6h23m against the pooled run's 8h46m, scaling
sensibly with n.

This is a different question from §6.6, and the two can disagree without conflict.
§6.6 tested whether the two cohorts have exchangeable *target distributions* — the means
of Q_C, Q_I and τ — and found every stratified interval containing zero. §7.5 tests
whether *probe deltas* reproduce, which additionally depends on how much training data a
3,584-dimensional probe has. A delta that shrinks on half the data (Q_continue,
+0.174 → +0.122) is a sample-size effect, not evidence that pooling was unsound.

**The arbiter replicates, and is slightly larger on the cleaner cohort.**

| C3 `risk = 1.00`, held-out games | L8 | L16 | L24 | seeds |
|---|---|---|---|---|
| pooled, n = 1,971 | +0.058 | +0.095 | +0.122 | 5/5 consistent |
| **single cohort, n = 928** | **+0.070** | **+0.123** | **+0.162** | **5/5 consistent** |

All fifteen per-seed deltas positive in both runs, monotone in depth in both. The
cohort with no pooling question at all returns +0.162 at L24 against the pooled
+0.122, on less than half the data. Nothing in the headline depended on pooling.

**Unconditional targets, held-out games:**

| target (L24) | pooled | single cohort |
|---|---|---|
| τ | +0.160 consistent | **+0.158 consistent** |
| Q_continue | +0.174 consistent | +0.122 consistent |
| Q_intervene | +0.120 consistent | +0.103 consistent |
| oracle_action | +0.046 consistent | +0.051 consistent |
| continue_success | +0.013 FLIPS | FLIPS |

τ at layer 24 reproducing to within 0.002 across independent cohorts is the single
strongest line in the study.

**C2 is null again, independently.** All three layers FLIP, with L8 and L16 now
carrying negative means (−0.009, −0.009) and L24 at +0.059, sd 0.086. This matters for
how §7.4's anomaly is read: the Q2 null reproduces on a dataset where the odd split
plays no part, so **the unexplained seed is not load-bearing for the conclusion.** The
pre-registered verdict rests on two independent runs, not on one fold.

**C1's artifact is larger at smaller n, which its stated mechanism predicts.** All
three layers are sign-consistent and strongly negative (−0.147, −0.157, −0.163) against
the pooled run's −0.057, −0.040, −0.077. The pre-registration attributes C1's behaviour
to a single ridge alpha being dragged up by 3,584 activation columns and shrinking the
standardized Q_C coefficient; halving n roughly doubles p/n, so the same penalty bites
harder and the artifact should grow. It does. That is a specific prediction of the
explanation, checked after the fact and matched — reported as consistency, not proof.

### 7.6 Measuring the channel rather than arguing about it

§7.2 raised the sibling-rollout channel as a limitation of the paper's split. A
post-hoc diagnostic (`scripts/game_id_ceiling.py`, outside the pre-registered study)
bounds it with a **game-identity one-hot** — a block that is pure channel and carries
no state information at all, not even step index. Ridge on it learns a per-game mean,
so under the paper's split it predicts each test row from its sibling rollout.

Both validity gates pass. Under held-out games the one-hot scores **−0.020 / −0.019 /
−0.024** with 0.0% train/test game overlap, as a correctly wired pure-leak block must;
and `obs+act_L24` reproduces `probe_study.py`'s own 0.732 / 0.706 / 0.561 exactly, so
the data prep did not drift.

**The channel is real and large.** Measured train/test game overlap under the paper's
split is **65.6%**. And on its own:

| block, scheme 1 | Q_continue | Q_intervene | τ |
|---|---|---|---|
| game identity alone | 0.407 | 0.317 | **0.320** |
| the 26 observables | 0.441 | 0.508 | **0.246** |

**On τ, pure game identity outscores the entire observable feature block.** Read this
narrowly. With 65.6% overlap the one-hot is not describing a property of τ — it is
reading the sibling rollout's τ mean for that game. The supportable statement is that
**under the paper's split the sibling-rollout channel is worth more on τ than 26
hand-built state features are**, which is a fact about the split, not about whether
intervention advantage is game-determined. It does establish that §7.2's concern was
worth the trouble of measuring.

**And it does not explain the activation advantage.**

| scheme 1 | activations over observables | over observables **+ game identity** | retained |
|---|---|---|---|
| Q_continue | +0.222 | +0.194 | **87%** |
| Q_intervene | +0.152 | +0.138 | **91%** |
| τ | +0.225 | +0.168 | **75%** |

Hand the model game identity for free and activations still add +0.138 to +0.194. The
channel accounts for at most a quarter of the advantage on τ — the most game-determined
target — and roughly a tenth on the Q targets.

Two honest limits on that bound. A one-hot encodes per-game *means*, so it ceilings the
**additive** game-identity channel; it does not bound interactions between game identity
and state, which would need game × step terms this dataset cannot support. And under
held-out games the one-hot is pure noise that consumes ridge capacity and degrades the
baseline, so that block's "over obs+game" figures (+0.251 / +0.197 / +0.219) are inflated
by a weakened baseline and are **not** comparable to the scheme 1 column above. They are
excluded from the argument.

### 7.7 Verdict on Q1–Q4

**Q1 — does the activation advantage survive at scale? Yes.** Every regression target
gains at every layer, monotone in depth, sign-consistent across all five splits, in
both the pooled run and the independent single-cohort run. The pilot's one negative
(`oracle_action`, −0.086 at n = 332) reversed to +0.067 pooled and +0.051 single-cohort.

**Q2 — does it survive conditioning on continuation value? Split, and the split is the
result.** In the regime the paper's argument is about — the `risk = 1.00` conflict set,
1,220 checkpoints where continuation *failure probability* is maximal and identical,
splitting 686 intervene / 534 quit — activations add signal on games never seen in training, 5/5 splits at all
three layers in both runs. Across the whole distribution, C2 fails its stability
standard in both runs independently. C1 is uninformative by pre-registered mechanism,
and its artifact grows at smaller n exactly as that mechanism predicts.

The two pre-registered arbiters for Q2 — C2, and scheme 2's `risk = 1.00` band —
**disagree, and this report presents both rather than promoting whichever reads better.**
The supportable claim is **"activations add information about intervention value in the
high-risk regime, on unseen games"** — not "conditional on Q_C across the board."

**Q3 — does it generalize to held-out games? Yes.** Removing every shared game costs
roughly a fifth to a third of the effect and leaves signs stable. Independently, the
leak-ceiling diagnostic shows 75–91% of the advantage survives handing the model game
identity for free. Two different instruments, same conclusion.

**Q4 — is the dataset statistically sufficient? For the main effects yes; for the
conditional question no.**

| quantity | sd / mean | verdict |
|---|---|---|
| unconditional deltas (L24) | ≈ 0.2–0.3 | sufficient; τ replicates across cohorts to 0.002 |
| C2, whole distribution | ≈ 0.6 | **not sufficient** — spread comparable to the effect |
| risk bands under ~20 test rows | unbounded | **not sufficient** — observable R² reaches −4.4 |

Two thousand checkpoints resolves the large effect and does not resolve the subtle one.
That is a concrete answer rather than a hedge: the next study needs more data for the
conditional test specifically, not more data in general.

**What died, recorded as plainly as what survived.** `continue_success` is a complete
null under held-out games in both runs — seed 0's `+0.039 *` was a lucky split.
`oracle_action` survives at layer 24 only. Three of four risk bands are noise. And one
split's behaviour on the conditional tests remains unexplained after a stated hypothesis
was tested and falsified.

---

### 7.8 Is the signal intervention-specific, or a Q_C proxy?

§7.7 left one ambiguity that none of the preceding tests can settle. Since
τ = Q_I − Q_C, a probe that merely represents **continuation** state better than 26
hand-built features will improve τ mechanically, knowing nothing whatever about the
expert. Every result above is compatible with both readings.

**The isolation.** Restrict to the `risk = 1.00` conflict set, hand the model Q_C
*as a feature*, and predict **Q_I directly** — the expert's value, not a difference
involving Q_C. Inside that band Q_C is pinned:

| within the conflict set | sd | share of variance in play |
|---|---|---|
| Q_C | **0.141** | ~2% |
| Q_I | **0.993** | ~98% |

So the proxy explanation is doubly unavailable here: Q_C is already supplied, and
there is almost none of it left to represent. Nested models, identical protocol
imported from `probe_study.py`, splits computed on the full cohort and intersected
with the band exactly as C3 does.

**Primary result — held-out games, ΔR² on Q_I of [obs + Q_C + activations] over
[obs + Q_C]:**

| layer | pooled, n = 1,220 | single cohort, n = 580 |
|---|---|---|
| 8 | +0.070 (sd 0.021) | +0.066 (sd 0.039) |
| 16 | +0.113 (sd 0.056) | +0.128 (sd 0.035) |
| **24** | **+0.151** (sd 0.060) | **+0.174** (sd 0.072) |

**All thirty per-split values are positive** — five splits × three layers × two
independent cohorts — monotone in depth in both, and replicating across cohorts at
every layer. Activations *alone*, with no observables and no Q_C, beat the full
baseline on all ten held-out-game splits (0.543 vs 0.403 pooled; 0.505 vs 0.338
single-cohort, both at layer 24).

**This answers the question. The residual stream carries information about how well
the expert will do that 26 observable features plus the measured continuation value
do not contain.** It is not a better picture of the actor's own prospects.

**The confound runs the safe way this time.** Model 3 shares one ridge alpha across
27 observable and 3,584 activation columns — the structure §7.4 pre-registered as
confounded. But that artifact biases model 3 **downward**: it shrinks the small
block's coefficients. Model 3 winning anyway means the artifact can only have made
these numbers an underestimate. The residualized companion was insurance against a
spurious *negative*, and no negative occurred, so it is not the arbiter here.

That companion is itself unstable in the smaller cohort (L8 −0.021, L16 −0.008, L24
+0.034, all FLIPS), with per-split values clustered at ±0.00x — for example L16 at
[+0.002, +0.003, −0.006, +0.007, −0.044]. That pattern is what ridge alpha collapsing
to near-mean predictions looks like on a noisy residual target at ~400 training rows.
Not verified: the companion's selected alphas are not printed.

**The control decision is a weaker result, and the reason is headroom.** On
intervene-versus-quit, the observable baseline already reaches AUROC **0.899**
(pooled) and 0.876 (single cohort). Activations add +0.017 to +0.041, sign-consistent
at layers 8 and 16 in the pooled cohort and FLIPS elsewhere; macro-F1 flips in five of
six held-out-game cells. Activations *alone* still reach 0.923 and 0.915. The honest
reading is that this binary decision is largely already determined by observables plus
Q_C, so there is little room left to demonstrate an increment — not that the
information is absent, given the regression result on the same rows.

**Secondary split**, grouped by rollout with the game channel open, is uniformly
stronger and stable at every layer (ΔR² +0.098 / +0.133 / +0.185; ΔAUROC +0.022 /
+0.033 / +0.040), as expected from §7.6.

---

## 8. Infrastructure findings

Recorded because they cost real time and would cost anyone else the same.

### 8.1 A silent disk leak in fast-downward

Two runs were destroyed by a filesystem filling with no visible cause: `du` reported
27 GB where `df` reported 170 GB used, growing 1–2 GB/min, with `lsof` finding
nothing.

**Cause.** Every PDDL solve extracts a fresh `libdownward.so` into its own temp
directory, `mmap`s it, unlinks the path, and never drops the mapping. The inode's
blocks stay allocated for the life of the process with no directory entry and no
open descriptor. One shard showed 2,760 deleted-but-mapped entries and
`write_bytes` of 19.16 GB. It is **one load per env construction** — ten per
checkpoint at five replicates per action, roughly 350 MB.

**Why every standard tool missed it.** `du` cannot see an unlinked inode. `lsof +L1`
selects on a stat link count it does not populate for mapped-deleted entries.
`/proc/PID/fd` shows nothing because the descriptor is closed before the unlink.
Only `/proc/PID/maps` and `map_files` show it — and process exit erases the
evidence, which is why the first occurrence left nothing to find.

**Mitigation without patching upstream.** Process exit frees the blocks, so bounding
process lifetime bounds the problem. `scripts/run_backfill_chunked.sh` runs each
shard for a fixed number of checkpoints, waits for every process to exit, and
restarts — and prints `df` on both sides of each round so the premise is measured
rather than assumed. The full backfill then completed in one round with free space
flat at 160 GB.

### 8.2 Failure handling that made a bad day worse

The original loss was not caused by the disk alone. A per-checkpoint
`except Exception` swallowed each `ENOSPC` and continued, so eight shards spent
hours executing branches they could not persist; one failed 146 of its 197. The
guard added afterwards was itself aimed at the wrong place — it watched the run
directory, which had 160 GB free throughout, while the root filesystem filled. Both
are fixed: disk exhaustion raises a type that is deliberately **not** an `OSError`
so a broad handler cannot swallow it, the guard watches every filesystem the
process can write to, and a consecutive-failure breaker stops on whatever systemic
failure comes next that we have not named.

### 8.3 Two bugs that would have corrupted the numbers silently

- **Duplicate replicates.** Branch ids are deterministic, so a resume reproduces
  them. A checkpoint with three leftover continue branches plus five fresh ones
  would have been averaged over eight, with the standard error understated to match.
  V9 would not have caught it: derivation purity asks whether all three actions are
  present and whether the result is order-independent, and both hold for a
  duplicated set. `RunDir.read_branches()` now selects one atomic attempt per
  checkpoint. *(In the event, zero checkpoints had multi-file records — the failures
  aborted before writing rather than mid-write — so this stands as insurance.)*
- **Short replicate counts.** Every consumer tested "all actions present and
  verified", which passes a checkpoint holding three branches where five are
  required. Completeness now includes the replicate count.

**Test coverage** grew from 91 to 132 across this work, including subprocess tests
that execute the analysis scripts end to end — added after a signature change with
a missed call site crashed `05_analyze` hours into the pipeline, invisible to both
the unit tests (which never import the scripts) and the import test (which never
calls `main()`).

---

## 9. Suitability for Stage 3

**The substrate is sound and the target is real.** τ is predictable from residual-stream
activations beyond a strong observable baseline, in the regime where scalar risk
prediction is provably insufficient, on games never seen in training, replicated across
two independent cohorts. Oracle actions are well balanced (28.5 / 43.9 / 27.5), τ spread
exceeds mean Monte-Carlo standard error by 12.6×, and 70% of τ's variance survives
conditioning on task type and step decile. Nothing here blocks Stage 3.

**Four things should change before it.**

*Resolve the conditional question with power, not rhetoric.* C2 is the one pre-registered
test that failed, in both runs, and §7.7 shows why: its spread is comparable to its
effect. That is an n problem in a specific cell, not a reason to abandon the question.

*Break the game-identity channel at the source.* 252 games × 2 rollouts gave 65.6% overlap
under the paper's split. More games with fewer rollouts each — or grouping by game from
the start — removes the confound rather than measuring it afterwards. §7.6 is a patch for
a design that should not need one.

*Raise N, knowing what it costs.* With five replicates `p_fail` takes only six values, so
a C3 "band" is a level rather than a bin: `risk = 1.00` is exactly the checkpoints where
all five continue branches failed, which is why it holds 1,220 of 1,971 and why the middle
levels are thin. N = 20 would give 21 levels and spread that mass out — **the 1.00 band
would shrink, and the study's strongest single result with it.** The trade is resolution
against concentration, and it is worth making, but it is not free. Raising N would also
cut the 2.8% oracle-flip rate from single-draw expert estimation. `MIN_BAND_TE = 10` was
set too permissively and should be at least 30.

*Fix the compute shape.* The study spent 15 hours on ~19,000 fits at roughly 8× off the
memory-bandwidth bound. Both hot operations are 3-column GEMMs, which no BLAS
parallelizes well, and 15 threads were sharing 8 cores with no limits set. The fix is
`OMP_NUM_THREADS=1` **plus** outer-loop process parallelism over folds or passes —
inner thread limits alone would serialize the only parallelism the job has and make it
roughly 8× slower.

**One method worth carrying forward.** The leakage objection in §7.2 was answerable only
because it was converted into an instrument: a feature block that is *purely* the
suspected confound and nothing else, plus a negative control in the regime where that
block must score zero. Both gates fired correctly, which is what made the 75–91% figure
usable rather than another argument. Stage 3 will face the same class of objection —
that a probe has learned an identity rather than a state — and the same construction
answers it. Building the control alongside the claim costs one script and settles what
would otherwise be an exchange of intuitions.

---

*Repository:* https://github.com/veer-1729/AIAlignment_Recoverability
