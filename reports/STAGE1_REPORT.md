# Stage 1 Report — Prefix-Branching Substrate for Runtime Control of LLM Agents

Reproduction of the data-generation and evaluation protocol behind *Calibration Is
Not Control* ([arXiv:2606.21399](https://arxiv.org/abs/2606.21399)) on ALFWorld with
Qwen2.5-7B-Instruct. **No training was performed.** The deliverable is a branched
dataset, a validation record, and an answer to two questions.

---

## 1. Headline result

**The paper's phenomenon reproduced.** At 332 checkpoints per arm, states with
*identical* measured continuation risk require *different* optimal actions:

| continuation risk | n | oracle action split |
| --- | --- | --- |
| 0.00 (continue succeeds) | 93 | continue 69 · **intervene 24** |
| 1.00 (continue fails) | 239 | **quit 80 · intervene 159** |

The second row is the paper's claim made concrete. All 239 states are equally
doomed under continuation, yet 159 are *recoverable* by the expert and 80 are not.
A controller routing on continuation risk alone cannot separate them — it must
mishandle one group or the other, no matter how well calibrated it is.

Scalar abstraction loss **Gap(g) = 0.303**, conflict-set mass **100%** (Arm A).

---

## 2. Validation — 11 PASS / 0 FAIL / 0 SKIP

| check | result |
| --- | --- |
| V1 replay fidelity | **3984/3984 = 100%** |
| V2 negative control | 6/6 planted corruptions detected |
| V3 cross-process determinism | 6/6 prefix hashes reproduced |
| V4 Arm A determinism (T=0) | 332 single-suffix branches, no variance |
| V5 Arm B stochasticity | 41/332 (0.12) differ across seeds |
| V6 expert sanity | **18/18 = 1.00** |
| V7 base success band | A **0.274**, B **0.286** (band 0.15–0.60, n=84 each) |
| V8 split leakage | clean, both schemes |
| V9 derivation purity | 664 usable, 0 order-dependent |
| V10 Stage-2 prompt/token round-trip | 30 checked, 0 mismatched |

**664/664 checkpoints replay-verified.** V1 matches the paper's reported 100%
retained match rate. V2 matters as much as V1: a validator that never fires proves
nothing, so a deliberately corrupted prefix must be caught — it was, every time.

Wall-clock 64 minutes. Root disk never moved off its 84 G baseline.

---

## 3. Setup

| | |
| --- | --- |
| Benchmark | ALFWorld `AlfredTWEnv`, train split, 42 games × 6 task types |
| Model | Qwen2.5-7B-Instruct, bf16, vLLM 0.28.0 |
| Actions | `continue` · `intervene` (defer-to-expert) · `quit` |
| Utility | paper Table 3: `w=1.0, c_int=0.05, s=0.01` |
| Arm A | T=0.0, 1 suffix/action → realized `U(s,a)` (paper-faithful) |
| Arm B | T=0.3, 5 suffixes/action → estimated `Q̂(s,a)` (extension) |
| Scale | 332 checkpoints/arm, 3,984 branches |

---

## 4. Answers to the two acceptance questions

### Q1 — Did the faithful arm reproduce the phenomenon? **Yes.**

- **Conflict-set mass 100%** — every state sits in a risk band whose oracle action
  is not unanimous.
- **Gap(g) = 0.303** — the value destroyed by routing through continuation risk.
- **Both risk bands straddle zero**, i.e. τ changes sign within a band.
- **Three-way oracle split**: continue 20.8% · intervene 55.1% · quit 24.1%. None
  degenerate.
- **τ = 1.023 ± 1.101**, range [−0.070, 2.400]. Intervention succeeds 0.759 of the
  time against continuation's 0.280.
- **Robust to utility choice**: across the 5×5 cost sweep, conflict mass stays at
  100% in all 25 cells and Gap(g) spans 0.043–0.444. This mirrors the paper's
  25/25 positive cells on ALFWorld.

### Q2 — Are the Monte-Carlo targets good enough for Stage 2? **Qualified yes.**

- **Signal dominates noise**: τ spread ÷ mean MC standard error = **10.29**. N=5 is
  adequate.
- **Arm B buys resolution Arm A cannot have**: 6 populated risk bands versus 2,
  because 5 replicates give graded `p̂_fail` where 1 gives a 0/1 indicator.
- **But the middle is thin.** Only 29 of 332 states fall in the intermediate bands
  (risk 0.2–0.8), and MC SE has median 0.000 — most checkpoints are deterministic.
  Only 41/332 (12%) vary across seeds at all.

The honest reading: Arm B adds a genuinely useful risk axis for Stage 2, but the
expectation it estimates is degenerate on ~88% of states.

---

## 5. The main scientific finding beyond replication

**ALFWorld outcomes are near-deterministic, and this is a property of the
environment, not of our sampling.** Measured directly: of 12 games run 3× each,
6 failed at the step cap on every seed and 3 succeeded in *identical* step counts.
Raising temperature from 0.3 to 0.5 produced *more* trajectory divergence (3/12
differing step counts) but *less* outcome variance (0/12). Game identity pins the
outcome far more tightly than sampling does.

This independently reproduces the paper's App. B.7 observation of zero
within-action variance on repeats — we reached it from the opposite direction,
having set out to *add* variance. It is also why the paper's single-rollout
protocol is sound here rather than a limitation.

**Implication for Stage 2:** deterministic outcomes are an advantage for probe
training — labels are exact, with no Monte-Carlo noise to average away.

---

## 6. Deviations and replication ambiguities

### Deviations (deliberate)
1. **Arm B exists at all.** The paper is single-rollout at T=0. Arm B changes the
   policy and therefore the data-generating distribution; it is reported separately
   and never pooled with Arm A.
2. **ALFWorld only.** The paper's other three benchmarks are out of Stage-1 scope.
3. **No controllers, no regret.** The paper's headline metric requires a *trained*
   witness. Stage 1 stops at values, oracle actions, and a training-free
   abstraction-loss diagnostic.
4. **Extra task-level split** alongside the paper's prefix-count split.
5. **Full histories retained**, versus the paper's truncated serialization.
6. **Expert-only intervention**; the ladder is unauthorized Gate D.

### Replication ambiguities resolved empirically
The paper states the 24/8-token generation budgets and the utility parameters. It
does **not** state the prompt format, shot count, whether admissible commands are
shown, the episode cap, or the prefix-selection rule, and released no code.

Two of our initial guesses were **measured to be wrong and corrected**:

| choice | initial | final | evidence |
| --- | --- | --- | --- |
| show admissible commands | false (ReAct convention) | **true** | 0/6 → 2/6 success; admissible rate 13.7% → 46.1% |
| Arm B temperature | 0.7 (our guess) | **0.3** | 0/6 at 0.7; one bad sample ruins a 50-step trajectory |

The paper-stated 24-token action budget was preserved throughout — this is why we
chose showing admissible commands over switching to ReAct, which would have
required breaking it.

---

## 7. Upstream findings

Four ALFWorld/TextWorld behaviours that materially affect anyone reproducing this:

1. **The handcoded expert is stochastic.** It breaks ties with unseeded
   `random.choice` and escapes loops with a random admissible action. The same game
   took 31 steps once and hit a 50-step cap the next. We seed per branch and give
   the intervention the same replicate count as `continue`. The paper calls this
   intervention an *oracle*, which sits awkwardly with the implementation.
2. **Neither library is thread-safe.** The same job produced 36/36 branches on one
   thread and 3 branches plus 11 errors on four — failing *quietly*. All parallelism
   is process-level sharding.
3. **A non-admissible expert proposal is not terminal.** Upstream falls back to
   `look`, restoring the expert's observation memory. Treating it as failure drops
   expert success from 6/6 to 3/6.
4. **`AlfredExpert` does not catch `HandCodedAgentFailed`**, and hard-codes a
   200-step expert budget that replayed prefix steps consume.

---

## 8. Suitability for Stage 2 and 3

**Suitable.** Specifically:

- **V10 passed against the real tokenizer** — every stored prompt re-tokenises
  exactly, so activations can be extracted later by a forward pass over the logged
  strings without re-running any environment.
- **664 verified checkpoints**, each with the exact rendered prompt, token ids,
  26-D handcrafted features, and full branch outcomes.
- **Non-degenerate targets**: three-way oracle split, τ spanning [−0.07, 2.40].
- **Provenance on every record**: model and tokenizer revision, chat-template hash,
  git SHA, seeds, and the scaffold config.

**Caveat to carry forward:** the intervention is strong (0.76 success), so
`Q(s,intervene)` varies less than `Q(s,continue)`. The oracle leans intervene at
55%. Gate D's intervention ladder — a weaker, non-privileged intervention — is the
natural way to make `Q(s,intervene)` vary more, and is the change I would prioritise
over simply scaling to more checkpoints.

---

## 9. Recommendation

Gate C (scale) is **not** the highest-value next step. The dataset already answers
Q1 decisively at n=332, and more of the same states would sharpen error bars on a
question already settled.

**Gate D (intervention ladder) is the better spend**, for the reason in §8: the
oracle expert succeeds nearly always, so intervention value barely varies with
state. A weaker intervention would restore that variation, which is what a Stage-2
probe most needs to predict.

Both remain unauthorized pending review.
