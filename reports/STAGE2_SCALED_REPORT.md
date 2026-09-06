# Runtime control of long-horizon LLM agents — scaled Stage 1 substrate and Stage 2 probe study

**Run:** `scale` · ALFWorld · Qwen2.5-7B-Instruct · Arm B (T=0.3, N=5)
**Dataset:** 252 games · 504 rollouts · **1,971 checkpoints** · 21,681 branch records
**Status:** Stage 1 complete and validated. Stage 2 probe results pending (§7).

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

> **Pending.** Two runs in progress: pooled (n = 1,971, primary) and the
> single-cohort robustness run (n = 928). Pre-registered design in
> `scripts/probe_study.py`.

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

**C2 is the arbiter, not C1.** C1 concatenates 27 observable columns with 3,584
activation dimensions under a single ridge alpha; the activation block drags alpha
up, shrinking the standardized Q_C coefficient — the most informative column for τ.
C1 can therefore lose for reasons unrelated to activation content. C2 residualizes
τ on `[observables, Q_C]` with its own alpha and asks whether activations predict
what remains. If they disagree, C2 is the one to believe.

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

Deferred to the final version pending §7.

---

*Repository:* https://github.com/veer-1729/AIAlignment_Recoverability
