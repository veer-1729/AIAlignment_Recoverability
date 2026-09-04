# Stage 1 — Prefix-Branching Substrate for Runtime Control of LLM Agents

Reproduction of the data-generation and evaluation substrate behind
*Calibration Is Not Control: Why LLM-Agent Oversight Needs Intervention*
([arXiv:2606.21399](https://arxiv.org/abs/2606.21399)), on ALFWorld with
Qwen2.5-7B-Instruct.

**Stage 1 produces a dataset, not a model.** Nothing here trains anything — no
random forests, no probes, no RL. That is Stage 2.

## What the protocol does

For each base trajectory we select decision prefixes, **replay** each one back
into the exact environment state the rollout reached (verifying step by step and
discarding any mismatch), and then execute every candidate control action from
that verified state:

| action | meaning | cost |
| --- | --- | --- |
| `continue` | the agent keeps acting | `n_replicates` × suffix of LLM calls |
| `intervene` | ALFWorld's handcoded expert takes over | **zero LLM calls** |
| `quit` | stop now | analytic, `U = 0` |

From the raw outcomes we derive `U(s,a)` (Arm A) or `Q̂(s,a)` (Arm B),
intervention advantage `τ`, and the same-prefix oracle action.

## Two arms, never pooled

| | Arm A `A_replication` | Arm B `B_montecarlo` |
| --- | --- | --- |
| temperature | 0.0 | 0.7 |
| suffixes per action | 1 | 5 |
| quantity | **realized `U(s,a)`** | **estimate `Q̂(s,a) = E[U|s,a]`** |
| role | paper-faithful replication | extension for Stage-2 targets |

Arm B changes the policy and therefore the data-generating distribution, so each
arm has its own base rollouts over the same game set. They are linkable by
`task_id` and kept analytically separate by the `arm` field on every record.

## Layout

```
src/cnc/
  env/factory.py      ALFWorld env + SafeAlfredExpert  (only module importing alfworld)
  env/state.py        canonical state signatures (obs + admissible + reward/done/won + PDDL facts)
  env/replay.py       ReplayValidator — step-wise prefix verification
  agent/prompts.py    scaffold + chat templates + action parsing
  agent/policy.py     act-only policy (24-token actions, 8-token confidence)
  agent/confidence.py self-reported confidence elicitation
  serving/client.py   provider-agnostic OpenAI-compatible client (+ offline mocks)
  interventions/expert.py  defer-to-expert
  branching/          checkpoint selection + the orchestrator
  features/prefix.py  the paper's 26-D prefix feature vector
  utility.py          U / Q̂ / τ / oracle action, derived from raw outcomes
  storage.py          schemas, provenance, sharded JSONL
scripts/01..05        rollouts → checkpoints → branches → validate → analyze
```

Raw records never contain a utility. Utilities are always re-derived under a
named `UtilityParams` set, which is what makes the paper's 5×5 cost sweep a pure
re-derivation rather than new rollouts.

## Running

Everything runs in a `linux/amd64` container — `textworld` publishes x86_64
wheels only, and its `jericho` / `fast-downward-textworld` dependencies build
from source.

```bash
docker build --platform linux/amd64 -f docker/Dockerfile -t cnc-alfworld:0.1 .

# one-time: ~2.2 GB of game files
docker run --rm --platform linux/amd64 -v "$PWD/data/alfworld_cache:/data" \
  -e ALFWORLD_DATA=/data/alfworld cnc-alfworld:0.1 alfworld-download

# full pipeline (both arms, 4 shards)
docker run --rm --platform linux/amd64 \
  -v "$PWD/data/alfworld_cache:/data" -v "$PWD/src:/app/src" \
  -v "$PWD/scripts:/app/scripts" -v "$PWD/configs:/app/configs" \
  -v "$PWD/data/runs:/app/data/runs" -v "$PWD/reports:/app/reports" \
  -e ALFWORLD_DATA=/data/alfworld cnc-alfworld:0.1 \
  bash scripts/run_pipeline.sh configs/dryrun.yaml 4
```

`configs/dryrun.yaml` needs no GPU and no network: a scripted client answers
`look` every turn. It exercises every code path end to end, and says nothing
about agent behaviour. `configs/pilot.yaml` is Gate A and needs a served
Qwen2.5-7B-Instruct endpoint.

Unit tests run natively on any machine (no ALFWorld needed):

```bash
python -m pytest tests/ -q     # 86 tests
```

## Parallelism: processes, not threads

ALFWorld and TextWorld carry module-level global state and are **not
thread-safe**. Measured on the same 12-checkpoint job:

| workers | branches | errors |
| --- | --- | --- |
| 1 thread | 36 / 36 | 0 |
| 4 threads | 3 | 11 (`pop from empty list`, `'NoneType' object is not iterable`) |

Those errors are caught per-checkpoint, so threading silently *shrinks* the
dataset instead of crashing. All parallelism is therefore process-level
sharding (`--shard i --num-shards N`), each shard writing its own JSONL file.

## Validation (V1–V10)

`scripts/04_validate.py` prints a pass/fail table and exits non-zero on failure.
Latest dry-run: **6 PASS / 0 FAIL / 5 SKIP** (every SKIP requires a real model).

Measured on real ALFWorld data:

- **V1 replay fidelity — 144/144 (100%)**, matching the paper's reported match rate.
- **V2 negative control — 6/6** planted prefix corruptions detected. Comparing PDDL
  facts, not just observation text, is what gives this teeth: a facts-only drift
  is invisible to a string comparison (`tests/test_replay.py`).
- **V3 cross-process determinism — 6/6** prefix state hashes reproduced.
- **V6 expert sanity — 17/18 (0.94)** over 6 games × 3 seeds; hardest task type is
  `pick_two_obj_and_place`.

## Things upstream that will bite you

1. **The ALFWorld expert is stochastic.** `HandCodedAgent` breaks ties with
   `random.choice(objs_of_interest)` and escapes repeat-action loops with
   `random.choice(admissible_commands)`, both on the unseeded global `random`.
   The same game finished in 31 steps on one run and hit a 50-step cap on
   another. We seed `random` before each branch's replay, and the intervention
   gets the same replicate count as `continue`.
2. **`AlfredExpert` does not catch `HandCodedAgentFailed`** — an expert breakdown
   would kill the rollout. `SafeAlfredExpert` turns it into a recorded outcome.
3. **A non-admissible expert proposal is not a failure.** Upstream falls back to
   `look`, which refreshes the expert's observation memory so it recovers.
   Treating it as terminal drops expert success from 6/6 to 3/6.
4. **`HandCodedTWAgent(max_steps=200)` is hard-coded**, and replayed prefix steps
   spend that budget. Raised to 1000.
5. **TextWorld's env registry is a growing global**, and `make()` reuses the same
   wrapper *instances* — two envs from one registration would share expert
   memory. We register per-env under a lock and drop the entry immediately.
6. **Python 3.9 cannot install the stack**: pip resolves `spacy` to a source
   tarball needing `thinc>=8.3.12`, which requires ≥3.10. The image uses 3.11.

## Attribution

- **ALFWorld** — Shridhar et al. 2020, [alfworld/alfworld](https://github.com/alfworld/alfworld).
  Game files are downloaded at runtime, never redistributed here.
- **ReAct prompts** — `src/cnc/agent/react_alfworld_prompts.json` is vendored verbatim
  from [ysymyth/ReAct](https://github.com/ysymyth/ReAct) (`prompts/alfworld.json`,
  MIT licensed). It is the de-facto standard ALFWorld exemplar set and is used
  because the paper does not publish its own prompts.
- **Paper under replication** — Zhang, Wan, Yu, Wu, Wen, Zhou, Zhao & Tsang,
  *Calibration Is Not Control: Why LLM-Agent Oversight Needs Intervention*,
  [arXiv:2606.21399](https://arxiv.org/abs/2606.21399). No code was released;
  this is a reconstruction from the paper text plus upstream sources.

## Replication ambiguities

The paper fixes the generation budgets (24-token actions, 8-token confidence) and
the utility parameters (`w=1.0, c_a=0.05, s=0.01`), and specifies split *sizes*
(8/4/8, 20/10/20 prefix states, trajectories kept whole). It does **not** specify
the prompt format, shot count, whether admissible commands are shown, the episode
cap, or how prefixes were chosen — and there is no code release. Those follow the
upstream ReAct convention (`ysymyth/ReAct`, `prompts/alfworld.json`), are
configurable, and are recorded in every file's provenance header. See
`configs/*.yaml` and `src/cnc/agent/prompts.py`.
