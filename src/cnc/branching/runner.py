"""Prefix-branching orchestrator: rollout -> checkpoint -> branch -> persist.

This implements the paper's protocol (Section 3, App. B.1):

1. Run a base trajectory and record it in full.
2. Select decision prefixes.
3. For each prefix, **replay into the exact state the base rollout reached**,
   verifying step by step and discarding any mismatch.
4. From that verified state, execute every candidate action -- ``continue``,
   ``intervene`` (defer-to-expert), ``quit``.

Two invariants are enforced rather than assumed:

* A branch that fails replay is recorded with ``replay_verified=False`` and is
  **never executed**. Rejections are counted, not hidden -- the paper's 100%
  match rate describes what survives this filter.
* The continue branch's first prompt must equal the checkpoint's stored prompt.
  If prompt construction and replay ever disagree, that assertion catches it
  immediately instead of silently poisoning the dataset.

Branch transcripts store actions and observations but not per-step rendered
prompts: prompt rendering is a pure function of the prefix, so they are exactly
reconstructible, and storing them would multiply dataset size for no information.
The *checkpoint* prompt is stored verbatim, because that is the one Stage 2
feeds through a forward pass.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..agent.policy import ActPolicy
from ..agent.prompts import ScaffoldConfig
from ..env.replay import ReplayValidator
from ..env.state import state_signature
from ..features import prefix as prefix_features
from ..interventions.expert import run_expert_branch
from ..storage import (
    ARM_MONTECARLO,
    ARM_REPLICATION,
    TERMINAL_GOAL,
    TERMINAL_PARSE_ERROR,
    TERMINAL_QUIT,
    TERMINAL_REPLAY_MISMATCH,
    TERMINAL_STEP_CAP,
    branch_record,
    checkpoint_record,
    derive_seed,
    env_state_record,
    rollout_record,
    step_record,
)
from ..utility import CONTINUE, INTERVENE, QUIT
from .checkpoints import select_checkpoint_indices


@dataclass
class RunnerConfig:
    """One arm's configuration.

    Arm A is ``temperature=0.0, n_replicates=1`` -- the paper's protocol,
    producing a realized ``U(s, a)``. Arm B is ``temperature=0.7,
    n_replicates=5`` -- the Monte-Carlo extension, producing ``Qhat(s, a)``.
    """

    arm: str = ARM_REPLICATION
    temperature: float = 0.0
    n_replicates: int = 1
    max_episode_steps: int = 50
    k_checkpoints: int = 4
    expert_max_steps: int = 1000
    elicit_confidence: bool = True
    scaffold: ScaffoldConfig = field(default_factory=ScaffoldConfig)

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "arm": self.arm,
            "temperature": self.temperature,
            "n_replicates": self.n_replicates,
            "max_episode_steps": self.max_episode_steps,
            "k_checkpoints": self.k_checkpoints,
            "expert_max_steps": self.expert_max_steps,
            "elicit_confidence": self.elicit_confidence,
        }
        d["scaffold"] = self.scaffold.as_dict()
        return d


ARM_A = RunnerConfig(arm=ARM_REPLICATION, temperature=0.0, n_replicates=1)
ARM_B = RunnerConfig(arm=ARM_MONTECARLO, temperature=0.7, n_replicates=5)

# An env factory: () -> AlfworldEnv, freshly reset-able. Injected so the runner
# stays testable without ALFWorld installed.
EnvFactory = Callable[[], Any]


# --------------------------------------------------------------------------
# Base rollouts
# --------------------------------------------------------------------------


def run_rollout(
    env: Any,
    task: Dict[str, Any],
    policy: ActPolicy,
    cfg: RunnerConfig,
    rollout_id: str,
    seed: int,
) -> Dict[str, Any]:
    """Play one base trajectory to termination or the step cap."""
    result = env.reset()
    initial_obs = result.obs
    initial = env_state_record(result)

    steps: List[Dict[str, Any]] = []
    history: List[Tuple[str, str]] = []
    prompt_tokens = 0
    completion_tokens = 0
    terminal = TERMINAL_STEP_CAP
    success = False

    for t in range(1, cfg.max_episode_steps + 1):
        step_seed = derive_seed(rollout_id, "base", t, seed)
        pstep = policy.act(
            task["task_type"],
            initial_obs,
            history,
            temperature=cfg.temperature,
            seed=step_seed,
            admissible_commands=result.admissible_commands,
        )
        prompt_tokens += pstep.prompt_tokens
        completion_tokens += pstep.completion_tokens

        if pstep.parse_failed:
            # An unparseable generation ends the episode. It is a real failure
            # mode of a 24-token budget, recorded as such rather than retried.
            steps.append(
                step_record(
                    t=t,
                    rendered_prompt=pstep.rendered_prompt,
                    prompt_token_ids=pstep.prompt_token_ids,
                    raw_completion=pstep.raw_completion,
                    action="",
                    env_state=env_state_record(result),
                    confidence_raw=pstep.confidence_raw,
                    confidence=pstep.confidence,
                    nothing_happens=False,
                )
            )
            terminal = TERMINAL_PARSE_ERROR
            break

        result = env.step(pstep.action)
        history.append((pstep.action, result.obs))
        steps.append(
            step_record(
                t=t,
                rendered_prompt=pstep.rendered_prompt,
                prompt_token_ids=pstep.prompt_token_ids,
                raw_completion=pstep.raw_completion,
                action=pstep.action,
                env_state=env_state_record(result),
                confidence_raw=pstep.confidence_raw,
                confidence=pstep.confidence,
                nothing_happens="nothing happens" in result.obs.lower(),
            )
        )

        if result.done:
            success = bool(result.won)
            terminal = TERMINAL_GOAL if success else TERMINAL_STEP_CAP
            break

    return rollout_record(
        rollout_id=rollout_id,
        task_id=task["task_id"],
        arm=cfg.arm,
        policy_seed=seed,
        temperature=cfg.temperature,
        initial=initial,
        steps=steps,
        success=success,
        n_steps=len(steps),
        terminal_reason=terminal,
        tokens={"prompt": prompt_tokens, "completion": completion_tokens},
    )


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------


def _signature_of(state: Dict[str, Any]) -> Dict[str, Any]:
    return state_signature(
        state["obs"],
        state.get("admissible_commands"),
        state.get("reward", 0.0),
        state.get("done", False),
        state.get("won", False),
        state.get("facts"),
    )


def expected_signatures(rollout: Dict[str, Any], t: int) -> List[Dict[str, Any]]:
    """The ``t + 1`` signatures a prefix of ``t`` actions must reproduce."""
    sigs = [_signature_of(rollout["initial"])]
    for step in rollout["steps"][:t]:
        sigs.append(_signature_of(step))
    return sigs


def prefix_actions(rollout: Dict[str, Any], t: int) -> List[str]:
    return [s["action"] for s in rollout["steps"][:t]]


def prefix_history(rollout: Dict[str, Any], t: int) -> List[Tuple[str, str]]:
    """The ``(action, observation)`` pairs the policy conditions on at ``s_t``."""
    return [(s["action"], s["obs"]) for s in rollout["steps"][:t]]


def build_checkpoints(
    rollout: Dict[str, Any],
    task: Dict[str, Any],
    policy: ActPolicy,
    cfg: RunnerConfig,
) -> List[Dict[str, Any]]:
    """Select prefixes from a rollout and materialise checkpoint records.

    The stored ``rendered_prompt`` is produced here without a model call --
    rendering is pure -- so checkpointing costs nothing beyond CPU.
    """
    n_steps = rollout["n_steps"]
    indices = select_checkpoint_indices(
        n_steps, seed=derive_seed(rollout["rollout_id"], "cp"), k=cfg.k_checkpoints
    )

    initial_obs = rollout["initial"]["obs"]
    out: List[Dict[str, Any]] = []
    for t in indices:
        step = rollout["steps"][t - 1]  # state s_t follows the t-th action
        history = prefix_history(rollout, t)
        prompt = policy.render(
            task["task_type"], initial_obs, history, step.get("admissible_commands")
        )

        feats = prefix_features.extract(
            step_index=t,
            history="".join(a + o for a, o in history),
            task_desc=initial_obs,
            observation=step["obs"],
            response=step.get("raw_completion", ""),
            admissible_commands=step.get("admissible_commands") or [],
            action=step["action"],
            task_type=task["task_type"],
            confidence=step.get("confidence"),
        )

        out.append(
            checkpoint_record(
                checkpoint_id="{}::t{}".format(rollout["rollout_id"], t),
                rollout_id=rollout["rollout_id"],
                task_id=task["task_id"],
                arm=cfg.arm,
                t=t,
                prefix_state_hash=step["state_hash"],
                rendered_prompt=prompt,
                prompt_token_ids=policy.token_ids(prompt),
                features_26d=feats,
                confidence_t=step.get("confidence"),
                admissible_commands=step.get("admissible_commands") or [],
                prefix_actions=prefix_actions(rollout, t),
                steps_remaining=max(0, cfg.max_episode_steps - t),
            )
        )
    return out


# --------------------------------------------------------------------------
# Branches
# --------------------------------------------------------------------------


def run_continue_branch(
    env: Any,
    policy: ActPolicy,
    task_type: str,
    initial_obs: str,
    history: Sequence[Tuple[str, str]],
    temperature: float,
    seed: int,
    max_steps: int,
    branch_id: str,
    expected_first_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Let the agent keep acting from the replayed checkpoint state."""
    hist: List[Tuple[str, str]] = list(history)
    result = env.last_result
    transcript: List[Dict[str, Any]] = []
    steps = 0
    success = False
    terminal = TERMINAL_STEP_CAP
    prompt_tokens = 0
    completion_tokens = 0

    while steps < max_steps:
        step_seed = derive_seed(branch_id, steps, seed)
        pstep = policy.act(
            task_type,
            initial_obs,
            hist,
            temperature=temperature,
            seed=step_seed,
            admissible_commands=result.admissible_commands,
        )
        prompt_tokens += pstep.prompt_tokens
        completion_tokens += pstep.completion_tokens

        if steps == 0 and expected_first_prompt is not None:
            # If replay and prompt construction ever disagree, the branch is not
            # starting from the state the checkpoint claims. Fail loudly.
            assert pstep.rendered_prompt == expected_first_prompt, (
                "continue branch {} does not resume from the checkpoint prompt".format(
                    branch_id
                )
            )

        if pstep.parse_failed:
            terminal = TERMINAL_PARSE_ERROR
            break

        result = env.step(pstep.action)
        steps += 1
        hist.append((pstep.action, result.obs))
        transcript.append(
            {
                "t": steps,
                "action": pstep.action,
                "raw_completion": pstep.raw_completion,
                "obs": result.obs,
                "reward": result.reward,
                "done": result.done,
                "won": result.won,
                "confidence": pstep.confidence,
                "source": "policy",
            }
        )

        if result.done:
            success = bool(result.won)
            terminal = TERMINAL_GOAL if success else TERMINAL_STEP_CAP
            break

    return {
        "success": success,
        "n_steps": steps,
        "terminal_reason": terminal,
        "transcript": transcript,
        "tokens": {"prompt": prompt_tokens, "completion": completion_tokens},
    }


def run_checkpoint_branches(
    checkpoint: Dict[str, Any],
    rollout: Dict[str, Any],
    task: Dict[str, Any],
    policy: ActPolicy,
    cfg: RunnerConfig,
    env_factory: EnvFactory,
    validator: Optional[ReplayValidator] = None,
) -> List[Dict[str, Any]]:
    """Execute every candidate action from one verified checkpoint state.

    Returns raw branch records. ``quit`` is analytic and consumes neither an env
    nor a model call; ``intervene`` consumes an env but no model calls.
    """
    validator = validator or ReplayValidator()
    t = checkpoint["t"]
    cid = checkpoint["checkpoint_id"]
    actions = prefix_actions(rollout, t)
    sigs = expected_signatures(rollout, t)
    history = prefix_history(rollout, t)
    initial_obs = rollout["initial"]["obs"]
    budget = checkpoint["steps_remaining"]

    records: List[Dict[str, Any]] = []

    # -- continue and intervene: n_replicates independent suffixes each -----
    #
    # The expert gets replicates too, because it is NOT deterministic. ALFWorld's
    # HandCodedAgent breaks ties with `random.choice(objs_of_interest)` and, when
    # it would repeat itself or proposes something inadmissible after a "look",
    # escapes the loop with `random.choice(admissible_commands)` -- an unseeded
    # random walk. Treating the intervention as a single deterministic suffix
    # would silently turn a Monte-Carlo estimate into one draw.
    for action_kind in (CONTINUE, INTERVENE):
        for rep in range(cfg.n_replicates):
            branch_id = "{}::{}::{}".format(cid, action_kind, rep)
            seed = derive_seed(cid, cfg.arm, action_kind, rep)

            # The expert reads the *global* `random`, and it draws during prefix
            # replay as well as during the branch, so seeding must happen before
            # the replay for the whole branch to be reproducible in isolation.
            random.seed(seed)

            env = env_factory()
            try:
                replay = validator.replay(env, actions, sigs)
                if not replay.verified:
                    records.append(
                        _rejected(branch_id, cid, cfg.arm, action_kind, rep, seed, replay)
                    )
                    continue

                if action_kind == CONTINUE:
                    outcome = run_continue_branch(
                        env, policy, task["task_type"], initial_obs, history,
                        temperature=cfg.temperature, seed=seed, max_steps=budget,
                        branch_id=branch_id,
                        expected_first_prompt=checkpoint["rendered_prompt"],
                    )
                else:
                    outcome = run_expert_branch(env, max_steps=budget)

                records.append(
                    branch_record(
                        branch_id=branch_id,
                        checkpoint_id=cid,
                        arm=cfg.arm,
                        action=action_kind,
                        replicate=rep,
                        branch_seed=seed,
                        replay_verified=True,
                        success=outcome["success"],
                        n_steps_in_branch=outcome["n_steps"],
                        terminal_reason=outcome["terminal_reason"],
                        transcript=outcome["transcript"],
                        tokens=outcome["tokens"],
                    )
                )
            finally:
                env.close()

    # -- quit: analytic, U == 0 -------------------------------------------
    records.append(
        branch_record(
            branch_id="{}::{}::0".format(cid, QUIT),
            checkpoint_id=cid,
            arm=cfg.arm,
            action=QUIT,
            replicate=0,
            branch_seed=derive_seed(cid, cfg.arm, QUIT, 0),
            replay_verified=True,
            success=False,
            n_steps_in_branch=0,
            terminal_reason=TERMINAL_QUIT,
            transcript=[],
        )
    )
    return records


def _rejected(
    branch_id: str,
    checkpoint_id: str,
    arm: str,
    action: str,
    replicate: int,
    seed: int,
    replay: Any,
) -> Dict[str, Any]:
    """Record a branch whose prefix failed verification. Never executed."""
    return branch_record(
        branch_id=branch_id,
        checkpoint_id=checkpoint_id,
        arm=arm,
        action=action,
        replicate=replicate,
        branch_seed=seed,
        replay_verified=False,
        success=False,
        n_steps_in_branch=0,
        terminal_reason=TERMINAL_REPLAY_MISMATCH,
        transcript=[],
        replay_mismatch=replay.mismatch,
    )


def usable_checkpoints(branches: Sequence[Dict[str, Any]]) -> List[str]:
    """Checkpoint ids where **every** branch replayed cleanly.

    A checkpoint with a partially verified action set cannot support a same-prefix
    comparison, so it is dropped whole rather than compared against a missing arm.
    """
    by_cp: Dict[str, bool] = {}
    for b in branches:
        cid = b["checkpoint_id"]
        by_cp[cid] = by_cp.get(cid, True) and bool(b["replay_verified"])
    return sorted(cid for cid, ok in by_cp.items() if ok)
