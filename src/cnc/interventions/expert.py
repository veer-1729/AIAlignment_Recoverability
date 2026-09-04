"""Defer-to-expert: the paper's ALFWorld intervention.

At a checkpoint the ALFWorld handcoded expert takes over and plays to
termination. Two properties make this the right implementation of the paper's
intervention rather than an approximation of it:

* The expert is **closed-loop**. ``HandCodedTWAgent.act`` reads the live PDDL
  facts and admissible commands and re-plans, so it is not replaying a fixed
  walkthrough that a mid-trajectory handoff would invalidate.
* It **recovers from off-policy states**. ALFWorld's own ``is_solvable`` check
  randomises the first 10 actions (then 15% thereafter) and marks a game
  solvable only if the expert still finishes; every shipped game carries that
  flag. Recovery from an arbitrary prefix is a property upstream already tests.

The expert costs **zero LLM calls**, which is why the expert regime is nearly
free once the continue branches are paid for.

It is not infallible -- upstream's own docstring says "Not guaranteed to succeed
or be optimal". Timeouts and planning failures are recorded as branch outcomes
with a distinguishing ``terminal_reason``, never raised.

**It is also not deterministic.** ``HandCodedAgent`` breaks ties with
``random.choice(objs_of_interest)`` when several objects satisfy a subgoal, and
escapes repeat-action loops with ``random.choice(admissible_commands)`` -- both on
the unseeded global ``random``. Measured consequence: the same
``pick_two_obj_and_place`` game finished in 31 steps on one run and hit a 50-step
cap on another. Callers therefore seed ``random`` before replay (see
``runner.run_checkpoint_branches``) and run the intervention for the same number
of replicates as ``continue``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..env.factory import EXPERT_TIMEOUT
from ..storage import (
    TERMINAL_EXPERT_FAILED,
    TERMINAL_EXPERT_TIMEOUT,
    TERMINAL_GOAL,
    TERMINAL_STEP_CAP,
)


def run_expert_branch(env: Any, max_steps: int) -> Dict[str, Any]:
    """Let the expert play from the env's current state until termination.

    ``env`` must already be replayed to the checkpoint state. Returns a raw
    outcome dict -- success, step count, terminal reason, transcript -- and
    deliberately no utility: utilities are derived later from these fields.
    """
    transcript: List[Dict[str, Any]] = []
    steps = 0
    success = False
    terminal = TERMINAL_STEP_CAP
    result = env.last_result  # state reached by the verified prefix replay

    while steps < max_steps:
        # Checking the expert's move at the top of the loop means a breakdown --
        # timeout, or a proposal the parser would reject -- ends the branch with
        # a distinguishing reason instead of an exception.
        action = result.expert_action
        if not action:
            terminal = (
                TERMINAL_EXPERT_TIMEOUT
                if result.expert_status == EXPERT_TIMEOUT
                else TERMINAL_EXPERT_FAILED
            )
            break

        result = env.step(action)
        steps += 1
        transcript.append(
            {
                "t": steps,
                "action": action,
                "obs": result.obs,
                "reward": result.reward,
                "done": result.done,
                "won": result.won,
                "expert_status": result.expert_status,
                "source": "expert",
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
        "tokens": {},  # zero LLM calls, by construction
    }
