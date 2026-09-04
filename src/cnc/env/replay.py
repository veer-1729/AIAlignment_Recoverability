"""Prefix replay and step-wise verification.

This is the integrity core of the whole protocol. Every counterfactual branch
claims to start from "the same state the base rollout reached at step t". That
claim is only worth something if it is *checked*, so we re-execute the logged
action sequence from ``reset()`` and compare the full state signature -- feedback
text, sorted admissible commands, reward, done, won, and the sorted PDDL facts --
at every single step, not just the last one.

Checking only the final state would let a compensating pair of divergences slip
through; checking every step means the first disagreement is caught where it
happens and reported with a field-level diff.

Any mismatch disqualifies the checkpoint. The paper reports a 100% match rate on
retained data, which is a statement about what survives this filter, not a
guarantee that nothing ever diverges -- so we count and report rejections rather
than assuming there will be none.

This module is free of ALFWorld imports: it drives anything exposing
``reset()``/``step()`` returning a :class:`~cnc.env.state.StepResult`, which is
what makes it unit-testable on a machine that cannot install the env stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .state import StepResult, diff_signatures


@dataclass
class ReplayResult:
    """Outcome of replaying a prefix.

    ``verified`` False means the checkpoint must be discarded. ``mismatch``
    always says *where* (step index) and *what* (field-level diff).
    """

    verified: bool
    n_steps_replayed: int
    final: Optional[StepResult] = None
    mismatch: Optional[Dict[str, Any]] = None
    signatures: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        """Fraction of compared steps that agreed before the first divergence."""
        total = len(self.signatures)
        if total == 0:
            return 1.0
        if self.verified:
            return 1.0
        idx = (self.mismatch or {}).get("step_index", 0)
        return float(idx) / float(total)


class ReplayValidator:
    """Replays a logged prefix and verifies it step by step."""

    def __init__(self, compare_facts: bool = True):
        # Facts comparison can be disabled only for environments that do not
        # expose them; for ALFWorld it is always on, and it is what makes this
        # check stronger than an observation-string comparison.
        self.compare_facts = compare_facts

    def _compare(
        self, expected: Mapping[str, Any], actual: Mapping[str, Any]
    ) -> Dict[str, Any]:
        exp = dict(expected)
        act = dict(actual)
        if not self.compare_facts:
            exp.pop("facts", None)
            act.pop("facts", None)
        return diff_signatures(exp, act)

    def replay(
        self,
        env: Any,
        prefix_actions: Sequence[str],
        expected_signatures: Sequence[Mapping[str, Any]],
    ) -> ReplayResult:
        """Re-execute ``prefix_actions`` and verify against recorded signatures.

        ``expected_signatures[i]`` is the state after ``i`` actions, so index 0 is
        the post-reset state and there must be exactly ``len(prefix_actions) + 1``
        of them. A length mismatch is a bug in the caller, not a replay failure,
        so it raises rather than returning ``verified=False``.
        """
        if len(expected_signatures) != len(prefix_actions) + 1:
            raise ValueError(
                "expected {} signatures for {} actions, got {}".format(
                    len(prefix_actions) + 1, len(prefix_actions), len(expected_signatures)
                )
            )

        seen: List[Dict[str, Any]] = []

        result = env.reset()
        seen.append(result.signature)
        diff = self._compare(expected_signatures[0], result.signature)
        if diff:
            return ReplayResult(
                verified=False,
                n_steps_replayed=0,
                final=result,
                mismatch={"step_index": 0, "action": None, "diff": diff},
                signatures=seen,
            )

        for i, action in enumerate(prefix_actions):
            result = env.step(action)
            seen.append(result.signature)
            diff = self._compare(expected_signatures[i + 1], result.signature)
            if diff:
                return ReplayResult(
                    verified=False,
                    n_steps_replayed=i + 1,
                    final=result,
                    mismatch={"step_index": i + 1, "action": action, "diff": diff},
                    signatures=seen,
                )

        return ReplayResult(
            verified=True,
            n_steps_replayed=len(prefix_actions),
            final=result,
            mismatch=None,
            signatures=seen,
        )


def signatures_from_steps(
    initial_signature: Mapping[str, Any],
    step_signatures: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Assemble the expected-signature list from a stored rollout."""
    return [dict(initial_signature)] + [dict(s) for s in step_signatures]
