"""Canonical environment-state signatures.

Replay validation (V1) compares the state reached by re-executing a logged
action sequence against the state recorded during the base rollout. To make that
comparison meaningful we need a *canonical* signature: two states are equal iff
their signatures are equal, independent of incidental ordering.

The signature covers everything the agent or the expert can observe:

* ``obs`` -- the feedback text
* ``admissible_commands`` -- sorted, because TextWorld does not promise a stable
  order and an order flip is not a state difference
* ``reward`` / ``done`` / ``won`` -- the episode-level outcome bits
* ``facts`` -- the PDDL world state, sorted. This is the strongest component:
  observations can coincide while the underlying world differs, so comparing
  facts is what makes V1 more than a string check.

This module is deliberately free of any ALFWorld import so it can be unit-tested
on a machine where the environment stack cannot be installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..storage import stable_hash


@dataclass
class StepResult:
    """One environment observation, plus the expert's read of the same state.

    ``expert_action`` is the ALFWorld handcoded expert's next move from *this*
    state -- the paper's defer-to-expert intervention. It is recorded at every
    step of the base rollout, not just at checkpoints, so the expert's internal
    memory stays in sync with the trajectory and a mid-episode handoff needs no
    cold start.

    ``expert_status`` is ``ok``/``timeout``/``failed``/``error:*``. Expert
    breakdown is an outcome we measure, never an exception that kills a rollout.
    """

    obs: str
    reward: float = 0.0
    done: bool = False
    won: bool = False
    admissible_commands: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    expert_action: Optional[str] = None
    expert_status: str = "ok"

    @property
    def signature(self) -> Dict[str, Any]:
        return state_signature(
            self.obs,
            self.admissible_commands,
            self.reward,
            self.done,
            self.won,
            self.facts,
        )

    @property
    def hash(self) -> str:
        return signature_hash(self.signature)


def canonical_facts(facts: Optional[Sequence[Any]]) -> List[str]:
    """Normalise TextWorld ``Proposition`` objects (or strings) to sorted text.

    Mirrors the rendering the ALFWorld handcoded expert itself uses --
    ``"{fact.name} " + " ".join(fact.names)`` -- so our notion of world state
    matches the one the expert plans against.
    """
    if not facts:
        return []
    out: List[str] = []
    for f in facts:
        if isinstance(f, str):
            out.append(f.strip())
            continue
        name = getattr(f, "name", None)
        names = getattr(f, "names", None)
        if name is not None and names is not None:
            out.append("{} {}".format(name, " ".join(str(n).strip() for n in names)))
        else:
            out.append(str(f).strip())
    return sorted(out)


def state_signature(
    obs: str,
    admissible_commands: Optional[Sequence[str]] = None,
    reward: float = 0.0,
    done: bool = False,
    won: bool = False,
    facts: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    return {
        "obs": (obs or "").strip(),
        "admissible_commands": sorted(str(c) for c in (admissible_commands or [])),
        "reward": float(reward),
        "done": bool(done),
        "won": bool(won),
        "facts": canonical_facts(facts),
    }


def signature_hash(signature: Mapping[str, Any]) -> str:
    return stable_hash(signature, length=32)


def state_hash(
    obs: str,
    admissible_commands: Optional[Sequence[str]] = None,
    reward: float = 0.0,
    done: bool = False,
    won: bool = False,
    facts: Optional[Sequence[Any]] = None,
) -> str:
    return signature_hash(
        state_signature(obs, admissible_commands, reward, done, won, facts)
    )


def diff_signatures(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> Dict[str, Any]:
    """Field-level diff, for reporting *why* a replay mismatched.

    Returns an empty dict when the signatures agree. A validator that only says
    "mismatch" is hard to act on; this says which component moved.
    """
    diffs: Dict[str, Any] = {}
    for key in ("obs", "reward", "done", "won"):
        if expected.get(key) != actual.get(key):
            diffs[key] = {"expected": expected.get(key), "actual": actual.get(key)}
    for key in ("admissible_commands", "facts"):
        e = list(expected.get(key) or [])
        a = list(actual.get(key) or [])
        if e != a:
            diffs[key] = {
                "only_expected": sorted(set(e) - set(a))[:10],
                "only_actual": sorted(set(a) - set(e))[:10],
                "n_expected": len(e),
                "n_actual": len(a),
            }
    return diffs
