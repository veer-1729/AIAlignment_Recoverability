"""Utility, value, and oracle-action derivation.

Nothing in the persisted branch records carries a utility. Utilities are always
*derived* from raw outcomes (success flag + branch length) under a named
``UtilityParams`` set, so the paper's 5x5 cost sweep and any reparameterisation
are pure re-derivations rather than new rollouts.

Terminology is enforced strictly (see plan section 3):

* Arm A (temperature 0, one suffix per action) yields a **realized utility**
  ``U(s, a)`` -- a single sample, not an expectation.
* Arm B (stochastic policy, N suffixes per action) estimates
  ``Qhat(s, a) = E[U | s, a]`` with a Monte-Carlo standard error.

The two are never conflated: ``realized_utility`` returns ``U`` for one branch,
``aggregate_action`` returns an ``ActionValue`` whose ``n`` records how many
suffixes went into it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

CONTINUE = "continue"
INTERVENE = "intervene"
QUIT = "quit"
ACTIONS = (CONTINUE, INTERVENE, QUIT)


@dataclass(frozen=True)
class UtilityParams:
    """Paper Table 3, ALFWorld row.

    ``U_a(h) = u(h, a) - c_a - s * m(h, a)`` for ``a != quit``, and
    ``U_quit(h) = 0``, where ``u`` is ``+1`` on success and ``-w`` on failure,
    ``m`` is the branch length in steps, ``c_a`` is a fixed per-action cost and
    ``s`` is the per-step cost.
    """

    wrong_penalty: float = 1.0  # w
    intervention_cost: float = 0.05  # c_intervene
    continue_cost: float = 0.0  # c_continue -- paper lists a single c_a for the intervention
    step_cost: float = 0.01  # s
    name: str = "paper_table3"

    def cost_of(self, action: str) -> float:
        if action == QUIT:
            return 0.0
        if action == INTERVENE:
            return self.intervention_cost
        if action == CONTINUE:
            return self.continue_cost
        raise ValueError("unknown action: {!r}".format(action))

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


PAPER_PARAMS = UtilityParams()


def realized_utility(
    action: str,
    success: bool,
    n_steps: int,
    params: UtilityParams = PAPER_PARAMS,
) -> float:
    """``U(s, a)`` for a single realized branch.

    ``quit`` is analytic and always exactly zero -- it consumes no steps and is
    never rolled out.
    """
    if action not in ACTIONS:
        raise ValueError("unknown action: {!r}".format(action))
    if action == QUIT:
        return 0.0
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative, got {}".format(n_steps))
    outcome = 1.0 if success else -params.wrong_penalty
    return outcome - params.cost_of(action) - params.step_cost * float(n_steps)


@dataclass(frozen=True)
class ActionValue:
    """Value of one action at one checkpoint, plus its Monte-Carlo uncertainty.

    ``n == 1`` means this is a realized ``U(s, a)`` (Arm A) and ``se`` is 0.0 by
    construction -- a single sample carries no variance information. ``n > 1``
    means this is ``Qhat(s, a)`` (Arm B).
    """

    action: str
    value: float
    n: int
    se: float
    p_success: float
    mean_steps: float

    @property
    def is_estimate(self) -> bool:
        """True when this is Qhat (Arm B), False when it is a realized U (Arm A)."""
        return self.n > 1


def aggregate_action(
    action: str,
    successes: Sequence[bool],
    step_counts: Sequence[int],
    params: UtilityParams = PAPER_PARAMS,
) -> ActionValue:
    """Aggregate one action's replicates into an ``ActionValue``.

    For ``quit`` the caller may pass empty sequences: the branch is analytic.
    """
    if action == QUIT:
        return ActionValue(action=QUIT, value=0.0, n=1, se=0.0, p_success=0.0, mean_steps=0.0)

    if len(successes) != len(step_counts):
        raise ValueError(
            "successes/step_counts length mismatch: {} vs {}".format(len(successes), len(step_counts))
        )
    n = len(successes)
    if n == 0:
        raise ValueError("no replicates for action {!r}".format(action))

    utilities = [
        realized_utility(action, s, m, params) for s, m in zip(successes, step_counts)
    ]
    mean = sum(utilities) / n
    if n > 1:
        var = sum((u - mean) ** 2 for u in utilities) / (n - 1)  # unbiased
        se = math.sqrt(var / n)
    else:
        se = 0.0
    return ActionValue(
        action=action,
        value=mean,
        n=n,
        se=se,
        p_success=sum(1.0 for s in successes if s) / n,
        mean_steps=sum(step_counts) / float(n),
    )


@dataclass(frozen=True)
class CheckpointValues:
    """All three action values at one checkpoint, plus the derived quantities."""

    checkpoint_id: str
    arm: str
    params_name: str
    values: Mapping[str, ActionValue]

    @property
    def tau(self) -> float:
        """Intervention advantage. ``U_I - U_C`` in Arm A, ``Qhat_I - Qhat_C`` in Arm B."""
        return self.values[INTERVENE].value - self.values[CONTINUE].value

    @property
    def tau_se(self) -> float:
        """SE of tau under independent branches. Exactly 0.0 in Arm A (n=1)."""
        a = self.values[INTERVENE].se
        b = self.values[CONTINUE].se
        return math.sqrt(a * a + b * b)

    @property
    def oracle_action(self) -> str:
        """argmax over the three actions.

        Ties are broken deterministically in ACTIONS order so the label is a pure
        function of the data, never of dict iteration order.
        """
        best = None
        best_value = -math.inf
        for action in ACTIONS:
            v = self.values[action].value
            if v > best_value:
                best_value = v
                best = action
        assert best is not None
        return best

    @property
    def oracle_value(self) -> float:
        return self.values[self.oracle_action].value

    @property
    def p_fail_continue(self) -> float:
        """The empirical continuation-risk scalar ``g``.

        This is the *only* scalar Stage 1 uses, and it is measured, never
        learned -- keeping the analysis on the correct side of the no-training
        boundary. In Arm A it is a 0/1 realized failure indicator; in Arm B it is
        ``1 - phat_continue``, an oracle-quality risk estimate.
        """
        return 1.0 - self.values[CONTINUE].p_success

    def regret_of(self, action: str) -> float:
        """Loss from taking ``action`` instead of the same-prefix oracle action."""
        return self.oracle_value - self.values[action].value

    def to_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "checkpoint_id": self.checkpoint_id,
            "arm": self.arm,
            "utility_params": self.params_name,
            "tau": self.tau,
            "tau_se": self.tau_se,
            "oracle_action": self.oracle_action,
            "oracle_value": self.oracle_value,
            "p_fail_continue": self.p_fail_continue,
        }
        for action in ACTIONS:
            v = self.values[action]
            row["value_{}".format(action)] = v.value
            row["se_{}".format(action)] = v.se
            row["n_{}".format(action)] = v.n
            row["p_success_{}".format(action)] = v.p_success
            row["mean_steps_{}".format(action)] = v.mean_steps
        return row


def values_from_branches(
    checkpoint_id: str,
    arm: str,
    branches: Iterable[Mapping[str, object]],
    params: UtilityParams = PAPER_PARAMS,
) -> CheckpointValues:
    """Build a ``CheckpointValues`` from raw branch records for one checkpoint.

    ``branches`` are raw rows as persisted by ``storage`` -- each carrying at
    least ``action``, ``success`` and ``n_steps_in_branch``. Only branches whose
    prefix replay was verified are eligible; the caller is responsible for
    filtering, and we assert rather than silently including unverified rows.
    """
    grouped: Dict[str, List[Mapping[str, object]]] = {a: [] for a in ACTIONS}
    for b in branches:
        action = str(b["action"])
        if action not in grouped:
            raise ValueError("unknown action in branch record: {!r}".format(action))
        if not b.get("replay_verified", False) and action != QUIT:
            raise ValueError(
                "unverified branch {!r} reached value derivation; filter first".format(
                    b.get("branch_id")
                )
            )
        grouped[action].append(b)

    values: Dict[str, ActionValue] = {}
    for action in ACTIONS:
        rows = grouped[action]
        if action == QUIT:
            values[QUIT] = aggregate_action(QUIT, [], [], params)
            continue
        if not rows:
            raise ValueError(
                "checkpoint {!r} has no {} branches".format(checkpoint_id, action)
            )
        # Sort by replicate so the aggregate is order-independent and reproducible.
        rows = sorted(rows, key=lambda r: int(r.get("replicate", 0)))
        values[action] = aggregate_action(
            action,
            [bool(r["success"]) for r in rows],
            [int(r["n_steps_in_branch"]) for r in rows],
            params,
        )

    return CheckpointValues(
        checkpoint_id=checkpoint_id,
        arm=arm,
        params_name=params.name,
        values=values,
    )


def cost_sweep_grid(
    intervention_costs: Optional[Sequence[float]] = None,
    wrong_penalties: Optional[Sequence[float]] = None,
) -> List[UtilityParams]:
    """The paper's 5x5 utility sensitivity grid (Table 6), centred on Table 3."""
    if intervention_costs is None:
        intervention_costs = (0.0, 0.05, 0.10, 0.15, 0.20)
    if wrong_penalties is None:
        wrong_penalties = (0.0, 0.5, 1.0, 1.5, 2.0)
    grid = []
    for c in intervention_costs:
        for w in wrong_penalties:
            grid.append(
                UtilityParams(
                    wrong_penalty=w,
                    intervention_cost=c,
                    name="c{:.2f}_w{:.2f}".format(c, w),
                )
            )
    return grid
