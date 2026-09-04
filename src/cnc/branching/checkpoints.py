"""Checkpoint (decision-prefix) selection.

The paper specifies split *sizes* (App. B.3) but never says how the particular
prefixes were chosen -- this is a documented replication ambiguity (plan section
5.2). Our rule: partition the valid step indices into K equal quantile bins and
draw one index uniformly per bin, seeded from the rollout id.

Stratification matters because step index is among the strongest features in the
paper's 26-D vector; uniform sampling over a length-varying range would
correlate checkpoint depth with episode length and confound any later probe.
"""

from __future__ import annotations

import random
from typing import List

DEFAULT_K = 4


def valid_indices(n_steps: int) -> List[int]:
    """Checkpointable states for a rollout that took ``n_steps`` actions.

    States are ``s_0 .. s_{n_steps}``. We exclude ``t=0`` (no evidence to
    condition on -- the controller would be deciding before the agent has acted)
    and ``t=n_steps`` (the episode has already terminated, so there is no
    downstream branch to run).
    """
    if n_steps <= 1:
        return []
    return list(range(1, n_steps))


def select_checkpoint_indices(
    n_steps: int,
    seed: int,
    k: int = DEFAULT_K,
) -> List[int]:
    """Pick up to ``k`` stratified checkpoint indices from a rollout.

    Returns a sorted list. If fewer than ``k`` valid indices exist, all of them
    are returned -- short rollouts contribute what they can rather than being
    dropped, which keeps early-termination episodes (a failure mode we care
    about) in the dataset.
    """
    if k <= 0:
        raise ValueError("k must be positive, got {}".format(k))
    valid = valid_indices(n_steps)
    if len(valid) <= k:
        return valid

    rng = random.Random(seed)
    chosen: List[int] = []
    n = len(valid)
    for i in range(k):
        # Equal-width quantile bins over the valid range; guaranteed non-empty
        # because len(valid) > k.
        lo = (i * n) // k
        hi = ((i + 1) * n) // k
        if hi <= lo:
            hi = lo + 1
        chosen.append(valid[rng.randrange(lo, hi)])
    return sorted(set(chosen))
