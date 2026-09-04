"""Two split schemes, computed and reported separately -- never pooled.

1. ``split_paper`` reproduces App. B.3: a fixed *count* of prefix states per run
   (8/4/8, or 20/10/20 for the enlarged suite), with **all prefixes from the same
   base trajectory assigned to the same split**. This is the split the
   replication analysis uses.

2. ``split_task_level`` holds out whole *games*, a strictly harder
   generalization test that the paper does not run. Reported alongside, never
   conflated with (1).

Both schemes enforce the same leakage invariant, checked by validation V8.
"""

from __future__ import annotations

import random
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

TRAIN = "train"
VAL = "val"
TEST = "test"
UNUSED = "unused"
SPLITS = (TRAIN, VAL, TEST)

PAPER_COUNTS_ORIGINAL: Tuple[int, int, int] = (8, 4, 8)
PAPER_COUNTS_ENLARGED: Tuple[int, int, int] = (20, 10, 20)


def _group_by(records: Sequence[Mapping[str, Any]], key: str) -> "OrderedDict[str, List[int]]":
    """Group record indices by a key, preserving first-appearance order.

    Deterministic ordering matters: the shuffle below is seeded, so a stable
    input order makes the whole assignment reproducible.
    """
    groups: "OrderedDict[str, List[int]]" = OrderedDict()
    for i, r in enumerate(records):
        groups.setdefault(str(r[key]), []).append(i)
    return groups


def assign_paper_split(
    checkpoints: Sequence[Mapping[str, Any]],
    counts: Tuple[int, int, int] = PAPER_COUNTS_ORIGINAL,
    seed: int = 0,
    group_key: str = "rollout_id",
) -> List[str]:
    """Assign checkpoints to train/val/test by prefix *count*, grouping by trajectory.

    Greedy fill in train -> val -> test order over a seeded shuffle of whole
    trajectories. A trajectory is placed in the first split that still has room
    for all of its prefixes; if none does, it goes to ``UNUSED``. Whole-group
    placement is what enforces the paper's leakage rule, and it means the
    realised counts can land slightly under target when group sizes do not
    divide evenly -- we accept that rather than splitting a trajectory.

    Returns a list of split labels parallel to ``checkpoints``.
    """
    targets = dict(zip(SPLITS, counts))
    groups = _group_by(checkpoints, group_key)

    order = list(groups.keys())
    random.Random(seed).shuffle(order)

    labels = [UNUSED] * len(checkpoints)
    filled = {s: 0 for s in SPLITS}
    for gid in order:
        idxs = groups[gid]
        size = len(idxs)
        for split in SPLITS:
            if filled[split] + size <= targets[split]:
                for i in idxs:
                    labels[i] = split
                filled[split] += size
                break
    return labels


def assign_task_level_split(
    checkpoints: Sequence[Mapping[str, Any]],
    fractions: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
    group_key: str = "task_id",
) -> List[str]:
    """Hold out whole games. Every checkpoint of a task shares one split.

    Because a rollout belongs to exactly one task, this also satisfies the
    trajectory-level invariant automatically.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1.0, got {}".format(fractions))

    groups = _group_by(checkpoints, group_key)
    order = list(groups.keys())
    random.Random(seed).shuffle(order)

    n = len(order)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    # Test takes the remainder so the partition is exhaustive regardless of rounding.
    bounds = {
        TRAIN: order[:n_train],
        VAL: order[n_train : n_train + n_val],
        TEST: order[n_train + n_val :],
    }

    labels = [UNUSED] * len(checkpoints)
    for split, gids in bounds.items():
        for gid in gids:
            for i in groups[gid]:
                labels[i] = split
    return labels


def check_no_leakage(
    checkpoints: Sequence[Mapping[str, Any]],
    split_field: str,
    group_key: str = "rollout_id",
) -> List[str]:
    """Return group ids that straddle more than one split. Empty list == clean.

    This is validation check V8. It is deliberately a *query*, not an assertion,
    so the validator can report every offending group rather than dying on the
    first one.
    """
    seen: Dict[str, set] = defaultdict(set)
    for r in checkpoints:
        label = str(r.get(split_field, UNUSED))
        if label == UNUSED:
            continue
        seen[str(r[group_key])].add(label)
    return sorted(gid for gid, labels in seen.items() if len(labels) > 1)
