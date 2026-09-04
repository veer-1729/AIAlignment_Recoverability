"""The paper's 26-D prefix-only feature vector for ALFWorld.

App. B.5 lists the ALFWorld continue-branch features as: "step index, history
length, task/observation lengths, number of admissible commands, action word
count, 10 action-type indicators, 'nothing happens' flag, 7-dimensional
task-type one-hot, and confidence", and states the prefix-only dimensionality is
**26**.

Counting the listed items gives 25 unless "lengths" is read against the paper's
own common-feature description one paragraph earlier -- "length statistics of the
task description, observation, **and model response**". Including the response
length yields exactly 26:

    1 step index + 1 history length + 3 lengths (task/obs/response)
  + 1 #admissible + 1 action word count + 10 action types
  + 1 nothing-happens + 7 task-type one-hot + 1 confidence  =  26

That reconstruction is documented here rather than buried, because the exact
composition is a replication ambiguity.

Stage 1 only *computes and stores* these features. Nothing here is trained --
they exist so Stage 2 can compare an activation probe against the paper's
handcrafted baseline on identical checkpoints.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

# Ten ALFWorld action verbs. Order is fixed and must not change: it defines
# feature column identity across runs.
ACTION_TYPES: Sequence[str] = (
    "goto",
    "take",
    "put",
    "open",
    "close",
    "use",
    "heat",
    "cool",
    "clean",
    "examine",
)

# Six ALFWorld task types (alfworld.agents.environment.alfred_tw_env.TASK_TYPES)
# plus an explicit "unknown" slot, giving the paper's 7-dimensional one-hot.
TASK_TYPES: Sequence[str] = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
    "unknown",
)

NOTHING_HAPPENS_MARKER = "nothing happens"


def feature_names() -> List[str]:
    """Ordered feature names. Length is asserted to be 26 by the unit tests."""
    names = [
        "step_index",
        "history_length",
        "task_desc_length",
        "observation_length",
        "response_length",
        "n_admissible_commands",
        "action_word_count",
    ]
    names += ["action_type_{}".format(a) for a in ACTION_TYPES]
    names.append("nothing_happens")
    names += ["task_type_{}".format(t) for t in TASK_TYPES]
    names.append("confidence")
    return names


FEATURE_NAMES = feature_names()
FEATURE_DIM = len(FEATURE_NAMES)


def classify_action(action: str) -> Optional[str]:
    """Map a raw ALFWorld action string to one of ``ACTION_TYPES``.

    Returns ``None`` for anything unrecognised (e.g. "look", "inventory", or a
    malformed generation), in which case all ten indicators are zero. That is
    deliberate: an unparseable action is informative, and forcing it into a
    bucket would erase the signal.
    """
    a = action.strip().lower()
    if not a:
        return None
    if a.startswith("go to"):
        return "goto"
    first = a.split()[0]
    # "put X in/on Y" is emitted by some scaffolds as "move X to Y".
    if first in ("put", "move"):
        return "put"
    if first == "take":
        return "take"
    if first == "open":
        return "open"
    if first == "close":
        return "close"
    if first in ("use", "toggle"):
        return "use"
    if first == "heat":
        return "heat"
    if first == "cool":
        return "cool"
    if first == "clean":
        return "clean"
    if first == "examine":
        return "examine"
    return None


def normalise_task_type(task_type: str) -> str:
    t = (task_type or "").strip()
    return t if t in TASK_TYPES else "unknown"


def extract(
    step_index: int,
    history: str,
    task_desc: str,
    observation: str,
    response: str,
    admissible_commands: Sequence[str],
    action: str,
    task_type: str,
    confidence: Optional[float],
) -> Dict[str, float]:
    """Compute the 26-D prefix feature vector at a checkpoint.

    All arguments describe the state *at* the checkpoint: ``action``/``response``
    are the most recent generated action and its raw completion, ``observation``
    is the resulting observation, and ``confidence`` is the agent's self-report
    at that step. Confidence is normalised to [0, 1]; a missing self-report
    becomes 0.5 (maximally uninformative) rather than 0.0, which would masquerade
    as a confident prediction of failure.
    """
    action_type = classify_action(action)
    task_key = normalise_task_type(task_type)

    feats: Dict[str, float] = {
        "step_index": float(step_index),
        "history_length": float(len(history)),
        "task_desc_length": float(len(task_desc)),
        "observation_length": float(len(observation)),
        "response_length": float(len(response)),
        "n_admissible_commands": float(len(admissible_commands)),
        "action_word_count": float(len(action.split())),
    }
    for a in ACTION_TYPES:
        feats["action_type_{}".format(a)] = 1.0 if action_type == a else 0.0
    feats["nothing_happens"] = (
        1.0 if NOTHING_HAPPENS_MARKER in observation.lower() else 0.0
    )
    for t in TASK_TYPES:
        feats["task_type_{}".format(t)] = 1.0 if task_key == t else 0.0
    if confidence is None:
        feats["confidence"] = 0.5
    else:
        feats["confidence"] = float(min(1.0, max(0.0, confidence)))

    assert len(feats) == FEATURE_DIM, (len(feats), FEATURE_DIM)
    return feats


def to_vector(feats: Dict[str, float]) -> List[float]:
    """Dict -> ordered list, using ``FEATURE_NAMES`` as the canonical order."""
    return [float(feats[name]) for name in FEATURE_NAMES]
