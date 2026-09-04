"""Self-reported confidence elicitation.

The paper budgets **8 tokens** for confidence generation on ALFWorld and lists
"the agent's self-reported confidence score, normalized to [0, 1]" among the
shared controller features. It also uses confidence as one of the two candidate
scalars in the calibration decomposition (Table 2), where Platt scaling takes its
ECE from 0.463 to 0.006 without moving control regret at all.

Stage 1 only elicits and stores this number. It is a feature and a candidate
scalar for Stage 2, never a training signal here.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple

from .prompts import ScaffoldConfig, render_user_message

CONFIDENCE_QUESTION = (
    "How confident are you, from 0.00 to 1.00, that you will complete this task "
    "successfully? Reply with only a number."
)

_NUMBER_RE = re.compile(r"(\d*\.?\d+)")


def render_confidence_prompt(
    template,
    task_type: str,
    initial_obs: str,
    steps: Sequence[Tuple[str, str]],
    chosen_action: str,
    cfg: ScaffoldConfig = ScaffoldConfig(),
) -> str:
    """Ask for confidence at the state reached after taking ``chosen_action``.

    Built from the same scaffold as the action call, with the just-chosen action
    appended, so the confidence report is conditioned on exactly the trajectory
    the agent committed to.
    """
    user = render_user_message(task_type, initial_obs, steps, cfg=cfg)
    # render_user_message ends with the ">" action cue; complete it and ask.
    user = "{} {}\n\n{}".format(user, chosen_action, CONFIDENCE_QUESTION)
    return template.render(cfg.system_prompt, user)


def parse_confidence(raw: str) -> Optional[float]:
    """Parse a confidence value from a short completion.

    Handles the shapes an 8-token budget actually produces: ``0.7``, ``.7``,
    ``70%``, ``Confidence: 0.7``. Returns ``None`` when nothing parseable is
    present -- callers store the raw text alongside, and the 26-D feature vector
    substitutes 0.5 for a missing report rather than 0.0, which would read as
    confident failure.
    """
    if not raw:
        return None
    text = raw.strip()
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None

    # "70%" and bare "70" both mean 0.70 under a 0-1 question.
    if "%" in text or value > 1.0:
        value = value / 100.0
    return min(1.0, max(0.0, value))
