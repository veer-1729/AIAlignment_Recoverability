"""Agent scaffold and chat-template rendering.

Everything in this module is a **replication ambiguity** (plan section 5.2). The
paper states only the generation budgets -- 24 tokens for an ALFWorld action, 8
for confidence -- and never gives the prompt format, the shot count, or whether
admissible commands are shown. There is no author code to recover them from.

We therefore adopt the upstream ReAct reference convention
(``ysymyth/ReAct``, ``prompts/alfworld.json``), which is the de-facto standard
for ALFWorld LLM agents, and make every choice configurable via
:class:`ScaffoldConfig`. The active config is hashed into run provenance, so a
later change shows up as a data diff instead of a silent drift.

Rendering is **stateless**: ``render`` is a pure function of the prefix. That is
what makes the stored ``rendered_prompt`` a faithful description of the model's
input at a checkpoint, which Stage 2 depends on when it replays these strings
through a HuggingFace forward pass.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "react_alfworld_prompts.json")

# ALFWorld task type -> ReAct prompt-file key stem.
TASK_TYPE_TO_KEY: Dict[str, str] = {
    "pick_and_place_simple": "put",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "look_at_obj_in_light": "examine",
    "pick_two_obj_and_place": "puttwo",
}

DEFAULT_SYSTEM = (
    "You are an agent solving a household task in a text environment. "
    "Reply with exactly one action and nothing else."
)

DEFAULT_HEADER = "Interact with a household to solve a task. Here are two examples."
DEFAULT_TASK_HEADER = "Here is the task."


@dataclass(frozen=True)
class ScaffoldConfig:
    """The replication-ambiguity knobs, in one place.

    Defaults are the ReAct convention. ``mode='act'`` is our reading of the
    paper's 24-token action budget, which is too small for chain-of-thought --
    an inference, not a stated fact, and recorded as such.
    """

    mode: str = "act"  # "act" (act-only) or "react" (with think: steps)
    n_shots: int = 2
    show_admissible_commands: bool = False
    system_prompt: str = DEFAULT_SYSTEM
    header: str = DEFAULT_HEADER
    task_header: str = DEFAULT_TASK_HEADER
    strip_welcome_block: bool = True
    max_action_tokens: int = 24  # paper-stated
    max_confidence_tokens: int = 8  # paper-stated

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_PROMPTS_CACHE: Optional[Dict[str, str]] = None


def load_prompts(path: str = _PROMPT_FILE) -> Dict[str, str]:
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is None:
        with open(path, "r", encoding="utf-8") as fh:
            _PROMPTS_CACHE = json.load(fh)
    return _PROMPTS_CACHE


def exemplars_for(task_type: str, cfg: ScaffoldConfig) -> List[str]:
    """Fetch the in-context examples for a task type."""
    key = TASK_TYPE_TO_KEY.get(task_type)
    if key is None:
        raise KeyError(
            "no ReAct exemplar key for task_type {!r}; known: {}".format(
                task_type, sorted(TASK_TYPE_TO_KEY)
            )
        )
    prompts = load_prompts()
    out = []
    for i in range(cfg.n_shots):
        name = "{}_{}_{}".format(cfg.mode, key, i)
        if name not in prompts:
            raise KeyError("exemplar {!r} not in prompt file".format(name))
        out.append(prompts[name])
    return out


def strip_welcome(obs: str) -> str:
    """Drop TextWorld's banner block, matching the ReAct notebook convention.

    ALFWorld's reset observation opens with "-= Welcome to TextWorld, ALFRED! =-"
    followed by a blank line, then the room description and task. The exemplars
    in the prompt file start at the room description, so the live observation is
    trimmed the same way.
    """
    parts = obs.split("\n\n")
    if len(parts) > 1 and "welcome" in parts[0].lower():
        return "\n".join(parts[1:]).strip()
    return obs.strip()


def render_user_message(
    task_type: str,
    initial_obs: str,
    steps: Sequence[Tuple[str, str]],
    cfg: ScaffoldConfig = ScaffoldConfig(),
    admissible_commands: Optional[Sequence[str]] = None,
) -> str:
    """Build the user-turn text for the state reached after ``steps``.

    ``steps`` is the ordered ``(action, resulting_observation)`` prefix. The
    string ends with the ReAct ``>`` action cue, so the model's next token
    begins the action.
    """
    obs0 = strip_welcome(initial_obs) if cfg.strip_welcome_block else initial_obs.strip()

    chunks = [cfg.header]
    for ex in exemplars_for(task_type, cfg):
        chunks.append(ex.strip())
    chunks.append(cfg.task_header)

    body = [obs0]
    for action, obs in steps:
        body.append("> {}".format(action))
        body.append(obs.strip())

    text = "\n".join(chunks) + "\n" + "\n".join(body)

    if cfg.show_admissible_commands and admissible_commands:
        text += "\nAvailable actions: {}".format(", ".join(admissible_commands))

    return text + "\n>"


# --------------------------------------------------------------------------
# Chat templates
# --------------------------------------------------------------------------


class ChatMLTemplate:
    """Qwen2.5's ChatML format, hardcoded.

    Used for offline development and unit tests, where pulling the tokenizer is
    undesirable. Production runs use :class:`HFChatTemplate` so the rendered
    string is byte-identical to the model's training format; the template hash
    in provenance records which one produced a given dataset.
    """

    name = "chatml_hardcoded"

    def render(self, system: str, user: str) -> str:
        return (
            "<|im_start|>system\n{}<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n".format(system, user)
        )

    def template_hash(self) -> str:
        from ..storage import text_hash

        return text_hash(self.render("S", "U"))


class HFChatTemplate:
    """Authoritative template, taken from the model's own tokenizer.

    Also the source of ``prompt_token_ids``. Stage 2 re-tokenises the stored
    prompt with the same pinned revision and asserts the ids match (V10).
    """

    name = "hf_tokenizer"

    def __init__(self, model_id: str, revision: Optional[str] = None):
        from transformers import AutoTokenizer  # imported lazily

        self.model_id = model_id
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def render(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def encode(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def template_hash(self) -> str:
        from ..storage import text_hash

        tpl = getattr(self.tokenizer, "chat_template", None) or ""
        return text_hash(tpl)


def render_prompt(
    template: Any,
    task_type: str,
    initial_obs: str,
    steps: Sequence[Tuple[str, str]],
    cfg: ScaffoldConfig = ScaffoldConfig(),
    admissible_commands: Optional[Sequence[str]] = None,
) -> str:
    """End-to-end: scaffold -> chat template -> the exact string sent to the model."""
    user = render_user_message(
        task_type, initial_obs, steps, cfg=cfg, admissible_commands=admissible_commands
    )
    return template.render(cfg.system_prompt, user)


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------


def parse_action(raw: str) -> str:
    """Extract a single ALFWorld action from a raw completion.

    With a 24-token cap the model can still emit a stray ``>`` prefix, a
    trailing observation line, or nothing at all. We take the first non-empty
    line, drop a leading ``>``, and lowercase (ALFWorld's admissible commands are
    all lowercase). An unparseable completion returns ``""``; the caller records
    it as a ``parse_error`` rather than substituting a guess -- the paper's
    "nothing happens" feature exists precisely because bad actions are real
    events worth measuring.
    """
    if not raw:
        return ""
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            line = line[1:].strip()
        if not line:
            continue
        return " ".join(line.lower().split())
    return ""
