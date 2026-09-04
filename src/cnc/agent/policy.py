"""The act-only ALFWorld policy.

One agent step is two model calls, matching the paper's per-step generation
budgets for ALFWorld: **24 tokens** to generate the action and **8 tokens** to
generate the self-reported confidence.

The policy is stateless. ``act`` is a pure function of
``(task_type, initial_obs, prefix_steps, temperature, seed)``, which is what
makes a branch reproducible in isolation and makes the stored
``rendered_prompt`` an honest description of the model's input at a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from ..serving.client import LLMClient
from .confidence import parse_confidence, render_confidence_prompt
from .prompts import ScaffoldConfig, parse_action, render_prompt

# The model must emit one action; anything past the first newline is the model
# hallucinating the environment's reply.
ACTION_STOP = ("\n", ">")
CONFIDENCE_STOP = ("\n",)


@dataclass
class PolicyStep:
    """Everything one agent decision produced, stored verbatim."""

    action: str
    raw_completion: str
    rendered_prompt: str
    prompt_token_ids: Optional[List[int]] = None
    confidence: Optional[float] = None
    confidence_raw: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def parse_failed(self) -> bool:
        return not self.action


class ActPolicy:
    """Act-only policy over an OpenAI-compatible completions endpoint."""

    def __init__(
        self,
        client: LLMClient,
        template: Any,
        cfg: ScaffoldConfig = ScaffoldConfig(),
        elicit_confidence: bool = True,
    ):
        self.client = client
        self.template = template
        self.cfg = cfg
        self.elicit_confidence = elicit_confidence

    def render(
        self,
        task_type: str,
        initial_obs: str,
        steps: Sequence[Tuple[str, str]],
        admissible_commands: Optional[Sequence[str]] = None,
    ) -> str:
        """The exact string the model sees at this state.

        Exposed separately from :meth:`act` because checkpoint records store this
        prompt without spending a model call, and because the continue branch
        asserts that its first prompt equals the checkpoint's.
        """
        return render_prompt(
            self.template,
            task_type,
            initial_obs,
            steps,
            cfg=self.cfg,
            admissible_commands=admissible_commands,
        )

    def token_ids(self, prompt: str) -> Optional[List[int]]:
        encode = getattr(self.template, "encode", None)
        if encode is None:
            return None
        try:
            return encode(prompt)
        except Exception:
            return None

    def act(
        self,
        task_type: str,
        initial_obs: str,
        steps: Sequence[Tuple[str, str]],
        temperature: float,
        seed: Optional[int] = None,
        admissible_commands: Optional[Sequence[str]] = None,
    ) -> PolicyStep:
        prompt = self.render(task_type, initial_obs, steps, admissible_commands)
        out = self.client.complete(
            prompt,
            max_tokens=self.cfg.max_action_tokens,
            temperature=temperature,
            seed=seed,
            stop=ACTION_STOP,
        )
        action = parse_action(out.text)

        step = PolicyStep(
            action=action,
            raw_completion=out.text,
            rendered_prompt=prompt,
            prompt_token_ids=self.token_ids(prompt),
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
        )

        if self.elicit_confidence and action:
            # Confidence is conditioned on the action just committed to, and uses
            # a separate seed so it cannot perturb the action's sampling path.
            cprompt = render_confidence_prompt(
                self.template, task_type, initial_obs, steps, action, cfg=self.cfg
            )
            cout = self.client.complete(
                cprompt,
                max_tokens=self.cfg.max_confidence_tokens,
                temperature=temperature,
                seed=None if seed is None else seed + 1,
                stop=CONFIDENCE_STOP,
            )
            step.confidence_raw = cout.text
            step.confidence = parse_confidence(cout.text)
            step.prompt_tokens += cout.prompt_tokens
            step.completion_tokens += cout.completion_tokens

        return step
