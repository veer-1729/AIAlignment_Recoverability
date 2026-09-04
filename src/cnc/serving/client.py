"""Provider-agnostic LLM client.

This is the *only* module that knows where the model runs. The experimental
design is deliberately independent of the serving provider (plan section 9), so
swapping vLLM-on-OpenRelay for any other OpenAI-compatible endpoint touches
nothing else.

Two hard rules, both Stage-2 requirements:

* We call ``/v1/completions`` with a **locally rendered** chat template, never
  ``/v1/chat/completions``. A server-side template could differ from the one we
  hash into provenance, and the stored ``rendered_prompt`` would then no longer
  describe the model's actual input.
* Every request carries an explicit ``seed`` derived from the branch identity,
  so a single branch can be re-run in isolation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class CompletionResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    raw: Optional[Dict[str, Any]] = None


class LLMClient:
    """Interface implemented by every backend."""

    def complete(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> CompletionResult:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    """Talks to any OpenAI-compatible ``/v1/completions`` endpoint (e.g. vLLM)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        import httpx  # imported lazily so offline tests need no network stack

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self._client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": "Bearer {}".format(api_key)},
        )

    def complete(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> CompletionResult:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = int(seed)
        if stop:
            payload["stop"] = list(stop)

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(self.base_url + "/v1/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})
                return CompletionResult(
                    text=choice.get("text", ""),
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    finish_reason=str(choice.get("finish_reason", "")),
                    raw=data,
                )
            except Exception as exc:  # pragma: no cover - network dependent
                last_err = exc
        raise RuntimeError(
            "completion failed after {} attempts: {}".format(self.max_retries, last_err)
        )

    def close(self) -> None:  # pragma: no cover
        self._client.close()


class MockClient(LLMClient):
    """Offline client driven by a callable.

    Lets the whole harness -- rollouts, checkpointing, branching, validation --
    run end-to-end on the laptop with no GPU and no network, which is how every
    code path gets exercised before any money is spent. The responder receives
    the rendered prompt and the requested seed, so tests can simulate both
    deterministic (Arm A) and seed-dependent (Arm B) behaviour.
    """

    def __init__(self, responder: Callable[[str, Optional[int]], str]):
        self.responder = responder
        self.calls: List[Dict[str, Any]] = []

    def complete(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> CompletionResult:
        text = self.responder(prompt, seed)
        self.calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed,
            }
        )
        # Emulate the server-side stop-sequence trim so offline behaviour
        # matches production parsing.
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx >= 0:
                    text = text[:idx]
        return CompletionResult(
            text=text,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(text) // 4),
            finish_reason="stop",
        )


class ScriptedClient(MockClient):
    """Returns a fixed sequence of completions, cycling if exhausted."""

    def __init__(self, responses: Sequence[str]):
        self._responses = list(responses)
        self._i = 0

        def responder(prompt: str, seed: Optional[int]) -> str:
            if not self._responses:
                return ""
            out = self._responses[self._i % len(self._responses)]
            self._i += 1
            return out

        super().__init__(responder)


def build_client(cfg: Dict[str, Any]) -> LLMClient:
    """Construct a client from a config block."""
    kind = cfg.get("kind", "openai_compatible")
    if kind == "openai_compatible":
        return OpenAICompatibleClient(
            base_url=cfg["base_url"],
            model=cfg["model"],
            api_key=cfg.get("api_key", "EMPTY"),
            timeout=float(cfg.get("timeout", 120.0)),
        )
    if kind == "scripted":
        return ScriptedClient(cfg.get("responses", []))
    raise ValueError("unknown client kind: {!r}".format(kind))
