"""Config loading and object construction shared by the pipeline scripts.

Keeps the scripts thin and, more importantly, keeps construction in one place so
that provenance always records the *actual* objects used -- template, scaffold,
utility params, seeds -- rather than a config file that may have drifted.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import yaml

from .agent.policy import ActPolicy
from .agent.prompts import ChatMLTemplate, HFChatTemplate, ScaffoldConfig
from .branching.runner import RunnerConfig
from .serving.client import build_client
from .storage import ARMS, Provenance, RunDir, stable_hash


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_path"] = path
    cfg["_hash"] = stable_hash({k: v for k, v in cfg.items() if not k.startswith("_")})
    return cfg


def build_template(cfg: Dict[str, Any]):
    """Chat template.

    ``hf`` is authoritative -- it is the model's own template and the source of
    ``prompt_token_ids``, which Stage 2 re-tokenises and checks (V10). ``chatml``
    is the offline fallback for laptop development; provenance records which one
    produced a dataset so the two can never be silently mixed.
    """
    model = cfg.get("model", {})
    kind = model.get("template", "chatml")
    if kind == "hf":
        return HFChatTemplate(model["id"], model.get("revision"))
    if kind == "chatml":
        return ChatMLTemplate()
    raise ValueError("unknown template kind: {!r}".format(kind))


def build_scaffold(cfg: Dict[str, Any]) -> ScaffoldConfig:
    return ScaffoldConfig(**(cfg.get("scaffold") or {}))


def build_policy(cfg: Dict[str, Any], template: Any) -> ActPolicy:
    client = build_client(dict(cfg["serving"]))
    return ActPolicy(
        client,
        template,
        build_scaffold(cfg),
        elicit_confidence=bool(cfg.get("elicit_confidence", True)),
    )


def build_runner_config(cfg: Dict[str, Any], arm_name: str) -> RunnerConfig:
    arms = {a["name"]: a for a in cfg.get("arms", [])}
    if arm_name not in arms:
        raise KeyError(
            "arm {!r} not in config; have {}".format(arm_name, sorted(arms))
        )
    if arm_name not in ARMS:
        raise ValueError("unknown arm {!r}; expected one of {}".format(arm_name, ARMS))
    arm = arms[arm_name]
    alf = cfg.get("alfworld", {})
    return RunnerConfig(
        arm=arm_name,
        temperature=float(arm["temperature"]),
        n_replicates=int(arm["n_replicates"]),
        max_episode_steps=int(alf.get("max_episode_steps", 50)),
        k_checkpoints=int((cfg.get("branching") or {}).get("k_checkpoints", 4)),
        expert_max_steps=int(alf.get("expert_max_steps", 1000)),
        elicit_confidence=bool(cfg.get("elicit_confidence", True)),
        scaffold=build_scaffold(cfg),
    )


def run_dir(cfg: Dict[str, Any]) -> RunDir:
    return RunDir(cfg.get("data_root", "data/runs"), cfg["run_id"])


def make_provenance(
    cfg: Dict[str, Any],
    rcfg: RunnerConfig,
    template: Any,
) -> Provenance:
    model = cfg.get("model", {})
    versions = _stack_versions()
    return Provenance(
        run_id=cfg["run_id"],
        arm=rcfg.arm,
        config_hash=cfg.get("_hash", ""),
        config={k: v for k, v in cfg.items() if not k.startswith("_")},
        model_id=str(model.get("id", "")),
        model_revision=str(model.get("revision") or ""),
        tokenizer_revision=str(model.get("revision") or ""),
        chat_template_hash=_template_hash(template),
        alfworld_version=versions.get("alfworld", ""),
        textworld_version=versions.get("textworld", ""),
        container_digest=os.environ.get("CNC_CONTAINER_DIGEST", ""),
        scaffold=rcfg.scaffold.as_dict(),
        temperature=rcfg.temperature,
        n_replicates=rcfg.n_replicates,
        base_seed=int((cfg.get("splits") or {}).get("seed", 0)),
    )


def _template_hash(template: Any) -> str:
    try:
        return template.template_hash()
    except Exception:
        return ""


def _stack_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in ("alfworld", "textworld"):
        try:
            import importlib.metadata as md

            out[name] = md.version(name)
        except Exception:
            out[name] = ""
    return out


def make_env_factory(game_file: str, rcfg: RunnerConfig) -> Callable[[], Any]:
    """A zero-arg factory producing a fresh, independent env for one branch.

    Fresh per branch is deliberate: TextWorld applies wrapper *instances*, so
    reusing an env would share the expert's internal memory across branches.
    """
    from .env.factory import AlfworldEnv

    def factory():
        return AlfworldEnv(
            game_file,
            max_episode_steps=rcfg.max_episode_steps,
            expert_max_steps=rcfg.expert_max_steps,
        )

    return factory


def parallel_map(
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    workers: int = 1,
    on_error: Optional[Callable[[Any, Exception], None]] = None,
) -> List[Any]:
    """Map with a thread pool, preserving input order and surviving failures.

    A single bad game or a transient serving hiccup must not abort a run that has
    already spent GPU time; failures become ``None`` and are reported by the
    caller rather than raised.
    """
    if workers <= 1:
        out = []
        for it in items:
            try:
                out.append(fn(it))
            except Exception as exc:
                if on_error:
                    on_error(it, exc)
                out.append(None)
        return out

    results: List[Any] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:
                if on_error:
                    on_error(items[i], exc)
                results[i] = None
    return results
