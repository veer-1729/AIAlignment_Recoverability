"""Schemas, provenance stamping, and append-only record writers.

Design rules that the rest of the codebase depends on:

1. **Raw outcomes only.** Branch records store ``success`` and
   ``n_steps_in_branch``; they never store a utility. Utilities live in
   ``derived/`` and are recomputed by :mod:`cnc.utility`.
2. **Full untruncated histories.** The paper's App. B.7 could only re-branch
   shallow prefixes because its saved rows held a truncated history string. We
   store every step in full so replay is possible at any depth.
3. **Provenance on every file.** Each JSONL file opens with a ``__provenance__``
   record. Anything that could change results -- model revision, tokenizer
   revision, chat-template hash, scaffold choices, utility params, seeds,
   container digest -- is captured there, so a silent config drift becomes a
   visible diff.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional

SCHEMA_VERSION = "stage1.1"

# Terminal reasons a branch or rollout can end with. Expert failures are
# outcomes, never crashes (plan section 2d).
TERMINAL_GOAL = "goal"
TERMINAL_STEP_CAP = "step_cap"
TERMINAL_EXPERT_FAILED = "expert_failed"
TERMINAL_EXPERT_TIMEOUT = "expert_timeout"
TERMINAL_PARSE_ERROR = "parse_error"
TERMINAL_QUIT = "quit"  # analytic branch: no rollout, U == 0 by definition
TERMINAL_REPLAY_MISMATCH = "replay_mismatch"  # prefix failed V1; branch not executed
TERMINAL_REASONS = (
    TERMINAL_GOAL,
    TERMINAL_STEP_CAP,
    TERMINAL_EXPERT_FAILED,
    TERMINAL_EXPERT_TIMEOUT,
    TERMINAL_PARSE_ERROR,
    TERMINAL_QUIT,
    TERMINAL_REPLAY_MISMATCH,
)

ARM_REPLICATION = "A_replication"
ARM_MONTECARLO = "B_montecarlo"
ARMS = (ARM_REPLICATION, ARM_MONTECARLO)


def stable_hash(obj: Any, length: int = 16) -> str:
    """Deterministic short hash of any JSON-serialisable object.

    Uses sorted keys and compact separators so the digest depends on content
    alone, never on dict insertion order or whitespace.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def text_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def derive_seed(*parts: Any) -> int:
    """Derive a reproducible 32-bit seed from arbitrary identifying parts.

    Used as ``derive_seed(checkpoint_id, arm, action, replicate)`` so that any
    single branch can be re-run in isolation and land on the same sampling path.
    """
    digest = hashlib.sha256(
        "|".join(str(p) for p in parts).encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


def _git_sha(cwd: Optional[str] = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # pragma: no cover - environment dependent
        pass
    return "unknown"


@dataclass
class Provenance:
    """Everything needed to interpret, and later reproduce, a run.

    ``model_revision``, ``tokenizer_revision`` and ``chat_template_hash`` are the
    fields Stage 2 depends on: activations will be extracted by a *separate*
    HuggingFace forward pass over the stored prompts, so the exact tokenizer and
    template that produced ``prompt_token_ids`` must be pinned here.
    """

    run_id: str
    arm: str
    config_hash: str
    config: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    git_sha: str = field(default_factory=_git_sha)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # Model / tokenizer pinning (Stage-2 critical)
    model_id: str = ""
    model_revision: str = ""
    tokenizer_revision: str = ""
    chat_template_hash: str = ""
    # Environment pinning
    alfworld_version: str = ""
    textworld_version: str = ""
    container_digest: str = ""
    # Replication ambiguities (plan section 5.2) -- recorded so a change is visible
    scaffold: Dict[str, Any] = field(default_factory=dict)
    # Sampling
    temperature: Optional[float] = None
    n_replicates: Optional[int] = None
    base_seed: Optional[int] = None

    def to_record(self) -> Dict[str, Any]:
        return {"__provenance__": asdict(self)}


class JsonlWriter:
    """Append-only JSONL writer that stamps provenance as the first record."""

    def __init__(self, path: str, provenance: Optional[Provenance] = None):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh = open(path, "a", encoding="utf-8")
        if is_new and provenance is not None:
            self._write_obj(provenance.to_record())

    def _write_obj(self, obj: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
        self._fh.flush()

    def write(self, record: Dict[str, Any]) -> None:
        if "__provenance__" in record:
            raise ValueError("use the constructor to write provenance")
        self._write_obj(record)

    def write_many(self, records: Iterable[Dict[str, Any]]) -> None:
        for r in records:
            self.write(r)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_jsonl(path: str, include_provenance: bool = False) -> Iterator[Dict[str, Any]]:
    """Stream records from a JSONL file, skipping the provenance header."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "__provenance__" in obj:
                if include_provenance:
                    yield obj
                continue
            yield obj


def read_provenance(path: str) -> Optional[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "__provenance__" in obj:
                return obj["__provenance__"]
            return None
    return None


class RunDir:
    """Filesystem layout for one run.

    ``data/<run_id>/{tasks,rollouts,checkpoints,branches}.jsonl`` plus
    ``derived/<arm>/`` for recomputed values. ``derived/`` is always safe to
    delete and regenerate -- that property is asserted by validation check V9.
    """

    def __init__(self, root: str, run_id: str):
        self.root = root
        self.run_id = run_id
        self.base = os.path.join(root, run_id)
        os.makedirs(self.base, exist_ok=True)

    def path(self, name: str) -> str:
        return os.path.join(self.base, name)

    @property
    def tasks(self) -> str:
        return self.path("tasks.jsonl")

    @property
    def rollouts(self) -> str:
        return self.path("rollouts.jsonl")

    @property
    def checkpoints(self) -> str:
        return self.path("checkpoints.jsonl")

    @property
    def branches(self) -> str:
        return self.path("branches.jsonl")

    # -- sharding ---------------------------------------------------------
    #
    # ALFWorld/TextWorld carry module-level global state, and running env
    # construction on multiple *threads* corrupts it: a measured 12-checkpoint
    # branching job yields 36/36 branches single-threaded but 3 branches and 11
    # errors at 4 threads ("pop from empty list", "'NoneType' object is not
    # iterable"). Those failures are caught and counted rather than raised, so
    # threading would have silently shrunk the dataset instead of crashing.
    #
    # Parallelism is therefore at the *process* level: each shard is its own
    # process with its own globals, writing its own file. These helpers keep the
    # shard files discoverable so readers never miss one.

    def shard_path(self, stem: str, shard: Optional[int], num_shards: int = 1) -> str:
        if num_shards <= 1 or shard is None:
            return self.path("{}.jsonl".format(stem))
        return self.path("{}.s{:03d}.jsonl".format(stem, shard))

    def all_shards(self, stem: str) -> List[str]:
        """Every file for a record type, sharded or not, in deterministic order."""
        import glob

        files = sorted(glob.glob(self.path("{}.s*.jsonl".format(stem))))
        plain = self.path("{}.jsonl".format(stem))
        if os.path.exists(plain):
            files = [plain] + files
        return files

    def read_all(self, stem: str) -> Iterator[Dict[str, Any]]:
        for path in self.all_shards(stem):
            for rec in read_jsonl(path):
                yield rec

    def derived(self, arm: str, name: str) -> str:
        d = os.path.join(self.base, "derived", arm)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, name)


# --------------------------------------------------------------------------
# Record constructors. These exist so field names are defined in exactly one
# place; every writer goes through them.
# --------------------------------------------------------------------------


def task_record(
    task_id: str,
    game_file: str,
    task_type: str,
    split_source: str,
    solvable: bool,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "game_file": game_file,
        "task_type": task_type,
        "split_source": split_source,
        "solvable": solvable,
    }


def env_state_record(result: Any) -> Dict[str, Any]:
    """Serialise a :class:`~cnc.env.state.StepResult` for replay verification.

    Stores the full signature components -- including the PDDL ``facts`` -- and
    not merely their hash. The hash alone would detect a replay divergence but
    could not say *which* field moved, and a validator you cannot debug is a
    validator you end up disabling.
    """
    return {
        "obs": result.obs,
        "reward": result.reward,
        "done": result.done,
        "won": result.won,
        "admissible_commands": list(result.admissible_commands),
        "facts": list(result.facts),
        "state_hash": result.hash,
        "expert_action": result.expert_action,
        "expert_status": result.expert_status,
    }


def step_record(
    t: int,
    rendered_prompt: str,
    prompt_token_ids: Optional[List[int]],
    raw_completion: str,
    action: str,
    env_state: Dict[str, Any],
    confidence_raw: str = "",
    confidence: Optional[float] = None,
    nothing_happens: bool = False,
) -> Dict[str, Any]:
    """One agent-environment interaction, stored in full.

    ``rendered_prompt`` is the byte-exact string sent to the model. Stage 2
    feeds this back through a HuggingFace forward pass, so it must never be
    truncated or reconstructed. ``env_state`` is the resulting state, carrying
    everything :class:`~cnc.env.replay.ReplayValidator` needs to verify that a
    re-execution landed in the same place.
    """
    rec = {
        "t": t,
        "rendered_prompt": rendered_prompt,
        "prompt_token_ids": prompt_token_ids,
        "raw_completion": raw_completion,
        "action": action,
        "confidence_raw": confidence_raw,
        "confidence": confidence,
        "nothing_happens": nothing_happens,
    }
    rec.update(env_state)
    return rec


def rollout_record(
    rollout_id: str,
    task_id: str,
    arm: str,
    policy_seed: int,
    temperature: float,
    initial: Dict[str, Any],
    steps: List[Dict[str, Any]],
    success: bool,
    n_steps: int,
    terminal_reason: str,
    tokens: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """A complete base trajectory.

    ``initial`` is the post-reset state (``s_0``); ``steps[i]`` is the state
    after ``i+1`` actions. Together they give the ``n+1`` signatures replay needs
    for an ``n``-action prefix.
    """
    if terminal_reason not in TERMINAL_REASONS:
        raise ValueError("bad terminal_reason: {!r}".format(terminal_reason))
    if arm not in ARMS:
        raise ValueError("bad arm: {!r}".format(arm))
    return {
        "rollout_id": rollout_id,
        "task_id": task_id,
        "arm": arm,
        "policy_seed": policy_seed,
        "temperature": temperature,
        "initial": initial,
        "steps": steps,
        "success": success,
        "n_steps": n_steps,
        "terminal_reason": terminal_reason,
        "tokens": tokens or {},
    }


def checkpoint_record(
    checkpoint_id: str,
    rollout_id: str,
    task_id: str,
    arm: str,
    t: int,
    prefix_state_hash: str,
    rendered_prompt: str,
    prompt_token_ids: Optional[List[int]],
    features_26d: Dict[str, float],
    confidence_t: Optional[float],
    admissible_commands: List[str],
    prefix_actions: List[str],
    steps_remaining: int,
    split_paper: str = "",
    split_task_level: str = "",
) -> Dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("bad arm: {!r}".format(arm))
    return {
        "checkpoint_id": checkpoint_id,
        "rollout_id": rollout_id,
        "task_id": task_id,
        "arm": arm,
        "t": t,
        "prefix_state_hash": prefix_state_hash,
        "rendered_prompt": rendered_prompt,
        "prompt_token_ids": prompt_token_ids,
        "features_26d": features_26d,
        "confidence_t": confidence_t,
        "admissible_commands": admissible_commands,
        "prefix_actions": prefix_actions,
        "steps_remaining": steps_remaining,
        "split_paper": split_paper,
        "split_task_level": split_task_level,
    }


def branch_record(
    branch_id: str,
    checkpoint_id: str,
    arm: str,
    action: str,
    replicate: int,
    branch_seed: int,
    replay_verified: bool,
    success: bool,
    n_steps_in_branch: int,
    terminal_reason: str,
    transcript: List[Dict[str, Any]],
    replay_mismatch: Optional[Dict[str, Any]] = None,
    tokens: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """One executed branch. Carries **no utility** -- see module docstring."""
    if arm not in ARMS:
        raise ValueError("bad arm: {!r}".format(arm))
    if terminal_reason not in TERMINAL_REASONS:
        raise ValueError("bad terminal_reason: {!r}".format(terminal_reason))
    return {
        "branch_id": branch_id,
        "checkpoint_id": checkpoint_id,
        "arm": arm,
        "action": action,
        "replicate": replicate,
        "branch_seed": branch_seed,
        "replay_verified": replay_verified,
        "success": success,
        "n_steps_in_branch": n_steps_in_branch,
        "terminal_reason": terminal_reason,
        "transcript": transcript,
        "replay_mismatch": replay_mismatch,
        "tokens": tokens or {},
    }
