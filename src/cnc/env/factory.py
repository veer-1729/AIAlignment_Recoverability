"""ALFWorld TextWorld environment construction.

The only module in the package that imports ALFWorld. Everything else --
state hashing, replay verification, value derivation -- is pure Python so it can
be tested on a machine where the env stack will not install.

We deliberately bypass ``AlfredTWEnv.init_env``. That helper attaches the expert
only when ``training_method == "dagger"`` *and* ``train_eval == "train"``, and it
always builds a *batched* gym env. We need the opposite on both counts: the
expert present on every run, and one unbatched env per worker so a single
episode can be replayed and branched deterministically.

Three upstream quirks are handled here, each of which silently breaks the
protocol otherwise (see plan section 2d):

1. ``AlfredExpert._gather_infos`` does not catch ``HandCodedAgentFailed``, so an
   expert breakdown propagates and kills the rollout. Expert failure is an
   *outcome* we need to measure, so :class:`SafeAlfredExpert` converts it into a
   status flag.
2. ``HandCodedTWAgent(max_steps=200)`` is hard-coded in ``AlfredExpert.load``.
   Replayed prefix steps consume that same budget, so a deep prefix plus an
   expert takeover can spuriously time out. We raise it.
3. Upstream writes ``AlfredExpert(expert_type)``, which passes the type string
   into the ``env`` positional slot. TextWorld's ``Wrapper.__call__(env)``
   re-attaches the wrapper later so this happens to work, but we construct with
   explicit keywords rather than rely on it.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional, Sequence

from .state import StepResult

# Guards TextWorld's global env registry (see AlfworldEnv.__init__).
_REGISTRY_LOCK = threading.Lock()

DEFAULT_MAX_EPISODE_STEPS = 50  # replication ambiguity; ReAct convention
DEFAULT_EXPERT_MAX_STEPS = 1000  # >> 200 default: prefix replay spends this budget

EXPERT_OK = "ok"
EXPERT_INIT = "init"
EXPERT_TIMEOUT = "timeout"
EXPERT_FAILED = "failed"
# Non-terminal: the expert proposed a momentarily unavailable command, so we fall
# back to "look" exactly as upstream does. Measured, but not a branch failure.
EXPERT_FALLBACK = "fallback_look"


def _alfworld_imports():
    """Import ALFWorld/TextWorld lazily and return the names we need."""
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler,
        AlfredExpert,
        AlfredExpertType,
        AlfredInfos,
        TASK_TYPES,
    )
    from alfworld.agents.expert import (
        HandCodedAgentFailed,
        HandCodedAgentTimeout,
        HandCodedTWAgent,
    )

    return {
        "textworld": textworld,
        "AlfredDemangler": AlfredDemangler,
        "AlfredExpert": AlfredExpert,
        "AlfredExpertType": AlfredExpertType,
        "AlfredInfos": AlfredInfos,
        "TASK_TYPES": TASK_TYPES,
        "HandCodedTWAgent": HandCodedTWAgent,
        "HandCodedAgentTimeout": HandCodedAgentTimeout,
        "HandCodedAgentFailed": HandCodedAgentFailed,
    }


def make_safe_expert_class():
    """Build :class:`SafeAlfredExpert` against the installed ALFWorld.

    Defined as a factory rather than a module-level class so importing
    :mod:`cnc.env.factory` does not require ALFWorld to be installed.
    """
    mods = _alfworld_imports()
    AlfredExpert = mods["AlfredExpert"]
    AlfredExpertType = mods["AlfredExpertType"]
    HandCodedTWAgent = mods["HandCodedTWAgent"]
    HandCodedAgentTimeout = mods["HandCodedAgentTimeout"]
    HandCodedAgentFailed = mods["HandCodedAgentFailed"]

    class SafeAlfredExpert(AlfredExpert):
        """``AlfredExpert`` that reports failure instead of raising it.

        The expert is kept in the wrapper chain from ``reset()`` onward, so it
        observes every step the agent takes. Its internal memory -- the receptacle
        map built from the "Welcome" banner, the inventory, the current receptacle
        -- therefore stays synchronised with the trajectory, and deferring to it
        mid-episode needs no cold start. That is precisely what makes the paper's
        defer-to-expert intervention implementable from an arbitrary prefix.
        """

        def __init__(self, env=None, expert_type=AlfredExpertType.HANDCODED,
                     max_steps: int = DEFAULT_EXPERT_MAX_STEPS):
            self._expert_max_steps = max_steps
            self.expert_status = EXPERT_INIT
            super().__init__(env=env, expert_type=expert_type)

        def load(self, gamefile):
            super().load(gamefile)
            # Quirk 2: replace the hard-coded 200-step budget.
            self._handcoded_expert = HandCodedTWAgent(max_steps=self._expert_max_steps)

        def _gather_infos(self):
            # Reimplemented rather than delegated: upstream re-raises a timeout as
            # a bare Exception("Timeout"), which is indistinguishable from a real
            # bug at the call site.
            self.state["extra.expert_plan"] = ["look"]
            if self.expert_type != AlfredExpertType.HANDCODED:
                self.state["extra.expert_plan"] = self.state["policy_commands"]
                self.state["extra.expert_status"] = EXPERT_OK
                return

            try:
                if not self.prev_command:
                    # Post-reset: the expert has seen no action yet, so it can only
                    # observe. "look" stands in for one step; we never branch at t=0.
                    self._handcoded_expert.observe(self.state["feedback"])
                    self.expert_status = EXPERT_INIT
                else:
                    nxt = self._handcoded_expert.act(
                        self.state, 0, self.state["won"], self.prev_command
                    )
                    if nxt in self.state["admissible_commands"]:
                        self.state["extra.expert_plan"] = [nxt]
                        self.expert_status = EXPERT_OK
                    else:
                        # The expert proposed a command the parser will not accept
                        # right now (e.g. "close microwave 1" before opening it).
                        # Upstream falls back to "look", and that is not a
                        # cosmetic detail: "look" refreshes the expert's
                        # observation memory and it recovers on the next step.
                        # Measured here (expert takeover from t=0 goes 3/6 -> 6/6
                        # with this fallback), but recorded so the rate is visible.
                        self.state["extra.expert_plan"] = ["look"]
                        self.expert_status = EXPERT_FALLBACK
            except HandCodedAgentTimeout:
                self.state["extra.expert_plan"] = []
                self.expert_status = EXPERT_TIMEOUT
            except HandCodedAgentFailed:
                self.state["extra.expert_plan"] = []
                self.expert_status = EXPERT_FAILED
            except Exception as exc:  # never let the expert kill a rollout
                self.state["extra.expert_plan"] = []
                self.expert_status = "error:{}".format(type(exc).__name__)

            self.state["extra.expert_status"] = self.expert_status

        def reset(self):
            self.expert_status = EXPERT_INIT
            return super().reset()

    return SafeAlfredExpert


class AlfworldEnv:
    """One unbatched ALFWorld TextWorld game with the expert attached.

    Deterministic by construction: ``AlfredDemangler(shuffle=False)`` is the only
    stochastic element in the stack, so re-executing an action sequence from
    ``reset()`` reproduces the state exactly. That is what
    :class:`~cnc.env.replay.ReplayValidator` verifies rather than assumes.
    """

    def __init__(
        self,
        game_file: str,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        expert_max_steps: int = DEFAULT_EXPERT_MAX_STEPS,
        seed: Optional[int] = None,
    ):
        mods = _alfworld_imports()
        textworld = mods["textworld"]
        import textworld.gym  # noqa: F401  (registers the gym namespace)

        SafeAlfredExpert = make_safe_expert_class()

        self.game_file = game_file
        self.max_episode_steps = max_episode_steps
        self._steps_taken = 0

        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            facts=True,  # required by the handcoded expert AND by state hashing
            extras=["gamefile", "expert_plan", "expert_status"],
        )
        wrappers = [
            mods["AlfredDemangler"](shuffle=False),
            mods["AlfredInfos"],
            SafeAlfredExpert(
                expert_type=mods["AlfredExpertType"].HANDCODED,
                max_steps=expert_max_steps,
            ),
        ]

        # Two upstream hazards, both handled under one lock:
        #
        # (1) `textworld.gym.registry` is a module-level global that gains a new
        #     versioned entry on every `register_games` call. Branching registers
        #     one env per branch, so leaving entries behind would grow the
        #     registry without bound over a full run.
        # (2) `_make_env` applies wrappers by calling `wrapper(env)`, and
        #     `Wrapper.__call__` mutates and returns *self*. Two envs made from
        #     one registration would therefore share a single SafeAlfredExpert
        #     instance and corrupt each other's expert memory.
        #
        # Building fresh wrapper instances per env (above) fixes (2); dropping
        # the registry entry immediately after `make` fixes (1). The env object
        # already holds everything it needs, so the entry is dead weight.
        with _REGISTRY_LOCK:
            env_id = textworld.gym.register_games(
                [game_file],
                request_infos,
                max_episode_steps=max_episode_steps,
                wrappers=wrappers,
            )
            self._env = textworld.gym.make(env_id)
            textworld.gym.utils.registry.pop(env_id, None)

        if seed is not None:
            try:
                self._env.seed(seed)
            except Exception:
                pass  # deterministic anyway with shuffle=False

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _expert_action(infos: Dict[str, Any]) -> Optional[str]:
        plan = infos.get("extra.expert_plan") or []
        if isinstance(plan, str):
            return plan
        return plan[0] if plan else None

    def _to_result(
        self, obs: str, reward: float, done: bool, infos: Dict[str, Any]
    ) -> StepResult:
        from .state import canonical_facts

        return StepResult(
            obs=obs,
            reward=float(reward),
            done=bool(done),
            won=bool(infos.get("won", False)),
            admissible_commands=list(infos.get("admissible_commands") or []),
            facts=canonical_facts(infos.get("facts")),
            expert_action=self._expert_action(infos),
            expert_status=str(infos.get("extra.expert_status", EXPERT_OK)),
        )

    # -- public API --------------------------------------------------------

    def reset(self) -> StepResult:
        self._steps_taken = 0
        obs, infos = self._env.reset()
        self.last_result = self._to_result(obs, 0.0, False, infos)
        return self.last_result

    def step(self, action: str) -> StepResult:
        obs, reward, done, infos = self._env.step(action)
        self._steps_taken += 1
        self.last_result = self._to_result(obs, reward, done, infos)
        return self.last_result

    @property
    def steps_taken(self) -> int:
        return self._steps_taken

    @property
    def steps_remaining(self) -> int:
        return max(0, self.max_episode_steps - self._steps_taken)

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass

    def __enter__(self) -> "AlfworldEnv":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --------------------------------------------------------------------------
# Game-file discovery
# --------------------------------------------------------------------------


def alfworld_data_dir() -> str:
    return os.environ.get(
        "ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")
    )


def task_type_of(game_file: str) -> str:
    """Read the task type from the game's sibling ``traj_data.json``."""
    traj = os.path.join(os.path.dirname(game_file), "traj_data.json")
    try:
        with open(traj, "r", encoding="utf-8") as fh:
            return str(json.load(fh).get("task_type", "unknown"))
    except Exception:
        return "unknown"


def collect_game_files(
    split: str = "train",
    data_dir: Optional[str] = None,
    task_types: Optional[Sequence[str]] = None,
    require_solvable: bool = True,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Find playable ALFWorld games, mirroring ``AlfredTWEnv.collect_game_files``.

    ``require_solvable`` keeps only games whose ``game.tw-pddl`` carries
    ``solvable: true``. That flag is produced upstream by perturbing the first 10
    actions at random (then 15% thereafter) and checking the handcoded expert
    still finishes -- which is exactly the "expert recovers from an arbitrary
    mid-trajectory state" property the defer-to-expert intervention relies on.

    Results are sorted by path so game selection is reproducible across machines.
    """
    root = data_dir or alfworld_data_dir()
    subdir = {
        "train": "json_2.1.1/train",
        "eval_in_distribution": "json_2.1.1/valid_seen",
        "eval_out_of_distribution": "json_2.1.1/valid_unseen",
    }.get(split, split)
    base = os.path.join(root, subdir)

    found: List[Dict[str, Any]] = []
    for dirpath, _dirnames, filenames in os.walk(base):
        if "traj_data.json" not in filenames:
            continue
        if "movable" in dirpath or "Sliced" in dirpath:
            continue  # unsupported upstream
        game_file = os.path.join(dirpath, "game.tw-pddl")
        if not os.path.exists(game_file):
            continue

        try:
            with open(game_file, "r", encoding="utf-8") as fh:
                gamedata = json.load(fh)
        except Exception:
            continue
        if require_solvable and not gamedata.get("solvable", False):
            continue

        ttype = task_type_of(game_file)
        if task_types and ttype not in task_types:
            continue

        found.append(
            {
                "game_file": game_file,
                "task_type": ttype,
                "split_source": split,
                "solvable": bool(gamedata.get("solvable", False)),
            }
        )

    found.sort(key=lambda g: g["game_file"])
    if limit is not None:
        found = found[:limit]
    return found


def sample_games_per_task_type(
    games: Sequence[Dict[str, Any]], per_type: int = 1
) -> List[Dict[str, Any]]:
    """Take the first ``per_type`` games of each task type, deterministically.

    Used by the Gate A pilot so all six task types are exercised -- each has a
    distinct expert policy class upstream, so covering them is how we find
    task-type-specific env bugs early.
    """
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for g in games:
        by_type.setdefault(g["task_type"], []).append(g)
    out: List[Dict[str, Any]] = []
    for ttype in sorted(by_type):
        out.extend(by_type[ttype][:per_type])
    return out
