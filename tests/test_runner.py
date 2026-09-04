"""End-to-end integration test of the branching protocol, with no GPU and no ALFWorld.

A miniature deterministic text game stands in for ALFWorld. It reproduces the
properties the protocol actually depends on:

* determinism (so replay-by-re-execution is meaningful),
* a closed-loop expert that re-plans from whatever state it is handed,
* PDDL-ish facts in the state signature,
* a win condition.

That is enough to exercise rollout -> checkpoint -> replay -> branch -> value
derivation before spending anything on a GPU.
"""

from __future__ import annotations

import pytest

from cnc.agent.policy import ActPolicy
from cnc.agent.prompts import ChatMLTemplate, ScaffoldConfig
from cnc.branching import runner
from cnc.env.state import StepResult
from cnc.serving.client import MockClient
from cnc.storage import ARM_MONTECARLO, ARM_REPLICATION
from cnc.utility import values_from_branches

LOCATIONS = ["cabinet 1", "desk 1"]
OPTIMAL = [
    "go to cabinet 1",
    "take mug 1 from cabinet 1",
    "go to desk 1",
    "put mug 1 in/on desk 1",
]


class MiniAlfworld:
    """Deterministic four-step 'put the mug on the desk' game."""

    def __init__(self, max_episode_steps=12):
        self.max_episode_steps = max_episode_steps
        self.reset()

    # -- internals -----------------------------------------------------
    def _facts(self):
        f = ["at agent {}".format(self.loc or "middle")]
        if self.holding:
            f.append("holds agent mug 1")
        if self.on_desk:
            f.append("inreceptacle mug 1 desk 1")
        elif not self.holding:
            f.append("inreceptacle mug 1 cabinet 1")
        return sorted(f)

    def _expert(self):
        """Closed-loop optimal policy: re-derived from state, not a fixed plan."""
        if self.on_desk:
            return None
        if self.holding:
            return "put mug 1 in/on desk 1" if self.loc == "desk 1" else "go to desk 1"
        if self.loc == "cabinet 1":
            return "take mug 1 from cabinet 1"
        return "go to cabinet 1"

    def _admissible(self):
        cmds = ["look", "inventory"] + ["go to {}".format(l) for l in LOCATIONS]
        if self.loc == "cabinet 1" and not self.holding and not self.on_desk:
            cmds.append("take mug 1 from cabinet 1")
        if self.loc == "desk 1" and self.holding:
            cmds.append("put mug 1 in/on desk 1")
        return sorted(cmds)

    def _result(self, obs, reward=0.0):
        done = self.on_desk or self.steps >= self.max_episode_steps
        self.last_result = StepResult(
            obs=obs,
            reward=reward,
            done=done,
            won=self.on_desk,
            admissible_commands=self._admissible(),
            facts=self._facts(),
            expert_action=self._expert(),
            expert_status="ok" if self._expert() or self.on_desk else "failed",
        )
        return self.last_result

    # -- public API ----------------------------------------------------
    def reset(self):
        self.loc = None
        self.holding = False
        self.on_desk = False
        self.steps = 0
        return self._result(
            "-= Welcome to TextWorld, ALFRED! =-\n\n"
            "You are in the middle of a room. Looking quickly around you, you see "
            "a cabinet 1, and a desk 1.\nYour task is to: put a mug on the desk."
        )

    def step(self, action):
        self.steps += 1
        obs = "Nothing happens."
        if action.startswith("go to "):
            target = action[len("go to ") :]
            if target in LOCATIONS:
                self.loc = target
                obs = "On the {}, you see a mug 1.".format(target) if (
                    target == "cabinet 1" and not self.holding and not self.on_desk
                ) else "On the {}, you see nothing.".format(target)
        elif action == "take mug 1 from cabinet 1" and self.loc == "cabinet 1" and not self.holding:
            self.holding = True
            obs = "You pick up the mug 1 from the cabinet 1."
        elif action == "put mug 1 in/on desk 1" and self.loc == "desk 1" and self.holding:
            self.holding = False
            self.on_desk = True
            obs = "You put the mug 1 in/on the desk 1."
        return self._result(obs, reward=1.0 if self.on_desk else 0.0)

    def close(self):
        pass


TASK = {"task_id": "mini1", "task_type": "pick_and_place_simple"}


def make_policy(script_fn):
    client = MockClient(script_fn)
    return ActPolicy(client, ChatMLTemplate(), ScaffoldConfig(), elicit_confidence=False)


def optimal_responder(prompt, seed):
    """Follow the optimal plan by counting actions already in the prompt."""
    taken = prompt.count("\n> ")
    # subtract the exemplars' own "> " lines by anchoring on the task section
    body = prompt.split("Here is the task.", 1)[-1]
    taken = body.count("\n> ")
    return OPTIMAL[taken] if taken < len(OPTIMAL) else "look"


def dithering_responder(prompt, seed):
    """Never makes progress -- the base agent fails, so intervention matters."""
    return "look"


def _run(cfg, responder):
    policy = make_policy(responder)
    # The env's own cap must match the config's, or the rollout stops short of
    # the horizon the checkpoint budgets are computed against.
    env = MiniAlfworld(cfg.max_episode_steps)
    rollout = runner.run_rollout(env, TASK, policy, cfg, "roll1", seed=1)
    cps = runner.build_checkpoints(rollout, TASK, policy, cfg)
    return policy, rollout, cps


# ---------------------------------------------------------------- rollouts


def test_optimal_agent_solves_and_records_a_full_trajectory():
    cfg = runner.RunnerConfig(arm=ARM_REPLICATION, temperature=0.0, n_replicates=1,
                              max_episode_steps=12, k_checkpoints=4)
    _, rollout, _ = _run(cfg, optimal_responder)
    assert rollout["success"] is True
    assert rollout["terminal_reason"] == "goal"
    assert rollout["n_steps"] == 4
    assert [s["action"] for s in rollout["steps"]] == OPTIMAL
    # every step carries what replay needs
    for s in rollout["steps"]:
        assert s["state_hash"] and "facts" in s and "admissible_commands" in s
    assert rollout["initial"]["state_hash"]


def test_dithering_agent_fails_at_the_step_cap():
    cfg = runner.RunnerConfig(max_episode_steps=6, k_checkpoints=4)
    _, rollout, _ = _run(cfg, dithering_responder)
    assert rollout["success"] is False
    assert rollout["n_steps"] == 6
    assert all(s["nothing_happens"] for s in rollout["steps"])


# ---------------------------------------------------------------- checkpoints


def test_checkpoints_are_stratified_and_carry_stage2_fields():
    cfg = runner.RunnerConfig(max_episode_steps=12, k_checkpoints=4)
    _, rollout, cps = _run(cfg, dithering_responder)
    assert 1 <= len(cps) <= 4
    for cp in cps:
        assert 1 <= cp["t"] < rollout["n_steps"]
        assert cp["rendered_prompt"].endswith("<|im_start|>assistant\n")
        assert len(cp["features_26d"]) == 26
        assert len(cp["prefix_actions"]) == cp["t"]
        assert cp["steps_remaining"] == 12 - cp["t"]


def test_checkpoint_prompt_matches_the_prefix_it_claims():
    cfg = runner.RunnerConfig(max_episode_steps=12, k_checkpoints=4)
    policy, rollout, cps = _run(cfg, dithering_responder)
    cp = cps[0]
    expected = policy.render(
        TASK["task_type"],
        rollout["initial"]["obs"],
        runner.prefix_history(rollout, cp["t"]),
        cp["admissible_commands"],
    )
    assert cp["rendered_prompt"] == expected


# ---------------------------------------------------------------- branching


def test_full_branching_produces_all_three_actions_and_verifies_replay():
    cfg = runner.RunnerConfig(arm=ARM_REPLICATION, temperature=0.0, n_replicates=1,
                              max_episode_steps=12, k_checkpoints=4)
    policy, rollout, cps = _run(cfg, dithering_responder)

    cp = cps[0]
    branches = runner.run_checkpoint_branches(
        cp, rollout, TASK, policy, cfg, env_factory=lambda: MiniAlfworld(12)
    )
    actions = sorted(b["action"] for b in branches)
    assert actions == ["continue", "intervene", "quit"]
    assert all(b["replay_verified"] for b in branches)
    assert runner.usable_checkpoints(branches) == [cp["checkpoint_id"]]


def _all_checkpoints_by_t(responder, max_steps=20):
    """Checkpoint at every valid index, so tests can pick a specific depth."""
    cfg = runner.RunnerConfig(max_episode_steps=max_steps, k_checkpoints=max_steps)
    policy, rollout, cps = _run(cfg, responder)
    return cfg, policy, rollout, {cp["t"]: cp for cp in cps}


def _branches_at(cfg, policy, rollout, cp, max_steps=20):
    return runner.run_checkpoint_branches(
        cp, rollout, TASK, policy, cfg, env_factory=lambda: MiniAlfworld(max_steps)
    )


def test_expert_rescues_a_failing_trajectory_from_an_early_prefix():
    """The core property the paper's ALFWorld intervention relies on."""
    cfg, policy, rollout, by_t = _all_checkpoints_by_t(dithering_responder)
    cp = by_t[2]  # plenty of budget left for the expert's 4-step plan
    branches = _branches_at(cfg, policy, rollout, cp)

    cont = [b for b in branches if b["action"] == "continue"][0]
    interv = [b for b in branches if b["action"] == "intervene"][0]

    assert cont["success"] is False  # dithering agent never finishes
    assert interv["success"] is True  # expert takes over and wins
    assert interv["tokens"] == {}  # zero LLM calls, by construction

    cv = values_from_branches(cp["checkpoint_id"], cfg.arm, branches)
    assert cv.oracle_action == "intervene"
    assert cv.tau > 0
    assert cv.p_fail_continue == 1.0


def test_late_prefixes_can_be_irrecoverable_and_quit_becomes_optimal():
    """Same continuation risk, different optimal action -- the paper's mechanism.

    At t=2 and at t=19 the dithering agent is equally doomed, so continuation
    risk is identical (1.0). But only the early state is *recoverable*: by t=19
    there is not enough budget left for even a perfect expert, so quit wins.
    A controller routing on continuation risk alone cannot separate these.
    """
    cfg, policy, rollout, by_t = _all_checkpoints_by_t(dithering_responder)
    early = values_from_branches(
        by_t[2]["checkpoint_id"], cfg.arm, _branches_at(cfg, policy, rollout, by_t[2])
    )
    late = values_from_branches(
        by_t[19]["checkpoint_id"], cfg.arm, _branches_at(cfg, policy, rollout, by_t[19])
    )

    assert early.p_fail_continue == late.p_fail_continue == 1.0  # identical risk
    assert early.oracle_action == "intervene"
    assert late.oracle_action == "quit"  # different optimal action
    assert early.tau > 0 and late.tau < early.tau


def test_arm_b_runs_multiple_continue_replicates():
    cfg = runner.RunnerConfig(arm=ARM_MONTECARLO, temperature=0.7, n_replicates=5,
                              max_episode_steps=12, k_checkpoints=4)
    policy, rollout, cps = _run(cfg, dithering_responder)
    branches = runner.run_checkpoint_branches(
        cps[0], rollout, TASK, policy, cfg, env_factory=lambda: MiniAlfworld(12)
    )
    conts = [b for b in branches if b["action"] == "continue"]
    assert len(conts) == 5
    assert sorted(b["replicate"] for b in conts) == [0, 1, 2, 3, 4]
    # seeds are derived per (checkpoint, arm, action, replicate) and must differ
    assert len({b["branch_seed"] for b in conts}) == 5


def test_intervene_gets_replicates_too_because_the_expert_is_stochastic():
    """ALFWorld's expert uses random.choice for tie-breaks and loop escapes, so a
    single expert suffix would be one draw masquerading as a value."""
    cfg = runner.RunnerConfig(arm=ARM_MONTECARLO, temperature=0.7, n_replicates=5,
                              max_episode_steps=12, k_checkpoints=4)
    policy, rollout, cps = _run(cfg, dithering_responder)
    branches = runner.run_checkpoint_branches(
        cps[0], rollout, TASK, policy, cfg, env_factory=lambda: MiniAlfworld(12)
    )
    interv = [b for b in branches if b["action"] == "intervene"]
    assert len(interv) == 5
    assert sorted(b["replicate"] for b in interv) == [0, 1, 2, 3, 4]
    assert len({b["branch_seed"] for b in interv}) == 5
    # quit stays analytic: exactly one record, no rollout
    quits = [b for b in branches if b["action"] == "quit"]
    assert len(quits) == 1 and quits[0]["n_steps_in_branch"] == 0


def test_arm_a_keeps_exactly_one_suffix_per_action():
    """Arm A is the paper's protocol: one realized suffix, so U(s,a) not Qhat."""
    cfg = runner.RunnerConfig(arm=ARM_REPLICATION, temperature=0.0, n_replicates=1,
                              max_episode_steps=12, k_checkpoints=4)
    policy, rollout, cps = _run(cfg, dithering_responder)
    branches = runner.run_checkpoint_branches(
        cps[0], rollout, TASK, policy, cfg, env_factory=lambda: MiniAlfworld(12)
    )
    assert len(branches) == 3
    cv = values_from_branches(cps[0]["checkpoint_id"], cfg.arm, branches)
    for a in ("continue", "intervene"):
        assert cv.values[a].n == 1
        assert cv.values[a].is_estimate is False
        assert cv.values[a].se == 0.0
    assert cv.tau_se == 0.0  # a difference of two single samples has no MC error


def test_branch_seeds_are_reproducible_across_runs():
    cfg = runner.RunnerConfig(arm=ARM_MONTECARLO, n_replicates=3, max_episode_steps=12)
    policy, rollout, cps = _run(cfg, dithering_responder)
    f = lambda: runner.run_checkpoint_branches(
        cps[0], rollout, TASK, policy, cfg, env_factory=lambda: MiniAlfworld(12)
    )
    a, b = f(), f()
    assert [x["branch_seed"] for x in a] == [x["branch_seed"] for x in b]


def test_a_corrupted_prefix_is_rejected_and_never_executed():
    """Planted replay failure: the branch must be recorded, not run."""
    cfg = runner.RunnerConfig(max_episode_steps=12, k_checkpoints=4)
    policy, rollout, cps = _run(cfg, dithering_responder)
    cp = cps[-1]

    corrupted = dict(rollout)
    corrupted["steps"] = [dict(s) for s in rollout["steps"]]
    corrupted["steps"][0]["action"] = "go to desk 1"  # diverges from the log

    branches = runner.run_checkpoint_branches(
        cp, corrupted, TASK, policy, cfg, env_factory=lambda: MiniAlfworld(12)
    )
    executed = [b for b in branches if b["action"] != "quit"]
    assert all(not b["replay_verified"] for b in executed)
    assert all(b["terminal_reason"] == "replay_mismatch" for b in executed)
    assert all(b["n_steps_in_branch"] == 0 for b in executed)
    assert all(b["replay_mismatch"] is not None for b in executed)
    # a checkpoint with any unverified branch is dropped whole
    assert runner.usable_checkpoints(branches) == []


def test_unverified_branches_cannot_reach_value_derivation():
    cfg = runner.RunnerConfig(max_episode_steps=12)
    policy, rollout, cps = _run(cfg, dithering_responder)
    corrupted = dict(rollout)
    corrupted["steps"] = [dict(s) for s in rollout["steps"]]
    corrupted["steps"][0]["action"] = "go to desk 1"
    branches = runner.run_checkpoint_branches(
        cps[-1], corrupted, TASK, policy, cfg, env_factory=lambda: MiniAlfworld(12)
    )
    with pytest.raises(ValueError, match="unverified"):
        values_from_branches(cps[-1]["checkpoint_id"], cfg.arm, branches)


def test_showing_admissible_commands_keeps_prompts_consistent():
    """`show_admissible_commands` is the first V7 remediation lever.

    Pulling it changes the prompt, so the checkpoint prompt (rendered from the
    stored step) and the continue branch's first prompt (rendered from the
    replayed env state) must still agree -- otherwise the assertion inside
    run_continue_branch fires and every branch dies.
    """
    scaffold = ScaffoldConfig(show_admissible_commands=True)
    cfg = runner.RunnerConfig(max_episode_steps=12, k_checkpoints=4, scaffold=scaffold)
    client = MockClient(dithering_responder)
    policy = ActPolicy(client, ChatMLTemplate(), scaffold, elicit_confidence=False)
    env = MiniAlfworld(12)
    rollout = runner.run_rollout(env, TASK, policy, cfg, "roll_adm", seed=1)
    cps = runner.build_checkpoints(rollout, TASK, policy, cfg)

    assert "Available actions:" in cps[0]["rendered_prompt"]
    for cp in cps:
        branches = runner.run_checkpoint_branches(
            cp, rollout, TASK, policy, cfg, env_factory=lambda: MiniAlfworld(12)
        )
        assert all(b["replay_verified"] for b in branches)


def test_confidence_is_elicited_and_stored_end_to_end():
    """The dry run leaves this on so Gate A's path is not first exercised in prod."""
    scaffold = ScaffoldConfig()
    cfg = runner.RunnerConfig(max_episode_steps=6, k_checkpoints=2, scaffold=scaffold)
    # action call -> "look"; confidence call -> "0.42"
    seq = iter([])

    def responder(prompt, seed):
        return "0.42" if "confident" in prompt else "look"

    policy = ActPolicy(MockClient(responder), ChatMLTemplate(), scaffold,
                       elicit_confidence=True)
    rollout = runner.run_rollout(MiniAlfworld(6), TASK, policy, cfg, "roll_conf", seed=1)
    assert all(s["confidence"] == 0.42 for s in rollout["steps"])
    assert all(s["confidence_raw"] == "0.42" for s in rollout["steps"])

    cps = runner.build_checkpoints(rollout, TASK, policy, cfg)
    assert all(c["confidence_t"] == 0.42 for c in cps)
    assert all(c["features_26d"]["confidence"] == 0.42 for c in cps)


def test_unparseable_confidence_becomes_none_not_zero():
    scaffold = ScaffoldConfig()
    cfg = runner.RunnerConfig(max_episode_steps=4, k_checkpoints=2, scaffold=scaffold)
    policy = ActPolicy(MockClient(lambda p, s: "look"), ChatMLTemplate(), scaffold,
                       elicit_confidence=True)
    rollout = runner.run_rollout(MiniAlfworld(4), TASK, policy, cfg, "roll_c2", seed=1)
    assert all(s["confidence"] is None for s in rollout["steps"])
    cps = runner.build_checkpoints(rollout, TASK, policy, cfg)
    # 0.5 (uninformative), never 0.0 (which would read as confident failure)
    assert all(c["features_26d"]["confidence"] == 0.5 for c in cps)


def test_continue_branch_resumes_from_the_checkpoint_prompt():
    """Guards the replay/prompt-construction agreement invariant."""
    cfg = runner.RunnerConfig(max_episode_steps=12, k_checkpoints=4)
    policy, rollout, cps = _run(cfg, dithering_responder)
    for cp in cps:
        # the assertion lives inside run_continue_branch; absence of raise is the test
        runner.run_checkpoint_branches(
            cp, rollout, TASK, policy, cfg, env_factory=lambda: MiniAlfworld(12)
        )


def test_successful_base_rollout_yields_continue_as_oracle():
    """Success mid-states supply the 'continue is optimal' class."""
    cfg = runner.RunnerConfig(max_episode_steps=12, k_checkpoints=4)
    policy, rollout, cps = _run(cfg, optimal_responder)
    assert rollout["success"] is True
    cp = cps[0]
    branches = runner.run_checkpoint_branches(
        cp, rollout, TASK, policy, cfg, env_factory=lambda: MiniAlfworld(12)
    )
    cv = values_from_branches(cp["checkpoint_id"], cfg.arm, branches)
    assert cv.values["continue"].p_success == 1.0
    # continue is free; intervene pays 0.05, so continue must win here
    assert cv.oracle_action == "continue"
    assert cv.tau < 0
