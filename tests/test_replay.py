"""Tests for state signatures and prefix replay verification.

These run without ALFWorld installed: :mod:`cnc.env.state` and
:mod:`cnc.env.replay` are pure, and the fake env below implements the same tiny
``reset()``/``step()`` contract that :class:`cnc.env.factory.AlfworldEnv` does.
"""

from __future__ import annotations

import pytest

from cnc.env.replay import ReplayValidator, signatures_from_steps
from cnc.env.state import (
    StepResult,
    canonical_facts,
    diff_signatures,
    state_hash,
    state_signature,
)


class FakeProposition:
    """Stands in for a TextWorld ``Proposition``."""

    def __init__(self, name, names):
        self.name = name
        self.names = names


class FakeEnv:
    """Deterministic scripted environment.

    ``drift_at`` injects a divergence at a given step so the negative control
    (validation V2) can be exercised: a validator that never fails proves nothing.
    """

    def __init__(self, script, drift_at=None, drift_field="obs"):
        self.script = script  # list of StepResult, index 0 == post-reset
        self.drift_at = drift_at
        self.drift_field = drift_field
        self.i = 0

    def _maybe_drift(self, result, idx):
        if self.drift_at is None or idx != self.drift_at:
            return result
        if self.drift_field == "obs":
            return StepResult(
                obs=result.obs + " (drifted)",
                admissible_commands=list(result.admissible_commands),
                facts=list(result.facts),
            )
        if self.drift_field == "facts":
            return StepResult(
                obs=result.obs,
                admissible_commands=list(result.admissible_commands),
                facts=list(result.facts) + ["holds agent ghost"],
            )
        raise AssertionError(self.drift_field)

    def reset(self):
        self.i = 0
        return self._maybe_drift(self.script[0], 0)

    def step(self, action):
        self.i += 1
        return self._maybe_drift(self.script[self.i], self.i)


def _script(n=4):
    out = []
    for i in range(n + 1):
        out.append(
            StepResult(
                obs="state {}".format(i),
                admissible_commands=["look", "go to cabinet {}".format(i)],
                facts=["at agent loc{}".format(i)],
                reward=0.0,
                done=False,
            )
        )
    return out


# ---------------------------------------------------------------- signatures


def test_admissible_command_order_is_not_a_state_difference():
    a = state_signature("obs", ["b", "a"], facts=["y", "x"])
    b = state_signature("obs", ["a", "b"], facts=["x", "y"])
    assert a == b
    assert state_hash("obs", ["b", "a"], facts=["y", "x"]) == state_hash(
        "obs", ["a", "b"], facts=["x", "y"]
    )


def test_facts_are_part_of_the_signature():
    """Observations can coincide while the world differs -- facts catch that."""
    a = state_hash("same obs", ["look"], facts=["holds agent mug"])
    b = state_hash("same obs", ["look"], facts=["holds agent apple"])
    assert a != b


def test_canonical_facts_renders_propositions_like_the_expert_does():
    facts = [FakeProposition("inreceptacle", ["mug 1", "desk 1"]), "at agent loc0"]
    assert canonical_facts(facts) == [
        "at agent loc0",
        "inreceptacle mug 1 desk 1",
    ]


def test_diff_reports_which_field_moved():
    a = state_signature("x", ["look"], facts=["f1"])
    b = state_signature("y", ["look"], facts=["f1"])
    d = diff_signatures(a, b)
    assert set(d) == {"obs"}
    assert d["obs"] == {"expected": "x", "actual": "y"}
    assert diff_signatures(a, a) == {}


def test_step_result_hash_is_stable():
    r = StepResult(obs="o", admissible_commands=["a"], facts=["f"])
    assert r.hash == StepResult(obs="o", admissible_commands=["a"], facts=["f"]).hash


# ---------------------------------------------------------------- replay


def test_clean_replay_verifies():
    script = _script(4)
    expected = [s.signature for s in script]
    env = FakeEnv(script)
    res = ReplayValidator().replay(env, ["a1", "a2", "a3", "a4"], expected)
    assert res.verified is True
    assert res.n_steps_replayed == 4
    assert res.mismatch is None
    assert res.match_rate == 1.0


def test_replay_catches_a_mid_prefix_observation_drift():
    """V2 negative control: the validator must have teeth."""
    script = _script(4)
    expected = [s.signature for s in script]
    env = FakeEnv(script, drift_at=2, drift_field="obs")
    res = ReplayValidator().replay(env, ["a1", "a2", "a3", "a4"], expected)
    assert res.verified is False
    assert res.mismatch["step_index"] == 2
    assert res.mismatch["action"] == "a2"
    assert "obs" in res.mismatch["diff"]


def test_replay_catches_a_hidden_world_state_drift():
    """Same observation text, different facts -- only the facts check finds this."""
    script = _script(4)
    expected = [s.signature for s in script]
    env = FakeEnv(script, drift_at=3, drift_field="facts")
    res = ReplayValidator().replay(env, ["a1", "a2", "a3", "a4"], expected)
    assert res.verified is False
    assert res.mismatch["step_index"] == 3
    assert "facts" in res.mismatch["diff"]


def test_a_facts_only_drift_is_invisible_without_facts_comparison():
    """Justifies comparing facts rather than just the observation string."""
    script = _script(4)
    expected = [s.signature for s in script]
    env = FakeEnv(script, drift_at=3, drift_field="facts")
    res = ReplayValidator(compare_facts=False).replay(env, ["a1", "a2", "a3", "a4"], expected)
    assert res.verified is True  # slipped through
    env2 = FakeEnv(_script(4), drift_at=3, drift_field="facts")
    assert ReplayValidator(compare_facts=True).replay(env2, ["a1", "a2", "a3", "a4"], expected).verified is False


def test_divergence_at_reset_is_caught_before_any_action():
    script = _script(4)
    expected = [s.signature for s in script]
    env = FakeEnv(script, drift_at=0, drift_field="obs")
    res = ReplayValidator().replay(env, ["a1", "a2"], expected[:3])
    assert res.verified is False
    assert res.mismatch["step_index"] == 0
    assert res.mismatch["action"] is None


def test_replay_of_a_partial_prefix():
    """Branching replays only up to the checkpoint, not the whole trajectory."""
    script = _script(4)
    expected = [s.signature for s in script]
    env = FakeEnv(script)
    res = ReplayValidator().replay(env, ["a1", "a2"], expected[:3])
    assert res.verified is True
    assert res.n_steps_replayed == 2
    assert res.final.obs == "state 2"


def test_signature_count_mismatch_is_a_caller_bug_not_a_replay_failure():
    script = _script(4)
    expected = [s.signature for s in script]
    with pytest.raises(ValueError, match="expected 3 signatures"):
        ReplayValidator().replay(FakeEnv(script), ["a1", "a2"], expected)


def test_signatures_from_steps_assembles_the_expected_list():
    init = state_signature("s0")
    steps = [state_signature("s1"), state_signature("s2")]
    out = signatures_from_steps(init, steps)
    assert len(out) == 3
    assert out[0]["obs"] == "s0" and out[2]["obs"] == "s2"
