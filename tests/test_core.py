"""Unit tests for the Phase-1 pure-python core (no ALFWorld dependency)."""

from __future__ import annotations

import os
import tempfile

import pytest

from cnc import splits, storage, utility
from cnc.agent import confidence, prompts
from cnc.branching import checkpoints
from cnc.features import prefix
from cnc.serving.client import MockClient, ScriptedClient


# ---------------------------------------------------------------- utility


def test_quit_is_exactly_zero_under_every_parameterisation():
    for params in utility.cost_sweep_grid():
        assert utility.realized_utility(utility.QUIT, False, 0, params) == 0.0
        assert utility.realized_utility(utility.QUIT, True, 99, params) == 0.0


def test_realized_utility_matches_paper_table3():
    p = utility.PAPER_PARAMS
    # success on continue: +1, no intervention cost, 10 steps * 0.01
    assert utility.realized_utility("continue", True, 10, p) == pytest.approx(0.90)
    # failure on continue: -w = -1
    assert utility.realized_utility("continue", False, 10, p) == pytest.approx(-1.10)
    # intervene pays the 0.05 handoff cost on top
    assert utility.realized_utility("intervene", True, 10, p) == pytest.approx(0.85)


def test_arm_a_single_sample_has_no_standard_error():
    """n=1 is a realized U, not an estimate -- it carries no variance information."""
    av = utility.aggregate_action("continue", [True], [5], utility.PAPER_PARAMS)
    assert av.n == 1
    assert av.se == 0.0
    assert av.is_estimate is False


def test_arm_b_multi_sample_is_an_estimate_with_positive_se():
    av = utility.aggregate_action(
        "continue", [True, False, True, False, True], [5, 5, 5, 5, 5]
    )
    assert av.n == 5
    assert av.is_estimate is True
    assert av.se > 0.0
    assert av.p_success == pytest.approx(0.6)


def _branch(action, success, steps, rep=0):
    return {
        "branch_id": "b{}{}".format(action, rep),
        "action": action,
        "success": success,
        "n_steps_in_branch": steps,
        "replicate": rep,
        "replay_verified": True,
    }


def test_values_from_branches_derives_tau_and_oracle():
    rows = [
        _branch("continue", False, 10),
        _branch("intervene", True, 8),
    ]
    cv = utility.values_from_branches("cp1", storage.ARM_REPLICATION, rows)
    # continue: -1 - 0.10 = -1.10 ; intervene: 1 - 0.05 - 0.08 = 0.87 ; quit: 0
    assert cv.values["continue"].value == pytest.approx(-1.10)
    assert cv.values["intervene"].value == pytest.approx(0.87)
    assert cv.tau == pytest.approx(1.97)
    assert cv.oracle_action == "intervene"
    assert cv.p_fail_continue == 1.0
    assert cv.regret_of("continue") == pytest.approx(1.97)
    assert cv.regret_of("intervene") == 0.0


def test_quit_wins_when_both_branches_are_bad():
    """The three-way structure is real: quit=0 beats two failing branches."""
    rows = [_branch("continue", False, 20), _branch("intervene", False, 20)]
    cv = utility.values_from_branches("cp2", storage.ARM_REPLICATION, rows)
    assert cv.oracle_action == "quit"
    assert cv.oracle_value == 0.0


def test_unverified_branches_are_refused_not_silently_included():
    bad = _branch("continue", True, 3)
    bad["replay_verified"] = False
    with pytest.raises(ValueError, match="unverified"):
        utility.values_from_branches("cp3", storage.ARM_REPLICATION, [bad])


def test_oracle_tie_break_is_deterministic():
    """Equal values must resolve by ACTIONS order, never by dict iteration."""
    rows = [_branch("continue", False, 100), _branch("intervene", False, 95)]
    cv = utility.values_from_branches("cp4", storage.ARM_REPLICATION, rows)
    # both branches are worse than 0, so quit wins regardless
    assert cv.oracle_action == "quit"


def test_cost_sweep_is_the_papers_5x5():
    grid = utility.cost_sweep_grid()
    assert len(grid) == 25
    assert len({g.name for g in grid}) == 25


def test_aggregation_is_order_independent():
    a = utility.values_from_branches(
        "cp",
        storage.ARM_MONTECARLO,
        [_branch("continue", True, 4, 0), _branch("continue", False, 6, 1), _branch("intervene", True, 3)],
    )
    b = utility.values_from_branches(
        "cp",
        storage.ARM_MONTECARLO,
        [_branch("continue", False, 6, 1), _branch("intervene", True, 3), _branch("continue", True, 4, 0)],
    )
    assert a.to_row() == b.to_row()


# ---------------------------------------------------------------- features


def test_feature_vector_is_26_dimensional():
    """The paper states 26; our reconstruction is documented in features/prefix.py."""
    assert prefix.FEATURE_DIM == 26
    assert len(set(prefix.FEATURE_NAMES)) == 26


def test_extract_produces_exactly_the_named_features():
    feats = prefix.extract(
        step_index=3,
        history="a" * 40,
        task_desc="put a mug on the desk",
        observation="On the desk 1, you see a mug 1.",
        response="go to desk 1",
        admissible_commands=["go to desk 1", "look"],
        action="go to desk 1",
        task_type="pick_and_place_simple",
        confidence=0.8,
    )
    assert set(feats) == set(prefix.FEATURE_NAMES)
    assert feats["action_type_goto"] == 1.0
    assert feats["action_type_take"] == 0.0
    assert feats["task_type_pick_and_place_simple"] == 1.0
    assert feats["nothing_happens"] == 0.0
    assert feats["n_admissible_commands"] == 2.0
    assert len(prefix.to_vector(feats)) == 26


def test_nothing_happens_flag_and_unknown_task_type():
    feats = prefix.extract(
        step_index=1, history="", task_desc="t", observation="Nothing happens.",
        response="", admissible_commands=[], action="frobnicate the widget",
        task_type="not_a_real_type", confidence=None,
    )
    assert feats["nothing_happens"] == 1.0
    assert feats["task_type_unknown"] == 1.0
    # unparseable action -> all ten indicators zero, deliberately
    assert sum(feats["action_type_{}".format(a)] for a in prefix.ACTION_TYPES) == 0.0
    # missing confidence is uninformative (0.5), not confident-failure (0.0)
    assert feats["confidence"] == 0.5


@pytest.mark.parametrize(
    "action,expected",
    [
        ("go to cabinet 1", "goto"),
        ("take mug 1 from desk 1", "take"),
        ("put mug 1 in/on desk 1", "put"),
        ("move mug 1 to desk 1", "put"),
        ("open fridge 1", "open"),
        ("use desklamp 1", "use"),
        ("heat egg 1 with microwave 1", "heat"),
        ("clean lettuce 1 with sinkbasin 1", "clean"),
        ("examine desklamp 1", "examine"),
        ("look", None),
        ("", None),
    ],
)
def test_classify_action(action, expected):
    assert prefix.classify_action(action) == expected


# ---------------------------------------------------------------- checkpoints


def test_terminal_and_initial_states_are_never_checkpoints():
    """t=0 has no evidence; t=n_steps has no downstream branch."""
    v = checkpoints.valid_indices(10)
    assert v == list(range(1, 10))
    assert 0 not in v and 10 not in v
    assert checkpoints.valid_indices(1) == []
    assert checkpoints.valid_indices(0) == []


def test_selection_is_stratified_and_deterministic():
    picks = checkpoints.select_checkpoint_indices(40, seed=123, k=4)
    assert picks == sorted(picks)
    assert len(picks) == 4
    assert picks == checkpoints.select_checkpoint_indices(40, seed=123, k=4)
    # one pick per quartile of the valid range [1, 39]
    assert picks[0] < picks[1] < picks[2] < picks[3]
    assert picks[0] <= 11 and picks[-1] >= 29


def test_short_rollouts_contribute_what_they_can():
    assert checkpoints.select_checkpoint_indices(3, seed=1, k=4) == [1, 2]


def test_different_seeds_explore_different_checkpoints():
    a = checkpoints.select_checkpoint_indices(40, seed=1, k=4)
    b = checkpoints.select_checkpoint_indices(40, seed=2, k=4)
    assert a != b


# ---------------------------------------------------------------- splits


def _cps(n_tasks=6, n_rollouts_per_task=2, n_cp=4):
    out = []
    for t in range(n_tasks):
        for r in range(n_rollouts_per_task):
            rid = "task{}_roll{}".format(t, r)
            for c in range(n_cp):
                out.append(
                    {
                        "checkpoint_id": "{}_cp{}".format(rid, c),
                        "rollout_id": rid,
                        "task_id": "task{}".format(t),
                    }
                )
    return out


def test_paper_split_keeps_whole_trajectories_together():
    cps = _cps()
    labels = splits.assign_paper_split(cps, counts=(8, 4, 8), seed=7)
    for cp, lab in zip(cps, labels):
        cp["split_paper"] = lab
    assert splits.check_no_leakage(cps, "split_paper", "rollout_id") == []


def test_paper_split_respects_target_counts():
    cps = _cps()
    labels = splits.assign_paper_split(cps, counts=(8, 4, 8), seed=7)
    for split, target in zip(splits.SPLITS, (8, 4, 8)):
        assert labels.count(split) <= target


def test_task_level_split_holds_out_whole_games():
    cps = _cps()
    labels = splits.assign_task_level_split(cps, seed=3)
    for cp, lab in zip(cps, labels):
        cp["split_task_level"] = lab
    assert splits.check_no_leakage(cps, "split_task_level", "task_id") == []
    # holding out whole games implies whole trajectories too
    assert splits.check_no_leakage(cps, "split_task_level", "rollout_id") == []
    assert splits.UNUSED not in labels


def test_leakage_check_actually_catches_a_planted_violation():
    """A validator that never fails is worthless; prove V8 has teeth."""
    cps = _cps()
    labels = splits.assign_task_level_split(cps, seed=3)
    for cp, lab in zip(cps, labels):
        cp["split_task_level"] = lab
    cps[0]["split_task_level"] = "train"
    cps[1]["split_task_level"] = "test"
    assert cps[0]["rollout_id"] in splits.check_no_leakage(
        cps, "split_task_level", "rollout_id"
    )


# ---------------------------------------------------------------- prompts


def test_every_alfworld_task_type_has_exemplars():
    cfg = prompts.ScaffoldConfig()
    for task_type in prompts.TASK_TYPE_TO_KEY:
        ex = prompts.exemplars_for(task_type, cfg)
        assert len(ex) == 2
        assert all(e.strip() for e in ex)


def test_rendered_user_message_ends_with_the_action_cue():
    cfg = prompts.ScaffoldConfig()
    msg = prompts.render_user_message(
        "pick_and_place_simple",
        "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in a room.\nYour task is to: put a mug on the desk.",
        [("go to desk 1", "On the desk 1, you see a mug 1.")],
        cfg=cfg,
    )
    assert msg.endswith("\n>")
    assert "Welcome to TextWorld" not in msg  # banner stripped
    assert "> go to desk 1" in msg
    assert "On the desk 1, you see a mug 1." in msg


def test_admissible_commands_are_hidden_by_default():
    """Default matches the ReAct convention; showing them is an explicit opt-in."""
    obs = "You are in a room.\nYour task is to: put a mug on the desk."
    hidden = prompts.render_user_message(
        "pick_and_place_simple", obs, [], admissible_commands=["look", "inventory"]
    )
    assert "Available actions" not in hidden

    shown = prompts.render_user_message(
        "pick_and_place_simple",
        obs,
        [],
        cfg=prompts.ScaffoldConfig(show_admissible_commands=True),
        admissible_commands=["look", "inventory"],
    )
    assert "Available actions: look, inventory" in shown


def test_rendering_is_a_pure_function_of_the_prefix():
    """Stage 2 replays these strings; identical prefixes must render identically."""
    args = (
        "pick_and_place_simple",
        "You are in a room.\nYour task is to: put a mug on the desk.",
        [("go to desk 1", "You see a mug 1.")],
    )
    assert prompts.render_user_message(*args) == prompts.render_user_message(*args)


def test_chatml_template_shape_and_hash_stability():
    t = prompts.ChatMLTemplate()
    out = t.render("SYS", "USR")
    assert out.startswith("<|im_start|>system\nSYS<|im_end|>")
    assert out.endswith("<|im_start|>assistant\n")
    assert t.template_hash() == prompts.ChatMLTemplate().template_hash()


@pytest.mark.parametrize(
    "raw,expected",
    [
        (" go to cabinet 1", "go to cabinet 1"),
        ("> go to cabinet 1", "go to cabinet 1"),
        ("GO TO Cabinet 1\nOn the cabinet...", "go to cabinet 1"),
        ("\n\n  take mug 1 from desk 1  ", "take mug 1 from desk 1"),
        ("", ""),
        ("   \n  ", ""),
    ],
)
def test_parse_action(raw, expected):
    assert prompts.parse_action(raw) == expected


# ---------------------------------------------------------------- confidence


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.7", 0.7),
        (" .85", 0.85),
        ("Confidence: 0.25", 0.25),
        ("70%", 0.7),
        ("70", 0.7),
        ("1", 1.0),
        ("0", 0.0),
        ("no idea", None),
        ("", None),
    ],
)
def test_parse_confidence(raw, expected):
    got = confidence.parse_confidence(raw)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


def test_confidence_prompt_conditions_on_the_chosen_action():
    t = prompts.ChatMLTemplate()
    p = confidence.render_confidence_prompt(
        t,
        "pick_and_place_simple",
        "You are in a room.\nYour task is to: put a mug on the desk.",
        [],
        "go to desk 1",
    )
    assert "> go to desk 1" in p
    assert confidence.CONFIDENCE_QUESTION in p


# ---------------------------------------------------------------- storage


def test_hash_and_seed_derivation_are_deterministic_and_order_free():
    assert storage.stable_hash({"a": 1, "b": 2}) == storage.stable_hash({"b": 2, "a": 1})
    assert storage.derive_seed("cp1", "A", "continue", 0) == storage.derive_seed(
        "cp1", "A", "continue", 0
    )
    assert storage.derive_seed("cp1", "A", "continue", 0) != storage.derive_seed(
        "cp1", "A", "continue", 1
    )
    assert 0 <= storage.derive_seed("x") < 2**32


def test_jsonl_roundtrip_stamps_and_skips_provenance():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "recs.jsonl")
        prov = storage.Provenance(
            run_id="r1", arm=storage.ARM_REPLICATION, config_hash="abc",
            model_id="Qwen/Qwen2.5-7B-Instruct", temperature=0.0, n_replicates=1,
        )
        with storage.JsonlWriter(path, prov) as w:
            w.write({"i": 1})
            w.write({"i": 2})
        rows = list(storage.read_jsonl(path))
        assert [r["i"] for r in rows] == [1, 2]
        got = storage.read_provenance(path)
        assert got["run_id"] == "r1"
        assert got["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
        assert got["schema_version"] == storage.SCHEMA_VERSION


def test_record_constructors_reject_bad_enums():
    with pytest.raises(ValueError, match="bad arm"):
        storage.branch_record("b", "c", "not_an_arm", "continue", 0, 1, True, True, 1, "goal", [])
    with pytest.raises(ValueError, match="terminal_reason"):
        storage.branch_record(
            "b", "c", storage.ARM_REPLICATION, "continue", 0, 1, True, True, 1, "exploded", []
        )


def test_branch_records_never_carry_a_utility():
    """Raw storage must stay parameter-free so cost sweeps are re-derivations."""
    rec = storage.branch_record(
        "b1", "cp1", storage.ARM_REPLICATION, "continue", 0, 42, True, False, 7, "step_cap", []
    )
    assert not any("util" in k.lower() for k in rec)
    assert rec["success"] is False and rec["n_steps_in_branch"] == 7


# ---------------------------------------------------------------- client


def test_mock_client_applies_stop_sequences_like_the_server_would():
    c = MockClient(lambda prompt, seed: "go to desk 1\nOn the desk 1, you see")
    out = c.complete("p", max_tokens=24, temperature=0.0, seed=1, stop=["\n"])
    assert out.text == "go to desk 1"
    assert c.calls[0]["seed"] == 1


def test_mock_client_can_be_seed_dependent_for_arm_b():
    c = MockClient(lambda prompt, seed: "action {}".format(seed))
    assert c.complete("p", 24, 0.7, seed=1).text != c.complete("p", 24, 0.7, seed=2).text


def test_scripted_client_cycles():
    c = ScriptedClient(["a", "b"])
    got = [c.complete("p", 8, 0.0).text for _ in range(4)]
    assert got == ["a", "b", "a", "b"]
