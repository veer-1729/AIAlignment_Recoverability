"""The cohort check must absorb task composition and still detect execution drift.

The scaled dataset was assembled in two executions against two server processes,
and which checkpoints landed in which is not random -- it follows how far each
shard got before the disk filled, which tracks position in a task-sorted list. So
the cohorts genuinely contain different games, and a raw difference in means
between them is expected and uninformative. A check that fires on that would send
us chasing an artifact; one that cannot fire at all would miss a real problem.
"""

import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "cohort_check", os.path.join(ROOT, "scripts", "cohort_check.py"))
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

TASK_TYPES = ["pick", "clean", "heat", "cool", "look", "puttwo"]


def _sample(rng, base, n, weights, shift=0.0, noise=0.3):
    """Values driven entirely by task type, plus an optional cohort shift."""
    strata = rng.choice(TASK_TYPES, n, p=np.array(weights, dtype=float) / sum(weights))
    vals = np.array([base[t] + shift + noise * rng.standard_normal() for t in strata])
    return vals, list(strata)


@pytest.fixture
def base():
    rng = np.random.default_rng(0)
    return {t: rng.uniform(-1, 1) for t in TASK_TYPES}


def test_cohort_of_reads_the_tag_from_the_filename():
    assert cc.cohort_of("branches.s003.jsonl") == "original"
    assert cc.cohort_of("branches.s003.rescue2.jsonl") == "rescue2"
    assert cc.cohort_of("branches.jsonl") == "original"


def test_composition_difference_alone_does_not_fire(base):
    """Cohorts with very different task mixes but identical behaviour.

    The raw difference is large and entirely an artifact of which games each
    cohort holds. Stratifying must absorb it.
    """
    rng = np.random.default_rng(1)
    a, sa = _sample(rng, base, 900, [6, 6, 1, 1, 1, 1])
    b, sb = _sample(rng, base, 900, [1, 1, 6, 6, 1, 1])
    assert abs(a.mean() - b.mean()) > 0.15, "confound too small to be a real test"

    _, lo, hi = cc.stratified_bootstrap(a, sa, b, sb, np.random.default_rng(2), n_boot=500)
    assert lo <= 0 <= hi, "false positive: stratification did not absorb the confound"


def test_a_genuine_cohort_shift_is_detected(base):
    """Same skewed mixes, plus a real shift from the second execution."""
    rng = np.random.default_rng(1)
    a, sa = _sample(rng, base, 900, [6, 6, 1, 1, 1, 1])
    b, sb = _sample(rng, base, 900, [1, 1, 6, 6, 1, 1], shift=0.25)

    mean, lo, hi = cc.stratified_bootstrap(a, sa, b, sb, np.random.default_rng(2), n_boot=500)
    assert lo > 0 or hi < 0, "false negative: missed a real cohort effect"
    assert mean == pytest.approx(-0.25, abs=0.08), "effect size should be recovered"


def test_identical_cohorts_do_not_fire(base):
    rng = np.random.default_rng(3)
    a, sa = _sample(rng, base, 600, [1] * 6)
    b, sb = _sample(rng, base, 600, [1] * 6)
    _, lo, hi = cc.stratified_bootstrap(a, sa, b, sb, np.random.default_rng(4), n_boot=500)
    assert lo <= 0 <= hi


def test_strata_present_in_only_one_cohort_are_skipped(base):
    """A game only one cohort covers offers no like-for-like comparison."""
    rng = np.random.default_rng(5)
    a, sa = _sample(rng, base, 400, [1, 1, 0, 0, 0, 0])
    b, sb = _sample(rng, base, 400, [1, 1, 1, 1, 0, 0])
    mean, lo, hi = cc.stratified_bootstrap(a, sa, b, sb, np.random.default_rng(6), n_boot=300)
    assert np.isfinite(mean) and lo <= 0 <= hi


# ---------------------------------------------------------------------------
# Game-level stratification. The real dataset's cohorts differ in WHICH GAMES
# they hold, not only in task-type proportions, and games differ in difficulty
# within a task type. Stratifying on task type cannot absorb that; stratifying on
# game can, wherever a game appears in both cohorts.
# ---------------------------------------------------------------------------

def _by_game(rng, n, games, difficulty, shift=0.0, noise=0.25):
    g = rng.choice(games, n)
    v = np.array([difficulty[x] + shift + noise * rng.standard_normal() for x in g])
    return v, list(g)


def test_game_strata_absorb_what_task_type_strata_cannot():
    """Same task types on both sides, different game mixes, no execution effect.

    Task-type stratification sees identical task-type proportions and still finds
    a difference, because the games underneath differ in difficulty. Game
    stratification compares like with like and finds nothing.
    """
    rng = np.random.default_rng(7)
    games = ["g{}".format(i) for i in range(12)]
    difficulty = {g: rng.uniform(-1, 1) for g in games}
    # every game is the same task type, so task-type strata are a single bucket
    a, ga = _by_game(rng, 700, games[:8], difficulty)
    b, gb = _by_game(rng, 700, games[4:], difficulty)
    tt_a, tt_b = ["pick"] * len(a), ["pick"] * len(b)

    _, lo_t, hi_t = cc.stratified_bootstrap(a, tt_a, b, tt_b,
                                            np.random.default_rng(8), n_boot=500)
    _, lo_g, hi_g = cc.stratified_bootstrap(a, ga, b, gb,
                                            np.random.default_rng(8), n_boot=500)
    assert lo_t > 0 or hi_t < 0, "task-type strata should be fooled by a game-mix confound"
    assert lo_g <= 0 <= hi_g, "game strata should absorb it"


def test_game_strata_still_detect_a_real_execution_shift():
    """Same game-mix confound, plus a genuine shift. Game strata must still fire."""
    rng = np.random.default_rng(7)
    games = ["g{}".format(i) for i in range(12)]
    difficulty = {g: rng.uniform(-1, 1) for g in games}
    a, ga = _by_game(rng, 700, games[:8], difficulty)
    b, gb = _by_game(rng, 700, games[4:], difficulty, shift=0.30)
    _, lo, hi = cc.stratified_bootstrap(a, ga, b, gb, np.random.default_rng(8), n_boot=500)
    assert lo > 0 or hi < 0, "a real shift must survive game stratification"
