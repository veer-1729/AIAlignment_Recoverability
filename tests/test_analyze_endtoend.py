"""Run the analysis scripts end to end on a synthetic dataset.

Added after 05_analyze crashed on the real 1,971-checkpoint run, hours into the
pipeline, on a call site left behind when a function signature changed:
`derive(branches, cps_by_id, params)` against
`derive(branches, cps_by_id, reps_by_arm, params)`, so a UtilityParams landed in
the replicate-counts slot. It sat inside the cost-sweep loop near the end of
main(), which nothing exercised.

Neither existing test could have caught it. The unit tests never import the
scripts; the import test imports them but never calls main(). A signature change
with a missed call site is invisible to both, and the cost of finding it on the
box is a full round trip.

So these run the actual scripts as subprocesses against a small synthetic run
directory. They assert nothing about the science -- the data is fabricated -- only
that every code path executes and writes its output.
"""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_REP = 5
ARMS = ("A_replication", "B_montecarlo")

CONFIG = """
run_id: synth
data_root: {data_root}
alfworld:
  split: train
  data_dir: /nonexistent
  games_per_task_type: 1
  max_episode_steps: 50
  expert_max_steps: 1000
model:
  id: Qwen/Qwen2.5-7B-Instruct
  revision: null
  template: hf
serving:
  kind: openai_compatible
  base_url: http://localhost:9
  model: Qwen/Qwen2.5-7B-Instruct
  api_key: EMPTY
scaffold:
  mode: act
  n_shots: 2
  show_admissible_commands: false
  max_action_tokens: 24
  max_confidence_tokens: 8
elicit_confidence: true
arms:
  - name: A_replication
    temperature: 0.0
    n_replicates: 1
  - name: B_montecarlo
    temperature: 0.7
    n_replicates: 5
rollouts:
  per_game: 2
branching:
  k_checkpoints: 4
splits:
  paper_counts: [8, 4, 8]
  task_level_fractions: [0.6, 0.2, 0.2]
  seed: 0
"""

TASK_TYPES = ["pick_and_place", "pick_clean_then_place", "pick_heat_then_place",
              "pick_cool_then_place", "look_at_obj", "pick_two_obj"]


def _write(path, records):
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """A small but structurally complete run: 12 games, 2 rollouts, 4 checkpoints."""
    root = tmp_path_factory.mktemp("synth")
    data_root = root / "runs"
    base = data_root / "synth"
    base.mkdir(parents=True)

    cfg = root / "synth.yaml"
    cfg.write_text(CONFIG.format(data_root=str(data_root)))

    import random
    rng = random.Random(0)

    tasks, cps, branches = [], [], []
    for gi in range(12):
        tid = "task_{:04d}".format(gi)
        tt = TASK_TYPES[gi % len(TASK_TYPES)]
        tasks.append({"task_id": tid, "game_file": "/dev/null", "task_type": tt,
                      "split_source": "train", "solvable": True})
        for arm in ARMS:
            n_rep = 1 if arm == "A_replication" else N_REP
            for ri in range(2):
                rid = "{}::{}::r{}".format(tid, arm, ri)
                for t in range(4):
                    cid = "{}::t{}".format(rid, t)
                    cps.append({
                        "checkpoint_id": cid, "rollout_id": rid, "task_id": tid,
                        "arm": arm, "t": t, "confidence_t": rng.random(),
                        "features_26d": {"step_index": t}, "rendered_prompt": "p",
                        "split_paper": "train" if gi < 8 else "test",
                        "split_task_level": "train" if gi < 8 else "test",
                    })
                    for action, n in (("continue", n_rep), ("intervene", n_rep), ("quit", 1)):
                        for rep in range(n):
                            branches.append({
                                "branch_id": "{}::{}::{}".format(cid, action, rep),
                                "checkpoint_id": cid, "arm": arm, "action": action,
                                "replicate": rep, "branch_seed": rep,
                                "replay_verified": True,
                                "success": (action != "quit") and rng.random() < (
                                    0.3 if action == "continue" else 0.75),
                                "n_steps_in_branch": rng.randint(1, 20),
                                "terminal_reason": "goal" if action != "quit" else "quit",
                            })

    _write(base / "tasks.jsonl", tasks)
    _write(base / "rollouts.jsonl", [])
    _write(base / "checkpoints.jsonl", cps)
    _write(base / "branches.jsonl", branches)
    return {"config": str(cfg), "base": str(base), "n_cps": len(cps)}


def _run(script, synth, *extra):
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", script),
         "--config", synth["config"], *extra],
        cwd=ROOT, capture_output=True, text=True, env=env, timeout=600)


def test_05_analyze_runs_every_path_including_the_cost_sweep(synth):
    r = _run("05_analyze.py", synth)
    assert r.returncode == 0, "05_analyze failed:\n{}\n{}".format(r.stdout[-3000:], r.stderr[-3000:])
    # The cost sweep is the section that was broken and it is last, so its
    # presence in the report is the proof that main() ran to the end.
    report = os.path.join(ROOT, "reports", "analysis_synth.md")
    assert os.path.exists(report)
    text = open(report).read()
    assert "cost sweep" in text.lower(), "cost-sweep section missing -- main() exited early"
    for arm in ARMS:
        assert arm in text


def test_audit_branches_runs(synth):
    r = _run("audit_branches.py", synth, "--arm", "B_montecarlo")
    assert r.returncode == 0, r.stderr[-3000:]
    assert "complete" in r.stdout


def test_cohort_check_runs_on_a_single_cohort(synth):
    r = _run("cohort_check.py", synth, "--arm", "B_montecarlo")
    assert r.returncode == 0, r.stderr[-3000:]
    assert "single cohort" in r.stdout


def test_variance_report_runs(synth):
    r = _run("variance_report.py", synth)
    assert r.returncode == 0, "variance_report failed:\n{}".format(r.stderr[-3000:])
