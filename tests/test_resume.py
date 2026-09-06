"""Resuming an interrupted run must not corrupt the checkpoints it already has.

The failure this guards against is silent. Branch ids are deterministic, so a
rerun reproduces them exactly; if a checkpoint's leftover records from a dead
attempt are concatenated with the full set from the rerun, the derivation
averages duplicated replicates and reports a standard error for a sample size it
does not have. Nothing raises. The numbers are simply wrong.
"""

import json

import pytest

from cnc.branching.runner import branch_census
from cnc.storage import (
    DiskExhausted,
    JsonlWriter,
    MALFORMED_LINES,
    RunDir,
    check_disk,
    read_jsonl,
)

N_REP = 5


def _branch(cid, action, rep, verified=True):
    return {
        "branch_id": "{}::{}::{}".format(cid, action, rep),
        "checkpoint_id": cid,
        "arm": "B_montecarlo",
        "action": action,
        "replicate": rep,
        "replay_verified": verified,
        "success": rep % 2,
        "n_steps_in_branch": 3,
    }


def _full(cid, **kw):
    return (
        [_branch(cid, "continue", i, **kw) for i in range(N_REP)]
        + [_branch(cid, "intervene", i, **kw) for i in range(N_REP)]
        + [_branch(cid, "quit", 0)]
    )


@pytest.fixture
def rd(tmp_path):
    return RunDir(str(tmp_path), "run1")


# ---------------------------------------------------------------- census


def test_census_marks_a_full_verified_checkpoint_complete(rd):
    assert branch_census(_full("cp"), N_REP)["cp"]["complete"]


def test_census_rejects_a_checkpoint_that_died_mid_write(rd):
    got = branch_census(_full("cp")[:3], N_REP)["cp"]
    assert not got["complete"]
    assert got["reason"].startswith("short:")


def test_census_rejects_an_unverified_replay(rd):
    got = branch_census(_full("cp", verified=False), N_REP)["cp"]
    assert not got["complete"] and got["reason"] == "unverified_replay"


def test_census_rejects_an_over_count_rather_than_calling_it_complete():
    """Two attempts concatenated must never read as complete.

    Counting alone would say 8 continue branches >= the 5 required and pass it.
    """
    got = branch_census(_full("cp")[:3] + _full("cp"), N_REP)["cp"]
    assert got["counts"]["continue"] == 8
    assert not got["complete"]
    assert got["reason"].startswith("DUPLICATE_RECORDS")


# ---------------------------------------------------------------- selection


def test_read_branches_prefers_the_complete_attempt_over_the_partial(rd):
    with JsonlWriter(rd.shard_path("branches", 7, 10)) as w:
        w.write_many(_full("cp_a"))
        w.write_many(_full("cp_b")[:3])          # died mid-write
    with JsonlWriter(rd.shard_path("branches", 3, 10, tag="rescue")) as w:
        w.write_many(_full("cp_b"))

    naive = branch_census(list(rd.read_all("branches")), N_REP)
    assert naive["cp_b"]["counts"]["continue"] == 8, "the hazard is real"

    report = {}
    atomic = branch_census(list(rd.read_branches(report)), N_REP)
    assert atomic["cp_b"]["complete"]
    assert atomic["cp_b"]["counts"] == {"continue": 5, "intervene": 5, "quit": 1}
    assert atomic["cp_a"]["complete"]
    assert report["checkpoints_with_multiple_sources"] == 1
    assert report["records_superseded"] == 3


def test_selection_does_not_depend_on_filename_order(rd):
    """The rescue file here sorts BEFORE the file holding the partial attempt.

    A "last file wins" rule would pick the corrupt one. Selection is by record
    count, which a partial attempt can never win.
    """
    with JsonlWriter(rd.shard_path("branches", 9, 10)) as w:
        w.write_many(_full("cp")[:4])
    with JsonlWriter(rd.shard_path("branches", 0, 10, tag="rescue")) as w:
        w.write_many(_full("cp"))
    assert branch_census(list(rd.read_branches()), N_REP)["cp"]["complete"]


def test_selection_is_deterministic_across_repeated_reads(rd):
    with JsonlWriter(rd.shard_path("branches", 1, 10)) as w:
        w.write_many(_full("cp"))
    with JsonlWriter(rd.shard_path("branches", 2, 10, tag="rescue")) as w:
        w.write_many(_full("cp"))                # both complete: pick one, always the same
    first = [r["branch_id"] for r in rd.read_branches()]
    assert first == [r["branch_id"] for r in rd.read_branches()]
    assert len(first) == 2 * N_REP + 1, "one attempt only, never both"


def test_untouched_run_is_unaffected_by_the_selection_rule(rd):
    """The common case -- one file per checkpoint -- must pass through verbatim."""
    with JsonlWriter(rd.shard_path("branches", 0, 2)) as w:
        w.write_many(_full("cp_a"))
    with JsonlWriter(rd.shard_path("branches", 1, 2)) as w:
        w.write_many(_full("cp_b"))
    report = {}
    got = list(rd.read_branches(report))
    assert len(got) == 2 * (2 * N_REP + 1)
    assert report["checkpoints_with_multiple_sources"] == 0
    assert report["records_superseded"] == 0


# ---------------------------------------------------------------- durability


def test_a_truncated_final_line_does_not_destroy_the_records_before_it(rd):
    path = rd.shard_path("branches", 0, 2)
    with JsonlWriter(path) as w:
        w.write_many(_full("cp"))
    with open(path, "a") as fh:
        fh.write('{"checkpoint_id": "cp_x", "action": "cont')     # killed mid-write
    MALFORMED_LINES.clear()
    assert len(list(read_jsonl(path))) == 2 * N_REP + 1
    assert MALFORMED_LINES[path] == 1


def test_strict_reads_still_raise_on_a_truncated_line(rd):
    path = rd.shard_path("branches", 0, 2)
    with open(path, "w") as fh:
        fh.write('{"a": 1}\n{"b": ')
    with pytest.raises(ValueError):
        list(read_jsonl(path, strict=True))


def test_sync_commits_records_to_disk(rd):
    path = rd.shard_path("branches", 0, 2)
    w = JsonlWriter(path)
    w.write_many(_full("cp"))
    w.sync()
    with open(path) as fh:
        assert sum(1 for _ in fh) == 2 * N_REP + 1


# ---------------------------------------------------------------- disk guards


def test_disk_guard_fires_on_bytes(tmp_path):
    with pytest.raises(DiskExhausted):
        check_disk(str(tmp_path), min_free_gb=10 ** 9)


def test_disk_guard_fires_on_inodes(tmp_path):
    with pytest.raises(DiskExhausted):
        check_disk(str(tmp_path), min_free_gb=0, min_free_inodes=10 ** 15)


def test_disk_exhaustion_cannot_be_swallowed_as_an_oserror(tmp_path):
    """The runner catches Exception per checkpoint. That is exactly how the
    original run burned hours failing 146 checkpoints against a full disk, so
    this must not be an OSError."""
    assert not issubclass(DiskExhausted, OSError)
    with pytest.raises(DiskExhausted):
        try:
            check_disk(str(tmp_path), min_free_gb=10 ** 9)
        except OSError:                                    # must not catch it
            pytest.fail("DiskExhausted was swallowed as an OSError")


def test_healthy_disk_passes_and_reports_headroom(tmp_path):
    h = check_disk(str(tmp_path), min_free_gb=0, min_free_inodes=0)
    assert h["free_bytes"] > 0 and 0 <= h["pct_bytes_free"] <= 100


# ---------------------------------------------------------------- paths


def test_tagged_shard_files_are_found_by_the_reader_glob(rd):
    plain = rd.shard_path("branches", 3, 10)
    tagged = rd.shard_path("branches", 3, 10, tag="rescue")
    assert plain != tagged
    for p in (plain, tagged):
        with JsonlWriter(p) as w:
            w.write({"checkpoint_id": "c", "arm": "x", "action": "quit",
                     "branch_id": "c::quit::0"})
    assert set(rd.all_shards("branches")) == {plain, tagged}
