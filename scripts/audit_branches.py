#!/usr/bin/env python
"""What survived the run, according to the data rather than the logs.

Written after a scaled branching run hit ENOSPC and lost roughly half its
checkpoints. The shard logs could not answer "what do we actually have?" -- one
shard's process was killed mid-run, so its log simply stops, and the remaining
logs report per-checkpoint errors without saying whether a partial write landed.

So this reads the branch records themselves and classifies every planned
checkpoint into exactly one of three states:

    complete    -- full same-prefix comparison present and replay-verified
    partial     -- some records, but not a usable comparison; must be re-run
    absent      -- no records at all

Partial and absent together are the re-run list, written to a file that
``03_run_branches.py --only-missing`` reproduces independently. It also reports
malformed JSONL lines, which is how a truncated write shows up on disk.

    python scripts/audit_branches.py --config configs/scale.yaml --arm B_montecarlo
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cnc import experiment, storage
from cnc.branching.runner import branch_census


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="B_montecarlo")
    ap.add_argument("--out", default=None, help="where to write the re-run list")
    args = ap.parse_args()

    cfg = experiment.load_config(args.config)
    rcfg = experiment.build_runner_config(cfg, args.arm)
    rd = experiment.run_dir(cfg)

    planned = [c["checkpoint_id"] for c in rd.read_all("checkpoints") if c["arm"] == args.arm]
    planned_set = set(planned)
    print("planned checkpoints (arm {}): {}".format(args.arm, len(planned)))

    # -- per file, so a damaged shard is visible individually -----------------
    print("\nPER SHARD FILE")
    print("  {:<38} {:>8} {:>10} {:>10}".format("file", "records", "malformed", "complete"))
    per_file_records = {}
    for path in rd.all_shards("branches"):
        recs = [r for r in storage.read_jsonl(path) if r.get("arm") == args.arm]
        per_file_records[path] = recs
        cen = branch_census(recs, rcfg.n_replicates)
        bad = storage.MALFORMED_LINES.get(path, 0)
        print("  {:<38} {:>8} {:>10} {:>10}".format(
            os.path.basename(path), len(recs), bad,
            sum(1 for v in cen.values() if v["complete"])))

    malformed_total = sum(storage.MALFORMED_LINES.values())
    if malformed_total:
        print("\n  !! {} malformed line(s) skipped -- truncated writes. "
              "Those checkpoints are re-run below.".format(malformed_total))

    # -- global census, over the ATOMIC selection.
    #
    # After a resume a checkpoint can hold records in two files: a few leftovers
    # from the attempt that died mid-write, and the full set from the rerun.
    # Unioning them would count eight continue branches where five exist and
    # would read as "complete". read_branches picks one attempt per checkpoint;
    # the report it fills in says how much was superseded.
    sel_report: dict = {}
    all_recs = [r for r in rd.read_branches(sel_report) if r.get("arm") == args.arm]
    census = branch_census(all_recs, rcfg.n_replicates)

    if sel_report.get("checkpoints_with_multiple_sources"):
        print("\nRESUME OVERLAP")
        print("  checkpoints present in >1 file  {}".format(
            sel_report["checkpoints_with_multiple_sources"]))
        print("  records superseded (dropped)    {}".format(sel_report["records_superseded"]))
        print("  duplicate branch ids dropped    {}".format(
            sel_report["duplicate_branch_ids_dropped"]))
        print("  kept per source file: {}".format(sel_report["records_per_source"]))

    complete = {cid for cid, v in census.items() if v["complete"]}
    partial = {cid for cid, v in census.items() if not v["complete"]}
    absent = planned_set - set(census)

    # Records for a checkpoint that was never planned means the config and the
    # data disagree -- worth knowing before anything is analysed.
    orphan = set(census) - planned_set

    print("\nCENSUS")
    print("  complete  {:>6}  ({:.1f}% of planned)".format(
        len(complete), 100.0 * len(complete) / max(1, len(planned))))
    print("  partial   {:>6}  (have records, unusable -- re-run)".format(len(partial)))
    print("  absent    {:>6}  (no records at all -- re-run)".format(len(absent)))
    if orphan:
        print("  orphan    {:>6}  !! records for checkpoints not in this config".format(len(orphan)))

    if partial:
        print("\n  why partial:")
        for reason, n in Counter(census[c]["reason"] for c in partial).most_common(10):
            print("    {:<40} {}".format(reason, n))

    # -- how the damage is distributed. Concentrated in a few games means those
    #    games are the problem; spread evenly means the run was.
    by_task = defaultdict(lambda: [0, 0])
    for c in rd.read_all("checkpoints"):
        if c["arm"] != args.arm:
            continue
        slot = by_task[c["task_id"]]
        slot[0] += 1
        if c["checkpoint_id"] in complete:
            slot[1] += 1
    fully_lost = [t for t, (n, ok) in by_task.items() if ok == 0]
    print("\n  games with zero usable checkpoints: {}/{}".format(len(fully_lost), len(by_task)))

    rerun = sorted(partial | absent)
    out = args.out or rd.path("rerun_{}.txt".format(args.arm))
    with open(out, "w") as fh:
        fh.write("\n".join(rerun) + ("\n" if rerun else ""))
    print("\nwrote re-run list ({} checkpoints): {}".format(len(rerun), out))

    h = storage.disk_headroom(rd.base)
    print("disk holding the run dir: {:.1f} GB free ({:.1f}%), {} inodes free ({:.1f}%)".format(
        h["free_gb"], h["pct_bytes_free"], h["free_inodes"], h["pct_inodes_free"]))
    if h["total_inodes"] and h["pct_inodes_free"] < 10:
        print("  !! inodes nearly exhausted -- THIS is what returns ENOSPC while df -h looks fine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
