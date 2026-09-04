#!/usr/bin/env python
"""Why did the agent fail? Separates 'model is bad at ALFWorld' from 'scaffold is broken'.

A 0% base success rate has two very different causes, and they demand opposite
responses. If the agent emits well-formed, admissible ALFWorld actions and still
loses, that is genuine capability and the fix is the scaffold's strength. If it
emits unparseable text or inadmissible commands, something is wired wrong and no
amount of prompt tuning is the right answer.

The discriminating metric is the admissible rate: what fraction of generated
actions were actually available in that state.

    python scripts/diagnose_rollouts.py --config configs/pilot.yaml
"""
from __future__ import annotations
import argparse, os, sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cnc import experiment
from cnc.storage import ARMS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()

    cfg = experiment.load_config(args.config)
    rd = experiment.run_dir(cfg)
    rollouts = list(rd.read_all("rollouts"))
    print("rollouts: {}".format(len(rollouts)))

    for arm in ARMS:
        rs = [r for r in rollouts if r["arm"] == arm]
        if not rs:
            continue
        steps = [r["n_steps"] for r in rs]
        print("\n" + "=" * 70)
        print("ARM {}  n={}  success={}/{}".format(
            arm, len(rs), sum(r["success"] for r in rs), len(rs)))
        print("  steps: min={} mean={:.1f} max={}".format(
            min(steps), sum(steps) / len(steps), max(steps)))
        print("  terminal_reason: {}".format(
            dict(Counter(r["terminal_reason"] for r in rs))))

        all_steps = [s for r in rs for s in r["steps"]]
        if not all_steps:
            continue
        n = len(all_steps)

        # THE discriminating metric.
        admissible = sum(
            1 for s in all_steps
            if s["action"] and s["action"] in (s.get("admissible_commands") or [])
        )
        empty = sum(1 for s in all_steps if not s["action"])
        nothing = sum(1 for s in all_steps if s.get("nothing_happens"))
        print("  steps total: {}".format(n))
        print("  ADMISSIBLE actions:   {}/{} = {:.3f}   <-- the key number".format(
            admissible, n, admissible / n))
        print("  empty/unparseable:    {}/{} = {:.3f}".format(empty, n, empty / n))
        print("  'nothing happens':    {}/{} = {:.3f}".format(nothing, n, nothing / n))

        conf = [s["confidence"] for s in all_steps if s.get("confidence") is not None]
        print("  confidence parsed:    {}/{}{}".format(
            len(conf), n,
            "  mean={:.3f}".format(sum(conf) / len(conf)) if conf else ""))

        print("\n  most common generated actions:")
        for a, c in Counter(s["action"] for s in all_steps).most_common(12):
            ok = "  (admissible somewhere)" if any(
                a in (s.get("admissible_commands") or []) for s in all_steps) else "  <-- NEVER admissible"
            print("    {:>4}x  {!r}{}".format(c, a, ok))

        print("\n  sample raw completions (what the model literally emitted):")
        for s in all_steps[: args.samples]:
            print("    raw={!r}\n      -> parsed={!r}\n      -> obs={!r}".format(
                s.get("raw_completion", "")[:120], s["action"], s["obs"][:110]))

        shortest = min(rs, key=lambda r: r["n_steps"])
        if shortest["n_steps"] <= 5:
            print("\n  SHORTEST ROLLOUT ({} steps, terminal={}) -- likely the outlier:".format(
                shortest["n_steps"], shortest["terminal_reason"]))
            for s in shortest["steps"]:
                print("    t={} action={!r} raw={!r}".format(
                    s["t"], s["action"], s.get("raw_completion", "")[:80]))

    # One full prompt, so I can eyeball the scaffold the model actually sees.
    r0 = rollouts[0]
    p = r0["steps"][0]["rendered_prompt"]
    print("\n" + "=" * 70)
    print("RENDERED PROMPT AT t=1  ({} chars)".format(len(p)))
    print("---- first 700 ----\n{}".format(p[:700]))
    print("---- last 700 ----\n{}".format(p[-700:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
