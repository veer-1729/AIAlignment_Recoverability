#!/usr/bin/env python
"""Stage 1, step 5: descriptive analysis. Strictly training-free.

The scalar ``g`` used throughout is the **empirical** continuation-failure
signal measured from the branches -- Arm A's realized failure indicator, Arm B's
``1 - phat_continue``. Nothing is fitted. In Arm B this ``g`` is deliberately the
strongest possible continuation-risk scalar (an oracle one), so any abstraction
loss against it is a conservative lower bound on what a learned score would incur.

Two questions are answered separately, per the review:

  Q1 (Arm A, replication) -- did we reproduce the paper's same-risk /
      different-control phenomenon?
  Q2 (Arm B, extension)   -- are the Qhat targets stable and non-degenerate
      enough for Stage-2 probes?

Usage:
    python scripts/05_analyze.py --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from cnc import experiment, utility
from cnc.storage import ARMS, read_jsonl

N_BINS = 10


def derive(branches, cps_by_id, params=utility.PAPER_PARAMS):
    """Raw branch records -> per-checkpoint values. Pure re-derivation."""
    grouped = defaultdict(list)
    for b in branches:
        grouped[(b["checkpoint_id"], b["arm"])].append(b)

    rows = []
    for (cid, arm), rs in sorted(grouped.items()):
        if not all(x["replay_verified"] for x in rs):
            continue  # checkpoint dropped whole; see runner.usable_checkpoints
        if {x["action"] for x in rs} != set(utility.ACTIONS):
            continue
        cv = utility.values_from_branches(cid, arm, rs, params)
        row = cv.to_row()
        cp = cps_by_id.get(cid, {})
        row["t"] = cp.get("t")
        row["task_id"] = cp.get("task_id")
        row["rollout_id"] = cp.get("rollout_id")
        row["task_type"] = cp.get("task_type", "")
        row["confidence_t"] = cp.get("confidence_t")
        row["split_paper"] = cp.get("split_paper", "")
        row["split_task_level"] = cp.get("split_task_level", "")
        rows.append(row)
    return rows


def abstraction_loss(rows, n_bins=N_BINS):
    """Gap(g) = E[max_a V_a] - E[max_a mean(V_a | bin(g))].

    The oracle picks per state; the scalar controller must first average all
    states sharing a scalar value and only then choose. The difference is the
    value destroyed by routing through ``g``.
    """
    if not rows:
        return 0.0, 0.0
    g = np.array([r["p_fail_continue"] for r in rows])
    V = np.array([[r["value_{}".format(a)] for a in utility.ACTIONS] for r in rows])

    oracle = V.max(axis=1).mean()
    # Unique values when g is coarse (Arm A is 0/1); quantile bins otherwise.
    uniq = np.unique(g)
    if len(uniq) <= n_bins:
        bins = np.searchsorted(uniq, g)
    else:
        edges = np.quantile(g, np.linspace(0, 1, n_bins + 1)[1:-1])
        bins = np.searchsorted(edges, g)

    total = 0.0
    for b in np.unique(bins):
        m = bins == b
        best = V[m].mean(axis=0).argmax()  # one action for the whole bin
        total += V[m][:, best].sum()
    return float(oracle - total / len(rows)), float(oracle)


def conflict_mass(rows, n_bins=N_BINS):
    """Probability mass in risk bins where the oracle action is not unanimous.

    This is the operational form of "same continuation risk, different optimal
    action". Non-trivial mass means a controller routing on continuation risk
    alone must be wrong somewhere, no matter how well calibrated it is.
    """
    if not rows:
        return 0.0, {}
    g = np.array([r["p_fail_continue"] for r in rows])
    act = [r["oracle_action"] for r in rows]
    uniq = np.unique(g)
    if len(uniq) <= n_bins:
        bins = np.searchsorted(uniq, g)
    else:
        edges = np.quantile(g, np.linspace(0, 1, n_bins + 1)[1:-1])
        bins = np.searchsorted(edges, g)

    conflicted = 0
    detail = {}
    for b in np.unique(bins):
        m = bins == b
        acts = Counter(a for a, keep in zip(act, m) if keep)
        detail["bin{}".format(int(b))] = dict(acts)
        if len(acts) > 1:
            conflicted += int(m.sum())
    return conflicted / len(rows), detail


def tau_spread_within_bands(rows, n_bins=N_BINS):
    """Spread of intervention advantage inside narrow continuation-risk bands."""
    if not rows:
        return {}
    g = np.array([r["p_fail_continue"] for r in rows])
    tau = np.array([r["tau"] for r in rows])
    uniq = np.unique(g)
    if len(uniq) <= n_bins:
        bins = np.searchsorted(uniq, g)
    else:
        edges = np.quantile(g, np.linspace(0, 1, n_bins + 1)[1:-1])
        bins = np.searchsorted(edges, g)

    out = {}
    for b in np.unique(bins):
        m = bins == b
        if m.sum() < 2:
            continue
        out["bin{}".format(int(b))] = {
            "n": int(m.sum()),
            "risk_mean": float(g[m].mean()),
            "tau_mean": float(tau[m].mean()),
            "tau_sd": float(tau[m].std(ddof=1)),
            "tau_min": float(tau[m].min()),
            "tau_max": float(tau[m].max()),
            "straddles_zero": bool((tau[m] > 0).any() and (tau[m] < 0).any()),
        }
    return out


def plot(rows, arm, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not rows:
        return None

    colors = {"continue": "#2b7bba", "intervene": "#d1495b", "quit": "#8d99ae"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for a in utility.ACTIONS:
        sel = [r for r in rows if r["oracle_action"] == a]
        if sel:
            ax.scatter([r["p_fail_continue"] for r in sel], [r["tau"] for r in sel],
                       s=22, alpha=0.7, label="oracle={}".format(a), color=colors[a])
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("empirical continuation failure risk")
    ax.set_ylabel(r"intervention advantage  $\tau$")
    ax.set_title("Same risk, different control ({})".format(arm))
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist([r["tau"] for r in rows], bins=25, color="#5b8c5a", edgecolor="white")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("checkpoints")
    ax.set_title("Intervention advantage distribution")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def summarise(rows, arm, lines):
    lines.append("\n## Arm: {}  (n = {} checkpoints)\n".format(arm, len(rows)))
    if not rows:
        lines.append("_No usable checkpoints._\n")
        return {}

    est = rows[0]["n_continue"] > 1
    lines.append("Quantity reported: **{}**\n".format(
        r"$\hat{Q}(s,a)$ (Monte-Carlo estimate)" if est else "$U(s,a)$ (realized, single suffix)"))

    oa = Counter(r["oracle_action"] for r in rows)
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for a in utility.ACTIONS:
        lines.append("| oracle = `{}` | {} ({:.1%}) |".format(a, oa[a], oa[a] / len(rows)))
    for a in ("continue", "intervene"):
        v = [r["p_success_{}".format(a)] for r in rows]
        lines.append("| mean success `{}` | {:.3f} |".format(a, float(np.mean(v))))
        lines.append("| mean value `{}` | {:.3f} |".format(
            a, float(np.mean([r["value_{}".format(a)] for r in rows]))))
    tau = np.array([r["tau"] for r in rows])
    lines.append("| tau mean / sd | {:.3f} / {:.3f} |".format(tau.mean(), tau.std(ddof=1) if len(tau) > 1 else 0.0))
    lines.append("| tau > 0 | {:.1%} |".format(float((tau > 0).mean())))
    lines.append("| tau range | [{:.3f}, {:.3f}] |".format(tau.min(), tau.max()))

    gap, oracle_v = abstraction_loss(rows)
    cm, cdetail = conflict_mass(rows)
    lines.append("| oracle value | {:.3f} |".format(oracle_v))
    lines.append("| **scalar abstraction loss Gap(g)** | **{:.3f}** |".format(gap))
    lines.append("| **conflict-set mass** | **{:.1%}** |".format(cm))

    if est:
        se = np.array([r["tau_se"] for r in rows])
        lines.append("| tau MC SE p50 / p90 | {:.3f} / {:.3f} |".format(
            float(np.percentile(se, 50)), float(np.percentile(se, 90))))
        ratio = float(tau.std(ddof=1) / max(1e-9, np.mean(se))) if len(tau) > 1 else 0.0
        lines.append("| tau spread / mean MC SE | {:.2f} |".format(ratio))

    bands = tau_spread_within_bands(rows)
    straddling = [b for b, v in bands.items() if v["straddles_zero"]]
    lines.append("\nRisk bands where tau straddles zero (same risk, opposite sign of "
                 "intervention advantage): **{}/{}**\n".format(len(straddling), len(bands)))
    lines.append("| risk band | n | risk | tau mean | tau sd | tau min | tau max |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for b, v in sorted(bands.items()):
        lines.append("| {} | {} | {:.2f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            b, v["n"], v["risk_mean"], v["tau_mean"], v["tau_sd"], v["tau_min"], v["tau_max"]))

    lines.append("\nOracle action by risk bin: `{}`\n".format(json.dumps(cdetail)))

    by_type = defaultdict(list)
    for r in rows:
        by_type[r.get("task_type") or "?"].append(r["tau"])
    if len(by_type) > 1:
        lines.append("\nBy task type:\n")
        lines.append("| task type | n | tau mean |")
        lines.append("| --- | --- | --- |")
        for k in sorted(by_type):
            lines.append("| {} | {} | {:.3f} |".format(k, len(by_type[k]), float(np.mean(by_type[k]))))

    return {"gap": gap, "conflict_mass": cm, "n": len(rows),
            "oracle_actions": dict(oa), "tau_mean": float(tau.mean()),
            "tau_sd": float(tau.std(ddof=1)) if len(tau) > 1 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = experiment.load_config(args.config)
    rd = experiment.run_dir(cfg)
    cps_by_id = {c["checkpoint_id"]: c for c in rd.read_all("checkpoints")}
    tasks = {t["task_id"]: t for t in read_jsonl(rd.tasks)}
    for c in cps_by_id.values():
        c["task_type"] = tasks.get(c["task_id"], {}).get("task_type", "")
    branches = list(rd.read_all("branches"))

    os.makedirs("reports/figures", exist_ok=True)
    lines = ["# Stage 1 descriptive analysis - run `{}`".format(cfg["run_id"]),
             "",
             "Training-free. The scalar `g` is the **measured** continuation-failure",
             "rate, not a learned score; no model is fitted anywhere in this report.",
             ""]
    summary = {}

    for arm in ARMS:
        rows = derive(branches, cps_by_id)
        rows = [r for r in rows if r["arm"] == arm]
        summary[arm] = summarise(rows, arm, lines)
        if rows:
            p = plot(rows, arm, "reports/figures/{}_{}.png".format(cfg["run_id"], arm))
            if p:
                lines.append("\n![{}]({})\n".format(arm, os.path.relpath(p, "reports")))
            # persist derived values
            out = rd.derived(arm, "checkpoint_stats.json")
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=2)

    # ---- utility sensitivity (paper Table 6 analogue) --------------------
    lines.append("\n## Utility sensitivity (5x5 cost sweep)\n")
    lines.append("| arm | c_a | w | conflict mass | Gap(g) | oracle=intervene |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for arm in ARMS:
        for params in utility.cost_sweep_grid():
            rows = [r for r in derive(branches, cps_by_id, params) if r["arm"] == arm]
            if not rows:
                continue
            cm, _ = conflict_mass(rows)
            gap, _ = abstraction_loss(rows)
            frac_i = sum(1 for r in rows if r["oracle_action"] == "intervene") / len(rows)
            lines.append("| {} | {:.2f} | {:.2f} | {:.1%} | {:.3f} | {:.1%} |".format(
                arm, params.intervention_cost, params.wrong_penalty, cm, gap, frac_i))

    lines.append("\n## Verdict\n")
    a, b = summary.get(ARMS[0], {}), summary.get(ARMS[1], {})
    lines.append("- **Q1 (replication, Arm A)**: conflict-set mass "
                 "{:.1%}, Gap(g) {:.3f} over n={}.".format(
                     a.get("conflict_mass", 0.0), a.get("gap", 0.0), a.get("n", 0)))
    lines.append("- **Q2 (extension, Arm B)**: n={}, tau sd {:.3f}.".format(
        b.get("n", 0), b.get("tau_sd", 0.0)))
    lines.append("")

    out = "reports/analysis_{}.md".format(cfg["run_id"])
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))
    print("\n[05] wrote {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
