"""Quick summary of the latest experiment run.

Usage:
    python scripts/show_results.py            # summary table
    python scripts/show_results.py --report   # full report.md
    python scripts/show_results.py --csv      # show CSV header + first rows
"""
import json
import sys
from pathlib import Path


def get_latest_output():
    outputs = sorted(d for d in Path("outputs").iterdir() if d.is_dir() and d.name[0].isdigit())
    if not outputs:
        print("No experiment outputs found.")
        sys.exit(1)
    return outputs[-1]


def show_summary(out):
    with open(out / "aggregate_metrics.json", encoding="utf-8") as f:
        agg = json.load(f)
    overall = next(m for m in agg if m["aggregation_level"] == "overall")

    run_id = out.name.split("_")[-1]
    print("=" * 60)
    print(f"  EXPERIMENT RESULTS - Run {run_id}")
    print(f"  Output: {out}")
    print("=" * 60)
    print(f"\n  Accuracy:     {overall['n_instances']} instances")
    print(f"  ECS-adj (complete, N={overall.get('n_ecs_adj_complete', 0)}): {overall.get('mean_ecs_adj_complete', 0):.4f}  <- HEADLINE (PRIMARY estimand)")
    print(f"  ECS-adj (available-component, N={overall.get('n_ecs_adj', 0)}): {overall.get('mean_ecs_adj', 0):.4f}")
    print(f"  [legacy/deprecated] Mean ECS: {overall['mean_ecs']:.4f}  [{overall['ecs_ci_lower']:.4f}, {overall['ecs_ci_upper']:.4f}]")
    print(f"  [legacy/deprecated] ECS lift/chance: {overall.get('mean_ecs_lift', 0):+.4f}  (random={overall.get('mean_ecs_random', 0):.4f})")
    print(f"  [legacy/deprecated] Complete-case ECS (N={overall['n_complete_cases']}): {overall['mean_ecs_complete']:.4f}")
    print(f"  ECS by correctness: correct={overall.get('mean_ecs_correct', 0):.4f} (N={overall.get('n_correct', 0)})  "
          f"incorrect={overall.get('mean_ecs_incorrect', 0):.4f} (N={overall.get('n_incorrect', 0)})")
    print(f"  Kendall tau:  {overall['mean_kendall_H_RO']:.4f}")
    print(f"  CC3:          {overall['pct_instances_with_cc3']:.0f}%  CC4: {overall['pct_instances_with_cc4']:.0f}%")
    print(f"  Introduced-concept rate (R): {overall.get('introduced_concept_rate', 0):.3f}")
    print(f"  CF validity:  minimal(canonical)={overall.get('cf_canonical_validity_rate', 0)*100:.0f}%  free(contrast)={overall.get('cf_contrast_validity_rate', 0)*100:.0f}%"
          f"   |  minimality: minimal={overall.get('mean_cf_canonical_minimality', 0):.3f} free={overall.get('mean_cf_contrast_minimality', 0):.3f}")

    print("\n  -- Sampling Log --")
    print(f"  {'Dataset':<10} {'Req':>3} {'Got':>3} {'Wrong':>5}")
    for m in agg:
        if m["aggregation_level"] == "model_dataset":
            parts = m["group_name"].split("_", 1)
            ds = parts[1] if len(parts) > 1 else m["group_name"]
            # wrong_pred_kept is the current field name (review P1.3); dropped_wrong_pred
            # is the pre-rename name in aggregate_metrics.json from older run directories.
            wrong_pred = m.get('wrong_pred_kept', m.get('dropped_wrong_pred', 0))
            print(f"  {ds:<10} {m['requested_samples']:>3} {m['sampled_samples']:>3} {wrong_pred:>5}")

    print("\n  -- Per Dataset --")
    print(f"  {'Dataset':<10} {'n':>3} {'ECS':>6} {'H':>4} {'R':>4} {'CF':>4} {'RO':>4}")
    for m in agg:
        if m["aggregation_level"] == "model_dataset":
            parts = m["group_name"].split("_", 1)
            ds = parts[1] if len(parts) > 1 else m["group_name"]
            print(f"  {ds:<10} {m['n_instances']:>3} {m['mean_ecs']:>6.3f}", end="")
            for k in ["highlighting_success_rate", "rationale_success_rate",
                       "counterfactual_success_rate", "rank_ordering_success_rate"]:
                print(f" {m[k]*100:>3.0f}%", end="")
            print()

    print("\n  -- ECS by Input Length --")
    print(f"  {'Length':<15} {'N':>3} {'Mean ECS':>8}")
    for bucket in ["short", "medium", "long"]:
        print(f"  {bucket:<15} {overall[f'n_{bucket}']:>3} {overall[f'mean_ecs_{bucket}']:>8.4f}")

    print("\n  -- Pairwise Jaccard --")
    for p in ["H_R", "H_CF", "H_RO", "R_CF", "R_RO", "CF_RO"]:
        print(f"  {p:<7}  {overall['mean_jaccard_'+p]:.4f}")

    with open(out / "instance_results.jsonl", encoding="utf-8") as f:
        insts = [json.loads(l) for l in f if l.strip()]
    valid = [i for i in insts if i.get("ecs") is not None]
    valid.sort(key=lambda x: x["ecs"])
    print(f"\n  -- Extreme Examples --")
    lo, hi = valid[0], valid[-1]
    print(f"  LOWEST ECS:  {lo['instance_id']} ({lo['ecs']:.4f})  {lo['ground_truth_label']} -> {lo['predicted_label']}")
    print(f"  HIGHEST ECS: {hi['instance_id']} ({hi['ecs']:.4f})  {hi['ground_truth_label']} -> {hi['predicted_label']}")

    # Erasure / faithfulness anchor (separate consistency axis), if the pass has run
    erasure_path = out / "aggregate_erasure.json"
    if erasure_path.exists():
        with open(erasure_path, encoding="utf-8") as f:
            er = json.load(f)
        o = er.get("overall", {})
        ops = o.get("operators", [])
        fmt = lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "  — "
        print("\n  -- Erasure / Faithfulness Anchor (separate consistency axis) --")
        for op in ops:
            cc = o.get("cc3_flip_rate", {}).get(op)
            rnd = o.get("random_flip_rate", {}).get(op)
            gap = o.get("cc3_minus_random", {}).get(op)
            print(f"  [{op:>6}] CC3 flip={fmt(cc)}  random={fmt(rnd)}  gap(CC3-rand)={fmt(gap)}")
        tiers = er.get("by_ecs_lift_tier", {})
        if tiers and ops:
            print(f"  CC3-minus-random gap by ECS-lift tier (operator={ops[0]}):")
            print("    CAVEAT: tertiles are data-dependent thresholds over this run's own")
            print("    lifts, no significance test is applied to the gap or its trend across")
            print("    tiers — treat as descriptive only, especially at N per tier this small.")
            for tier in ["low", "mid", "high"]:
                if tier in tiers:
                    t = tiers[tier]
                    print(f"    {tier:<5} (N={t.get('n')}): gap={fmt(t.get(f'gap_{ops[0]}'))}")
    else:
        print("\n  (No erasure pass yet — run:  python scripts/run_validity_tests.py)")

    files = sorted(f for f in out.iterdir() if f.is_file())
    print(f"\n  -- Files ({len(files)}) --")
    for f in files:
        sz = f.stat().st_size
        print(f"  {f.name:<30s} {sz:>8,} B")
    print()


def main():
    out = get_latest_output()
    if "--report" in sys.argv:
        print(open(out / "report.md", encoding="utf-8").read())
    elif "--csv" in sys.argv:
        lines = open(out / "instance_metrics.csv", encoding="utf-8").readlines()
        print("".join(lines[:6]))
        print(f"  ... ({len(lines) - 1} data rows)")
    else:
        show_summary(out)


if __name__ == "__main__":
    main()
