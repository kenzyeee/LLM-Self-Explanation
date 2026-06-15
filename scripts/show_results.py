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
    with open(out / "aggregate_metrics.json") as f:
        agg = json.load(f)
    overall = next(m for m in agg if m["aggregation_level"] == "overall")

    run_id = out.name.split("_")[-1]
    print("=" * 60)
    print(f"  EXPERIMENT RESULTS - Run {run_id}")
    print(f"  Output: {out}")
    print("=" * 60)
    print(f"\n  Accuracy:     {overall['n_instances']} instances")
    print(f"  Mean ECS:     {overall['mean_ecs']:.4f}  [{overall['ecs_ci_lower']:.4f}, {overall['ecs_ci_upper']:.4f}]")
    print(f"  Spearman rho: {overall['spearman_rho']:.4f}  (p={overall['spearman_p_value']:.4f})")
    print(f"  Kendall tau:  {overall['mean_kendall_H_RO']:.4f}")
    print(f"  CC3:          {overall['pct_instances_with_cc3']:.0f}%  CC4: {overall['pct_instances_with_cc4']:.0f}%")

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

    print("\n  -- Pairwise Jaccard --")
    for p in ["H_R", "H_CF", "H_RO", "R_CF", "R_RO", "CF_RO"]:
        print(f"  {p:<7}  {overall['mean_jaccard_'+p]:.4f}")

    with open(out / "instance_results.jsonl") as f:
        insts = [json.loads(l) for l in f if l.strip()]
    valid = [i for i in insts if i.get("ecs") is not None]
    valid.sort(key=lambda x: x["ecs"])
    print(f"\n  -- Extreme Examples --")
    lo, hi = valid[0], valid[-1]
    print(f"  LOWEST ECS:  {lo['instance_id']} ({lo['ecs']:.4f})  {lo['ground_truth_label']} -> {lo['predicted_label']}")
    print(f"  HIGHEST ECS: {hi['instance_id']} ({hi['ecs']:.4f})  {hi['ground_truth_label']} -> {hi['predicted_label']}")

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

