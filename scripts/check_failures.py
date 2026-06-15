import json
outdir = "outputs/20260611_142721_4e2f69bf"
lines = open(f"{outdir}/instance_results.jsonl").readlines()
for l in lines:
    d = json.loads(l)
    failures = []
    if not d.get("highlighting_valid"): failures.append("H")
    if not d.get("rationale_valid"): failures.append("R")
    if not d.get("counterfactual_valid"): failures.append("CF")
    if not d.get("rank_ordering_valid"): failures.append("RO")
    if failures:
        print(f"{d['instance_id']}: invalid={failures}")
        if "CF" in failures:
            raw = d.get("raw_counterfactual", "")
            print(f"  CF raw: {raw[:150]}")
        if "RO" in failures:
            raw = d.get("raw_rank_ordering", "")
            print(f"  RO raw: {raw[:150]}")
        print()
