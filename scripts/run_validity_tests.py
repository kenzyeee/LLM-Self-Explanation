"""Erasure / faithfulness pass — the SEPARATE consistency axis.

This is the canonical erasure anchor for the study. It is deliberately a
*separate, post-hoc* pass over a completed collection run (instance_results.jsonl),
embodying the framing that faithfulness is an independently-validated quantity —
NOT ground truth, but a second behavioural consistency axis (stated-vs-revealed
input sensitivity, in the spirit of ERASER comprehensiveness; DeYoung et al. 2020).

For each instance it measures, relative to the model's OWN prediction:
  * Per-strategy erasure: erase a strategy's evidence token set, does the
    prediction flip? Run for H, R, CF, RO under BOTH operators (mask, delete).
  * Consensus-Core erasure: erase CC3 / CC4 token sets under both operators.
  * Random control: erase a same-size random token set, AVERAGED over
    n_random_baseline_trials, under both operators.

The headline result is CC-erasure flip rate MINUS random-erasure flip rate,
bucketed by ECS-lift tier: does cross-strategy consensus localize causally
important tokens better than chance, and does the gap grow with agreement?

Usage:
    python scripts/run_validity_tests.py                       # latest run, both operators
    python scripts/run_validity_tests.py --results-dir outputs/<run>
    python scripts/run_validity_tests.py --max-instances 6 --trials 3   # cheap smoke
"""
import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.inference.inference_engine import InferenceEngine
from src.parsing.parser import Parser
from src.utils.config_loader import load_and_validate_config, parse_command_line_args
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

_PUNCT = '.,!?;:()[]{}\'"'


def load_instance_results(filepath: str) -> List[Dict[str, Any]]:
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def erase(text: str, tokens: Set[str], operator: str, mask_token: str = "[MASK]") -> str:
    """Erase whole-word occurrences of `tokens` from `text`.

    operator="mask" -> replace with [MASK]; operator="delete" -> drop entirely.
    Matching is case-insensitive on the punctuation-stripped surface word.
    """
    toks = {t.lower() for t in tokens}
    out = []
    for w in text.split():
        clean = w.strip(_PUNCT).lower()
        if clean and clean in toks:
            if operator == "mask":
                out.append(mask_token)
            # delete: append nothing
        else:
            out.append(w)
    return " ".join(out)


async def classify(engine: InferenceEngine, parser: Parser, class_prompt: str,
                   text: str, label_set: List[str]) -> str:
    prompt = class_prompt.format(input_text=text, label_set=", ".join(label_set))
    try:
        resp = await engine._make_request(prompt, max_tokens=50)
        return parser.parse_classification(resp, label_set)
    except Exception as e:
        # Surface failures rather than silently turning them into "no flip" —
        # a rate-limited / unparseable call is UNKNOWN (None downstream), not evidence.
        logger.warning(f"erasure classify failed ({type(e).__name__}: {str(e)[:120]})")
        return ""


async def flip_after_erasure(engine, parser, class_prompt, text, tokens, operator,
                             label_set, original) -> Optional[bool]:
    """Erase `tokens` under `operator`; True if the prediction flipped. None if
    no tokens or the re-classification was unparseable."""
    if not tokens:
        return None
    pred = await classify(engine, parser, class_prompt, erase(text, set(tokens), operator), label_set)
    if not pred:
        return None
    return pred != original


async def random_flip_rate(engine, parser, class_prompt, text, n, operator,
                           label_set, original, trials, seed) -> Optional[float]:
    """Flip rate when erasing `n` RANDOM unique tokens, averaged over `trials`."""
    words = list(dict.fromkeys(w.strip(_PUNCT).lower() for w in text.split() if w.strip(_PUNCT)))
    if n <= 0 or not words:
        return None
    rng = random.Random(seed)
    k = min(n, len(words))
    flips = []
    for _ in range(trials):
        sample = set(rng.sample(words, k))
        pred = await classify(engine, parser, class_prompt, erase(text, sample, operator), label_set)
        if pred:
            flips.append(1 if pred != original else 0)
    return (sum(flips) / len(flips)) if flips else None


def _ro_tokens(data: Dict[str, Any]) -> List[str]:
    return [t for t, _ in data.get("rank_ordering_tokens", [])]


async def process_instance_erasure(data, engine, parser, class_prompt, label_set,
                                   operators, trials, seed) -> Dict[str, Any]:
    text = data["text"]
    original = data.get("predicted_label", "")
    strat_sets = {
        "H": set(data.get("highlighting_tokens", [])),
        "R": set(data.get("rationale_tokens", [])),
        "CF": set(data.get("counterfactual_tokens", [])),
        "RO": set(_ro_tokens(data)),
    }
    cc3 = set(data.get("cc3_tokens", []))
    cc4 = set(data.get("cc4_tokens", []))

    rec: Dict[str, Any] = {
        "instance_id": data["instance_id"],
        "dataset": data.get("dataset", ""),
        "model": data.get("model", ""),
        "correct": data.get("correct"),
        "ecs": data.get("ecs"),
        "ecs_lift": data.get("ecs_lift"),
        "original_prediction": original,
        "strategy_erasure": {},
        "cc3": {"size": len(cc3)},
        "cc4": {"size": len(cc4)},
        "random_cc3": {"n": len(cc3) if cc3 else len(cc4)},
    }
    if not original:
        return rec

    for s, toks in strat_sets.items():
        entry = {"n": len(toks)}
        for op in operators:
            entry[op] = await flip_after_erasure(engine, parser, class_prompt, text, toks, op, label_set, original)
        rec["strategy_erasure"][s] = entry

    for op in operators:
        rec["cc3"][op] = await flip_after_erasure(engine, parser, class_prompt, text, cc3, op, label_set, original)
        rec["cc4"][op] = await flip_after_erasure(engine, parser, class_prompt, text, cc4, op, label_set, original)

    n_random = len(cc3) if cc3 else len(cc4)
    for op in operators:
        rec["random_cc3"][f"{op}_rate"] = await random_flip_rate(
            engine, parser, class_prompt, text, n_random, op, label_set, original, trials, seed)
    return rec


def _rate(vals: List[Optional[bool]]) -> Optional[float]:
    xs = [1 if v else 0 for v in vals if v is not None]
    return (sum(xs) / len(xs)) if xs else None


def _mean(vals: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return (float(np.mean(xs)) if xs else None)


def _tier(lift: Optional[float], lo: float, hi: float) -> Optional[str]:
    if lift is None:
        return None
    if lift <= lo:
        return "low"
    if lift <= hi:
        return "mid"
    return "high"


def aggregate(records: List[Dict[str, Any]], operators: List[str]) -> Dict[str, Any]:
    overall: Dict[str, Any] = {"n": len(records), "operators": operators, "strategy_flip_rate": {}}
    for s in ["H", "R", "CF", "RO"]:
        overall["strategy_flip_rate"][s] = {
            op: _rate([r["strategy_erasure"].get(s, {}).get(op) for r in records]) for op in operators
        }
    overall["cc3_flip_rate"] = {op: _rate([r["cc3"].get(op) for r in records]) for op in operators}
    overall["cc4_flip_rate"] = {op: _rate([r["cc4"].get(op) for r in records]) for op in operators}
    overall["random_flip_rate"] = {op: _mean([r["random_cc3"].get(f"{op}_rate") for r in records]) for op in operators}
    overall["cc3_minus_random"] = {}
    for op in operators:
        cc = overall["cc3_flip_rate"][op]
        rnd = overall["random_flip_rate"][op]
        overall["cc3_minus_random"][op] = (cc - rnd) if (cc is not None and rnd is not None) else None

    # ECS-lift tiers (tertiles over available lifts)
    lifts = sorted(r["ecs_lift"] for r in records if r.get("ecs_lift") is not None)
    by_tier: Dict[str, Any] = {}
    if len(lifts) >= 3:
        lo = lifts[len(lifts) // 3]
        hi = lifts[2 * len(lifts) // 3]
        for tier in ["low", "mid", "high"]:
            sub = [r for r in records if _tier(r.get("ecs_lift"), lo, hi) == tier]
            if not sub:
                continue
            by_tier[tier] = {"n": len(sub)}
            for op in operators:
                cc = _rate([r["cc3"].get(op) for r in sub])
                rnd = _mean([r["random_cc3"].get(f"{op}_rate") for r in sub])
                by_tier[tier][f"cc3_flip_{op}"] = cc
                by_tier[tier][f"random_flip_{op}"] = rnd
                by_tier[tier][f"gap_{op}"] = (cc - rnd) if (cc is not None and rnd is not None) else None
        by_tier["_thresholds"] = {"low<=": lo, "high>": hi}
    return {"overall": overall, "by_ecs_lift_tier": by_tier}


async def run(config, args):
    base = Path(config.output.base_dir)
    results_file = None
    if args.results_dir:
        cand = Path(args.results_dir)
        results_file = cand if cand.suffix == ".jsonl" else cand / "instance_results.jsonl"
    else:
        dirs = sorted((d for d in base.iterdir() if d.is_dir() and (d / "instance_results.jsonl").exists()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if dirs:
            results_file = dirs[0] / "instance_results.jsonl"
    if not results_file or not results_file.exists():
        logger.error("No instance_results.jsonl found. Run the experiment first.")
        return

    out_dir = results_file.parent
    setup_logging(log_dir=out_dir / "logs", console_level=config.output.log_level)

    instances = load_instance_results(str(results_file))
    if args.max_instances:
        instances = instances[:args.max_instances]
    logger.info(f"Erasure pass over {len(instances)} instances from {results_file}")

    operators = list(config.validity.erasure_operators)
    trials = args.trials if args.trials else config.validity.n_random_baseline_trials

    engine = InferenceEngine(
        model_name=config.models[0].model_id,
        max_retries=config.inference.max_retries,
        concurrent_requests=config.inference.concurrent_requests,
    )
    parser = Parser()

    prompt_cache: Dict[str, str] = {}

    def class_prompt_for(ds: str) -> str:
        if ds not in prompt_cache:
            p = Path(f"prompts/classification_{ds}.txt")
            if not p.exists():
                p = Path("prompts/classification.txt")
            prompt_cache[ds] = p.read_text(encoding="utf-8")
        return prompt_cache[ds]

    records = []
    for i, data in enumerate(instances):
        ds = data.get("dataset", "")
        ds_cfg = config.get_dataset_by_name(ds)
        label_set = ds_cfg.labels if ds_cfg else ["positive", "negative"]
        rec = await process_instance_erasure(
            data, engine, parser, class_prompt_for(ds), label_set,
            operators, trials, seed=config.experiment.seed + i)
        records.append(rec)
        logger.info(f"[{i+1}/{len(instances)}] {rec['instance_id']} erasure done")

    with open(out_dir / "erasure_instances.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    agg = aggregate(records, operators)
    with open(out_dir / "aggregate_erasure.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    logger.info(f"Erasure pass complete -> {out_dir / 'aggregate_erasure.json'}")
    o = agg["overall"]
    for op in operators:
        logger.info(f"[{op}] CC3 flip={o['cc3_flip_rate'][op]} random={o['random_flip_rate'][op]} "
                    f"gap={o['cc3_minus_random'][op]}")
    return records, agg


def main():
    from dotenv import load_dotenv
    load_dotenv()
    p = argparse.ArgumentParser(description="Erasure / faithfulness pass")
    p.add_argument("--results-dir", type=str, help="Run dir or instance_results.jsonl path")
    p.add_argument("--max-instances", type=int, help="Process only the first N instances (cheap smoke)")
    p.add_argument("--trials", type=int, help="Random-control draws (overrides config n_random_baseline_trials)")
    args, _ = p.parse_known_args()
    # Use config defaults — this script reads a completed run, it does not sample.
    # Pass [] so the shared config parser ignores our custom flags in sys.argv.
    config = load_and_validate_config(args=parse_command_line_args([]))
    asyncio.run(run(config, args))


if __name__ == "__main__":
    main()
