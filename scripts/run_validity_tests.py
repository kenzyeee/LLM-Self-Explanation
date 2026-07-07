"""Erasure pass — the SEPARATE second consistency axis.

This is the canonical erasure instrument for the study. It is deliberately a
*separate, post-hoc* pass over a completed collection run (instance_results.jsonl),
embodying the framing that erasure behaviour is an independently-measured quantity —
NOT ground truth and NOT "faithfulness", but a second behavioural consistency axis
(stated-vs-revealed input sensitivity, in the spirit of ERASER comprehensiveness;
DeYoung et al. 2020 — with the OOD caveats of Bastings & Filippova 2020 / Hooker et
al. 2019 acknowledged).

Every record is re-classified by the SAME model that produced its original
prediction (grouped by the record's `model` field; one engine per model). Erasing
with a different model would measure cross-model transfer, not the within-model
stated-vs-revealed sensitivity the construct requires (review §8.1).

For each instance it measures, relative to the model's OWN prediction:
  * Per-strategy erasure: erase a strategy's evidence token set, does the
    prediction flip? Run for H, R, CF, RO under BOTH operators (mask, delete).
  * Consensus-Core erasure: erase CC3 / CC4 token sets under both operators.
  * Random control: erase a same-size random CONTENT-word token set, AVERAGED
    over n_random_baseline_trials, under both operators.
  * Held-out CF flip verification (optional, config validity.heldout_cf_verification):
    re-classify each self-verified counterfactual with a DIFFERENT configured model —
    a judge-choice robustness check (arXiv:2505.13972). The construct-defining flip
    stays self-verified.

The headline result is CC-erasure flip rate MINUS random-erasure flip rate,
bucketed by ECS-lift tier — with the pre-registered test family (b): a one-sided
sign-flip permutation test on the per-instance paired differences
(cc3_flip − random_rate), per model per operator, Holm-corrected within each
model's operator family. Everything else is descriptive.

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
from src.normalization.normalizer import Normalizer
from src.parsing.parser import Parser
from src.statistics.statistical_tests import sign_flip_permutation_test, holm_correction
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


def erase(text: str, tokens: Set[str], operator: str, normalizer: Optional[Normalizer] = None,
         mask_token: str = "[MASK]") -> str:
    """Erase whole-word occurrences of `tokens` from `text`.

    operator="mask" -> replace with [MASK]; operator="delete" -> drop entirely.
    Matching is case-insensitive on the punctuation-stripped surface word. With a
    normalizer, a word also matches under two morphology-aware criteria:

      1. Shared single-pass WordNet lemma with a token to erase (the same criterion
         Normalizer.is_anchored used to anchor evidence to the input) — handles simple
         inflection, e.g. evidence "movie" erasing the input occurrence "movies".
      2. Full FIXED-POINT normalization equality (review P0.3): normalization v3.0
         lemmatizes to a fixed point that can take >=2 WordNet passes ("grounds" ->
         "grind", "pass" -> "pa"), and evidence tokens are stored ALREADY fixed-point
         normalized. A single-pass anchor-lemma set for the input word ({"grounds",
         "ground"}) never intersects the evidence token ("grind"), so criterion 1
         alone silently skips exactly these tokens — understating CC/strategy flip
         rates in the headline instrument while the random control (which erases the
         surface words it sampled) is unaffected, biasing the paired CC-vs-random gap.
         Re-normalizing the input word the SAME way the evidence token was produced
         and comparing for equality closes that gap.
    """
    toks = {t.lower() for t in tokens}
    lemma_pool = set()
    if normalizer is not None:
        for t in toks:
            lemma_pool |= normalizer._anchor_lemmas(t)
    out = []
    for w in text.split():
        clean = w.strip(_PUNCT).lower()
        is_match = bool(clean) and clean in toks
        if not is_match and normalizer is not None and clean:
            is_match = bool(normalizer._anchor_lemmas(clean) & lemma_pool)
        if not is_match and normalizer is not None and clean:
            norm = normalizer.normalize(clean)
            is_match = norm is not None and norm in toks
        if is_match:
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
                             label_set, original, normalizer: Optional[Normalizer] = None) -> Optional[bool]:
    """Erase `tokens` under `operator`; True if the prediction flipped. None if
    no tokens or the re-classification was unparseable."""
    if not tokens:
        return None
    pred = await classify(engine, parser, class_prompt, erase(text, set(tokens), operator, normalizer), label_set)
    if not pred:
        return None
    return pred != original


async def random_flip_rate(engine, parser, class_prompt, text, n, operator,
                           label_set, original, trials, seed,
                           normalizer: Optional[Normalizer] = None) -> Optional[float]:
    """Flip rate when erasing `n` RANDOM unique CONTENT-word tokens, averaged over
    `trials`. The control must be matched in token type to what it's compared
    against: CC tokens are normalized content-word lemmas (stopwords/discourse
    words already excluded), not raw surface words. Drawing the random sample from
    ALL surface words — including "the", "a", "is" — makes stopword draws common;
    stopwords rarely carry the prediction, so the random flip rate is
    under-estimated and the CC-minus-random gap is inflated. Restricting the pool
    to words that survive the same content-word filter (normalizer.normalize)
    removes that mismatch.
    """
    surface_words = list(dict.fromkeys(w.strip(_PUNCT).lower() for w in text.split() if w.strip(_PUNCT)))
    if normalizer is not None:
        words = [w for w in surface_words if normalizer.normalize(w) is not None]
    else:
        words = surface_words
    if n <= 0 or not words:
        return None
    rng = random.Random(seed)
    k = min(n, len(words))
    flips = []
    for _ in range(trials):
        sample = set(rng.sample(words, k))
        pred = await classify(engine, parser, class_prompt, erase(text, sample, operator, normalizer), label_set)
        if pred:
            flips.append(1 if pred != original else 0)
    return (sum(flips) / len(flips)) if flips else None


def _ro_tokens(data: Dict[str, Any]) -> List[str]:
    # Prefer the top-k RO evidence set ECS scored (review P1.1); fall back to the full
    # ranked list for legacy records that predate rank_ordering_set.
    ro_set = data.get("rank_ordering_set")
    if ro_set:
        return list(ro_set)
    return [t for t, _ in data.get("rank_ordering_tokens", [])]


async def process_instance_erasure(data, engine, parser, class_prompt, label_set,
                                   operators, trials, seed,
                                   normalizer: Optional[Normalizer] = None,
                                   heldout_engine: Optional[InferenceEngine] = None) -> Dict[str, Any]:
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
            entry[op] = await flip_after_erasure(engine, parser, class_prompt, text, toks, op, label_set,
                                                  original, normalizer)
        rec["strategy_erasure"][s] = entry

    for op in operators:
        rec["cc3"][op] = await flip_after_erasure(engine, parser, class_prompt, text, cc3, op, label_set,
                                                   original, normalizer)
        rec["cc4"][op] = await flip_after_erasure(engine, parser, class_prompt, text, cc4, op, label_set,
                                                   original, normalizer)

    n_random = len(cc3) if cc3 else len(cc4)
    for op in operators:
        rec["random_cc3"][f"{op}_rate"] = await random_flip_rate(
            engine, parser, class_prompt, text, n_random, op, label_set, original, trials, seed, normalizer)

    # Held-out CF flip verification (judge-choice robustness, arXiv:2505.13972):
    # does a DIFFERENT model also classify the self-verified counterfactual away
    # from the original label? None = not applicable / unparseable.
    rec["cf_flip_heldout"] = None
    cf_text = data.get("cf_counterfactual_text", "")
    if heldout_engine is not None and cf_text and data.get("cf_flip_verified"):
        pred = await classify(heldout_engine, parser, class_prompt, cf_text, label_set)
        rec["cf_flip_heldout"] = (pred != original) if pred else None
        rec["cf_heldout_model"] = heldout_engine.model_name
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


def _paired_cc_random_diffs(records: List[Dict[str, Any]], op: str) -> List[float]:
    """Per-instance paired differences (cc3_flip − random_rate) for one operator.
    Pairing is within-instance; instances missing either side are excluded."""
    diffs = []
    for r in records:
        cc = r.get("cc3", {}).get(op)
        rnd = r.get("random_cc3", {}).get(f"{op}_rate")
        if cc is not None and rnd is not None:
            diffs.append((1.0 if cc else 0.0) - rnd)
    return diffs


def aggregate(records: List[Dict[str, Any]], operators: List[str],
              n_permutations: int = 10000, min_n_for_test: int = 6,
              seed: int = 42) -> Dict[str, Any]:
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

    # Held-out CF verification rate (robustness; construct-defining flip is self-verified).
    heldout_vals = [r.get("cf_flip_heldout") for r in records]
    overall["cf_flip_heldout_rate"] = _rate(heldout_vals)
    overall["n_cf_heldout_checked"] = sum(1 for v in heldout_vals if v is not None)

    # Pre-registered test family (b): CC3-erasure vs random control, per operator.
    # One-sided sign-flip permutation on within-instance paired differences; Holm
    # across the operator family. Below min_n_for_test: estimate only, no p.
    raw_ps: List[Optional[float]] = []
    diffs_by_op = {}
    for op in operators:
        diffs = _paired_cc_random_diffs(records, op)
        diffs_by_op[op] = diffs
        if len(diffs) >= min_n_for_test:
            raw_ps.append(sign_flip_permutation_test(diffs, n_permutations=n_permutations,
                                                     seed=seed, alternative="greater"))
        else:
            raw_ps.append(None)
    adj_ps = holm_correction(raw_ps)
    overall["cc3_vs_random_test"] = {
        op: {"n_paired": len(diffs_by_op[op]), "p_raw": raw_ps[i], "p_holm": adj_ps[i]}
        for i, op in enumerate(operators)
    }

    # ECS-lift tiers (tertiles over available lifts) — DESCRIPTIVE ONLY: tertile
    # thresholds are data-dependent and no trend test is applied.
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

    # Group records by the model that produced them. Erasure re-classification MUST
    # use the SAME model as the original prediction (stated-vs-revealed sensitivity
    # of ONE model); re-classifying with config.models[0] for every record measured
    # cross-model transfer for 2/3 of a multi-model run (review §8.1). Unknown model
    # ids are a hard error — silently substituting a model would corrupt the axis.
    configured_ids = {m.model_id for m in config.models}
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for data in instances:
        mid = data.get("model", "")
        by_model.setdefault(mid, []).append(data)
    unknown = sorted(set(by_model) - configured_ids)
    if unknown:
        raise RuntimeError(
            f"instance_results.jsonl contains records from model(s) not in the current "
            f"config: {unknown}. Erasure must re-classify with the SAME model that made "
            f"each prediction — add the model(s) back to config/experiment.yaml or run "
            f"against a matching results file.")

    engines: Dict[str, InferenceEngine] = {
        mid: InferenceEngine(
            model_name=mid,
            max_retries=config.inference.max_retries,
            concurrent_requests=config.inference.concurrent_requests,
        ) for mid in by_model
    }

    # Held-out CF judge: the next DIFFERENT model in config order (round-robin).
    # Only available when >=2 models are configured.
    heldout_for: Dict[str, Optional[InferenceEngine]] = {}
    use_heldout = getattr(config.validity, "heldout_cf_verification", False)
    ordered_ids = [m.model_id for m in config.models]
    for mid in by_model:
        heldout_for[mid] = None
        if use_heldout and len(ordered_ids) >= 2 and mid in ordered_ids:
            nxt = ordered_ids[(ordered_ids.index(mid) + 1) % len(ordered_ids)]
            if nxt != mid:
                heldout_for[mid] = engines.get(nxt) or InferenceEngine(
                    model_name=nxt,
                    max_retries=config.inference.max_retries,
                    concurrent_requests=config.inference.concurrent_requests,
                )

    parser = Parser()
    # Mirror the live collection run's normalization settings so erasure matches
    # evidence tokens the same way they were anchored to the input during extraction.
    normalizer = Normalizer(
        use_lemmatization=config.normalization.use_lemmatization,
        remove_stopwords=config.normalization.remove_stopwords,
        lemmatizer=config.normalization.lemmatizer,
    )

    prompt_cache: Dict[str, str] = {}

    def class_prompt_for(ds: str) -> str:
        if ds not in prompt_cache:
            p = Path(f"prompts/classification_{ds}.txt")
            if not p.exists():
                p = Path("prompts/classification.txt")
            prompt_cache[ds] = p.read_text(encoding="utf-8")
        return prompt_cache[ds]

    records = []
    i_global = 0
    for mid in ordered_ids:
        model_records = by_model.get(mid, [])
        if not model_records:
            continue
        engine = engines[mid]
        logger.info(f"Erasure: {len(model_records)} record(s) re-classified by their own model {mid}")
        for data in model_records:
            ds = data.get("dataset", "")
            ds_cfg = config.get_dataset_by_name(ds)
            label_set = ds_cfg.labels if ds_cfg else ["positive", "negative"]
            rec = await process_instance_erasure(
                data, engine, parser, class_prompt_for(ds), label_set,
                operators, trials, seed=config.experiment.seed + i_global,
                normalizer=normalizer, heldout_engine=heldout_for.get(mid))
            records.append(rec)
            i_global += 1
            logger.info(f"[{i_global}/{len(instances)}] {rec['instance_id']} ({mid}) erasure done")

    with open(out_dir / "erasure_instances.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    n_perms = getattr(config.metrics, "permutation_tests", 10000)
    min_n = getattr(config.metrics, "min_n_for_test", 6)
    # Per-model aggregates (primary reporting unit) + pooled overall (descriptive).
    agg: Dict[str, Any] = {"per_model": {}, "pooled": None}
    for mid in sorted(by_model):
        model_recs = [r for r in records if r["model"] == mid]
        if model_recs:
            agg["per_model"][mid] = aggregate(model_recs, operators, n_permutations=n_perms,
                                              min_n_for_test=min_n, seed=config.experiment.seed)
    agg["pooled"] = aggregate(records, operators, n_permutations=n_perms,
                              min_n_for_test=min_n, seed=config.experiment.seed)
    agg["_notes"] = (
        "Second consistency axis (stated-vs-revealed input sensitivity), NOT faithfulness "
        "ground truth. Each record re-classified by its OWN model. per_model is the primary "
        "reporting unit; pooled mixes models/datasets and is descriptive. cc3_vs_random_test "
        "is the pre-registered family (b): one-sided sign-flip permutation on within-instance "
        "paired differences, Holm-corrected across operators. ECS-lift tiers are descriptive "
        "(data-dependent tertiles, no trend test)."
    )
    with open(out_dir / "aggregate_erasure.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    logger.info(f"Erasure pass complete -> {out_dir / 'aggregate_erasure.json'}")
    for mid, a in agg["per_model"].items():
        o = a["overall"]
        for op in operators:
            t = o["cc3_vs_random_test"][op]
            logger.info(f"[{mid}][{op}] CC3 flip={o['cc3_flip_rate'][op]} random={o['random_flip_rate'][op]} "
                        f"gap={o['cc3_minus_random'][op]} p_holm={t['p_holm']}")
    return records, agg


def main():
    from dotenv import load_dotenv
    load_dotenv()
    p = argparse.ArgumentParser(description="Erasure pass (second consistency axis)")
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
