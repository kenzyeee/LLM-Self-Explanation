"""Dataset curation pipeline (no model).

Produces a frozen, reproducible, hand-vetted **clean-gold** set of N instances per
dataset for the full experiment, replacing the per-run random
``DatasetLoader.sample_balanced`` draw. Two deterministic stages with human (Claude)
judgment in between:

  build:
    1. Load + clean      — reuse DatasetLoader / clean_text; drop MNLI label == -1
    2. Quality filters   — length floor/ceiling, exact dedup, junk/non-English
    3. Stratified pool   — oversample ~1.5x balanced across label x [genre] x length
                           (near-dup removal applied here, at pool granularity)
       → freeze_candidates writes {dataset}_candidates.jsonl for review

  (Claude hand-vets the candidates and authors {dataset}_decisions.jsonl:
   keep/drop + reason, dropping mislabeled / ambiguous / low-quality instances.)

  finalize:
    4. apply_decisions   — keep only the vetted-clean candidates
    5. select_balanced   — stratify the kept set down to exactly `target`
    6. write_outputs     — {dataset}_curated.jsonl + {dataset}_datasheet.json

No API calls anywhere: the curated set is purely clean gold + Claude's judgment, so it
stays valid regardless of which experiment model is later used. Reproducibility rests
on the seeded candidate pool plus the committed decisions file.
"""

import json
import logging
import string
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.load.dataset_loader import Instance, clean_text
from src.load.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)

# Curation defaults — overridden per dataset by the `curation:` block in datasets.yaml.
DEFAULTS = {
    "target": 200,
    "min_content_words": 3,
    "max_chars": 2000,
    "oversample_factor": 1.5,
    "near_dup_jaccard": 0.9,
    "stratify_by": ["label", "length"],
}


def content_word_count(text: str) -> int:
    """Count content words using the same rule as ``parser.dynamic_k``:
    whitespace tokens that are not pure punctuation."""
    return sum(1 for w in text.split() if w.strip(string.punctuation))


def _normalized_key(text: str) -> str:
    """Lowercased, punctuation-stripped, whitespace-collapsed key for exact dedup."""
    toks = [w.strip(string.punctuation).lower() for w in text.split()]
    return " ".join(t for t in toks if t)


def _token_set(text: str) -> frozenset:
    return frozenset(w.strip(string.punctuation).lower()
                     for w in text.split() if w.strip(string.punctuation))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class CurationReport:
    dataset: str
    huggingface_id: str
    split: str
    seed: int
    target: int
    raw_count: int = 0
    drop_counts: Dict[str, int] = field(default_factory=dict)
    pool_size: int = 0
    vetted_kept: int = 0
    vetted_dropped: Dict[str, int] = field(default_factory=dict)
    final_count: int = 0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    genre_distribution: Dict[str, int] = field(default_factory=dict)
    length_distribution: Dict[str, int] = field(default_factory=dict)
    shortfalls: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class DatasetCurator:
    def __init__(self, seed: int = 42):
        import random
        self.seed = seed
        self.rng = random.Random(seed)
        self.loader = DatasetLoader(seed=seed)

    # ---- Stage 1: load + clean -------------------------------------------------
    def build_instances(self, dataset, ds_cfg) -> List[Instance]:
        """Map raw rows to cleaned Instances with curation metadata. Stable,
        original-position instance_ids so the decisions file lines up on re-run."""
        name = ds_cfg.name
        split = ds_cfg.split
        text_field = getattr(ds_cfg, "text_field", "text")
        label_field = getattr(ds_cfg, "label_field", "label")
        secondary_field = getattr(ds_cfg, "secondary_text_field", None)
        label_names = list(ds_cfg.labels)

        instances: List[Instance] = []
        for idx, item in enumerate(dataset):
            raw_label = str(item[label_field])
            if raw_label.isdigit():  # "-1" (MNLI no-gold) is not isdigit → stays unmapped → dropped below
                li = int(raw_label)
                label = label_names[li] if 0 <= li < len(label_names) else raw_label
            else:
                label = raw_label
            if label not in label_names:
                continue  # drops MNLI label == -1 and any unmapped label

            primary = clean_text(str(item[text_field]))
            hypothesis = ""
            if secondary_field and item.get(secondary_field):
                hypothesis = clean_text(str(item[secondary_field]))
                text = f"Premise: {primary}\nHypothesis: {hypothesis}"
            else:
                text = primary

            genre = str(item["genre"]) if "genre" in item and item["genre"] else ""
            # Floor field: hypothesis for NLI pairs, else the primary text.
            floor_text = hypothesis if hypothesis else primary
            meta = {
                "genre": genre,
                "content_words": content_word_count(floor_text),
                "char_len": len(text),
            }
            instances.append(Instance(
                instance_id=f"{name}_{split}_{idx:06d}",
                text=text, label=label, dataset=name, split=split, metadata=meta,
            ))
        return instances

    # ---- Stage 2: quality filters ---------------------------------------------
    def quality_filter(self, instances: List[Instance], cfg: dict) -> Tuple[List[Instance], Dict[str, int]]:
        drops = {"empty": 0, "too_short": 0, "too_long": 0, "junk": 0, "exact_dup": 0}
        min_words = cfg["min_content_words"]
        max_chars = cfg["max_chars"]
        seen_keys = set()
        kept: List[Instance] = []
        for inst in instances:
            t = inst.text
            if not t or not t.strip():
                drops["empty"] += 1
                continue
            if inst.metadata["content_words"] < min_words:
                drops["too_short"] += 1
                continue
            if len(t) > max_chars:
                drops["too_long"] += 1
                continue
            # Junk: leftover markup after cleaning, or a high non-ASCII ratio.
            import re
            if re.search(r'<[^>]+>|&[a-zA-Z]+;|#\d+;', t):
                drops["junk"] += 1
                continue
            non_ascii = sum(1 for c in t if ord(c) > 127)
            if len(t) and non_ascii / len(t) > 0.3:
                drops["junk"] += 1
                continue
            key = _normalized_key(t)
            if key in seen_keys:
                drops["exact_dup"] += 1
                continue
            seen_keys.add(key)
            kept.append(inst)
        return kept, drops

    # ---- length buckets --------------------------------------------------------
    def assign_length_buckets(self, instances: List[Instance]) -> None:
        counts = sorted(i.metadata["content_words"] for i in instances)
        if not counts:
            return
        n = len(counts)
        t1 = counts[n // 3]
        t2 = counts[(2 * n) // 3]
        for inst in instances:
            c = inst.metadata["content_words"]
            if c <= t1:
                bucket = "short"
            elif c <= t2:
                bucket = "medium"
            else:
                bucket = "long"
            inst.metadata["length_bucket"] = bucket

    def _stratum_key(self, inst: Instance, stratify_by: List[str]) -> tuple:
        parts = []
        for axis in stratify_by:
            if axis == "label":
                parts.append(inst.label)
            elif axis == "genre":
                parts.append(inst.metadata.get("genre", ""))
            elif axis == "length":
                parts.append(inst.metadata.get("length_bucket", "medium"))
        return tuple(parts)

    # ---- Stage 3: stratified oversampled pool ---------------------------------
    def stratified_pool(self, instances: List[Instance], cfg: dict) -> Tuple[List[Instance], int]:
        stratify_by = cfg["stratify_by"]
        pool_target = min(int(round(cfg["target"] * cfg["oversample_factor"])), len(instances))
        strata: Dict[tuple, List[Instance]] = defaultdict(list)
        for inst in instances:
            strata[self._stratum_key(inst, stratify_by)].append(inst)
        keys = sorted(strata.keys())

        selected: List[Instance] = []
        leftovers: List[Instance] = []
        base, rem = divmod(pool_target, len(keys))
        for i, k in enumerate(keys):
            quota = base + (1 if i < rem else 0)
            items = strata[k][:]
            self.rng.shuffle(items)
            selected.extend(items[:quota])
            leftovers.extend(items[quota:])

        # Near-dup removal at pool granularity (full-set O(n^2) is intractable).
        selected, n_near = self._dedup_near(selected, cfg["near_dup_jaccard"])

        # Top up to pool_target from leftovers (dup-checked) if strata were thin.
        if len(selected) < pool_target and leftovers:
            self.rng.shuffle(leftovers)
            kept_sets = [_token_set(i.text) for i in selected]
            for cand in leftovers:
                if len(selected) >= pool_target:
                    break
                cs = _token_set(cand.text)
                if all(_jaccard(cs, ks) < cfg["near_dup_jaccard"] for ks in kept_sets):
                    selected.append(cand)
                    kept_sets.append(cs)
        return selected, n_near

    def _dedup_near(self, instances: List[Instance], threshold: float) -> Tuple[List[Instance], int]:
        kept: List[Instance] = []
        kept_sets: List[frozenset] = []
        n_dropped = 0
        for inst in instances:
            ts = _token_set(inst.text)
            if any(_jaccard(ts, ks) >= threshold for ks in kept_sets):
                n_dropped += 1
                continue
            kept.append(inst)
            kept_sets.append(ts)
        return kept, n_dropped

    # ---- Stage: freeze candidate pool for review ------------------------------
    def freeze_candidates(self, pool: List[Instance], path: Path) -> Path:
        """Write the candidate pool for hand-vetting. One record per candidate with
        everything needed to judge gold-label correctness/clarity and to line up the
        decisions file by instance_id."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for inst in pool:
                f.write(json.dumps(self.to_record(inst)) + "\n")
        logger.info(f"Wrote {len(pool)} candidates -> {path}")
        return path

    # ---- Stage 4: apply hand-vetting decisions --------------------------------
    @staticmethod
    def load_decisions(path: Path) -> Dict[str, dict]:
        """Read {instance_id: {decision, reason}} from a decisions JSONL file."""
        path = Path(path)
        decisions: Dict[str, dict] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                decisions[d["instance_id"]] = d
        return decisions

    def apply_decisions(self, pool: List[Instance],
                        decisions: Dict[str, dict]) -> Tuple[List[Instance], Dict[str, int]]:
        """Keep only candidates explicitly marked keep; tally drops by reason.
        A candidate with no decision is treated as a drop (reason: 'undecided')."""
        kept: List[Instance] = []
        dropped: Dict[str, int] = defaultdict(int)
        for inst in pool:
            d = decisions.get(inst.instance_id)
            if d is None:
                dropped["undecided"] += 1
                continue
            if str(d.get("decision", "")).lower() == "keep":
                kept.append(inst)
            else:
                reason = d.get("reason") or "unspecified"
                dropped[reason] += 1
        return kept, dict(sorted(dropped.items()))

    def _stratify_pick(self, items: List[Instance], n: int, axes: List[str]) -> List[Instance]:
        """Pick `n` items spread evenly across the strata defined by `axes`
        (deterministic), topping up from leftovers if a stratum is thin."""
        n = min(n, len(items))
        if n <= 0:
            return []
        if not axes:
            pool = items[:]
            self.rng.shuffle(pool)
            return pool[:n]
        strata: Dict[tuple, List[Instance]] = defaultdict(list)
        for inst in items:
            strata[self._stratum_key(inst, axes)].append(inst)
        keys = sorted(strata.keys())
        for k in keys:
            self.rng.shuffle(strata[k])
        base, rem = divmod(n, len(keys))
        picked: List[Instance] = []
        leftovers: List[Instance] = []
        for i, k in enumerate(keys):
            q = base + (1 if i < rem else 0)
            picked.extend(strata[k][:q])
            leftovers.extend(strata[k][q:])
        if len(picked) < n and leftovers:
            self.rng.shuffle(leftovers)
            picked.extend(leftovers[:n - len(picked)])
        return picked[:n]

    # ---- Stage 5: balance the vetted-clean set down to `target` ---------------
    def select_balanced(self, kept: List[Instance], cfg: dict) -> Tuple[List[Instance], List[str]]:
        """Reduce the kept (clean-gold) set to exactly `target`, balanced by label.

        Label balance is the priority (it matters most for the study), so when
        ``label`` is a stratify axis we allocate the target evenly across labels
        first, then spread each label's allotment across the remaining axes
        (e.g. genre x length). Otherwise we stratify evenly over all axes."""
        target = min(cfg["target"], len(kept))
        stratify_by = cfg["stratify_by"]
        shortfalls: List[str] = []
        if len(kept) < cfg["target"]:
            shortfalls.append(f"only {len(kept)} vetted-clean instances available for target {cfg['target']}")

        if "label" in stratify_by:
            sub_axes = [a for a in stratify_by if a != "label"]
            by_label: Dict[str, List[Instance]] = defaultdict(list)
            for inst in kept:
                by_label[inst.label].append(inst)
            labels = sorted(by_label)
            # Even target across labels, capped by availability; redistribute deficit.
            base, rem = divmod(target, len(labels))
            quota = {l: base + (1 if i < rem else 0) for i, l in enumerate(labels)}
            deficit = 0
            for l in labels:
                if len(by_label[l]) < quota[l]:
                    deficit += quota[l] - len(by_label[l])
                    quota[l] = len(by_label[l])
            while deficit > 0:
                progressed = False
                for l in labels:
                    if deficit <= 0:
                        break
                    if quota[l] < len(by_label[l]):
                        quota[l] += 1
                        deficit -= 1
                        progressed = True
                if not progressed:
                    break
            if deficit > 0:
                shortfalls.append(f"could not fill {deficit} slots while keeping label balance")
            selected: List[Instance] = []
            for l in labels:
                selected.extend(self._stratify_pick(by_label[l], quota[l], sub_axes))
        else:
            selected = self._stratify_pick(kept, target, stratify_by)

        self.rng.shuffle(selected)
        return selected, shortfalls

    # ---- output ----------------------------------------------------------------
    @staticmethod
    def to_record(inst: Instance) -> dict:
        return {
            "instance_id": inst.instance_id,
            "text": inst.text,
            "label": inst.label,
            "dataset": inst.dataset,
            "split": inst.split,
            "length_bucket": inst.metadata.get("length_bucket", ""),
            "genre": inst.metadata.get("genre", ""),
            "content_words": inst.metadata.get("content_words", 0),
        }

    def write_outputs(self, instances: List[Instance], report: CurationReport,
                      out_dir: Path) -> Tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        curated_path = out_dir / f"{report.dataset}_curated.jsonl"
        with open(curated_path, "w", encoding="utf-8") as f:
            for inst in instances:
                f.write(json.dumps(self.to_record(inst)) + "\n")

        # Final distributions for the datasheet.
        report.final_count = len(instances)
        report.label_distribution = self._dist(instances, lambda i: i.label)
        report.genre_distribution = self._dist(instances, lambda i: i.metadata.get("genre", "") or "n/a")
        report.length_distribution = self._dist(instances, lambda i: i.metadata.get("length_bucket", ""))

        datasheet_path = out_dir / f"{report.dataset}_datasheet.json"
        with open(datasheet_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Wrote {len(instances)} curated instances -> {curated_path}")
        logger.info(f"Wrote datasheet -> {datasheet_path}")
        return curated_path, datasheet_path

    @staticmethod
    def _dist(instances: List[Instance], keyfn) -> Dict[str, int]:
        d: Dict[str, int] = defaultdict(int)
        for i in instances:
            d[str(keyfn(i))] += 1
        return dict(sorted(d.items()))
