import html
import logging
import random
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from src.utils.exceptions import DataLoadError

logger = logging.getLogger(__name__)


def load_dataset(huggingface_id: str, split: str = "train", cache_dir: str = None):
    from datasets import load_dataset as hf_load_dataset
    return hf_load_dataset(huggingface_id, split=split, cache_dir=cache_dir)


def clean_text(text: str) -> str:
    """Strip HTML entities and markup from raw dataset text.

    Single source of truth shared by ``sample_balanced`` and the curation
    pipeline (``src/load/curator.py``). Mirrors ``pre_clean_text`` in
    ``scripts/run_experiment.py``: unescape entities, drop tags, recover orphaned
    numeric entities that lost their ``&`` prefix, then remove any leftover named
    entities. AG News in particular is riddled with these artifacts.
    """
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'#(\d+);', lambda m: chr(int(m.group(1))), text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    return text


@dataclass
class Instance:
    instance_id: str = ""
    text: str = ""
    label: str = ""
    dataset: str = ""
    split: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'instance_id': self.instance_id,
            'text': self.text,
            'label': self.label,
            'dataset': self.dataset,
            'split': self.split,
        }


@dataclass
class DatasetStats:
    dataset_name: str
    total_count: int
    label_distribution: Dict[str, int]
    average_text_length: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dataset_name': self.dataset_name,
            'total_count': self.total_count,
            'label_distribution': self.label_distribution,
            'average_text_length': self.average_text_length,
        }


class DatasetLoader:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def load_dataset(self, huggingface_id: str, split: str) -> Any:
        try:
            dataset = load_dataset(huggingface_id, split=split, cache_dir=None)
            logger.info(f"Loaded dataset {huggingface_id} (split={split}): {len(dataset)} instances")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset {huggingface_id}: {e}")
            raise DataLoadError(f"Failed to load dataset {huggingface_id}: {e}", error_code="DLE001")

    def sample_balanced(self, dataset, n_samples: int, label_field: str = "label",
                        text_field: str = "text", dataset_name: str = "", split: str = "",
                        secondary_text_field: Optional[str] = None,
                        label_names: List[str] = None) -> List[Instance]:
        labels_to_instances = {}
        for item in dataset:
            raw_label = str(item[label_field])
            if label_names and raw_label.isdigit():
                idx = int(raw_label)
                label = label_names[idx] if 0 <= idx < len(label_names) else raw_label
            else:
                label = raw_label
            if label not in labels_to_instances:
                labels_to_instances[label] = []
            text = clean_text(str(item[text_field]))
            if secondary_text_field and secondary_text_field in item and item[secondary_text_field]:
                secondary = clean_text(str(item[secondary_text_field]))
                text = f"Premise: {text}\nHypothesis: {secondary}"
            labels_to_instances[label].append(
                Instance(text=text, label=label, dataset=dataset_name, split=split)
            )

        labels = sorted(labels_to_instances.keys())
        if not labels:
            logger.warning("No instances found for any label after filtering")
            return []
        target_per_label = max(n_samples // len(labels), 1)

        available_per_label = [len(labels_to_instances[l]) for l in labels]
        adjusted_per_label = min(target_per_label, *available_per_label)

        result = []
        for label in labels:
            pool = labels_to_instances[label]
            if len(pool) <= adjusted_per_label:
                result.extend(pool)
            else:
                result.extend(self.rng.sample(pool, adjusted_per_label))

        self.rng.shuffle(result)
        for i, inst in enumerate(result):
            inst.instance_id = f"{dataset_name}_{split}_{i:04d}" if dataset_name else f"inst_{i:04d}"

        logger.info(f"Sampled {len(result)} balanced instances ({n_samples} target, {len(labels)} labels)")
        return result

    def export_to_file(self, instances: List[Instance], filepath: str) -> None:
        import json
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            for inst in instances:
                f.write(json.dumps(inst.to_dict()) + '\n')
        logger.info(f"Exported {len(instances)} instances to {filepath}")

    def load_curated(self, filepath: str) -> List[Instance]:
        """Load a frozen curated dataset (produced by src/load/curator.py).

        Each line is an Instance dict plus curation metadata (predicted_label,
        correct, length_bucket, genre); the extra keys are stashed in
        ``Instance.metadata`` so downstream code that only reads the core fields
        is unaffected.
        """
        import json
        path = Path(filepath)
        if not path.exists():
            raise DataLoadError(f"Curated dataset not found: {filepath}", error_code="DLE007")
        core = {'instance_id', 'text', 'label', 'dataset', 'split'}
        instances: List[Instance] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                meta = {k: v for k, v in d.items() if k not in core}
                instances.append(Instance(
                    instance_id=d.get('instance_id', ''),
                    text=d.get('text', ''),
                    label=d.get('label', ''),
                    dataset=d.get('dataset', ''),
                    split=d.get('split', ''),
                    metadata=meta,
                ))
        logger.info(f"Loaded {len(instances)} curated instances from {filepath}")
        return instances

    def compute_statistics(self, instances: List[Instance]) -> DatasetStats:
        if not instances:
            raise DataLoadError("Cannot compute statistics for empty dataset", error_code="DLE006")
        label_dist = {}
        total_len = 0
        for inst in instances:
            label_dist[inst.label] = label_dist.get(inst.label, 0) + 1
            total_len += len(inst.text)
        return DatasetStats(
            dataset_name=instances[0].dataset,
            total_count=len(instances),
            label_distribution=label_dist,
            average_text_length=total_len / len(instances),
        )

    def load_and_prepare_dataset(self, config: dict, output_dir: Path):
        huggingface_id = config['huggingface_id']
        split = config.get('split', 'train')
        sample_size = config.get('sample_size', 100)
        label_field = config.get('label_field', 'label')
        text_field = config.get('text_field', 'text')
        dataset_name = config.get('name', huggingface_id)
        secondary_text_field = config.get('secondary_text_field')
        dataset = self.load_dataset(huggingface_id, split)
        instances = self.sample_balanced(
            dataset=dataset, n_samples=sample_size, label_field=label_field,
            text_field=text_field, secondary_text_field=secondary_text_field,
            dataset_name=dataset_name, split=split,
        )
        stats = self.compute_statistics(instances)
        export_path = Path(output_dir) / f"{dataset_name}_sampled.jsonl"
        self.export_to_file(instances, str(export_path))
        return instances, stats
