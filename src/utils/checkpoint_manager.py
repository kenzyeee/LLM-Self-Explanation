import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, checkpoint_file: Path, force_restart: bool = False):
        self.checkpoint_file = Path(checkpoint_file)
        self.force_restart = force_restart
        if self.force_restart and self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Force restart: deleted existing checkpoint file")

    def save_checkpoint(self, results: List[Dict[str, Any]]) -> None:
        if not results:
            logger.debug("No results to save in checkpoint")
            return
        with open(self.checkpoint_file, 'a', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        logger.info(f"Saved {len(results)} instances to checkpoint ({self.checkpoint_file})")

    def load_checkpoint(self) -> List[Dict[str, Any]]:
        if not self.checkpoint_file.exists():
            logger.info("No checkpoint file found, starting fresh")
            return []
        results = []
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
            logger.info(f"Loaded {len(results)} instances from checkpoint ({self.checkpoint_file})")
        except json.JSONDecodeError as e:
            logger.error(f"Checkpoint file corrupted (invalid JSON): {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        return results

    def validate_checkpoint(self) -> bool:
        if not self.checkpoint_file.exists():
            return True
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if 'instance_id' not in data:
                            logger.warning("Checkpoint entry missing instance_id")
                            return False
            return True
        except json.JSONDecodeError:
            logger.error("Checkpoint file contains invalid JSON")
            return False
        except Exception as e:
            logger.error(f"Checkpoint validation failed: {e}")
            return False

    def skip_processed_instances(self, all_instances: List[Any], processed_ids: set) -> List[Any]:
        skipped = [inst for inst in all_instances if getattr(inst, 'instance_id', None) not in processed_ids]
        n_skipped = len(all_instances) - len(skipped)
        if n_skipped > 0:
            logger.info(f"Skipping {n_skipped} already-processed instances")
        return skipped
