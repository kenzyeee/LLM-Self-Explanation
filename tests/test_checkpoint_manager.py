import json
import pytest
from pathlib import Path
from unittest.mock import patch
from src.utils.checkpoint_manager import CheckpointManager


class TestCheckpointManager:
    def test_init_creates_path(self, tmp_path):
        f = tmp_path / "checkpoint.jsonl"
        mgr = CheckpointManager(f)
        assert mgr.checkpoint_file == f
        assert not mgr.force_restart

    def test_force_restart_removes_existing(self, tmp_path):
        f = tmp_path / "checkpoint.jsonl"
        f.write_text('{"instance_id": "1"}\n')
        mgr = CheckpointManager(f, force_restart=True)
        assert not f.exists()

    def test_force_restart_no_existing(self, tmp_path):
        f = tmp_path / "checkpoint.jsonl"
        mgr = CheckpointManager(f, force_restart=True)
        # Should not crash if file doesn't exist
        assert not f.exists()

    def test_save_checkpoint(self, tmp_path):
        f = tmp_path / "checkpoint.jsonl"
        mgr = CheckpointManager(f)
        mgr.save_checkpoint([{"instance_id": "1"}, {"instance_id": "2"}])
        assert f.exists()
        content = f.read_text()
        assert "instance_id" in content
        assert content.count("\n") == 2

    def test_save_checkpoint_empty(self, tmp_path):
        f = tmp_path / "checkpoint.jsonl"
        mgr = CheckpointManager(f)
        mgr.save_checkpoint([])
        # Should not create file for empty results
        assert not f.exists()

    def test_load_checkpoint(self, tmp_path):
        f = tmp_path / "checkpoint.jsonl"
        f.write_text('{"instance_id": "1"}\n{"instance_id": "2"}\n')
        mgr = CheckpointManager(f)
        results = mgr.load_checkpoint()
        assert len(results) == 2
        assert results[0]["instance_id"] == "1"

    def test_load_checkpoint_not_exists(self, tmp_path):
        f = tmp_path / "nonexistent.jsonl"
        mgr = CheckpointManager(f)
        results = mgr.load_checkpoint()
        assert results == []

    def test_load_checkpoint_corrupted_json(self, tmp_path):
        f = tmp_path / "corrupt.jsonl"
        f.write_text('{"instance_id": "1"}\ninvalid json\n')
        mgr = CheckpointManager(f)
        with pytest.raises(json.JSONDecodeError):
            mgr.load_checkpoint()

    def test_load_checkpoint_general_error(self, tmp_path):
        f = tmp_path / "checkpoint.jsonl"
        # Create a directory with the same name to cause an error
        mgr = CheckpointManager(f)
        # Write a binary file that will cause an encoding error
        f.write_bytes(b'\x00\x01\x02')
        with pytest.raises(Exception):
            mgr.load_checkpoint()

    def test_validate_checkpoint_valid(self, tmp_path):
        f = tmp_path / "checkpoint.jsonl"
        f.write_text('{"instance_id": "1"}\n{"instance_id": "2"}\n')
        mgr = CheckpointManager(f)
        assert mgr.validate_checkpoint() is True

    def test_validate_checkpoint_not_exists(self, tmp_path):
        f = tmp_path / "nonexistent.jsonl"
        mgr = CheckpointManager(f)
        assert mgr.validate_checkpoint() is True

    def test_validate_checkpoint_missing_id(self, tmp_path):
        f = tmp_path / "checkpoint.jsonl"
        f.write_text('{"not_id": "1"}\n')
        mgr = CheckpointManager(f)
        assert mgr.validate_checkpoint() is False

    def test_validate_checkpoint_corrupted(self, tmp_path):
        f = tmp_path / "corrupt.jsonl"
        f.write_text("invalid json\n")
        mgr = CheckpointManager(f)
        assert mgr.validate_checkpoint() is False

    def test_validate_checkpoint_general_error(self, tmp_path):
        f = tmp_path / "checkpoint.jsonl"
        f.write_bytes(b'\x00\x01\x02')
        mgr = CheckpointManager(f)
        assert mgr.validate_checkpoint() is False

    def test_skip_processed_instances(self, tmp_path):
        class MockInstance:
            def __init__(self, id):
                self.instance_id = id

        instances = [MockInstance("1"), MockInstance("2"), MockInstance("3")]
        processed = {"1", "3"}
        f = tmp_path / "checkpoint.jsonl"
        mgr = CheckpointManager(f)
        remaining = mgr.skip_processed_instances(instances, processed)
        assert len(remaining) == 1
        assert remaining[0].instance_id == "2"

    def test_skip_processed_instances_none_skipped(self, tmp_path):
        class MockInstance:
            def __init__(self, id):
                self.instance_id = id

        instances = [MockInstance("1"), MockInstance("2")]
        processed = set()
        f = tmp_path / "checkpoint.jsonl"
        mgr = CheckpointManager(f)
        remaining = mgr.skip_processed_instances(instances, processed)
        assert len(remaining) == 2

    def test_skip_processed_instances_no_instance_id(self, tmp_path):
        class MockInstance:
            pass

        instances = [MockInstance()]
        processed = {"1"}
        f = tmp_path / "checkpoint.jsonl"
        mgr = CheckpointManager(f)
        remaining = mgr.skip_processed_instances(instances, processed)
        assert len(remaining) == 1

    def test_save_then_load_checkpoint(self, tmp_path):
        f = tmp_path / "checkpoint.jsonl"
        mgr = CheckpointManager(f)
        mgr.save_checkpoint([{"instance_id": "1", "data": "test"}])
        results = mgr.load_checkpoint()
        assert len(results) == 1
        assert results[0]["data"] == "test"


class TestCheckpointManagerEdgeCases:
    def test_load_checkpoint_generic_exception(self, tmp_path):
        import json
        f = tmp_path / "checkpoint.jsonl"
        f.write_text('{"instance_id": "1"}\n')
        mgr = CheckpointManager(f)
        with patch('builtins.open', side_effect=PermissionError("denied")):
            with pytest.raises(Exception):
                mgr.load_checkpoint()

    def test_validate_checkpoint_generic_exception(self, tmp_path):
        f = tmp_path / "checkpoint.jsonl"
        f.write_text('{"instance_id": "1"}\n')
        mgr = CheckpointManager(f)
        with patch('builtins.open', side_effect=PermissionError("denied")):
            assert mgr.validate_checkpoint() is False

    def test_save_checkpoint_with_results(self, tmp_path):
        f = tmp_path / "checkpoint.jsonl"
        mgr = CheckpointManager(f)
        mgr.save_checkpoint([{"instance_id": "1"}])
        assert f.exists()
        assert 'instance_id' in f.read_text()
