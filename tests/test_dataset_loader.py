"""
Unit tests for dataset loader.

Tests cover:
- Dataset loading from Hugging Face
- Balanced sampling with various label distributions
- Export to JSONL format
- Statistics computation

Requirements: 23.1
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from collections import Counter

from src.load.dataset_loader import DatasetLoader, Instance, DatasetStats
from src.utils.exceptions import DataLoadError


class TestDatasetLoader:
    """Test suite for DatasetLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Create a DatasetLoader instance with fixed seed."""
        return DatasetLoader(seed=42)
    
    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset with balanced labels."""
        # Create mock dataset with 100 instances, 2 labels (0 and 1)
        mock_data = []
        for i in range(100):
            mock_data.append({
                'sentence': f'This is test sentence {i}',
                'label': i % 2  # Alternating labels 0, 1
            })
        
        mock_ds = Mock()
        mock_ds.__len__ = Mock(return_value=100)
        mock_ds.__iter__ = Mock(return_value=iter(mock_data))
        mock_ds.__getitem__ = Mock(side_effect=lambda idx: mock_data[idx])
        
        return mock_ds
    
    @pytest.fixture
    def mock_imbalanced_dataset(self):
        """Create a mock dataset with imbalanced labels."""
        # 80 instances of label 0, 20 instances of label 1
        mock_data = []
        for i in range(80):
            mock_data.append({
                'text': f'Text {i}',
                'label': 0
            })
        for i in range(20):
            mock_data.append({
                'text': f'Text {80+i}',
                'label': 1
            })
        
        mock_ds = Mock()
        mock_ds.__len__ = Mock(return_value=100)
        mock_ds.__iter__ = Mock(return_value=iter(mock_data))
        mock_ds.__getitem__ = Mock(side_effect=lambda idx: mock_data[idx])
        
        return mock_ds
    
    @pytest.fixture
    def mock_mnli_dataset(self):
        """Create a mock MNLI dataset with premise and hypothesis."""
        mock_data = []
        for i in range(90):
            mock_data.append({
                'premise': f'Premise {i}',
                'hypothesis': f'Hypothesis {i}',
                'label': i % 3  # Three labels: 0, 1, 2
            })
        
        mock_ds = Mock()
        mock_ds.__len__ = Mock(return_value=90)
        mock_ds.__iter__ = Mock(return_value=iter(mock_data))
        mock_ds.__getitem__ = Mock(side_effect=lambda idx: mock_data[idx])
        
        return mock_ds
    
    def test_initialization(self, loader):
        """Test DatasetLoader initialization with seed."""
        assert loader.seed == 42
    
    @patch('src.load.dataset_loader.load_dataset')
    def test_load_dataset_success(self, mock_load, loader):
        """Test successful dataset loading from Hugging Face."""
        # Setup mock
        mock_ds = Mock()
        mock_ds.__len__ = Mock(return_value=872)
        mock_load.return_value = mock_ds
        
        # Load dataset
        result = loader.load_dataset('stanfordnlp/sst2', 'validation')
        
        # Verify
        mock_load.assert_called_once_with(
            'stanfordnlp/sst2',
            split='validation',
            cache_dir=None
        )
        assert result == mock_ds
    
    @patch('src.load.dataset_loader.load_dataset')
    def test_load_dataset_failure(self, mock_load, loader):
        """Test dataset loading failure raises DataLoadError."""
        # Setup mock to raise exception
        mock_load.side_effect = Exception("Dataset not found")
        
        # Verify exception
        with pytest.raises(DataLoadError) as exc_info:
            loader.load_dataset('invalid/dataset', 'train')
        
        assert exc_info.value.error_code == "DLE001"
        assert "invalid/dataset" in str(exc_info.value)
    
    def test_sample_balanced_equal_distribution(self, loader, mock_dataset):
        """Test balanced sampling with equal label distribution."""
        # Sample 40 instances (should get 20 per label)
        instances = loader.sample_balanced(
            dataset=mock_dataset,
            n_samples=40,
            label_field='label',
            text_field='sentence',
            dataset_name='test',
            split='validation'
        )
        
        # Verify total count
        assert len(instances) == 40
        
        # Verify label distribution is balanced
        label_counts = Counter(inst.label for inst in instances)
        assert label_counts['0'] == 20
        assert label_counts['1'] == 20
        
        # Verify instance structure
        for inst in instances:
            assert inst.instance_id.startswith('test_')
            assert inst.dataset == 'test'
            assert inst.split == 'validation'
            assert inst.label in ['0', '1']
    
    def test_sample_balanced_imbalanced_source(self, loader, mock_imbalanced_dataset):
        """Test balanced sampling from imbalanced dataset."""
        # Sample 40 instances from dataset with 80/20 split
        instances = loader.sample_balanced(
            dataset=mock_imbalanced_dataset,
            n_samples=40,
            label_field='label',
            text_field='text',
            dataset_name='imbalanced',
            split='train'
        )
        
        # Should get 20 per label (balanced despite imbalanced source)
        assert len(instances) == 40
        
        label_counts = Counter(inst.label for inst in instances)
        assert label_counts['0'] == 20
        assert label_counts['1'] == 20
    
    def test_sample_balanced_insufficient_data(self, loader, mock_imbalanced_dataset):
        """Test handling of insufficient data for requested sample size."""
        # Request more samples than available for minority class
        # Dataset has 80 of label 0, 20 of label 1
        # Requesting 60 means 30 per label, but only 20 available for label 1
        instances = loader.sample_balanced(
            dataset=mock_imbalanced_dataset,
            n_samples=60,
            label_field='label',
            text_field='text',
            dataset_name='test',
            split='train'
        )
        
        # Should adjust to 20 per label (40 total)
        assert len(instances) == 40
        
        label_counts = Counter(inst.label for inst in instances)
        assert label_counts['0'] == 20
        assert label_counts['1'] == 20
    
    def test_sample_balanced_three_labels(self, loader, mock_mnli_dataset):
        """Test balanced sampling with three labels (MNLI case)."""
        instances = loader.sample_balanced(
            dataset=mock_mnli_dataset,
            n_samples=60,
            label_field='label',
            text_field='premise',
            secondary_text_field='hypothesis',
            dataset_name='mnli',
            split='validation'
        )
        
        # Should get 20 per label (60 total)
        assert len(instances) == 60
        
        label_counts = Counter(inst.label for inst in instances)
        assert label_counts['0'] == 20
        assert label_counts['1'] == 20
        assert label_counts['2'] == 20
        
        # Verify secondary text field is combined with [SEP]
        for inst in instances:
            assert '[SEP]' in inst.text
            assert inst.text.startswith('Premise')
            assert 'Hypothesis' in inst.text
    
    def test_export_to_file(self, loader, tmp_path):
        """Test export to JSONL file."""
        # Create sample instances
        instances = [
            Instance(
                instance_id='test_001',
                text='Sample text 1',
                label='0',
                dataset='test',
                split='train'
            ),
            Instance(
                instance_id='test_002',
                text='Sample text 2',
                label='1',
                dataset='test',
                split='train'
            )
        ]
        
        # Export
        output_path = tmp_path / 'test_export.jsonl'
        loader.export_to_file(instances, output_path)
        
        # Verify file exists
        assert output_path.exists()
        
        # Verify content
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        # Parse and verify first instance
        inst1 = json.loads(lines[0])
        assert inst1['instance_id'] == 'test_001'
        assert inst1['text'] == 'Sample text 1'
        assert inst1['label'] == '0'
        assert inst1['dataset'] == 'test'
        assert inst1['split'] == 'train'
        
        # Parse and verify second instance
        inst2 = json.loads(lines[1])
        assert inst2['instance_id'] == 'test_002'
        assert inst2['label'] == '1'
    
    def test_export_to_file_creates_directory(self, loader, tmp_path):
        """Test that export creates output directory if it doesn't exist."""
        instances = [
            Instance(
                instance_id='test_001',
                text='Sample',
                label='0',
                dataset='test',
                split='train'
            )
        ]
        
        # Use nested directory path that doesn't exist
        output_path = tmp_path / 'nested' / 'dir' / 'output.jsonl'
        loader.export_to_file(instances, output_path)
        
        # Verify file was created
        assert output_path.exists()
    
    def test_compute_statistics(self, loader):
        """Test statistics computation."""
        instances = [
            Instance('id1', 'Short', '0', 'test', 'train'),
            Instance('id2', 'A longer text sample', '0', 'test', 'train'),
            Instance('id3', 'Medium length', '1', 'test', 'train'),
            Instance('id4', 'Another medium text', '1', 'test', 'train'),
        ]
        
        stats = loader.compute_statistics(instances)
        
        # Verify statistics
        assert stats.dataset_name == 'test'
        assert stats.total_count == 4
        assert stats.label_distribution == {'0': 2, '1': 2}
        
        # Verify average text length
        expected_avg = (len('Short') + len('A longer text sample') + 
                       len('Medium length') + len('Another medium text')) / 4
        assert stats.average_text_length == expected_avg
    
    def test_compute_statistics_empty_dataset(self, loader):
        """Test statistics computation with empty dataset raises error."""
        with pytest.raises(DataLoadError) as exc_info:
            loader.compute_statistics([])
        
        assert exc_info.value.error_code == "DLE006"
        assert "empty" in str(exc_info.value).lower()
    
    def test_instance_to_dict(self):
        """Test Instance serialization to dictionary."""
        inst = Instance(
            instance_id='test_001',
            text='Sample text',
            label='positive',
            dataset='sst2',
            split='validation'
        )
        
        d = inst.to_dict()
        
        assert d['instance_id'] == 'test_001'
        assert d['text'] == 'Sample text'
        assert d['label'] == 'positive'
        assert d['dataset'] == 'sst2'
        assert d['split'] == 'validation'
    
    def test_dataset_stats_to_dict(self):
        """Test DatasetStats serialization to dictionary."""
        stats = DatasetStats(
            dataset_name='sst2',
            total_count=200,
            label_distribution={'0': 100, '1': 100},
            average_text_length=52.5
        )
        
        d = stats.to_dict()
        
        assert d['dataset_name'] == 'sst2'
        assert d['total_count'] == 200
        assert d['label_distribution'] == {'0': 100, '1': 100}
        assert d['average_text_length'] == 52.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestModuleLevelLoadDataset:
    def test_load_dataset_calls_hf(self):
        from src.load.dataset_loader import load_dataset
        with patch('datasets.load_dataset') as mock_hf:
            mock_hf.return_value = "dataset"
            result = load_dataset("test/ds", split="train", cache_dir="/tmp")
            mock_hf.assert_called_once_with("test/ds", split="train", cache_dir="/tmp")
            assert result == "dataset"
