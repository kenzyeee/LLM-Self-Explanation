import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.metrics.validity_checker import ValidityChecker, FlipResult


class TestValidityChecker:
    @pytest.fixture
    def checker(self):
        mock_engine = Mock()
        mock_engine.classify_with_mask = AsyncMock()
        mock_engine.classify_with_mask.return_value.predicted_label = "different"
        return ValidityChecker(mock_engine)

    @pytest.mark.asyncio
    async def test_consensus_core_removal_flip(self, checker):
        result = await checker.test_consensus_core_removal(
            "This movie was great", "positive", {"great"}
        )
        assert result.flipped is True
        assert "great" in result.masked_tokens

    @pytest.mark.asyncio
    async def test_consensus_core_removal_no_flip(self):
        mock_engine = Mock()
        mock_engine.classify_with_mask = AsyncMock()
        mock_engine.classify_with_mask.return_value.predicted_label = "positive"
        checker = ValidityChecker(mock_engine)
        result = await checker.test_consensus_core_removal(
            "This movie was great", "positive", {"great"}
        )
        assert result.flipped is False

    @pytest.mark.asyncio
    async def test_random_removal_baseline(self, checker):
        result = await checker.test_random_removal_baseline(
            "This movie was great and wonderful", "positive", 2
        )
        assert len(result.masked_tokens) == 2
        assert isinstance(result.flipped, bool)

    @pytest.mark.asyncio
    async def test_random_removal_limited_tokens(self, checker):
        result = await checker.test_random_removal_baseline(
            "short", "positive", 10
        )
        assert len(result.masked_tokens) <= 1  # only 1 token available

    def test_flip_detection(self):
        result = FlipResult(original_prediction="a", masked_prediction="a", flipped=False, masked_tokens=set())
        assert result.flipped is False

        result = FlipResult(original_prediction="a", masked_prediction="b", flipped=True, masked_tokens=set())
        assert result.flipped is True

    def test_flip_rate_computation(self):
        results = [
            FlipResult("a", "a", False, set()),
            FlipResult("a", "b", True, set()),
            FlipResult("a", "c", True, set()),
        ]
        rate = sum(1 for r in results if r.flipped) / len(results)
        assert rate == 2.0 / 3.0

    def test_flip_rate_all_false(self):
        results = [
            FlipResult("a", "a", False, set()),
            FlipResult("a", "a", False, set()),
        ]
        rate = sum(1 for r in results if r.flipped) / len(results)
        assert rate == 0.0

    def test_flip_rate_all_true(self):
        results = [
            FlipResult("a", "b", True, set()),
            FlipResult("a", "c", True, set()),
        ]
        rate = sum(1 for r in results if r.flipped) / len(results)
        assert rate == 1.0

    def test_flip_rate_empty(self):
        assert sum(1 for r in [] if r.flipped) == 0.0

    def test_flip_result_to_dict(self):
        result = FlipResult("pos", "neg", True, {"great", "good"})
        d = result.to_dict()
        assert d["original_prediction"] == "pos"
        assert d["masked_prediction"] == "neg"
        assert d["flipped"] is True
        assert "great" in d["masked_tokens"]
        assert "good" in d["masked_tokens"]

    def test_flip_result_to_dict_empty_tokens(self):
        result = FlipResult()
        d = result.to_dict()
        assert d["flipped"] is False
        assert d["masked_tokens"] == []

    @pytest.mark.asyncio
    async def test_consensus_core_removal_exception(self):
        mock_engine = Mock()
        mock_engine.classify_with_mask = AsyncMock(side_effect=RuntimeError("API error"))
        checker = ValidityChecker(mock_engine)
        result = await checker.test_consensus_core_removal(
            "This movie was great", "positive", {"great"}
        )
        assert result.masked_prediction == ""
        assert result.flipped is False

    @pytest.mark.asyncio
    async def test_random_removal_baseline_exception(self):
        mock_engine = Mock()
        mock_engine.classify_with_mask = AsyncMock(side_effect=RuntimeError("API error"))
        checker = ValidityChecker(mock_engine)
        result = await checker.test_random_removal_baseline(
            "This movie was great and wonderful", "positive", 2
        )
        assert result.masked_prediction == ""
        assert result.flipped is False
