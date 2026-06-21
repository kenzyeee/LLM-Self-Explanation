import pytest
from src.metrics.redaction_test import RedactionTest, RedactionTestResult


class TestRedactionTestResult:
    def test_default_construction(self):
        r = RedactionTestResult()
        assert r.faithfulness == 0.0
        assert r.flip_at_k is None
        assert r.n_tokens == 0
        assert r.predictions == []

    def test_to_dict_roundtrip(self):
        r = RedactionTestResult(faithfulness=0.6, flip_at_k=2, n_tokens=5,
                                predictions=["pos", "pos", "neg", "neg", "neg", "neg"])
        d = r.to_dict()
        assert d["faithfulness"] == 0.6
        assert d["flip_at_k"] == 2
        assert d["n_tokens"] == 5
        r2 = RedactionTestResult.from_dict(d)
        assert r2.faithfulness == 0.6
        assert r2.flip_at_k == 2
        assert r2.n_tokens == 5

    def test_no_flip(self):
        r = RedactionTestResult(faithfulness=0.0, flip_at_k=None, n_tokens=3)
        d = r.to_dict()
        assert d["faithfulness"] == 0.0
        assert d["flip_at_k"] is None


class TestRedactionTest:
    @pytest.mark.asyncio
    async def test_empty_tokens_returns_zero(self):
        async def classify(_):
            return "pos"
        rt = RedactionTest(classify)
        result = await rt.run([], "some text", "pos")
        assert result.faithfulness == 0.0
        assert result.flip_at_k is None
        assert result.n_tokens == 0

    @pytest.mark.asyncio
    async def test_flip_at_first_token(self):
        call_count = 0

        async def classify(text):
            nonlocal call_count
            call_count += 1
            return "neg"

        rt = RedactionTest(classify)
        result = await rt.run(["bad", "awful", "terrible"], "this is a bad awful terrible movie", "pos")
        assert result.faithfulness == pytest.approx(1.0 - 1.0 / 3.0)
        assert result.flip_at_k == 1
        assert result.n_tokens == 3
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_flip_at_third_token(self):
        call_count = 0

        async def classify(text):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                return "neg"
            return "pos"

        rt = RedactionTest(classify)
        result = await rt.run(["a", "b", "c"], "a b c d e", "pos")
        assert result.faithfulness == pytest.approx(1.0 - 3.0 / 3.0)
        assert result.flip_at_k == 3
        assert result.n_tokens == 3
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_flip(self):
        async def classify(_):
            return "pos"

        rt = RedactionTest(classify)
        result = await rt.run(["a", "b", "c"], "a b c", "pos")
        assert result.faithfulness == 0.0
        assert result.flip_at_k is None
        assert result.n_tokens == 3

    @pytest.mark.asyncio
    async def test_predictions_tracked(self):
        calls = []

        async def classify(text):
            if "[MASK]" in text:
                calls.append("neg")
                return "neg"
            calls.append("pos")
            return "pos"

        rt = RedactionTest(classify)
        result = await rt.run(["bad"], "this is bad", "pos")
        assert len(result.predictions) == 2
        assert result.predictions[0] == "pos"
        assert len(calls) == 1

    def test_redact_tokens_simple(self):
        text = "This is a great movie"
        redacted = RedactionTest._redact_tokens(text, ["great"])
        assert redacted == "This is a [MASK] movie"

    def test_redact_multiple_tokens(self):
        text = "The acting was superb and the plot compelling"
        redacted = RedactionTest._redact_tokens(text, ["superb", "plot"])
        assert redacted == "The acting was [MASK] and the [MASK] compelling"

    def test_redact_case_insensitive(self):
        text = "This is GREAT"
        redacted = RedactionTest._redact_tokens(text, ["great"])
        assert redacted == "This is [MASK]"

    def test_redact_with_punctuation(self):
        text = "This is great!"
        redacted = RedactionTest._redact_tokens(text, ["great"])
        assert redacted == "This is [MASK]"

    def test_redact_no_match_leaves_unchanged(self):
        text = "This is fine"
        redacted = RedactionTest._redact_tokens(text, ["great"])
        assert redacted == "This is fine"

    def test_redact_delete_operator(self):
        text = "This is a great movie"
        redacted = RedactionTest._redact_tokens(text, ["great"], "delete")
        assert redacted == "This is a movie"

    def test_redact_delete_multiple(self):
        text = "The acting was superb and the plot compelling"
        redacted = RedactionTest._redact_tokens(text, ["superb", "plot"], "delete")
        assert redacted == "The acting was and the compelling"

    def test_mask_vs_delete_differ(self):
        text = "a great movie"
        assert RedactionTest._redact_tokens(text, ["great"], "mask") == "a [MASK] movie"
        assert RedactionTest._redact_tokens(text, ["great"], "delete") == "a movie"

    def test_invalid_operator_raises(self):
        async def classify(_):
            return "pos"
        with pytest.raises(ValueError):
            RedactionTest(classify, operator="blank")
