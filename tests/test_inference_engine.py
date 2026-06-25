import pytest
from unittest.mock import AsyncMock, patch, MagicMock, PropertyMock
from src.inference.inference_engine import InferenceEngine, ClassificationResult, ExplanationResult, TokenUsage
from src.utils.exceptions import APIError
from unittest.mock import MagicMock
from openai import RateLimitError, APIError as OpenAIAPIError


def make_rate_limit_error(msg="rate limited"):
    mock_response = MagicMock()
    return RateLimitError(msg, response=mock_response, body=None)


# A realistic Groq per-day quota message (note the misleadingly parseable "try again in").
DAILY_LIMIT_MSG = (
    "Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_x` "
    "service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 100000, "
    "Requested 500. Please try again in 2h31m12s. Visit the docs for more info."
)


def make_api_error(msg="api error"):
    mock_request = MagicMock()
    return OpenAIAPIError(msg, request=mock_request, body=None)


def make_completion(content, finish_reason="stop", prompt_tokens=0, completion_tokens=0):
    """Build a mock chat-completion with realistic usage + finish_reason."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_choice.finish_reason = finish_reason
    type(mock_response).choices = PropertyMock(return_value=[mock_choice])
    mock_response.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return mock_response


class TestInferenceEngineInit:
    def test_init_with_api_key(self):
        engine = InferenceEngine(api_key="test-key", model_name="llama-3.3-70b")
        assert engine.api_key == "test-key"
        assert engine.model_name == "llama-3.3-70b"
        assert engine.client is not None

    @patch.dict("os.environ", {"GROQ_API_KEY": "env-key"})
    def test_init_from_env(self):
        engine = InferenceEngine(model_name="llama-3.3-70b")
        assert engine.api_key == "env-key"

    @patch.dict("os.environ", {}, clear=True)
    def test_init_no_key(self):
        engine = InferenceEngine()
        assert engine.api_key is None
        assert engine.client is None

    def test_custom_params(self):
        engine = InferenceEngine(api_key="k", max_retries=5, concurrent_requests=10, request_timeout=60)
        assert engine.max_retries == 5
        assert engine.request_timeout == 60


class TestInferenceEngineMakeRequest:
    @pytest.mark.asyncio
    async def test_no_client_raises_error(self):
        engine = InferenceEngine()
        with pytest.raises(APIError, match="GROQ_API_KEY is not set"):
            await engine._make_request("prompt")

    @pytest.mark.asyncio
    async def test_successful_request(self):
        engine = InferenceEngine(api_key="test-key")
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "test response"
        type(mock_response).choices = PropertyMock(return_value=[mock_choice])

        engine.client = AsyncMock()
        engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await engine._make_request("hello", max_tokens=100)
        assert result == "test response"
        engine.client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limit_then_retry(self):
        import groq
        engine = InferenceEngine(api_key="test-key", max_retries=2)
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "success"
        type(mock_response).choices = PropertyMock(return_value=[mock_choice])

        client_mock = AsyncMock()
        client_mock.chat.completions.create = AsyncMock(
            side_effect=[make_rate_limit_error(), mock_response]
        )
        engine.client = client_mock

        result = await engine._make_request("hello")
        assert result == "success"
        assert client_mock.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_exhausted(self):
        engine = InferenceEngine(api_key="test-key", max_retries=1)
        client_mock = AsyncMock()
        client_mock.chat.completions.create = AsyncMock(
            side_effect=make_rate_limit_error("always rate limited")
        )
        engine.client = client_mock

        with pytest.raises(APIError, match="Rate limit exceeded"):
            await engine._make_request("hello")
        assert client_mock.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_daily_rate_limit_stops_without_retrying(self):
        """A per-day quota must abort immediately, not wait out a misleading backoff."""
        from src.utils.exceptions import DailyRateLimitExhausted
        engine = InferenceEngine(api_key="test-key", max_retries=5)
        client_mock = AsyncMock()
        client_mock.chat.completions.create = AsyncMock(
            side_effect=make_rate_limit_error(DAILY_LIMIT_MSG)
        )
        engine.client = client_mock

        with pytest.raises(DailyRateLimitExhausted) as exc_info:
            await engine._make_request("hello")

        # No retries despite max_retries=5 — exactly one call, then stop.
        assert client_mock.chat.completions.create.call_count == 1
        # The real reset time is surfaced, not the misleading short retry-after.
        assert "2h 31m" in str(exc_info.value)
        assert exc_info.value.details["scope"] == "daily"
        assert exc_info.value.details["reset_seconds"] == pytest.approx(2 * 3600 + 31 * 60 + 12)

    @pytest.mark.asyncio
    async def test_api_error_then_retry(self):
        engine = InferenceEngine(api_key="test-key", max_retries=2)
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "success after api error"
        type(mock_response).choices = PropertyMock(return_value=[mock_choice])

        client_mock = AsyncMock()
        client_mock.chat.completions.create = AsyncMock(
            side_effect=[make_api_error(), mock_response]
        )
        engine.client = client_mock

        result = await engine._make_request("hello")
        assert result == "success after api error"

    @pytest.mark.asyncio
    async def test_api_error_exhausted(self):
        engine = InferenceEngine(api_key="test-key", max_retries=1)
        client_mock = AsyncMock()
        client_mock.chat.completions.create = AsyncMock(
            side_effect=make_api_error("persistent api error")
        )
        engine.client = client_mock

        with pytest.raises(APIError, match="API request failed"):
            await engine._make_request("hello")

    @pytest.mark.asyncio
    async def test_unexpected_error(self):
        engine = InferenceEngine(api_key="test-key", max_retries=1)
        client_mock = AsyncMock()
        client_mock.chat.completions.create = AsyncMock(
            side_effect=ValueError("unexpected")
        )
        engine.client = client_mock

        with pytest.raises(APIError, match="Unexpected error"):
            await engine._make_request("hello")


class TestRateLimitClassification:
    @pytest.mark.parametrize("msg", [
        "... on requests per day (RPD): Limit 1000 ...",
        "... on tokens per day (TPD): Limit 100000 ...",
        "rate limit reached on tokens per day",
    ])
    def test_detects_daily_limit(self, msg):
        assert InferenceEngine._is_daily_rate_limit(make_rate_limit_error(msg)) is True

    @pytest.mark.parametrize("msg", [
        "... on requests per minute (RPM): Limit 30 ...",
        "... on tokens per minute (TPM): Limit 6000 ...",
        "rate limited",
    ])
    def test_minute_limit_not_classified_daily(self, msg):
        assert InferenceEngine._is_daily_rate_limit(make_rate_limit_error(msg)) is False

    def test_duration_to_seconds(self):
        assert InferenceEngine._duration_to_seconds("7m12.5s") == pytest.approx(432.5)
        assert InferenceEngine._duration_to_seconds("1h2m3s") == pytest.approx(3723)
        assert InferenceEngine._duration_to_seconds("512ms") == pytest.approx(0.512)
        assert InferenceEngine._duration_to_seconds("no duration here") is None

    def test_reset_seconds_prefers_message_over_header(self):
        err = make_rate_limit_error("on tokens per day (TPD). Please try again in 2m3s.")
        assert InferenceEngine._reset_seconds(err) == pytest.approx(123)

    def test_format_duration(self):
        assert InferenceEngine._format_duration(123) == "2m 3s"
        assert InferenceEngine._format_duration(3723) == "1h 2m"
        assert InferenceEngine._format_duration(None) == "an unknown amount of time"
        assert InferenceEngine._format_duration(0) == "an unknown amount of time"


class TestInferenceEnginePublicMethods:
    @pytest.mark.asyncio
    async def test_classify(self):
        engine = InferenceEngine(api_key="test-key")
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "positive"
        type(mock_response).choices = PropertyMock(return_value=[mock_choice])
        engine.client = AsyncMock()
        engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await engine.classify("classify this")
        assert isinstance(result, ClassificationResult)
        assert result.raw_response == "positive"

    @pytest.mark.asyncio
    async def test_explain(self):
        engine = InferenceEngine(api_key="test-key")
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "explanation text"
        type(mock_response).choices = PropertyMock(return_value=[mock_choice])
        engine.client = AsyncMock()
        engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await engine.explain("explain this", strategy="R")
        assert isinstance(result, ExplanationResult)
        assert result.strategy == "R"
        assert result.raw_response == "explanation text"

    @pytest.mark.asyncio
    async def test_classify_with_mask(self):
        engine = InferenceEngine(api_key="test-key")
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "negative"
        type(mock_response).choices = PropertyMock(return_value=[mock_choice])
        engine.client = AsyncMock()
        engine.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await engine.classify_with_mask("masked input", {"bad"})
        assert isinstance(result, ClassificationResult)
        assert result.raw_response == "negative"


class TestInferenceEngineUsageAccounting:
    @pytest.mark.asyncio
    async def test_usage_captured_and_accumulated(self):
        engine = InferenceEngine(api_key="test-key")
        engine.client = AsyncMock()
        engine.client.chat.completions.create = AsyncMock(
            return_value=make_completion("ok", finish_reason="stop", prompt_tokens=37, completion_tokens=11)
        )

        content, usage = await engine.chat_with_usage([{"role": "user", "content": "hi"}], max_tokens=100)
        assert content == "ok"
        assert isinstance(usage, TokenUsage)
        assert usage.prompt_tokens == 37
        assert usage.completion_tokens == 11
        assert usage.truncated is False
        # Cumulative engine counters reflect the real API usage.
        assert engine.total_prompt_tokens == 37
        assert engine.total_completion_tokens == 11

    @pytest.mark.asyncio
    async def test_classify_returns_usage(self):
        engine = InferenceEngine(api_key="test-key")
        engine.client = AsyncMock()
        engine.client.chat.completions.create = AsyncMock(
            return_value=make_completion('{"label":"positive"}', prompt_tokens=20, completion_tokens=5)
        )
        result = await engine.classify("classify this")
        assert result.usage is not None
        assert result.usage.prompt_tokens == 20
        assert result.usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_truncation_triggers_retry_then_succeeds(self):
        engine = InferenceEngine(api_key="test-key", context_window=8192)
        engine.client = AsyncMock()
        # First call truncated (finish_reason=length), retry returns a complete response.
        engine.client.chat.completions.create = AsyncMock(side_effect=[
            make_completion("partial", finish_reason="length", prompt_tokens=30, completion_tokens=100),
            make_completion("complete", finish_reason="stop", prompt_tokens=30, completion_tokens=250),
        ])
        content, usage = await engine.chat_with_usage([{"role": "user", "content": "x"}], max_tokens=100)
        assert content == "complete"
        assert usage.truncated is False
        # Both attempts consumed tokens and are accounted for.
        assert engine.client.chat.completions.create.call_count == 2
        assert engine.total_completion_tokens == 350
        assert engine.n_truncated == 0

    @pytest.mark.asyncio
    async def test_truncation_flagged_when_still_truncated(self):
        # context_window equal to max_tokens => no room to expand => stays truncated.
        engine = InferenceEngine(api_key="test-key", context_window=100)
        engine.client = AsyncMock()
        engine.client.chat.completions.create = AsyncMock(
            return_value=make_completion("partial", finish_reason="length", prompt_tokens=30, completion_tokens=70)
        )
        content, usage = await engine.chat_with_usage([{"role": "user", "content": "x"}], max_tokens=100)
        assert usage.truncated is True
        assert engine.n_truncated == 1
        # No expansion was possible, so only one call was made.
        assert engine.client.chat.completions.create.call_count == 1
