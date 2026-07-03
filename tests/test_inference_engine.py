import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from botocore.exceptions import ClientError, ReadTimeoutError, NoCredentialsError

from src.inference.inference_engine import (
    InferenceEngine, ClassificationResult, ExplanationResult, TokenUsage,
)
from src.utils.exceptions import APIError, RateLimitExhausted


def make_engine(**kwargs):
    """Construct an engine with a stubbed Bedrock client (credentials + client mocked).

    boto3.Session is patched only for construction; the resulting ``engine.client`` is a
    plain MagicMock whose ``converse`` each test configures with return values / errors.
    """
    with patch("src.inference.inference_engine.boto3.Session") as mock_session:
        session = MagicMock()
        session.get_credentials.return_value = MagicMock()  # truthy => credentials present
        session.client.return_value = MagicMock()           # the bedrock-runtime client
        mock_session.return_value = session
        return InferenceEngine(**kwargs)


def make_response(text, stop_reason="end_turn", input_tokens=0, output_tokens=0):
    """A minimal Bedrock Converse response dict."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": stop_reason,
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        },
    }


def make_client_error(code="ThrottlingException", msg="rate exceeded", op="Converse"):
    """A botocore ClientError with the given service error code."""
    return ClientError({"Error": {"Code": code, "Message": msg}}, op)


class TestInferenceEngineInit:
    def test_init_with_credentials(self):
        engine = make_engine(model_name="us.meta.llama3-3-70b-instruct-v1:0")
        assert engine.model_name == "us.meta.llama3-3-70b-instruct-v1:0"
        assert engine.client is not None

    def test_region_from_env(self):
        with patch.dict("os.environ", {"AWS_REGION": "eu-west-1"}):
            engine = make_engine()
        assert engine.region == "eu-west-1"

    def test_explicit_region_overrides_env(self):
        with patch.dict("os.environ", {"AWS_REGION": "eu-west-1"}):
            engine = make_engine(region="ap-southeast-2")
        assert engine.region == "ap-southeast-2"

    def test_init_no_credentials(self):
        with patch("src.inference.inference_engine.boto3.Session") as mock_session, \
             patch.dict("os.environ", {"AWS_BEARER_TOKEN_BEDROCK": ""}, clear=False):
            session = MagicMock()
            session.get_credentials.return_value = None
            mock_session.return_value = session
            engine = InferenceEngine()
        assert engine.client is None

    def test_bearer_token_alone_is_sufficient(self):
        """A Bedrock API key (bearer token) is accepted even with no SigV4 creds."""
        with patch("src.inference.inference_engine.boto3.Session") as mock_session, \
             patch.dict("os.environ", {"AWS_BEARER_TOKEN_BEDROCK": "ABSKtest"}, clear=False):
            session = MagicMock()
            session.get_credentials.return_value = None   # no SigV4 creds
            session.client.return_value = MagicMock()
            mock_session.return_value = session
            engine = InferenceEngine()
        assert engine.client is not None
        session.client.assert_called_once()               # a client WAS built

    def test_custom_params(self):
        engine = make_engine(max_retries=5, concurrent_requests=10, request_timeout=60)
        assert engine.max_retries == 5
        assert engine.request_timeout == 60


class TestInferenceEngineMakeRequest:
    @pytest.mark.asyncio
    async def test_no_client_raises_error(self):
        with patch("src.inference.inference_engine.boto3.Session") as mock_session, \
             patch.dict("os.environ", {"AWS_BEARER_TOKEN_BEDROCK": ""}, clear=False):
            session = MagicMock()
            session.get_credentials.return_value = None
            mock_session.return_value = session
            engine = InferenceEngine()
        with pytest.raises(APIError, match="No Bedrock credentials found"):
            await engine._make_request("prompt")

    @pytest.mark.asyncio
    async def test_successful_request(self):
        engine = make_engine()
        engine.client.converse = MagicMock(return_value=make_response("test response"))

        result = await engine._make_request("hello", max_tokens=100)
        assert result == "test response"
        engine.client.converse.assert_called_once()
        # The max_tokens budget is threaded into the Converse inferenceConfig.
        _, kwargs = engine.client.converse.call_args
        assert kwargs["inferenceConfig"]["maxTokens"] == 100
        assert kwargs["modelId"] == engine.model_name

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", [
        "eu.anthropic.claude-haiku-4-5-20251001-v1:0",  # Anthropic rejects temp+topP together
        "eu.amazon.nova-pro-v1:0",
        "qwen.qwen3-235b-a22b-2507-v1:0",
    ])
    async def test_inference_config_sends_temperature_only(self, model_name):
        """Only temperature is sent (never topP) for every provider — topP is a no-op at
        temperature=0 and Anthropic rejects the two together."""
        engine = make_engine(model_name=model_name)
        engine.client.converse = MagicMock(return_value=make_response("ok"))
        await engine._make_request("hi", max_tokens=50)
        _, kwargs = engine.client.converse.call_args
        cfg = kwargs["inferenceConfig"]
        assert cfg["temperature"] == 0.0
        assert "topP" not in cfg

    @pytest.mark.asyncio
    @patch("src.inference.inference_engine.asyncio.sleep", new_callable=AsyncMock)
    async def test_throttle_then_retry(self, mock_sleep):
        engine = make_engine(max_retries=2)
        engine.client.converse = MagicMock(side_effect=[
            make_client_error("ThrottlingException"),
            make_response("success"),
        ])

        result = await engine._make_request("hello")
        assert result == "success"
        assert engine.client.converse.call_count == 2
        mock_sleep.assert_awaited()  # backoff was applied between attempts

    @pytest.mark.asyncio
    @patch("src.inference.inference_engine.asyncio.sleep", new_callable=AsyncMock)
    async def test_throttle_exhausted_raises_rate_limit(self, mock_sleep):
        engine = make_engine(max_retries=1)
        engine.client.converse = MagicMock(side_effect=make_client_error("ThrottlingException", "always"))

        with pytest.raises(RateLimitExhausted, match="throttled"):
            await engine._make_request("hello")
        # 1 initial attempt + 1 retry.
        assert engine.client.converse.call_count == 2

    @pytest.mark.asyncio
    @patch("src.inference.inference_engine.asyncio.sleep", new_callable=AsyncMock)
    async def test_retryable_server_error_then_success(self, mock_sleep):
        engine = make_engine(max_retries=2)
        engine.client.converse = MagicMock(side_effect=[
            make_client_error("InternalServerException"),
            make_response("recovered"),
        ])

        result = await engine._make_request("hi")
        assert result == "recovered"
        assert engine.client.converse.call_count == 2

    @pytest.mark.asyncio
    @patch("src.inference.inference_engine.asyncio.sleep", new_callable=AsyncMock)
    async def test_retryable_server_error_exhausted(self, mock_sleep):
        engine = make_engine(max_retries=1)
        engine.client.converse = MagicMock(side_effect=make_client_error("ModelTimeoutException"))

        with pytest.raises(APIError, match="Bedrock request failed"):
            await engine._make_request("hi")
        assert engine.client.converse.call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_client_error_stops_immediately(self):
        """A ValidationException/AccessDenied is a client bug — fail fast, do not retry."""
        engine = make_engine(max_retries=5)
        engine.client.converse = MagicMock(side_effect=make_client_error("ValidationException", "bad input"))

        with pytest.raises(APIError, match="ValidationException"):
            await engine._make_request("hi")
        # Exactly one call despite max_retries=5.
        assert engine.client.converse.call_count == 1

    @pytest.mark.asyncio
    async def test_missing_credentials_at_call_time_maps_to_api_error(self):
        engine = make_engine(max_retries=3)
        engine.client.converse = MagicMock(side_effect=NoCredentialsError())

        with pytest.raises(APIError, match="No Bedrock credentials found"):
            await engine._make_request("hi")
        assert engine.client.converse.call_count == 1  # not retried

    @pytest.mark.asyncio
    @patch("src.inference.inference_engine.asyncio.sleep", new_callable=AsyncMock)
    async def test_connection_error_then_retry(self, mock_sleep):
        engine = make_engine(max_retries=2)
        engine.client.converse = MagicMock(side_effect=[
            ReadTimeoutError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"),
            make_response("ok after timeout"),
        ])

        result = await engine._make_request("hi")
        assert result == "ok after timeout"
        assert engine.client.converse.call_count == 2

    @pytest.mark.asyncio
    @patch("src.inference.inference_engine.asyncio.sleep", new_callable=AsyncMock)
    async def test_unexpected_error(self, mock_sleep):
        engine = make_engine(max_retries=1)
        engine.client.converse = MagicMock(side_effect=ValueError("boom"))

        with pytest.raises(APIError, match="Unexpected error"):
            await engine._make_request("hi")


class TestBedrockHelpers:
    def test_to_converse_splits_system_prompt(self):
        system, conv = InferenceEngine._to_converse([
            {"role": "system", "content": "you are a bot"},
            {"role": "user", "content": "hi"},
        ])
        assert system == [{"text": "you are a bot"}]
        assert conv == [{"role": "user", "content": [{"text": "hi"}]}]

    def test_to_converse_user_only(self):
        system, conv = InferenceEngine._to_converse([{"role": "user", "content": "hello"}])
        assert system == []
        assert conv == [{"role": "user", "content": [{"text": "hello"}]}]

    def test_parse_response_extracts_text_and_usage(self):
        content, usage = InferenceEngine._parse_response(
            make_response("hello world", stop_reason="end_turn", input_tokens=5, output_tokens=2)
        )
        assert content == "hello world"
        assert usage.prompt_tokens == 5
        assert usage.completion_tokens == 2
        assert usage.finish_reason == "end_turn"
        assert usage.truncated is False

    def test_parse_response_flags_truncation(self):
        _, usage = InferenceEngine._parse_response(make_response("cut", stop_reason="max_tokens"))
        assert usage.truncated is True

    def test_parse_response_tolerates_missing_fields(self):
        content, usage = InferenceEngine._parse_response({})
        assert content == ""
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.truncated is False

    def test_error_code_from_client_error(self):
        assert InferenceEngine._error_code(make_client_error("ThrottlingException")) == "ThrottlingException"

    def test_error_code_falls_back_to_class_name(self):
        assert InferenceEngine._error_code(ValueError("x")) == "ValueError"

    def test_backoff_grows_and_caps(self):
        assert InferenceEngine._backoff(0) == 2.0
        assert InferenceEngine._backoff(1) == 4.0
        assert InferenceEngine._backoff(100) == 60.0


class TestInferenceEnginePublicMethods:
    @pytest.mark.asyncio
    async def test_classify(self):
        engine = make_engine()
        engine.client.converse = MagicMock(return_value=make_response("positive"))

        result = await engine.classify("classify this")
        assert isinstance(result, ClassificationResult)
        assert result.raw_response == "positive"

    @pytest.mark.asyncio
    async def test_explain(self):
        engine = make_engine()
        engine.client.converse = MagicMock(return_value=make_response("explanation text"))

        result = await engine.explain("explain this", strategy="R")
        assert isinstance(result, ExplanationResult)
        assert result.strategy == "R"
        assert result.raw_response == "explanation text"


class TestInferenceEngineUsageAccounting:
    @pytest.mark.asyncio
    async def test_usage_captured_and_accumulated(self):
        engine = make_engine()
        engine.client.converse = MagicMock(
            return_value=make_response("ok", stop_reason="end_turn", input_tokens=37, output_tokens=11)
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
        engine = make_engine()
        engine.client.converse = MagicMock(
            return_value=make_response('{"label":"positive"}', input_tokens=20, output_tokens=5)
        )
        result = await engine.classify("classify this")
        assert result.usage is not None
        assert result.usage.prompt_tokens == 20
        assert result.usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_truncation_triggers_retry_then_succeeds(self):
        engine = make_engine(context_window=8192)
        # First call truncated (stopReason=max_tokens), retry returns a complete response.
        engine.client.converse = MagicMock(side_effect=[
            make_response("partial", stop_reason="max_tokens", input_tokens=30, output_tokens=100),
            make_response("complete", stop_reason="end_turn", input_tokens=30, output_tokens=250),
        ])

        content, usage = await engine.chat_with_usage([{"role": "user", "content": "x"}], max_tokens=100)
        assert content == "complete"
        assert usage.truncated is False
        # Both attempts consumed tokens and are accounted for.
        assert engine.client.converse.call_count == 2
        assert engine.total_completion_tokens == 350
        assert engine.n_truncated == 0

    @pytest.mark.asyncio
    async def test_truncation_flagged_when_still_truncated(self):
        # context_window equal to max_tokens => no room to expand => stays truncated.
        engine = make_engine(context_window=100)
        engine.client.converse = MagicMock(
            return_value=make_response("partial", stop_reason="max_tokens", input_tokens=30, output_tokens=70)
        )

        content, usage = await engine.chat_with_usage([{"role": "user", "content": "x"}], max_tokens=100)
        assert usage.truncated is True
        assert engine.n_truncated == 1
        # No expansion was possible, so only one call was made.
        assert engine.client.converse.call_count == 1
