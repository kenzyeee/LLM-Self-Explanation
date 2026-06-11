import pytest
from unittest.mock import AsyncMock, patch, MagicMock, PropertyMock
from src.inference.inference_engine import InferenceEngine, ClassificationResult, ExplanationResult
from src.utils.exceptions import APIError
from unittest.mock import MagicMock


def make_rate_limit_error(msg="rate limited"):
    import groq
    mock_response = MagicMock()
    return groq.RateLimitError(msg, response=mock_response, body=None)


def make_api_error(msg="api error"):
    import groq
    mock_request = MagicMock()
    return groq.APIError(msg, request=mock_request, body=None)


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
