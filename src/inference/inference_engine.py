import os
import re
import asyncio
import logging
from typing import Set, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, BotoCoreError, NoCredentialsError

from src.utils.exceptions import APIError, RateLimitExhausted

logger = logging.getLogger(__name__)

# Two auth mechanisms are accepted for Bedrock: a Bedrock API key (bearer token) in
# AWS_BEARER_TOKEN_BEDROCK, which boto3 applies to Bedrock calls automatically, or the
# standard AWS SigV4 credential chain. This string names both wherever creds are missing.
_CREDS_HELP = (
    "Set a Bedrock API key in AWS_BEARER_TOKEN_BEDROCK, or provide AWS SigV4 credentials "
    "(AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, an AWS profile, or an IAM role)."
)


def _as_int(value) -> int:
    """Coerce an API usage field to a non-negative int (mocks/None → 0)."""
    return value if isinstance(value, int) and value >= 0 else 0


@dataclass
class TokenUsage:
    """Real token accounting for a single completion, taken from the API response."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    truncated: bool = False


@dataclass
class ClassificationResult:
    predicted_label: str
    confidence: float
    raw_response: str
    timestamp: datetime
    usage: Optional[TokenUsage] = None


@dataclass
class ExplanationResult:
    strategy: str
    raw_response: str
    timestamp: datetime


class InferenceEngine:
    """Async wrapper over the AWS Bedrock runtime Converse API.

    Bedrock's SDK (boto3) is synchronous, so each call is offloaded to a worker
    thread with ``asyncio.to_thread`` and gated by a semaphore for concurrency —
    the public interface stays fully async. Authentication uses the standard AWS
    credential chain (env vars, shared config/credentials file, or an IAM role);
    there is no single API-key env var like the previous provider had.
    """

    def __init__(self, model_name: str = "us.meta.llama3-3-70b-instruct-v1:0",
                 region: str = None, max_retries: int = 3, concurrent_requests: int = 5,
                 request_timeout: int = 30, context_window: int = 8192):
        self.model_name = model_name
        self.region = (region or os.environ.get("AWS_REGION")
                       or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.context_window = context_window
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        # Cumulative real token usage across every API call this engine makes —
        # the authoritative source for the run's total token consumption.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_truncated = 0
        self.client = self._build_client()

    def _build_client(self):
        """Create a bedrock-runtime client, or None if no credentials can be resolved.

        Accepts either a Bedrock API key (bearer token in AWS_BEARER_TOKEN_BEDROCK, which
        boto3 applies to Bedrock calls automatically) or the standard AWS SigV4 chain.
        Fails fast with a clear message instead of an opaque first-call error. botocore's
        own retries are disabled (``total_max_attempts=1``) because ``_call_once`` owns backoff.
        """
        session = boto3.Session(region_name=self.region)
        has_bearer = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))
        if not has_bearer and session.get_credentials() is None:
            logger.warning(f"No Bedrock credentials found; client not created. {_CREDS_HELP}")
            return None
        cfg = BotoConfig(
            region_name=self.region,
            read_timeout=self.request_timeout,
            connect_timeout=self.request_timeout,
            retries={"total_max_attempts": 1},
        )
        return session.client("bedrock-runtime", config=cfg)

    # Bedrock (botocore ClientError) codes that mean "throttled / over a quota" —
    # transient, worth retrying with backoff.
    _THROTTLING_CODES = frozenset({
        "ThrottlingException", "TooManyRequestsException",
        "ServiceQuotaExceededException", "ProvisionedThroughputExceededException",
    })
    # Server-side errors that are also worth a retry, but are not rate limits.
    _RETRYABLE_CODES = frozenset({
        "ModelTimeoutException", "ModelNotReadyException", "InternalServerException",
        "ServiceUnavailableException", "ModelErrorException",
    })

    @staticmethod
    def _error_code(error) -> str:
        """The botocore error code for a ClientError (e.g. 'ThrottlingException'),
        falling back to the exception class name for connection-level errors."""
        resp = getattr(error, "response", None)
        if isinstance(resp, dict):
            code = resp.get("Error", {}).get("Code")
            if isinstance(code, str) and code:
                return code
        return error.__class__.__name__

    @staticmethod
    def _backoff(attempt: int) -> float:
        """Exponential backoff for transient throttling, capped at 60s.

        Bedrock does not return a Retry-After hint, so we grow the wait per attempt."""
        return min(60.0, 2.0 ** (attempt + 1))

    @staticmethod
    def _to_converse(messages: List[dict]) -> Tuple[List[dict], List[dict]]:
        """Convert OpenAI-style ``{role, content}`` messages into Converse form.

        Returns ``(system_blocks, conversation)``. Converse takes system prompts in a
        top-level ``system`` field (not inline in ``messages``) and wraps each turn's
        text in a content-block list, so both are reshaped here."""
        system_blocks: List[dict] = []
        conversation: List[dict] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            text = content if isinstance(content, str) else str(content)
            if role == "system":
                system_blocks.append({"text": text})
            else:
                conversation.append({"role": role, "content": [{"text": text}]})
        return system_blocks, conversation

    @staticmethod
    def _parse_response(response: dict) -> Tuple[str, TokenUsage]:
        """Extract text + real token usage from a Converse response.

        ``stopReason == "max_tokens"`` is Bedrock's truncation signal (the analogue of
        the previous provider's ``finish_reason == "length"``)."""
        message = (response.get("output", {}) or {}).get("message", {}) or {}
        blocks = message.get("content", []) or []
        content = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        stop_reason = response.get("stopReason", "")
        if not isinstance(stop_reason, str):
            stop_reason = ""
        api_usage = response.get("usage", {}) or {}
        usage = TokenUsage(
            prompt_tokens=_as_int(api_usage.get("inputTokens", 0)),
            completion_tokens=_as_int(api_usage.get("outputTokens", 0)),
            finish_reason=stop_reason,
            truncated=(stop_reason == "max_tokens"),
        )
        return content, usage

    def _inference_config(self, max_tokens: int) -> dict:
        """Build the Converse ``inferenceConfig`` for this model.

        Only ``temperature`` (0.0) is sent — deliberately NOT ``topP``. Anthropic models
        on Bedrock reject the two together ("temperature and top_p cannot both be specified
        for this model"), and other providers vary in what they accept. Since the study
        runs greedy (temperature=0), ``topP`` is a no-op for every model, so omitting it
        entirely sidesteps that whole class of ValidationException across all providers.
        """
        return {"maxTokens": max_tokens, "temperature": 0.0}

    def _invoke(self, conversation: List[dict], system_blocks: List[dict], max_tokens: int) -> dict:
        """Synchronous Converse call — run inside a worker thread by ``_call_once``."""
        kwargs = {
            "modelId": self.model_name,
            "messages": conversation,
            "inferenceConfig": self._inference_config(max_tokens),
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        return self.client.converse(**kwargs)

    async def _call_once(self, messages: List[dict], max_tokens: int, retries: int) -> Tuple[str, TokenUsage]:
        """Single Converse call with retry/backoff. Returns (content, usage).

        Token usage and stopReason are read from the API response. Numbers are coerced
        defensively so non-numeric mock responses count as zero rather than crashing
        the accumulator.
        """
        system_blocks, conversation = self._to_converse(messages)
        attempt = 0
        while True:
            try:
                response = await asyncio.to_thread(
                    self._invoke, conversation, system_blocks, max_tokens
                )
                content, usage = self._parse_response(response)
                if not content or not content.strip():
                    logger.warning(f"Bedrock returned empty content for model={self.model_name} "
                                   f"n_messages={len(conversation)} stop_reason={usage.finish_reason}")
                else:
                    logger.info("Bedrock request successful", extra={
                        'model': self.model_name, 'n_messages': len(conversation), 'status': 'success'
                    })
                return content, usage
            except ClientError as e:
                code = self._error_code(e)
                if code in self._THROTTLING_CODES:
                    if attempt >= retries:
                        logger.error("Bedrock throttling retries exhausted", extra={'error': str(e)})
                        raise RateLimitExhausted(f"Bedrock throttled after {retries} retries: {e}")
                    wait = self._backoff(attempt)
                    logger.warning(f"Bedrock throttled ({code}), retrying in {wait}s (attempt {attempt + 1})")
                    attempt += 1
                    await asyncio.sleep(wait)
                elif code in self._RETRYABLE_CODES:
                    if attempt >= retries:
                        logger.error("Bedrock request failed", extra={'error': str(e)})
                        raise APIError(f"Bedrock request failed after {retries} retries ({code}): {e}")
                    attempt += 1
                    await asyncio.sleep(2 ** attempt)
                else:
                    # Non-retryable: ValidationException, AccessDeniedException,
                    # ResourceNotFoundException (e.g. model access not enabled), etc.
                    logger.error("Bedrock request failed (non-retryable)",
                                 extra={'error': str(e), 'code': code})
                    raise APIError(f"Bedrock request failed ({code}): {e}")
            except BotoCoreError as e:
                # Client-side failures: missing credentials, connection/read timeouts.
                if isinstance(e, NoCredentialsError):
                    raise APIError(f"No Bedrock credentials found. {_CREDS_HELP}")
                if attempt >= retries:
                    logger.error("Bedrock connection error", extra={'error': str(e)})
                    raise APIError(f"Bedrock connection error after {retries} retries: {e}")
                attempt += 1
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt >= retries:
                    logger.error("Bedrock request failed", extra={'error': str(e)})
                    raise APIError(f"Unexpected error during Bedrock request: {e}")
                attempt += 1
                await asyncio.sleep(2 ** attempt)

    def _account(self, usage: TokenUsage) -> None:
        """Fold a call's real usage into the engine's cumulative counters."""
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens

    async def _complete(self, messages: List[dict], max_tokens: int, retries: int) -> Tuple[str, TokenUsage]:
        """Shared Converse entry point. Accounts real token usage and recovers from
        token-limit truncation by retrying once with a larger budget."""
        if not self.client:
            raise APIError(f"No Bedrock credentials found. {_CREDS_HELP}")
        async with self.semaphore:
            content, usage = await self._call_once(messages, max_tokens, retries)
            # Truncation recovery: a response cut off at max_tokens (stopReason ==
            # "max_tokens") is incomplete and will fail downstream parsing. Retry once
            # with a larger budget (capped at the model context window). Both attempts
            # cost tokens, so account for both.
            if usage.truncated:
                bigger = min(max_tokens * 2, self.context_window)
                if bigger > max_tokens:
                    logger.warning(f"Response truncated (stopReason=max_tokens) at max_tokens={max_tokens} "
                                   f"for model={self.model_name}; retrying with max_tokens={bigger}")
                    self._account(usage)  # the truncated first attempt still consumed tokens
                    content, usage = await self._call_once(messages, bigger, retries)
            self._account(usage)
            if usage.truncated:
                self.n_truncated += 1
                logger.warning(f"Response still truncated after retry for model={self.model_name} "
                               f"(max_tokens ceiling {self.context_window} reached)")
            return content, usage

    async def _make_request(self, prompt: str, max_tokens: int = 512, max_retries: int = None) -> str:
        retries = max_retries if max_retries is not None else self.max_retries
        content, _ = await self._complete([{"role": "user", "content": prompt}], max_tokens, retries)
        return content

    async def chat(self, messages: List[dict], max_tokens: int = 512) -> str:
        content, _ = await self._complete(messages, max_tokens, self.max_retries)
        return content

    async def chat_with_usage(self, messages: List[dict], max_tokens: int = 512) -> Tuple[str, TokenUsage]:
        """Like chat(), but also returns the real TokenUsage for this call."""
        return await self._complete(messages, max_tokens, self.max_retries)

    async def classify(self, prompt: str, max_tokens: int = 1024) -> ClassificationResult:
        content, usage = await self._complete([{"role": "user", "content": prompt}], max_tokens, self.max_retries)
        return ClassificationResult(predicted_label="", confidence=0.0, raw_response=content,
                                    timestamp=datetime.now(), usage=usage)

    async def explain(self, prompt: str, strategy: str) -> ExplanationResult:
        raw_response = await self._make_request(prompt, max_tokens=512)
        return ExplanationResult(strategy=strategy, raw_response=raw_response, timestamp=datetime.now())

    async def classify_with_mask(self, prompt: str, masked_tokens: Set[str]) -> ClassificationResult:
        content, usage = await self._complete([{"role": "user", "content": prompt}], 1024, self.max_retries)
        label_match = re.search(r'"label"\s*:\s*"([^"]+)"', content)
        label = label_match.group(1) if label_match else ""
        return ClassificationResult(predicted_label=label, confidence=0.0, raw_response=content,
                                    timestamp=datetime.now(), usage=usage)
