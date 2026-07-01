import os
import re
import asyncio
import logging
from typing import Set, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from openai import AsyncOpenAI
from openai import APIError as OpenAIAPIError, RateLimitError

from src.utils.exceptions import APIError, RateLimitExhausted, DailyRateLimitExhausted

logger = logging.getLogger(__name__)


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
    def __init__(self, api_key: str = None, model_name: str = "gpt-4o-mini",
                 max_retries: int = 3, concurrent_requests: int = 5, request_timeout: int = 30,
                 context_window: int = 8192):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model_name = model_name
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.context_window = context_window
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        # Cumulative real token usage across every API call this engine makes —
        # the authoritative source for the run's total token consumption.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_truncated = 0
        if self.api_key:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        else:
            self.client = None

    # Markers Groq puts in the 429 message body to name the limit that was hit.
    # Per-day limits (RPD/TPD) won't free up for hours, so they must be handled
    # differently from transient per-minute (RPM/TPM) limits.
    _DAILY_LIMIT_MARKERS = ("per day", "(rpd)", "(tpd)", "requests per day", "tokens per day")
    _TRY_AGAIN_RE = re.compile(r"try again in\s+([0-9hms.\s]+)", re.IGNORECASE)
    _DURATION_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*(ms|h|m|s)")

    @staticmethod
    def _error_text(error) -> str:
        """Best-effort extraction of the human-readable message from an API error,
        spanning the exception string and any structured body Groq returns."""
        parts = [str(error)]
        msg = getattr(error, "message", None)
        if isinstance(msg, str):
            parts.append(msg)
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict) and isinstance(err.get("message"), str):
                parts.append(err["message"])
            elif isinstance(err, str):
                parts.append(err)
        return " ".join(parts)

    @classmethod
    def _is_daily_rate_limit(cls, error) -> bool:
        """True when the 429 is a per-day quota (RPD/TPD) rather than a per-minute one."""
        text = cls._error_text(error).lower()
        return any(marker in text for marker in cls._DAILY_LIMIT_MARKERS)

    @staticmethod
    def _retry_after_header(error) -> Optional[float]:
        """The server-provided Retry-After header in seconds, if present and numeric."""
        if hasattr(error, 'response') and error.response is not None:
            try:
                retry_after = error.response.headers.get('retry-after')
            except (AttributeError, TypeError):
                return None
            if retry_after is not None:
                try:
                    return float(retry_after)
                except (ValueError, TypeError):
                    pass
        return None

    @classmethod
    def _duration_to_seconds(cls, text: str) -> Optional[float]:
        """Parse a Groq duration like '7m12.5s', '1h2m3s', or '512ms' into seconds."""
        units = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
        total, matched = 0.0, False
        for value, unit in cls._DURATION_RE.findall(text):
            total += float(value) * units[unit]
            matched = True
        return total if matched else None

    @classmethod
    def _reset_seconds(cls, error) -> Optional[float]:
        """Seconds until the limit resets: prefer the message's 'try again in ...',
        fall back to the Retry-After header."""
        m = cls._TRY_AGAIN_RE.search(cls._error_text(error))
        if m:
            parsed = cls._duration_to_seconds(m.group(1))
            if parsed is not None:
                return parsed
        return cls._retry_after_header(error)

    @staticmethod
    def _format_duration(seconds: Optional[float]) -> str:
        """Human-readable reset estimate, e.g. '2h 3m' or '4m 1s'."""
        if seconds is None or seconds <= 0:
            return "an unknown amount of time"
        total = int(round(seconds))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        parts = []
        if h:
            parts.append(f"{h}h")
        if m:
            parts.append(f"{m}m")
        if s and not h:  # seconds only matter at sub-hour resolution
            parts.append(f"{s}s")
        return " ".join(parts) if parts else f"{total}s"

    @classmethod
    def _rate_limit_wait(cls, error, attempt: int) -> float:
        """Honor a server-provided Retry-After header, else exponential backoff (capped 300s)."""
        header = cls._retry_after_header(error)
        if header is not None:
            return header
        return min(300, 2 ** (attempt + 1))

    async def _call_once(self, messages: List[dict], max_tokens: int, retries: int) -> Tuple[str, TokenUsage]:
        """Single chat-completion call with retry/backoff. Returns (content, usage).

        Token usage and finish_reason are read from the API response. Numbers are
        coerced defensively so non-numeric mock responses count as zero rather than
        crashing the accumulator.
        """
        attempt = 0
        while True:
            try:
                chat_completion = await self.client.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=max_tokens,
                )
                choice = chat_completion.choices[0]
                content = choice.message.content
                finish_reason = choice.finish_reason if isinstance(choice.finish_reason, str) else ""
                api_usage = getattr(chat_completion, "usage", None)
                usage = TokenUsage(
                    prompt_tokens=_as_int(getattr(api_usage, "prompt_tokens", 0)),
                    completion_tokens=_as_int(getattr(api_usage, "completion_tokens", 0)),
                    finish_reason=finish_reason,
                    truncated=(finish_reason == "length"),
                )
                if not content or not content.strip():
                    logger.warning(f"API returned empty content for model={self.model_name} "
                                   f"n_messages={len(messages)} finish_reason={finish_reason}")
                else:
                    logger.info("API request successful", extra={
                        'model': self.model_name, 'n_messages': len(messages), 'status': 'success'
                    })
                return content, usage
            except RateLimitError as e:
                # A per-day quota (RPD/TPD) won't free up for hours — retrying after a
                # short backoff is futile and the Retry-After can be misleadingly small.
                # Identify it, surface the real reset time, and stop instead of waiting.
                if self._is_daily_rate_limit(e):
                    reset = self._reset_seconds(e)
                    reset_str = self._format_duration(reset)
                    logger.error(
                        f"Daily Groq quota exhausted for model={self.model_name} "
                        f"(per-day RPD/TPD limit); not retrying. Quota resets in ~{reset_str}.",
                        extra={'error': str(e)},
                    )
                    raise DailyRateLimitExhausted(
                        f"Daily rate limit hit for model '{self.model_name}'; "
                        f"quota resets in ~{reset_str}. Stopping run — waiting it out is futile.",
                        details={'model': self.model_name, 'reset_seconds': reset, 'scope': 'daily'},
                    )
                if attempt >= retries:
                    logger.error("Rate limit exceeded", extra={'error': str(e)})
                    raise RateLimitExhausted(f"Rate limit exceeded after {retries} retries: {e}")
                wait = self._rate_limit_wait(e, attempt)
                logger.warning(f"Rate limited (per-minute), retrying in {wait}s (attempt {attempt + 1})")
                attempt += 1
                await asyncio.sleep(wait)
            except OpenAIAPIError as e:
                if attempt >= retries:
                    logger.error("API request failed", extra={'error': str(e)})
                    raise APIError(f"API request failed after {retries} retries: {e}")
                attempt += 1
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt >= retries:
                    logger.error("API request failed", extra={'error': str(e)})
                    raise APIError(f"Unexpected error during API request: {e}")
                attempt += 1
                await asyncio.sleep(2 ** attempt)

    def _account(self, usage: TokenUsage) -> None:
        """Fold a call's real usage into the engine's cumulative counters."""
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens

    async def _complete(self, messages: List[dict], max_tokens: int, retries: int) -> Tuple[str, TokenUsage]:
        """Shared chat-completion entry point. Accounts real token usage and recovers
        from token-limit truncation by retrying once with a larger budget."""
        if not self.client:
            raise APIError("GROQ_API_KEY is not set.")
        async with self.semaphore:
            content, usage = await self._call_once(messages, max_tokens, retries)
            # Truncation recovery: a response cut off at max_tokens (finish_reason ==
            # "length") is incomplete and will fail downstream parsing. Retry once with
            # a larger budget (capped at the model context window). Both attempts cost
            # tokens, so account for both.
            if usage.truncated:
                bigger = min(max_tokens * 2, self.context_window)
                if bigger > max_tokens:
                    logger.warning(f"Response truncated (finish_reason=length) at max_tokens={max_tokens} "
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
