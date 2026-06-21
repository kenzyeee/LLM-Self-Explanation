import os
import asyncio
import logging
from typing import Set, List, Dict
from dataclasses import dataclass
from datetime import datetime
from openai import AsyncOpenAI
from openai import APIError as OpenAIAPIError, RateLimitError

from src.utils.exceptions import APIError, RateLimitExhausted

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    predicted_label: str
    confidence: float
    raw_response: str
    timestamp: datetime


@dataclass
class ExplanationResult:
    strategy: str
    raw_response: str
    timestamp: datetime


class InferenceEngine:
    def __init__(self, api_key: str = None, model_name: str = "gpt-4o-mini",
                 max_retries: int = 3, concurrent_requests: int = 5, request_timeout: int = 30):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model_name = model_name
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        if self.api_key:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        else:
            self.client = None

    @staticmethod
    def _rate_limit_wait(error, attempt: int) -> float:
        """Honor a server-provided Retry-After header, else exponential backoff (capped 300s)."""
        retry_after = None
        if hasattr(error, 'response') and error.response is not None:
            retry_after = error.response.headers.get('retry-after')
        if retry_after is not None:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass
        return min(300, 2 ** (attempt + 1))

    async def _complete(self, messages: List[dict], max_tokens: int, retries: int) -> str:
        """Shared chat-completion call with retry/backoff for rate limits and API errors."""
        if not self.client:
            raise APIError("GROQ_API_KEY is not set.")
        async with self.semaphore:
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
                    content = chat_completion.choices[0].message.content
                    if not content or not content.strip():
                        logger.warning(f"API returned empty content for model={self.model_name} "
                                       f"n_messages={len(messages)} "
                                       f"finish_reason={chat_completion.choices[0].finish_reason}")
                    else:
                        logger.info("API request successful", extra={
                            'model': self.model_name, 'n_messages': len(messages), 'status': 'success'
                        })
                    return content
                except RateLimitError as e:
                    if attempt >= retries:
                        logger.error("Rate limit exceeded", extra={'error': str(e)})
                        raise RateLimitExhausted(f"Rate limit exceeded after {retries} retries: {e}")
                    wait = self._rate_limit_wait(e, attempt)
                    logger.warning(f"Rate limited, retrying in {wait}s (attempt {attempt + 1})")
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

    async def _make_request(self, prompt: str, max_tokens: int = 512, max_retries: int = None) -> str:
        retries = max_retries if max_retries is not None else self.max_retries
        return await self._complete([{"role": "user", "content": prompt}], max_tokens, retries)

    async def chat(self, messages: List[dict], max_tokens: int = 512) -> str:
        return await self._complete(messages, max_tokens, self.max_retries)

    async def classify(self, prompt: str) -> ClassificationResult:
        raw_response = await self._make_request(prompt, max_tokens=1024)
        return ClassificationResult(predicted_label="", confidence=0.0, raw_response=raw_response, timestamp=datetime.now())

    async def explain(self, prompt: str, strategy: str) -> ExplanationResult:
        raw_response = await self._make_request(prompt, max_tokens=512)
        return ExplanationResult(strategy=strategy, raw_response=raw_response, timestamp=datetime.now())

    async def classify_with_mask(self, prompt: str, masked_tokens: Set[str]) -> ClassificationResult:
        import re
        raw_response = await self._make_request(prompt, max_tokens=1024)
        label_match = re.search(r'"label"\s*:\s*"([^"]+)"', raw_response)
        label = label_match.group(1) if label_match else ""
        return ClassificationResult(predicted_label=label, confidence=0.0, raw_response=raw_response, timestamp=datetime.now())
