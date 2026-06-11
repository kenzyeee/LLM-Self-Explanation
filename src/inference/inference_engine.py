import os
import asyncio
import logging
from typing import Set
from dataclasses import dataclass
from datetime import datetime
from groq import AsyncGroq
import groq

from src.utils.exceptions import APIError

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
    def __init__(self, api_key: str = None, model_name: str = "llama3-8b-8192",
                 max_retries: int = 3, concurrent_requests: int = 5, request_timeout: int = 30):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model_name = model_name
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        if self.api_key:
            self.client = AsyncGroq(api_key=self.api_key)
        else:
            self.client = None

    async def _make_request(self, prompt: str, max_tokens: int = 512, max_retries: int = None) -> str:
        if not self.client:
            raise APIError("GROQ_API_KEY is not set.")
        retries = max_retries if max_retries is not None else self.max_retries
        async with self.semaphore:
            for attempt in range(retries + 1):
                try:
                    chat_completion = await self.client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model_name,
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=max_tokens,
                    )
                    logger.info("API request successful", extra={
                        'model': self.model_name, 'prompt_hash': hash(prompt), 'status': 'success'
                    })
                    return chat_completion.choices[0].message.content
                except groq.RateLimitError as e:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited, retrying in {wait}s (attempt {attempt+1}/{retries})")
                    if attempt == retries:
                        raise APIError(f"Rate limit exceeded after {retries} retries: {e}")
                    await asyncio.sleep(wait)
                except groq.APIError as e:
                    if attempt == retries:
                        logger.error("API request failed", extra={'error': str(e)})
                        raise APIError(f"API request failed after {retries} retries: {e}")
                    await asyncio.sleep(2 ** (attempt + 1))
                except Exception as e:
                    if attempt == retries:
                        logger.error("API request failed", extra={'error': str(e)})
                        raise APIError(f"Unexpected error during API request: {e}")
                    await asyncio.sleep(2 ** (attempt + 1))

    async def classify(self, prompt: str) -> ClassificationResult:
        raw_response = await self._make_request(prompt, max_tokens=50)
        return ClassificationResult(predicted_label="", confidence=0.0, raw_response=raw_response, timestamp=datetime.now())

    async def explain(self, prompt: str, strategy: str) -> ExplanationResult:
        raw_response = await self._make_request(prompt, max_tokens=512)
        return ExplanationResult(strategy=strategy, raw_response=raw_response, timestamp=datetime.now())

    async def classify_with_mask(self, prompt: str, masked_tokens: Set[str]) -> ClassificationResult:
        raw_response = await self._make_request(prompt, max_tokens=50)
        return ClassificationResult(predicted_label="", confidence=0.0, raw_response=raw_response, timestamp=datetime.now())
