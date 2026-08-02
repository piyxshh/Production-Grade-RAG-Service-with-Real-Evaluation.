"""
Embedder: turn a list of text strings into embedding vectors via the Cohere API.

Design:
- Concurrent, not sequential: batches are fanned out with ``asyncio.gather``
  and a semaphore caps how many requests are in flight at once. Embedding
  10,000 chunks one at a time would take minutes; concurrency keeps the
  latency down to roughly one request round-trip per batch.
- Batching: Cohere caps each ``/v2/embed`` request at 96 texts, so inputs are
  grouped into batches of ``BATCH_SIZE`` before being sent.
- Retries: ``tenacity`` retries on rate-limit and transient server errors
  (429 / 5xx) with exponential backoff plus jitter.
"""
import asyncio
import logging

import cohere
from cohere import (
    InternalServerError,
    ServiceUnavailableError,
    TooManyRequestsError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 96       # Cohere v2 embed max texts per request
MAX_CONCURRENCY = 10  # in-flight API requests at once

_client: cohere.AsyncClientV2 | None = None
_model: str | None = None


def _get_client() -> cohere.AsyncClientV2:
    """Build (once) an async Cohere client from application settings."""
    global _client, _model
    if _client is None:
        from src.config import settings

        _client = cohere.AsyncClientV2(api_key=settings.cohere_key)
        _model = settings.cohere_model
    return _client


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=1, max=30),
    retry=retry_if_exception_type(
        (TooManyRequestsError, ServiceUnavailableError, InternalServerError)
    ),
    reraise=True,
)
async def _embed_one_batch(
    client: cohere.AsyncClientV2, texts: list[str]
) -> list[list[float]]:
    response = await client.embed(
        model=_model,
        input_type="search_document",
        texts=texts,
        embedding_types=["float"],
    )
    return [list(map(float, vec)) for vec in response.embeddings.float_]


async def embed_batch(
    texts: list[str],
    *,
    batch_size: int = BATCH_SIZE,
    max_concurrency: int = MAX_CONCURRENCY,
) -> list[list[float]]:
    """Embed ``texts`` into vectors, preserving input order.

    Texts are grouped into batches of ``batch_size`` which are then embedded
    concurrently (``asyncio.gather``) with at most ``max_concurrency`` requests
    in flight at any moment. Returns one vector per input text.
    """
    if not texts:
        return []

    client = _get_client()
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    semaphore = asyncio.Semaphore(max_concurrency)

    async def guarded(batch: list[str]) -> list[list[float]]:
        async with semaphore:
            return await _embed_one_batch(client, batch)

    per_batch = await asyncio.gather(*(guarded(batch) for batch in batches))
    return [vector for batch in per_batch for vector in batch]
