"""LLM Generation: execute grounded inference calls against Groq (or OpenAI).

Uses OpenAI-compatible Async API endpoints to query models like Llama 3.1 via Groq.
"""
import logging
import os
from openai import AsyncOpenAI
from src.config import settings

logger = logging.getLogger(__name__)

_client = None


def _get_async_client() -> tuple[AsyncOpenAI | None, str, str]:
    """Resolve the active LLM client, model name, and provider."""
    provider = (settings.llm_provider or "groq").lower()
    
    if provider == "groq":
        api_key = settings.groq_api_key or os.environ.get("GROQ_API_KEY", "")
        base_url = "https://api.groq.com/openai/v1"
        model = settings.groq_model or "llama-3.1-8b-instant"
        if not api_key:
            return None, model, provider
        return AsyncOpenAI(api_key=api_key, base_url=base_url), model, provider
    
    # Default to OpenAI
    api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    model = settings.llm_model or "gpt-4o-mini"
    if not api_key:
        return None, model, provider
    return AsyncOpenAI(api_key=api_key), model, provider


async def generate_answer(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Send system and user prompts to the configured LLM and return the generated text.

    Args:
        system_prompt: Grounding instructions and citation rules.
        user_prompt: Context chunks and user query.
        temperature: Sampling temperature (0.0 for deterministic factual QA).
        max_tokens: Maximum tokens in response.

    Returns:
        Generated answer string.
    """
    client, model, provider = _get_async_client()

    if client is None:
        logger.warning(
            "No API key configured for LLM provider '%s'. Set GROQ_API_KEY or OPENAI_API_KEY in .env.local.",
            provider,
        )
        return (
            "[LLM Offline] No API key configured. "
            "Please set GROQ_API_KEY in your environment or .env.local to receive live model responses."
        )

    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = response.choices[0]
        return choice.message.content or ""
    except Exception as err:
        logger.error("LLM generation error with provider %s (%s): %s", provider, model, err)
        raise

