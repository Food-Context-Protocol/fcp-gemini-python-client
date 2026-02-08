"""Helper utilities for Gemini service."""

import json
import logging
from typing import Any, TypedDict

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fcp.config import Config
from fcp.services.gemini_constants import (
    COST_PER_INPUT_TOKEN,
    COST_PER_OUTPUT_TOKEN,
    RETRYABLE_EXCEPTIONS,
    THINKING_BUDGETS,
)
from fcp.utils.json_extractor import extract_json
from fcp.utils.metrics import record_gemini_usage

logger = logging.getLogger(__name__)


class GeminiThinkingResult(TypedDict):
    """Result shape when thinking output is included in response."""

    analysis: dict[str, Any] | list[Any]
    thinking: str | None


def _create_retry_decorator():
    """Create retry decorator for Gemini API calls.

    Retries on:
    - Network errors (connection, timeout)
    - HTTP 429 (rate limit) and 503 (service unavailable)

    Strategy:
    - Max attempts from config
    - Exponential backoff: min/max from config
    - Logs retry attempts at WARNING level
    """
    return retry(
        stop=stop_after_attempt(Config.RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=1,
            min=Config.RETRY_MIN_WAIT_SECONDS,
            max=Config.RETRY_MAX_WAIT_SECONDS,
        ),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# Retry decorator instance
gemini_retry = _create_retry_decorator()


def _get_thinking_budget(level: str) -> int:
    """Convert thinking level string to integer budget."""
    return THINKING_BUDGETS.get(level.lower(), 16384)  # Default to medium


def _extract_thinking_content(response) -> str | None:
    """Extract thinking/reasoning content from Gemini response.

    When include_thoughts=True is set, Gemini returns thinking tokens
    in separate parts of the response. This extracts and concatenates them.

    Args:
        response: The GenerateContentResponse from Gemini

    Returns:
        The thinking content as a string, or None if not available
    """
    # Defensive check for missing candidates
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return None

    thinking_parts = []
    for candidate in candidates:
        # Defensive check for missing content
        content = getattr(candidate, "content", None)
        if not content:
            continue

        # Defensive check for missing parts
        parts = getattr(content, "parts", None)
        if not parts:
            continue

        for part in parts:
            # Gemini marks thinking parts with the 'thought' attribute
            if getattr(part, "thought", False) and (text := getattr(part, "text", None)):
                thinking_parts.append(text)

    return "\n".join(thinking_parts) if thinking_parts else None


def _parse_json_response(text: str) -> dict[str, Any] | list[Any]:
    """Parse JSON from Gemini response with robust fallback handling.

    While JSON mode (response_mime_type="application/json") usually guarantees
    valid JSON, edge cases can occur:
    - Truncated responses due to token limits
    - API errors returning non-JSON
    - Markdown-wrapped JSON despite JSON mode
    - Whitespace or BOM characters

    Args:
        text: The raw response text from Gemini

    Returns:
        Parsed JSON as dict or list

    Raises:
        ValueError: If JSON cannot be extracted from the response.
            Note: This normalizes json.JSONDecodeError to ValueError for
            consistent error handling across all parsing strategies.
    """
    # Strip whitespace and BOM characters once at the start for efficiency
    stripped_text = text.strip().lstrip("\ufeff") if text else ""
    if not stripped_text:
        raise ValueError("Empty response from Gemini API")

    # Strategy 1: Direct JSON parsing (expected path for JSON mode)
    try:
        return json.loads(stripped_text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Use robust extractor for edge cases (markdown, embedded JSON)
    result = extract_json(stripped_text)
    if result is not None:
        return result

    # All strategies failed - log length only to avoid PII leakage
    logger.warning("Failed to parse JSON from Gemini response (length=%d)", len(stripped_text))
    raise ValueError("Failed to parse JSON from Gemini response")


def _extract_grounding_sources(response: Any) -> list[dict[str, str]]:
    """Extract grounding sources from a Gemini API response.

    Handles various edge cases:
    - No candidates
    - No grounding_metadata
    - Empty grounding_chunks
    - Chunks without web attribute

    Args:
        response: The Gemini API response object.

    Returns:
        List of source dicts with 'uri' and 'title' keys.
    """
    sources = []
    if not response.candidates:
        return sources

    candidate = response.candidates[0]
    if not candidate.grounding_metadata:
        return sources

    metadata = candidate.grounding_metadata
    if not hasattr(metadata, "grounding_chunks") or not metadata.grounding_chunks:
        return sources

    sources.extend(
        {
            "uri": chunk.web.uri,
            "title": chunk.web.title,
        }
        for chunk in metadata.grounding_chunks
        if hasattr(chunk, "web") and chunk.web
    )
    return sources


def _log_token_usage(
    response: Any,
    method_name: str,
    latency_seconds: float = 0.0,
    success: bool = True,
) -> dict[str, Any]:
    """Extract and log token usage from Gemini API response.

    Args:
        response: The Gemini API response object.
        method_name: Name of the calling method for logging context.
        latency_seconds: Request latency in seconds (for metrics).
        success: Whether the request succeeded (for metrics).

    Returns:
        Dict with input_tokens, output_tokens, total_tokens, and cost_usd.
    """
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        metadata = response.usage_metadata
        # Ensure token counts are integers (handles mock objects in tests)
        raw_input = getattr(metadata, "prompt_token_count", 0)
        raw_output = getattr(metadata, "candidates_token_count", 0)
        input_tokens = int(raw_input) if isinstance(raw_input, int | float) else 0
        output_tokens = int(raw_output) if isinstance(raw_output, int | float) else 0
        total_tokens = input_tokens + output_tokens

        # Calculate cost
        cost = (input_tokens * COST_PER_INPUT_TOKEN) + (output_tokens * COST_PER_OUTPUT_TOKEN)

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(cost, 6),
        }

        logger.info(
            "Gemini API usage [%s]: input=%d, output=%d, total=%d, cost=$%.6f",
            method_name,
            input_tokens,
            output_tokens,
            total_tokens,
            cost,
        )

        # Record Prometheus metrics
        record_gemini_usage(
            method=method_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_seconds=latency_seconds,
            success=success,
        )
    else:
        logger.warning("No usage metadata in response for %s", method_name)
        # Record metrics even without usage metadata (for error tracking)
        record_gemini_usage(
            method=method_name,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            latency_seconds=latency_seconds,
            success=success,
        )

    return usage
