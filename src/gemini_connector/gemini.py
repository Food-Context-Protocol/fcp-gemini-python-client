"""Gemini 3 API client for FCP.

This module provides a thin facade over smaller Gemini feature modules.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from google import genai
from google.genai import types

from fcp.security import validate_image_url
from fcp.services.gemini_async_ops import GeminiCacheMixin, GeminiDeepResearchMixin, GeminiVideoMixin
from fcp.services.gemini_base import GeminiBase
from fcp.services.gemini_constants import GEMINI_API_KEY, MAX_IMAGE_SIZE, MODEL_NAME
from fcp.services.gemini_constants import RETRYABLE_EXCEPTIONS as _RETRYABLE_EXCEPTIONS
from fcp.services.gemini_generation import (
    GeminiCodeExecutionMixin,
    GeminiCombinedToolsMixin,
    GeminiGenerationMixin,
    GeminiGroundingMixin,
    GeminiImageMixin,
    GeminiMediaMixin,
    GeminiThinkingMixin,
    GeminiToolingMixin,
)
from fcp.services.gemini_helpers import _create_retry_decorator as _create_retry_decorator_impl
from fcp.services.gemini_helpers import (
    _extract_grounding_sources,
    _extract_thinking_content,
    _get_thinking_budget,
    _log_token_usage,
    _parse_json_response,
    gemini_retry,
)
from fcp.services.gemini_live import GeminiLiveMixin

# Re-export constants/utilities for tests and callers that import from this module.
RETRYABLE_EXCEPTIONS = _RETRYABLE_EXCEPTIONS
_create_retry_decorator = _create_retry_decorator_impl

__all__ = [
    "GEMINI_API_KEY",
    "MAX_IMAGE_SIZE",
    "MODEL_NAME",
    "RETRYABLE_EXCEPTIONS",
    "gemini_retry",
    "GeminiClient",
    "_parse_json_response",
    "_get_thinking_budget",
    "types",
    "time",
    "genai",
    "validate_image_url",
    "_extract_thinking_content",
    "_extract_grounding_sources",
    "_log_token_usage",
    "set_gemini_client",
    "reset_gemini_client",
]


class GeminiClient(
    GeminiBase,
    GeminiGenerationMixin,
    GeminiToolingMixin,
    GeminiGroundingMixin,
    GeminiThinkingMixin,
    GeminiCodeExecutionMixin,
    GeminiMediaMixin,
    GeminiImageMixin,
    GeminiCombinedToolsMixin,
    GeminiCacheMixin,
    GeminiDeepResearchMixin,
    GeminiVideoMixin,
    GeminiLiveMixin,
):
    """Wrapper for Gemini 3 API with all advanced features."""


_gemini_client: GeminiClient | None = None
_gemini_lock = threading.Lock()


def get_gemini_client() -> GeminiClient:
    """Get or create the Gemini client singleton."""
    global _gemini_client
    if _gemini_client is None:
        with _gemini_lock:
            if _gemini_client is None:
                _gemini_client = GeminiClient()
    assert _gemini_client is not None
    return _gemini_client


def get_gemini() -> GeminiClient:
    """FastAPI dependency for Gemini client."""
    return get_gemini_client()


def set_gemini_client(client: GeminiClient | Any) -> None:
    """Override the shared Gemini client (tests only)."""
    global _gemini_client
    _gemini_client = client


def reset_gemini_client() -> None:
    """Reset the shared Gemini client (tests only)."""
    global _gemini_client
    _gemini_client = None


class _GeminiProxy:
    """Proxy that forwards attribute access to the active Gemini client."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_gemini_client(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return get_gemini_client()(*args, **kwargs)  # type: ignore[operator]

    def __repr__(self) -> str:
        return f"<GeminiProxy client={get_gemini_client()!r}>"


# For backwards compatibility (use get_gemini() with Depends() in new code)
gemini: GeminiClient = _GeminiProxy()  # type: ignore[assignment]
