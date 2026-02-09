"""Gemini constants for standalone connector.

This module imports shared constants from fcp-gemini-core and adds
environment-based overrides for standalone usage.
"""

import httpx
import os

# Import shared constants from core package
from fcp_gemini_core import (
    # Model identifiers
    GEMINI_3_FLASH_PREVIEW,
    GEMINI_3_FLASH,
    GEMINI_3_LIVE_PREVIEW,
    DEFAULT_MODEL as _DEFAULT_MODEL,
    # API Configuration
    DEFAULT_TIMEOUT_SECONDS,
    MAX_RETRIES,
    RETRY_INITIAL_DELAY,
    RETRY_MAX_DELAY,
    # Generation parameters
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_MAX_OUTPUT_TOKENS,
    # Media resolution
    MEDIA_RESOLUTION_LOW,
    MEDIA_RESOLUTION_MEDIUM,
    MEDIA_RESOLUTION_HIGH,
    # Thinking budgets
    THINKING_BUDGETS as _CORE_THINKING_BUDGETS,
    THINKING_BUDGET_MINIMAL,
    THINKING_BUDGET_LOW,
    THINKING_BUDGET_MEDIUM,
    THINKING_BUDGET_HIGH,
    # Limits
    MAX_IMAGE_SIZE_BYTES as _MAX_IMAGE_SIZE_BYTES,
    MAX_VIDEO_SIZE_BYTES,
    MAX_AUDIO_SIZE_BYTES,
    # Costs
    DEFAULT_COST_PER_INPUT_TOKEN as _DEFAULT_COST_PER_INPUT_TOKEN,
    DEFAULT_COST_PER_OUTPUT_TOKEN as _DEFAULT_COST_PER_OUTPUT_TOKEN,
)

# Environment-based configuration for standalone use
# These can be overridden via environment variables when using gemini-connector independently
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", _DEFAULT_MODEL)
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE_BYTES", str(_MAX_IMAGE_SIZE_BYTES)))
COST_PER_INPUT_TOKEN = float(os.getenv("GEMINI_COST_PER_INPUT_TOKEN", str(_DEFAULT_COST_PER_INPUT_TOKEN)))
COST_PER_OUTPUT_TOKEN = float(os.getenv("GEMINI_COST_PER_OUTPUT_TOKEN", str(_DEFAULT_COST_PER_OUTPUT_TOKEN)))

# Thinking budgets - can be overridden via env vars
THINKING_BUDGETS = {
    "minimal": int(os.getenv("THINKING_BUDGET_MINIMAL", str(THINKING_BUDGET_MINIMAL))),
    "low": int(os.getenv("THINKING_BUDGET_LOW", str(THINKING_BUDGET_LOW))),
    "medium": int(os.getenv("THINKING_BUDGET_MEDIUM", str(THINKING_BUDGET_MEDIUM))),
    "high": int(os.getenv("THINKING_BUDGET_HIGH", str(THINKING_BUDGET_HIGH))),
}

# Exceptions that warrant retry (transient failures)
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.HTTPStatusError,  # Includes 429 rate limit, 503 service unavailable
)
