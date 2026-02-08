"""Shared constants for Gemini service."""

import httpx

from fcp.config import Config

# Gemini pricing (per 1M tokens)
# See: https://ai.google.dev/pricing
COST_PER_INPUT_TOKEN = Config.GEMINI_COST_PER_INPUT_TOKEN
COST_PER_OUTPUT_TOKEN = Config.GEMINI_COST_PER_OUTPUT_TOKEN

# Configure Gemini
GEMINI_API_KEY = Config.GEMINI_API_KEY

# Use Gemini 3 Flash for speed and low latency
# See: https://ai.google.dev/gemini-api/docs/models
MODEL_NAME = Config.GEMINI_MODEL_NAME

# Maximum image size (10MB)
MAX_IMAGE_SIZE = Config.MAX_IMAGE_SIZE_BYTES

# Thinking budget mapping (string level to integer tokens)
THINKING_BUDGETS = {
    "minimal": Config.THINKING_BUDGET_MINIMAL,
    "low": Config.THINKING_BUDGET_LOW,
    "medium": Config.THINKING_BUDGET_MEDIUM,
    "high": Config.THINKING_BUDGET_HIGH,
}

# Exceptions that warrant retry (transient failures)
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.HTTPStatusError,  # Includes 429 rate limit, 503 service unavailable
)
