"""Gemini constants for fcp-gemini-python-client.

All constants are defined locally with environment-based overrides.
"""

import httpx
import os

# Model identifiers
GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"
GEMINI_3_FLASH = "gemini-3-flash"
GEMINI_3_LIVE_PREVIEW = "gemini-3-live-preview"
DEFAULT_MODEL = GEMINI_3_FLASH_PREVIEW

# API Configuration
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 3
RETRY_INITIAL_DELAY = 1
RETRY_MAX_DELAY = 60

# Generation parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40
DEFAULT_MAX_OUTPUT_TOKENS = 8192

# Media resolution
MEDIA_RESOLUTION_LOW = "MEDIA_RESOLUTION_LOW"
MEDIA_RESOLUTION_MEDIUM = "MEDIA_RESOLUTION_MEDIUM"
MEDIA_RESOLUTION_HIGH = "MEDIA_RESOLUTION_HIGH"

# Thinking budget defaults (tokens)
THINKING_BUDGET_MINIMAL = 512
THINKING_BUDGET_LOW = 1024
THINKING_BUDGET_MEDIUM = 2048
THINKING_BUDGET_HIGH = 4096

# Limits
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_AUDIO_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

# Costs (per token)
DEFAULT_COST_PER_INPUT_TOKEN = 0.0001
DEFAULT_COST_PER_OUTPUT_TOKEN = 0.0003

# Environment-based configuration overrides
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", DEFAULT_MODEL)
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE_BYTES", str(MAX_IMAGE_SIZE_BYTES)))
COST_PER_INPUT_TOKEN = float(os.getenv("GEMINI_COST_PER_INPUT_TOKEN", str(DEFAULT_COST_PER_INPUT_TOKEN)))
COST_PER_OUTPUT_TOKEN = float(os.getenv("GEMINI_COST_PER_OUTPUT_TOKEN", str(DEFAULT_COST_PER_OUTPUT_TOKEN)))

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
