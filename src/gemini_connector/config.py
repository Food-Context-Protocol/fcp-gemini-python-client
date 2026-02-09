"""Standalone configuration for gemini-connector.

Uses environment variables for all configuration.
"""

import os


class Config:
    """Configuration from environment variables."""

    # Gemini API Key (required)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    # Model configuration
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")

    # Thinking budgets (tokens)
    THINKING_BUDGET_MINIMAL = int(os.getenv("THINKING_BUDGET_MINIMAL", "512"))
    THINKING_BUDGET_LOW = int(os.getenv("THINKING_BUDGET_LOW", "1024"))
    THINKING_BUDGET_MEDIUM = int(os.getenv("THINKING_BUDGET_MEDIUM", "2048"))
    THINKING_BUDGET_HIGH = int(os.getenv("THINKING_BUDGET_HIGH", "4096"))

    # Costs (per 1M tokens)
    GEMINI_COST_PER_INPUT_TOKEN = float(os.getenv("GEMINI_COST_PER_INPUT_TOKEN", "0.0001"))
    GEMINI_COST_PER_OUTPUT_TOKEN = float(os.getenv("GEMINI_COST_PER_OUTPUT_TOKEN", "0.0003"))

    # Limits
    MAX_IMAGE_SIZE_BYTES = int(os.getenv("MAX_IMAGE_SIZE_BYTES", str(10 * 1024 * 1024)))

    # Timeouts
    HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "30.0"))
    DEEP_RESEARCH_TIMEOUT_SECONDS = int(os.getenv("DEEP_RESEARCH_TIMEOUT_SECONDS", "300"))
    VIDEO_GENERATION_TIMEOUT_SECONDS = int(os.getenv("VIDEO_GENERATION_TIMEOUT_SECONDS", "180"))
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

    # HTTP connection pooling
    HTTP_MAX_CONNECTIONS = int(os.getenv("HTTP_MAX_CONNECTIONS", "100"))
    HTTP_MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("HTTP_MAX_KEEPALIVE_CONNECTIONS", "20"))

    # Model names for specialized features
    DEEP_RESEARCH_AGENT = os.getenv("DEEP_RESEARCH_AGENT", "gemini-3-deep-research")
    VEO_MODEL_NAME = os.getenv("VEO_MODEL_NAME", "veo-3.1")
    GEMINI_LIVE_MODEL_NAME = os.getenv("GEMINI_LIVE_MODEL_NAME", "gemini-3-live-preview")

    # Service metadata
    SERVICE_NAME = os.getenv("SERVICE_NAME", "gemini-connector")
    API_VERSION = os.getenv("API_VERSION", "1.0.0")

    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
