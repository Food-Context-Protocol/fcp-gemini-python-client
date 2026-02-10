"""Tests for fcp.gemini.gemini_constants - Model constants and env overrides."""

import importlib

import httpx

import fcp.gemini.gemini_constants as constants_module


class TestModelConstants:
    """Test model identifier constants."""

    def test_flash_preview(self):
        assert constants_module.GEMINI_3_FLASH_PREVIEW == "gemini-3-flash-preview"

    def test_flash(self):
        assert constants_module.GEMINI_3_FLASH == "gemini-3-flash"

    def test_live_preview(self):
        assert constants_module.GEMINI_3_LIVE_PREVIEW == "gemini-3-live-preview"

    def test_default_model(self):
        assert constants_module.DEFAULT_MODEL == "gemini-3-flash-preview"


class TestApiConstants:
    """Test API configuration constants."""

    def test_timeout(self):
        assert constants_module.DEFAULT_TIMEOUT_SECONDS == 30.0

    def test_retries(self):
        assert constants_module.MAX_RETRIES == 3

    def test_retry_delays(self):
        assert constants_module.RETRY_INITIAL_DELAY == 1
        assert constants_module.RETRY_MAX_DELAY == 60


class TestGenerationConstants:
    """Test generation parameter constants."""

    def test_temperature(self):
        assert constants_module.DEFAULT_TEMPERATURE == 0.7

    def test_top_p(self):
        assert constants_module.DEFAULT_TOP_P == 0.95

    def test_top_k(self):
        assert constants_module.DEFAULT_TOP_K == 40

    def test_max_output_tokens(self):
        assert constants_module.DEFAULT_MAX_OUTPUT_TOKENS == 8192


class TestMediaConstants:
    """Test media resolution and size constants."""

    def test_media_resolutions(self):
        assert constants_module.MEDIA_RESOLUTION_LOW == "MEDIA_RESOLUTION_LOW"
        assert constants_module.MEDIA_RESOLUTION_MEDIUM == "MEDIA_RESOLUTION_MEDIUM"
        assert constants_module.MEDIA_RESOLUTION_HIGH == "MEDIA_RESOLUTION_HIGH"

    def test_size_limits(self):
        assert constants_module.MAX_IMAGE_SIZE_BYTES == 10 * 1024 * 1024
        assert constants_module.MAX_VIDEO_SIZE_BYTES == 100 * 1024 * 1024
        assert constants_module.MAX_AUDIO_SIZE_BYTES == 50 * 1024 * 1024


class TestThinkingBudgets:
    """Test thinking budget defaults."""

    def test_defaults(self):
        assert constants_module.THINKING_BUDGET_MINIMAL == 512
        assert constants_module.THINKING_BUDGET_LOW == 1024
        assert constants_module.THINKING_BUDGET_MEDIUM == 2048
        assert constants_module.THINKING_BUDGET_HIGH == 4096


class TestCostConstants:
    """Test cost per token constants."""

    def test_defaults(self):
        assert constants_module.DEFAULT_COST_PER_INPUT_TOKEN == 0.0001
        assert constants_module.DEFAULT_COST_PER_OUTPUT_TOKEN == 0.0003


class TestEnvOverrides:
    """Test environment variable overrides for runtime config."""

    def test_model_name_override(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL_NAME", "gemini-3-pro")
        importlib.reload(constants_module)
        assert constants_module.MODEL_NAME == "gemini-3-pro"

    def test_max_image_size_override(self, monkeypatch):
        monkeypatch.setenv("MAX_IMAGE_SIZE_BYTES", "5242880")
        importlib.reload(constants_module)
        assert constants_module.MAX_IMAGE_SIZE == 5242880

    def test_cost_overrides(self, monkeypatch):
        monkeypatch.setenv("GEMINI_COST_PER_INPUT_TOKEN", "0.0005")
        monkeypatch.setenv("GEMINI_COST_PER_OUTPUT_TOKEN", "0.001")
        importlib.reload(constants_module)
        assert constants_module.COST_PER_INPUT_TOKEN == 0.0005
        assert constants_module.COST_PER_OUTPUT_TOKEN == 0.001

    def test_thinking_budget_overrides(self, monkeypatch):
        monkeypatch.setenv("THINKING_BUDGET_MINIMAL", "256")
        monkeypatch.setenv("THINKING_BUDGET_HIGH", "8192")
        importlib.reload(constants_module)
        assert constants_module.THINKING_BUDGETS["minimal"] == 256
        assert constants_module.THINKING_BUDGETS["high"] == 8192


class TestRetryableExceptions:
    """Test retryable exception tuple."""

    def test_includes_connect_error(self):
        assert httpx.ConnectError in constants_module.RETRYABLE_EXCEPTIONS

    def test_includes_timeout(self):
        assert httpx.TimeoutException in constants_module.RETRYABLE_EXCEPTIONS

    def test_includes_http_status_error(self):
        assert httpx.HTTPStatusError in constants_module.RETRYABLE_EXCEPTIONS
