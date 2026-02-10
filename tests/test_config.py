"""Tests for fcp.gemini.config - Environment-based configuration."""

import importlib

import fcp.gemini.config as config_module


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        importlib.reload(config_module)
        assert config_module.Config.GEMINI_API_KEY == ""

    def test_default_model_name(self, monkeypatch):
        monkeypatch.delenv("GEMINI_MODEL_NAME", raising=False)
        importlib.reload(config_module)
        assert config_module.Config.GEMINI_MODEL_NAME == "gemini-3-flash-preview"

    def test_default_http_timeout(self, monkeypatch):
        monkeypatch.delenv("HTTP_TIMEOUT_SECONDS", raising=False)
        importlib.reload(config_module)
        assert config_module.Config.HTTP_TIMEOUT_SECONDS == 30.0

    def test_default_service_name(self, monkeypatch):
        monkeypatch.delenv("SERVICE_NAME", raising=False)
        importlib.reload(config_module)
        assert config_module.Config.SERVICE_NAME == "fcp-gemini-python-client"

    def test_default_environment(self, monkeypatch):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        importlib.reload(config_module)
        assert config_module.Config.ENVIRONMENT == "development"

    def test_default_thinking_budgets(self, monkeypatch):
        for key in ["THINKING_BUDGET_MINIMAL", "THINKING_BUDGET_LOW", "THINKING_BUDGET_MEDIUM", "THINKING_BUDGET_HIGH"]:
            monkeypatch.delenv(key, raising=False)
        importlib.reload(config_module)
        assert config_module.Config.THINKING_BUDGET_MINIMAL == 512
        assert config_module.Config.THINKING_BUDGET_LOW == 1024
        assert config_module.Config.THINKING_BUDGET_MEDIUM == 2048
        assert config_module.Config.THINKING_BUDGET_HIGH == 4096

    def test_default_costs(self, monkeypatch):
        monkeypatch.delenv("GEMINI_COST_PER_INPUT_TOKEN", raising=False)
        monkeypatch.delenv("GEMINI_COST_PER_OUTPUT_TOKEN", raising=False)
        importlib.reload(config_module)
        assert config_module.Config.GEMINI_COST_PER_INPUT_TOKEN == 0.0001
        assert config_module.Config.GEMINI_COST_PER_OUTPUT_TOKEN == 0.0003

    def test_default_max_image_size(self, monkeypatch):
        monkeypatch.delenv("MAX_IMAGE_SIZE_BYTES", raising=False)
        importlib.reload(config_module)
        assert config_module.Config.MAX_IMAGE_SIZE_BYTES == 10 * 1024 * 1024

    def test_default_deep_research_timeout(self, monkeypatch):
        monkeypatch.delenv("DEEP_RESEARCH_TIMEOUT_SECONDS", raising=False)
        importlib.reload(config_module)
        assert config_module.Config.DEEP_RESEARCH_TIMEOUT_SECONDS == 300

    def test_default_video_timeout(self, monkeypatch):
        monkeypatch.delenv("VIDEO_GENERATION_TIMEOUT_SECONDS", raising=False)
        importlib.reload(config_module)
        assert config_module.Config.VIDEO_GENERATION_TIMEOUT_SECONDS == 180

    def test_default_cache_ttl(self, monkeypatch):
        monkeypatch.delenv("CACHE_TTL_SECONDS", raising=False)
        importlib.reload(config_module)
        assert config_module.Config.CACHE_TTL_SECONDS == 3600

    def test_default_http_connections(self, monkeypatch):
        monkeypatch.delenv("HTTP_MAX_CONNECTIONS", raising=False)
        monkeypatch.delenv("HTTP_MAX_KEEPALIVE_CONNECTIONS", raising=False)
        importlib.reload(config_module)
        assert config_module.Config.HTTP_MAX_CONNECTIONS == 100
        assert config_module.Config.HTTP_MAX_KEEPALIVE_CONNECTIONS == 20

    def test_default_specialized_models(self, monkeypatch):
        monkeypatch.delenv("DEEP_RESEARCH_AGENT", raising=False)
        monkeypatch.delenv("VEO_MODEL_NAME", raising=False)
        monkeypatch.delenv("GEMINI_LIVE_MODEL_NAME", raising=False)
        importlib.reload(config_module)
        assert config_module.Config.DEEP_RESEARCH_AGENT == "gemini-3-deep-research"
        assert config_module.Config.VEO_MODEL_NAME == "veo-3.1"
        assert config_module.Config.GEMINI_LIVE_MODEL_NAME == "gemini-3-live-preview"

    def test_default_api_version(self, monkeypatch):
        monkeypatch.delenv("API_VERSION", raising=False)
        importlib.reload(config_module)
        assert config_module.Config.API_VERSION == "1.0.0"


class TestConfigOverrides:
    """Test environment variable overrides."""

    def test_api_key_override(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "my-test-key")
        importlib.reload(config_module)
        assert config_module.Config.GEMINI_API_KEY == "my-test-key"

    def test_model_name_override(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL_NAME", "gemini-3-pro")
        importlib.reload(config_module)
        assert config_module.Config.GEMINI_MODEL_NAME == "gemini-3-pro"

    def test_timeout_override(self, monkeypatch):
        monkeypatch.setenv("HTTP_TIMEOUT_SECONDS", "60.0")
        importlib.reload(config_module)
        assert config_module.Config.HTTP_TIMEOUT_SECONDS == 60.0

    def test_thinking_budget_override(self, monkeypatch):
        monkeypatch.setenv("THINKING_BUDGET_HIGH", "8192")
        importlib.reload(config_module)
        assert config_module.Config.THINKING_BUDGET_HIGH == 8192
