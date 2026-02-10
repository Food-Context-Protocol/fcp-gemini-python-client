# Test Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a comprehensive test suite achieving 100% code coverage with zero custom mock infrastructure, using stdlib `unittest.mock` for unit tests and the SDK's built-in `DebugConfig` replay system for integration tests.

**Architecture:** Three-layer testing strategy: (1) Pure function tests with no mocking, (2) Client wrapper tests using `unittest.mock.AsyncMock` to isolate our logic from the genai SDK, (3) Integration tests using `google-genai`'s `DebugConfig(client_mode="replay")` to replay recorded API interactions. Tests are separated by `@pytest.mark.integration` so CI runs unit tests by default and integration tests on demand.

**Tech Stack:** pytest, pytest-asyncio, pytest-cov, unittest.mock (stdlib), google-genai DebugConfig/ReplayApiClient, monkeypatch for env vars, respx for HTTP mocking (optional, httpx-native)

---

## Testing Strategy Summary

### Layer 1: Pure Unit Tests (no mocking)
Files: `utils.py`, `security.py`, `config.py`, `gemini_constants.py`
These are pure functions or env-var-driven config. Test by calling directly with `monkeypatch.setenv`.

### Layer 2: Helper/Client Tests (stdlib `unittest.mock`)
Files: `gemini_helpers.py`, `gemini.py`, `gemini_base.py`
Mock `genai.Client` constructor and `httpx.AsyncClient` where needed. All mocks are stdlib.

### Layer 3: Generation Method Tests (stdlib `AsyncMock`)
Files: `gemini_generation.py`, `gemini_async_ops.py`, `gemini_live.py`
Create a `GeminiClient` instance, replace `self.client` with an `AsyncMock`, test that our wrapper logic (retry, JSON parsing, grounding extraction, streaming) works correctly.

### Layer 4: Integration Tests (SDK Replay System)
Use `DebugConfig(client_mode="record")` with a real API key to record responses to `tests/replays/`.
CI runs with `client_mode="replay"` and a dummy key. Marked `@pytest.mark.integration`.

### Why NOT a FakeGeminiClient
- The SDK's replay system records real API responses - higher fidelity than any fake
- Custom fakes drift from the real API surface over time
- `unittest.mock.AsyncMock` already covers unit-level isolation
- Zero maintenance overhead vs. a hand-maintained fake

---

### Task 1: Test Infrastructure Setup

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/replays/.gitkeep`
- Modify: `pyproject.toml:65-82` (pytest config, coverage, markers)
- Modify: `Makefile` (test targets)

**Step 1: Add `fail_under = 100` to pyproject.toml and register markers**

In `pyproject.toml`, update:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
minversion = "8.0"
testpaths = ["tests"]
markers = [
    "integration: tests that require GEMINI_API_KEY or network access",
]

[tool.coverage.run]
source = ["src/fcp"]
omit = ["tests/*"]

[tool.coverage.report]
fail_under = 100
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

**Step 2: Create `tests/__init__.py`**

```python
```

(Empty file - marks tests as a package)

**Step 3: Create `tests/conftest.py`**

```python
"""Shared test fixtures for fcp-gemini-python-client."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Ensure tests don't leak env vars. Clear GEMINI_API_KEY by default."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)


@pytest.fixture
def mock_genai_client():
    """Create a mock google.genai.Client with async model methods."""
    client = MagicMock()

    # Mock aio.models.generate_content
    client.aio.models.generate_content = AsyncMock()
    client.aio.models.generate_content_stream = AsyncMock()
    client.aio.models.generate_images = AsyncMock()
    client.aio.models.generate_videos = AsyncMock()

    # Mock aio.caches
    client.aio.caches.create = AsyncMock()

    # Mock aio.interactions
    client.aio.interactions.create = AsyncMock()
    client.aio.interactions.get = AsyncMock()

    # Mock aio.operations
    client.aio.operations.get = AsyncMock()

    # Mock aio.live
    client.aio.live.connect = MagicMock()

    return client


@pytest.fixture
def gemini_client(mock_genai_client):
    """Create a GeminiClient with a mocked genai.Client injected."""
    with patch("fcp.gemini.gemini_base.genai.Client", return_value=mock_genai_client):
        with patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key"):
            from fcp.gemini.gemini import GeminiClient, reset_gemini_client

            reset_gemini_client()
            client = GeminiClient()
            client.client = mock_genai_client
            yield client
            reset_gemini_client()


@pytest.fixture
def mock_response():
    """Factory for creating mock Gemini API responses."""

    def _make(
        text="test response",
        input_tokens=10,
        output_tokens=20,
        candidates=None,
        thinking_parts=None,
    ):
        response = MagicMock()
        response.text = text

        # Usage metadata
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = input_tokens
        response.usage_metadata.candidates_token_count = output_tokens

        # Candidates (for grounding, thinking, code execution)
        if candidates is not None:
            response.candidates = candidates
        else:
            response.candidates = []

        return response

    return _make
```

**Step 4: Create `tests/replays/.gitkeep`**

```
```

**Step 5: Update Makefile test targets**

Add to `Makefile`:
```makefile
test:
	uv run pytest tests/ -m "not integration" -v

test-integration:
	uv run pytest tests/ -m "integration" -v

coverage:
	uv run pytest tests/ -m "not integration" --cov --cov-report=term-missing
```

**Step 6: Run tests to verify infrastructure**

Run: `uv run pytest tests/ -v --co`
Expected: 0 tests collected (no test files yet), no errors

**Step 7: Commit**

```bash
git add tests/ pyproject.toml Makefile
git commit -m "test: add test infrastructure with conftest, markers, and 100% coverage target"
```

---

### Task 2: Test `utils.py` - JSON Extraction (Pure Functions)

**Files:**
- Create: `tests/test_utils.py`

**Step 1: Write the tests**

```python
"""Tests for fcp.gemini.utils - JSON extraction utilities."""

import pytest

from fcp.gemini.utils import (
    _extract_balanced_json,
    extract_json,
    extract_json_with_key,
    record_gemini_usage,
)


class TestExtractJson:
    """Tests for extract_json()."""

    def test_none_input(self):
        assert extract_json(None) is None

    def test_empty_string(self):
        assert extract_json("") is None

    def test_non_string_input(self):
        assert extract_json(123) is None

    def test_pure_json_object(self):
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_pure_json_array(self):
        result = extract_json('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_json_in_markdown_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_json_in_generic_code_block(self):
        text = '```\n{"key": "value"}\n```'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_json_embedded_in_prose(self):
        text = 'Here is the result: {"key": "value"} and more text'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_array_before_object(self):
        text = 'Result: [{"a": 1}] then {"b": 2}'
        result = extract_json(text)
        assert result == [{"a": 1}]

    def test_object_before_array(self):
        text = 'Result: {"a": 1} then [2, 3]'
        result = extract_json(text)
        assert result == {"a": 1}

    def test_array_after_failed_object(self):
        """Strategy 6: array tried after object was attempted first."""
        text = '{invalid json} then [1, 2, 3]'
        result = extract_json(text)
        assert result == [1, 2, 3]

    def test_last_resort_object_pattern(self):
        text = 'blah blah {"name": "test", "value": 42} blah'
        result = extract_json(text)
        assert result == {"name": "test", "value": 42}

    def test_last_resort_array_of_objects(self):
        text = 'data: [{"id": 1}, {"id": 2}] end'
        result = extract_json(text)
        assert result == [{"id": 1}, {"id": 2}]

    def test_completely_invalid(self):
        assert extract_json("no json here at all") is None

    def test_whitespace_handling(self):
        result = extract_json('  \n  {"key": "value"}  \n  ')
        assert result == {"key": "value"}

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = extract_json(text)
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_json_with_escaped_quotes(self):
        text = '{"key": "value with \\"quotes\\""}'
        result = extract_json(text)
        assert result == {"key": 'value with "quotes"'}


class TestExtractBalancedJson:
    """Tests for _extract_balanced_json()."""

    def test_simple_object(self):
        result = _extract_balanced_json('{"a": 1}', "{", "}")
        assert result == '{"a": 1}'

    def test_simple_array(self):
        result = _extract_balanced_json("[1, 2]", "[", "]")
        assert result == "[1, 2]"

    def test_nested(self):
        result = _extract_balanced_json('{"a": {"b": 1}}', "{", "}")
        assert result == '{"a": {"b": 1}}'

    def test_no_match(self):
        assert _extract_balanced_json("no braces", "{", "}") is None

    def test_with_string_containing_braces(self):
        text = '{"key": "value with { and } inside"}'
        result = _extract_balanced_json(text, "{", "}")
        assert result == text

    def test_unbalanced_returns_none(self):
        assert _extract_balanced_json('{"unclosed', "{", "}") is None

    def test_with_escaped_backslash(self):
        text = '{"key": "value\\\\"}'
        result = _extract_balanced_json(text, "{", "}")
        assert result is not None


class TestExtractJsonWithKey:
    """Tests for extract_json_with_key()."""

    def test_none_input(self):
        assert extract_json_with_key(None, "key") is None

    def test_empty_string(self):
        assert extract_json_with_key("", "key") is None

    def test_non_string_input(self):
        assert extract_json_with_key(123, "key") is None

    def test_key_found(self):
        result = extract_json_with_key('{"target": "value"}', "target")
        assert result == {"target": "value"}

    def test_key_not_found(self):
        result = extract_json_with_key('{"other": "value"}', "target")
        assert result is None

    def test_key_in_embedded_json(self):
        text = 'some text {"target": "value", "extra": 1} more text'
        result = extract_json_with_key(text, "target")
        assert result is not None
        assert result["target"] == "value"

    def test_returns_list_without_key(self):
        """extract_json returns a list, which doesn't have dict keys."""
        result = extract_json_with_key("[1, 2, 3]", "key")
        assert result is None

    def test_fallback_pattern_match(self):
        """The regex fallback pattern after extract_json fails to find key."""
        text = 'prefix {"required_key": "found"} suffix'
        result = extract_json_with_key(text, "required_key")
        assert result == {"required_key": "found"}


class TestRecordGeminiUsage:
    """Tests for record_gemini_usage() no-op stub."""

    def test_no_op(self):
        """Should not raise - it's a no-op stub."""
        record_gemini_usage(
            method="test",
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.001,
            latency_seconds=0.5,
            success=True,
        )
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_utils.py -v`
Expected: All tests PASS

**Step 3: Check coverage for utils.py**

Run: `uv run pytest tests/test_utils.py --cov=fcp.gemini.utils --cov-report=term-missing`
Expected: 100% coverage on `utils.py`

**Step 4: Commit**

```bash
git add tests/test_utils.py
git commit -m "test: add utils.py tests - JSON extraction with 100% coverage"
```

---

### Task 3: Test `security.py` - URL Validation (Pure Functions + Env Vars)

**Files:**
- Create: `tests/test_security.py`

**Step 1: Write the tests**

```python
"""Tests for fcp.gemini.security - SSRF prevention and URL validation."""

import importlib

import pytest

import fcp.gemini.security as security_module
from fcp.gemini.security import (
    ImageURLError,
    _is_ip_blocked,
    validate_content_type,
    validate_image_url,
    verify_url_reachability,
)


class TestIsIpBlocked:
    """Tests for _is_ip_blocked()."""

    def test_private_10_range(self):
        assert _is_ip_blocked("10.0.0.1") is True

    def test_private_172_range(self):
        assert _is_ip_blocked("172.16.0.1") is True

    def test_private_192_range(self):
        assert _is_ip_blocked("192.168.1.1") is True

    def test_loopback(self):
        assert _is_ip_blocked("127.0.0.1") is True

    def test_link_local(self):
        assert _is_ip_blocked("169.254.169.254") is True

    def test_cgnat(self):
        assert _is_ip_blocked("100.64.0.1") is True

    def test_benchmarking(self):
        assert _is_ip_blocked("198.18.0.1") is True

    def test_ipv6_loopback(self):
        assert _is_ip_blocked("::1") is True

    def test_ipv6_link_local(self):
        assert _is_ip_blocked("fe80::1") is True

    def test_ipv6_ula(self):
        assert _is_ip_blocked("fd00::1") is True

    def test_public_ip(self):
        assert _is_ip_blocked("8.8.8.8") is False

    def test_not_an_ip(self):
        assert _is_ip_blocked("example.com") is False


class TestValidateImageUrl:
    """Tests for validate_image_url() in development mode."""

    def test_empty_url(self):
        with pytest.raises(ImageURLError, match="URL is required"):
            validate_image_url("")

    def test_none_url(self):
        with pytest.raises(ImageURLError, match="URL is required"):
            validate_image_url(None)

    def test_non_string_url(self):
        with pytest.raises(ImageURLError, match="URL is required"):
            validate_image_url(123)

    def test_file_scheme_blocked(self):
        with pytest.raises(ImageURLError, match="file://"):
            validate_image_url("file:///etc/passwd")

    def test_data_scheme_blocked(self):
        with pytest.raises(ImageURLError, match="data:"):
            validate_image_url("data:image/png;base64,abc")

    def test_ftp_scheme_blocked(self):
        with pytest.raises(ImageURLError, match="ftp://"):
            validate_image_url("ftp://example.com/file")

    def test_invalid_scheme(self):
        with pytest.raises(ImageURLError, match="scheme must be"):
            validate_image_url("gopher://example.com")

    def test_no_hostname(self):
        with pytest.raises(ImageURLError, match="hostname"):
            validate_image_url("https://")

    def test_blocked_hostname_metadata(self):
        with pytest.raises(ImageURLError, match="not allowed"):
            validate_image_url("https://metadata.google.internal/something")

    def test_blocked_hostname_169(self):
        with pytest.raises(ImageURLError, match="not allowed"):
            validate_image_url("https://169.254.169.254/latest/meta-data/")

    def test_blocked_private_ip(self):
        with pytest.raises(ImageURLError, match="private/internal"):
            validate_image_url("https://10.0.0.1/image.png", allow_any_domain=True)

    def test_domain_not_in_whitelist(self):
        with pytest.raises(ImageURLError, match="not in the allowed list"):
            validate_image_url("https://evil.com/image.png")

    def test_allowed_domain(self):
        result = validate_image_url("https://storage.googleapis.com/bucket/img.png")
        assert result == "https://storage.googleapis.com/bucket/img.png"

    def test_subdomain_of_allowed(self):
        result = validate_image_url("https://sub.storage.googleapis.com/img.png")
        assert result == "https://sub.storage.googleapis.com/img.png"

    def test_allow_any_domain(self):
        result = validate_image_url("https://any-domain.com/img.png", allow_any_domain=True)
        assert result == "https://any-domain.com/img.png"

    def test_additional_domains(self):
        result = validate_image_url(
            "https://custom.example.com/img.png",
            additional_domains={"custom.example.com"},
        )
        assert result == "https://custom.example.com/img.png"

    def test_credentials_in_url(self):
        with pytest.raises(ImageURLError, match="credentials"):
            validate_image_url("https://user:pass@storage.googleapis.com/img.png")

    def test_non_standard_port_blocked(self):
        with pytest.raises(ImageURLError, match="Non-standard port"):
            validate_image_url("https://storage.googleapis.com:9999/img.png")

    def test_allowed_dev_ports(self):
        """In dev mode, ports 80, 443, 8080, 8000, 3000 are allowed."""
        result = validate_image_url("http://localhost:8080/img.png")
        assert "8080" in result

    def test_localhost_allowed_in_dev(self):
        result = validate_image_url("http://localhost/img.png")
        assert result == "http://localhost/img.png"

    def test_whitespace_stripped(self):
        result = validate_image_url("  https://storage.googleapis.com/img.png  ")
        assert result == "https://storage.googleapis.com/img.png"


class TestValidateImageUrlProduction:
    """Tests for validate_image_url() in production mode."""

    def test_http_blocked_in_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        importlib.reload(security_module)
        try:
            with pytest.raises(ImageURLError, match="HTTP URLs are not allowed in production"):
                security_module.validate_image_url("http://storage.googleapis.com/img.png")
        finally:
            monkeypatch.delenv("ENVIRONMENT", raising=False)
            importlib.reload(security_module)

    def test_localhost_not_allowed_in_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        importlib.reload(security_module)
        try:
            with pytest.raises(ImageURLError, match="not in the allowed list"):
                security_module.validate_image_url("https://localhost/img.png")
        finally:
            monkeypatch.delenv("ENVIRONMENT", raising=False)
            importlib.reload(security_module)

    def test_production_ports_restricted(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        importlib.reload(security_module)
        try:
            with pytest.raises(ImageURLError, match="Non-standard port"):
                security_module.validate_image_url("https://storage.googleapis.com:8080/img.png")
        finally:
            monkeypatch.delenv("ENVIRONMENT", raising=False)
            importlib.reload(security_module)


class TestIsProduction:
    """Tests for _is_production()."""

    def test_production_env(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        importlib.reload(security_module)
        assert security_module._is_production() is True
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        importlib.reload(security_module)

    def test_prod_env(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "prod")
        importlib.reload(security_module)
        assert security_module._is_production() is True
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        importlib.reload(security_module)

    def test_k_service_env(self, monkeypatch):
        monkeypatch.setenv("K_SERVICE", "my-service")
        importlib.reload(security_module)
        assert security_module._is_production() is True
        monkeypatch.delenv("K_SERVICE", raising=False)
        importlib.reload(security_module)

    def test_development_env(self, monkeypatch):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        monkeypatch.delenv("K_SERVICE", raising=False)
        importlib.reload(security_module)
        assert security_module._is_production() is False


class TestValidateContentType:
    """Tests for validate_content_type()."""

    def test_none(self):
        assert validate_content_type(None) is False

    def test_empty(self):
        assert validate_content_type("") is False

    def test_jpeg(self):
        assert validate_content_type("image/jpeg") is True

    def test_png(self):
        assert validate_content_type("image/png") is True

    def test_gif(self):
        assert validate_content_type("image/gif") is True

    def test_webp(self):
        assert validate_content_type("image/webp") is True

    def test_heic(self):
        assert validate_content_type("image/heic") is True

    def test_heif(self):
        assert validate_content_type("image/heif") is True

    def test_with_charset(self):
        assert validate_content_type("image/jpeg; charset=utf-8") is True

    def test_text_html(self):
        assert validate_content_type("text/html") is False

    def test_application_json(self):
        assert validate_content_type("application/json") is False

    def test_case_insensitive(self):
        assert validate_content_type("IMAGE/JPEG") is True


class TestVerifyUrlReachability:
    """Tests for verify_url_reachability()."""

    async def test_invalid_url_returns_false(self):
        result = await verify_url_reachability("file:///etc/passwd")
        assert result is False

    async def test_unreachable_url_returns_false(self):
        result = await verify_url_reachability("https://this-domain-does-not-exist-xyz123.com/img.png")
        assert result is False
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_security.py -v`
Expected: All tests PASS

**Step 3: Check coverage**

Run: `uv run pytest tests/test_security.py --cov=fcp.gemini.security --cov-report=term-missing`
Expected: Close to 100% on `security.py` (the `verify_url_reachability` HEAD->GET fallback path may need a respx mock for full coverage)

**Step 4: Commit**

```bash
git add tests/test_security.py
git commit -m "test: add security.py tests - SSRF prevention with env-based prod/dev modes"
```

---

### Task 4: Test `config.py` - Configuration (Env Var Overrides)

**Files:**
- Create: `tests/test_config.py`

**Step 1: Write the tests**

```python
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
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_config.py
git commit -m "test: add config.py tests - default values and env var overrides"
```

---

### Task 5: Test `gemini_constants.py` - Constants and Env Overrides

**Files:**
- Create: `tests/test_constants.py`

**Step 1: Write the tests**

```python
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
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_constants.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_constants.py
git commit -m "test: add gemini_constants.py tests - defaults, env overrides, retryable exceptions"
```

---

### Task 6: Test `gemini_helpers.py` - Helper Functions

**Files:**
- Create: `tests/test_helpers.py`

**Step 1: Write the tests**

```python
"""Tests for fcp.gemini.gemini_helpers - retry, parsing, extraction helpers."""

from unittest.mock import MagicMock

import pytest

from fcp.gemini.gemini_helpers import (
    GeminiThinkingResult,
    _extract_grounding_sources,
    _extract_thinking_content,
    _get_thinking_budget,
    _log_token_usage,
    _parse_json_response,
    gemini_retry,
)


class TestGetThinkingBudget:
    """Tests for _get_thinking_budget()."""

    def test_minimal(self):
        assert _get_thinking_budget("minimal") == 512

    def test_low(self):
        assert _get_thinking_budget("low") == 1024

    def test_medium(self):
        assert _get_thinking_budget("medium") == 2048

    def test_high(self):
        assert _get_thinking_budget("high") == 4096

    def test_case_insensitive(self):
        assert _get_thinking_budget("HIGH") == 4096
        assert _get_thinking_budget("Medium") == 2048

    def test_unknown_returns_default(self):
        result = _get_thinking_budget("unknown")
        assert result == 16384  # Default fallback


class TestExtractThinkingContent:
    """Tests for _extract_thinking_content()."""

    def test_no_candidates(self):
        response = MagicMock()
        response.candidates = None
        assert _extract_thinking_content(response) is None

    def test_empty_candidates(self):
        response = MagicMock()
        response.candidates = []
        assert _extract_thinking_content(response) is None

    def test_candidate_with_no_content(self):
        candidate = MagicMock()
        candidate.content = None
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_thinking_content(response) is None

    def test_candidate_with_no_parts(self):
        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = None
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_thinking_content(response) is None

    def test_thinking_part_extracted(self):
        part = MagicMock()
        part.thought = True
        part.text = "I'm thinking about this..."
        candidate = MagicMock()
        candidate.content.parts = [part]
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_thinking_content(response) == "I'm thinking about this..."

    def test_non_thinking_parts_skipped(self):
        regular_part = MagicMock()
        regular_part.thought = False
        thinking_part = MagicMock()
        thinking_part.thought = True
        thinking_part.text = "thinking"
        candidate = MagicMock()
        candidate.content.parts = [regular_part, thinking_part]
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_thinking_content(response) == "thinking"

    def test_multiple_thinking_parts_joined(self):
        part1 = MagicMock()
        part1.thought = True
        part1.text = "first thought"
        part2 = MagicMock()
        part2.thought = True
        part2.text = "second thought"
        candidate = MagicMock()
        candidate.content.parts = [part1, part2]
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_thinking_content(response) == "first thought\nsecond thought"

    def test_thinking_part_with_no_text(self):
        part = MagicMock()
        part.thought = True
        part.text = None
        candidate = MagicMock()
        candidate.content.parts = [part]
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_thinking_content(response) is None

    def test_no_candidates_attr(self):
        """Response object without candidates attribute."""
        response = MagicMock(spec=[])
        assert _extract_thinking_content(response) is None


class TestParseJsonResponse:
    """Tests for _parse_json_response()."""

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Empty response"):
            _parse_json_response("")

    def test_none_text(self):
        with pytest.raises(ValueError, match="Empty response"):
            _parse_json_response(None)

    def test_valid_json_object(self):
        assert _parse_json_response('{"key": "value"}') == {"key": "value"}

    def test_valid_json_array(self):
        assert _parse_json_response("[1, 2, 3]") == [1, 2, 3]

    def test_whitespace_stripped(self):
        assert _parse_json_response('  {"key": "value"}  ') == {"key": "value"}

    def test_bom_stripped(self):
        assert _parse_json_response('\ufeff{"key": "value"}') == {"key": "value"}

    def test_markdown_wrapped(self):
        text = '```json\n{"key": "value"}\n```'
        assert _parse_json_response(text) == {"key": "value"}

    def test_completely_invalid(self):
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            _parse_json_response("not json at all")


class TestExtractGroundingSources:
    """Tests for _extract_grounding_sources()."""

    def test_no_candidates(self):
        response = MagicMock()
        response.candidates = []
        assert _extract_grounding_sources(response) == []

    def test_no_grounding_metadata(self):
        candidate = MagicMock()
        candidate.grounding_metadata = None
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_grounding_sources(response) == []

    def test_no_grounding_chunks_attr(self):
        candidate = MagicMock(spec=["grounding_metadata"])
        candidate.grounding_metadata = MagicMock(spec=[])
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_grounding_sources(response) == []

    def test_empty_grounding_chunks(self):
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = []
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_grounding_sources(response) == []

    def test_extracts_sources(self):
        chunk = MagicMock()
        chunk.web.uri = "https://example.com"
        chunk.web.title = "Example"
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk]
        response = MagicMock()
        response.candidates = [candidate]
        result = _extract_grounding_sources(response)
        assert result == [{"uri": "https://example.com", "title": "Example"}]

    def test_chunk_without_web_skipped(self):
        chunk = MagicMock(spec=[])  # No 'web' attribute
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk]
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_grounding_sources(response) == []


class TestLogTokenUsage:
    """Tests for _log_token_usage()."""

    def test_with_usage_metadata(self):
        response = MagicMock()
        response.usage_metadata.prompt_token_count = 100
        response.usage_metadata.candidates_token_count = 200
        result = _log_token_usage(response, "test_method")
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 200
        assert result["total_tokens"] == 300
        assert result["cost_usd"] > 0

    def test_without_usage_metadata(self):
        response = MagicMock()
        response.usage_metadata = None
        result = _log_token_usage(response, "test_method")
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["total_tokens"] == 0
        assert result["cost_usd"] == 0.0

    def test_no_usage_metadata_attr(self):
        response = MagicMock(spec=[])
        result = _log_token_usage(response, "test_method")
        assert result["input_tokens"] == 0

    def test_non_numeric_token_counts(self):
        """Handles mock objects that return non-int values."""
        response = MagicMock()
        response.usage_metadata.prompt_token_count = "not a number"
        response.usage_metadata.candidates_token_count = "not a number"
        result = _log_token_usage(response, "test_method")
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0

    def test_cost_calculation(self):
        response = MagicMock()
        response.usage_metadata.prompt_token_count = 1000
        response.usage_metadata.candidates_token_count = 500
        result = _log_token_usage(response, "test_method")
        expected_cost = (1000 * 0.0001) + (500 * 0.0003)
        assert result["cost_usd"] == round(expected_cost, 6)

    def test_latency_and_success_passed(self):
        """Verify latency/success args are accepted (metrics recording)."""
        response = MagicMock()
        response.usage_metadata.prompt_token_count = 10
        response.usage_metadata.candidates_token_count = 20
        result = _log_token_usage(response, "test_method", latency_seconds=1.5, success=False)
        assert result["input_tokens"] == 10


class TestGeminiRetryDecorator:
    """Tests for the retry decorator."""

    def test_gemini_retry_exists(self):
        """Verify the retry decorator is created."""
        assert gemini_retry is not None

    def test_thinking_result_type(self):
        """Verify GeminiThinkingResult TypedDict structure."""
        result: GeminiThinkingResult = {"analysis": {"key": "value"}, "thinking": "thoughts"}
        assert result["analysis"] == {"key": "value"}
        assert result["thinking"] == "thoughts"
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_helpers.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_helpers.py
git commit -m "test: add gemini_helpers.py tests - parsing, extraction, retry, token logging"
```

---

### Task 7: Test `gemini.py` - Client Singleton and Proxy

**Files:**
- Create: `tests/test_client.py`

**Step 1: Write the tests**

```python
"""Tests for fcp.gemini.gemini - Client singleton, proxy, and module exports."""

from unittest.mock import MagicMock, patch

from fcp.gemini.gemini import (
    GeminiClient,
    _GeminiProxy,
    get_gemini,
    get_gemini_client,
    reset_gemini_client,
    set_gemini_client,
)


class TestSingleton:
    """Tests for get_gemini_client() singleton pattern."""

    def setup_method(self):
        reset_gemini_client()

    def teardown_method(self):
        reset_gemini_client()

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key")
    @patch("fcp.gemini.gemini_base.genai.Client")
    def test_creates_client_on_first_call(self, mock_genai):
        client = get_gemini_client()
        assert client is not None
        assert isinstance(client, GeminiClient)

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key")
    @patch("fcp.gemini.gemini_base.genai.Client")
    def test_returns_same_instance(self, mock_genai):
        client1 = get_gemini_client()
        client2 = get_gemini_client()
        assert client1 is client2

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key")
    @patch("fcp.gemini.gemini_base.genai.Client")
    def test_get_gemini_returns_same_as_get_gemini_client(self, mock_genai):
        client1 = get_gemini_client()
        client2 = get_gemini()
        assert client1 is client2


class TestSetResetClient:
    """Tests for set_gemini_client() and reset_gemini_client()."""

    def setup_method(self):
        reset_gemini_client()

    def teardown_method(self):
        reset_gemini_client()

    def test_set_client(self):
        mock_client = MagicMock()
        set_gemini_client(mock_client)
        result = get_gemini_client()
        assert result is mock_client

    def test_reset_client(self):
        mock_client = MagicMock()
        set_gemini_client(mock_client)
        reset_gemini_client()
        # After reset, getting client should create a new one
        with patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key"):
            with patch("fcp.gemini.gemini_base.genai.Client"):
                new_client = get_gemini_client()
                assert new_client is not mock_client


class TestGeminiProxy:
    """Tests for _GeminiProxy lazy-access object."""

    def setup_method(self):
        reset_gemini_client()

    def teardown_method(self):
        reset_gemini_client()

    def test_getattr_forwards_to_client(self):
        mock_client = MagicMock()
        mock_client.some_method.return_value = "result"
        set_gemini_client(mock_client)

        proxy = _GeminiProxy()
        assert proxy.some_method() == "result"

    def test_repr(self):
        mock_client = MagicMock()
        set_gemini_client(mock_client)
        proxy = _GeminiProxy()
        repr_str = repr(proxy)
        assert "GeminiProxy" in repr_str

    def test_call_forwards(self):
        mock_client = MagicMock()
        mock_client.return_value = "called"
        set_gemini_client(mock_client)
        proxy = _GeminiProxy()
        result = proxy()
        assert result == "called"


class TestModuleExports:
    """Test __all__ exports are importable."""

    def test_all_exports(self):
        from fcp.gemini import gemini

        for name in gemini.__all__:
            assert hasattr(gemini, name), f"Missing export: {name}"
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_client.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_client.py
git commit -m "test: add gemini.py tests - singleton, proxy, set/reset, exports"
```

---

### Task 8: Test `gemini_base.py` - HTTP Client and Media Fetching

**Files:**
- Create: `tests/test_base.py`

**Step 1: Write the tests**

```python
"""Tests for fcp.gemini.gemini_base - HTTP client management and media fetching."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from fcp.gemini.gemini_base import GeminiBase, GeminiStreamingMixin


class TestHttpClientManagement:
    """Tests for _get_http_client() and close_http_client()."""

    def setup_method(self):
        GeminiBase.reset_http_client()

    def teardown_method(self):
        GeminiBase.reset_http_client()

    def test_creates_client_on_first_call(self):
        client = GeminiBase._get_http_client()
        assert isinstance(client, httpx.AsyncClient)

    def test_returns_same_client(self):
        client1 = GeminiBase._get_http_client()
        client2 = GeminiBase._get_http_client()
        assert client1 is client2

    async def test_close_http_client(self):
        client = GeminiBase._get_http_client()
        await GeminiBase.close_http_client()
        assert GeminiBase._http_client is None

    async def test_close_when_none(self):
        """Should not raise when no client exists."""
        await GeminiBase.close_http_client()

    def test_reset_http_client(self):
        GeminiBase._get_http_client()
        GeminiBase.reset_http_client()
        assert GeminiBase._http_client is None


class TestGeminiBaseInit:
    """Tests for GeminiBase.__init__()."""

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key")
    @patch("fcp.gemini.gemini_base.genai.Client")
    def test_init_with_api_key(self, mock_genai_client):
        base = GeminiBase()
        assert base.client is not None
        mock_genai_client.assert_called_once_with(api_key="test-key")

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "")
    def test_init_without_api_key(self):
        base = GeminiBase()
        assert base.client is None

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "")
    def test_require_client_raises(self):
        base = GeminiBase()
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY not configured"):
            base._require_client()

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key")
    @patch("fcp.gemini.gemini_base.genai.Client")
    def test_require_client_returns_client(self, mock_genai_client):
        base = GeminiBase()
        result = base._require_client()
        assert result is not None


class TestPrepareParts:
    """Tests for _prepare_parts()."""

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key")
    @patch("fcp.gemini.gemini_base.genai.Client")
    async def test_text_only(self, mock_genai_client):
        base = GeminiBase()
        parts = await base._prepare_parts("hello")
        assert len(parts) == 1
        assert parts[0].text == "hello"

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key")
    @patch("fcp.gemini.gemini_base.genai.Client")
    async def test_with_image_bytes(self, mock_genai_client):
        base = GeminiBase()
        parts = await base._prepare_parts(
            "describe", image_bytes=b"fake-image", image_mime_type="image/png"
        )
        assert len(parts) == 2

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key")
    @patch("fcp.gemini.gemini_base.genai.Client")
    async def test_with_image_url(self, mock_genai_client):
        base = GeminiBase()
        base._fetch_media = AsyncMock(return_value=(b"image-data", "image/jpeg"))
        parts = await base._prepare_parts("describe", image_url="https://example.com/img.jpg")
        assert len(parts) == 2

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key")
    @patch("fcp.gemini.gemini_base.genai.Client")
    async def test_with_media_url(self, mock_genai_client):
        base = GeminiBase()
        base._fetch_media = AsyncMock(return_value=(b"media-data", "video/mp4"))
        parts = await base._prepare_parts("analyze", media_url="https://example.com/vid.mp4")
        assert len(parts) == 2


class TestStreamingMixin:
    """Tests for GeminiStreamingMixin."""

    def test_async_text_stream_type(self):
        """Verify the type alias exists."""
        from collections.abc import AsyncIterator

        assert GeminiStreamingMixin.AsyncTextStream is AsyncIterator[str]
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_base.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_base.py
git commit -m "test: add gemini_base.py tests - HTTP client, init, prepare_parts"
```

---

### Task 9: Test `gemini_generation.py` - Generation Mixins

**Files:**
- Create: `tests/test_generation.py`

**Step 1: Write the tests**

```python
"""Tests for fcp.gemini.gemini_generation - all generation method mixins."""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestGeminiGenerationMixin:
    """Tests for basic text/JSON generation."""

    async def test_generate_content(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text="Hello world")
        mock_genai_client.aio.models.generate_content.return_value = resp
        result = await gemini_client.generate_content("Say hello")
        assert result == "Hello world"
        mock_genai_client.aio.models.generate_content.assert_awaited_once()

    async def test_generate_content_empty(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text=None)
        mock_genai_client.aio.models.generate_content.return_value = resp
        result = await gemini_client.generate_content("Say hello")
        assert result == ""

    async def test_generate_json(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text='{"key": "value"}')
        mock_genai_client.aio.models.generate_content.return_value = resp
        result = await gemini_client.generate_json("Get JSON")
        assert result == {"key": "value"}

    async def test_generate_json_list_wrapped(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text="[1, 2, 3]")
        mock_genai_client.aio.models.generate_content.return_value = resp
        result = await gemini_client.generate_json("Get list")
        assert result == {"items": [1, 2, 3]}

    async def test_generate_content_stream(self, gemini_client, mock_genai_client):
        chunk1 = MagicMock()
        chunk1.text = "Hello "
        chunk2 = MagicMock()
        chunk2.text = "world"

        async def mock_stream(*args, **kwargs):
            for chunk in [chunk1, chunk2]:
                yield chunk

        mock_genai_client.aio.models.generate_content_stream = mock_stream
        chunks = []
        async for chunk in gemini_client.generate_content_stream("Say hello"):
            chunks.append(chunk)
        assert chunks == ["Hello ", "world"]

    async def test_generate_json_stream(self, gemini_client, mock_genai_client):
        chunk1 = MagicMock()
        chunk1.text = '{"ke'
        chunk2 = MagicMock()
        chunk2.text = 'y": "value"}'
        chunk3 = MagicMock()
        chunk3.text = None  # Chunk with no text

        async def mock_stream(*args, **kwargs):
            for chunk in [chunk1, chunk2, chunk3]:
                yield chunk

        mock_genai_client.aio.models.generate_content_stream = mock_stream
        # Note: generate_json_stream uses cast + await, need to adapt mock
        gemini_client.client.aio.models.generate_content_stream = mock_stream
        chunks = []
        async for chunk in gemini_client.generate_json_stream("Get JSON"):
            chunks.append(chunk)
        assert len(chunks) == 2  # chunk3 with None text is skipped


class TestGeminiToolingMixin:
    """Tests for function calling."""

    async def test_generate_with_tools_function_call(self, gemini_client, mock_genai_client, mock_response):
        # Build a response with function call
        func_call = MagicMock()
        func_call.name = "search"
        func_call.args = {"query": "test"}

        part = MagicMock()
        part.function_call = func_call

        candidate = MagicMock()
        candidate.content.parts = [part]

        resp = mock_response(text="Using search tool")
        resp.candidates = [candidate]
        mock_genai_client.aio.models.generate_content.return_value = resp

        tools = [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        result = await gemini_client.generate_with_tools("Find something", tools=tools)
        assert result["text"] == "Using search tool"
        assert len(result["function_calls"]) == 1
        assert result["function_calls"][0]["name"] == "search"
        assert result["function_calls"][0]["args"] == {"query": "test"}

    async def test_generate_with_tools_no_function_calls(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text="No tools needed")
        resp.candidates = []
        mock_genai_client.aio.models.generate_content.return_value = resp

        tools = [{"name": "search", "description": "Search"}]
        result = await gemini_client.generate_with_tools("Just answer", tools=tools)
        assert result["function_calls"] == []


class TestGeminiGroundingMixin:
    """Tests for Google Search grounding."""

    async def test_generate_with_grounding(self, gemini_client, mock_genai_client, mock_response):
        chunk = MagicMock()
        chunk.web.uri = "https://example.com"
        chunk.web.title = "Example"

        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk]

        resp = mock_response(text="Grounded answer")
        resp.candidates = [candidate]
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_with_grounding("Search query")
        assert result["text"] == "Grounded answer"
        assert len(result["sources"]) == 1

    async def test_generate_json_with_grounding(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text='{"answer": "yes"}')
        resp.candidates = [MagicMock()]
        resp.candidates[0].grounding_metadata = None
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_json_with_grounding("Query")
        assert result["data"] == {"answer": "yes"}

    async def test_generate_json_with_grounding_empty_response(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text=None)
        resp.candidates = []
        mock_genai_client.aio.models.generate_content.return_value = resp

        with pytest.raises(ValueError, match="empty response"):
            await gemini_client.generate_json_with_grounding("Query")


class TestGeminiThinkingMixin:
    """Tests for thinking/reasoning mode."""

    async def test_generate_with_thinking(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text="Thoughtful answer")
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_with_thinking("Think about this")
        assert result == "Thoughtful answer"

    async def test_generate_json_with_thinking(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text='{"analysis": "deep"}')
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_json_with_thinking("Analyze this")
        assert result == {"analysis": "deep"}

    async def test_generate_json_with_thinking_include_output(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text='{"result": "yes"}')
        resp.candidates = []
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_json_with_thinking(
            "Analyze", include_thinking_output=True
        )
        assert "analysis" in result
        assert "thinking" in result

    async def test_generate_with_large_context(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text="Large context answer")
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_with_large_context("Big prompt")
        assert result == "Large context answer"

    async def test_generate_json_with_large_context(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text='{"key": "value"}')
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_json_with_large_context("Big JSON prompt")
        assert result == {"key": "value"}

    async def test_generate_json_with_large_context_empty(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text=None)
        mock_genai_client.aio.models.generate_content.return_value = resp

        with pytest.raises(ValueError, match="empty response"):
            await gemini_client.generate_json_with_large_context("Empty prompt")

    async def test_generate_json_with_large_context_list(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text="[1, 2, 3]")
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_json_with_large_context("List prompt")
        assert result == {"items": [1, 2, 3]}


class TestGeminiCodeExecutionMixin:
    """Tests for code execution features."""

    async def test_generate_with_code_execution(self, gemini_client, mock_genai_client, mock_response):
        code_part = MagicMock()
        code_part.executable_code = MagicMock()
        code_part.executable_code.code = "print(42)"
        code_part.code_execution_result = MagicMock()
        code_part.code_execution_result.output = "42"

        candidate = MagicMock()
        candidate.content.parts = [code_part]

        resp = mock_response(text="The answer is 42")
        resp.candidates = [candidate]
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_with_code_execution("Calculate 42")
        assert result["text"] == "The answer is 42"
        assert result["code"] == "print(42)"
        assert result["execution_result"] == "42"

    async def test_generate_with_code_execution_no_code(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text="Just text")
        resp.candidates = []
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_with_code_execution("No code needed")
        assert result["code"] is None
        assert result["execution_result"] is None

    async def test_generate_json_with_agentic_vision(self, gemini_client, mock_genai_client, mock_response):
        gemini_client._fetch_media = AsyncMock(return_value=(b"img", "image/png"))
        resp = mock_response(text='{"objects": ["cat"]}')
        resp.candidates = []
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_json_with_agentic_vision(
            "Describe", "https://example.com/img.png"
        )
        assert result["analysis"] == {"objects": ["cat"]}

    async def test_generate_json_with_agentic_vision_empty(self, gemini_client, mock_genai_client, mock_response):
        gemini_client._fetch_media = AsyncMock(return_value=(b"img", "image/png"))
        resp = mock_response(text=None)
        resp.candidates = []
        mock_genai_client.aio.models.generate_content.return_value = resp

        with pytest.raises(ValueError, match="empty response"):
            await gemini_client.generate_json_with_agentic_vision(
                "Describe", "https://example.com/img.png"
            )


class TestGeminiMediaMixin:
    """Tests for media resolution and URL context."""

    async def test_generate_json_with_media_resolution(self, gemini_client, mock_genai_client, mock_response):
        gemini_client._fetch_media = AsyncMock(return_value=(b"img", "image/png"))
        resp = mock_response(text='{"detail": "fine"}')
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_json_with_media_resolution(
            "Analyze", "https://example.com/img.png", resolution="high"
        )
        assert result == {"detail": "fine"}

    async def test_generate_json_with_media_resolution_list(self, gemini_client, mock_genai_client, mock_response):
        gemini_client._fetch_media = AsyncMock(return_value=(b"img", "image/png"))
        resp = mock_response(text="[1, 2]")
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_json_with_media_resolution(
            "Analyze", "https://example.com/img.png"
        )
        assert result == {"items": [1, 2]}

    async def test_generate_json_with_url_context(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text='{"summary": "content"}')
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_json_with_url_context(
            "Summarize", urls=["https://example.com/page1"]
        )
        assert result["data"] == {"summary": "content"}
        assert result["sources"] == ["https://example.com/page1"]


class TestGeminiImageMixin:
    """Tests for image generation."""

    async def test_generate_image(self, gemini_client, mock_genai_client):
        img = MagicMock()
        img.image.image_bytes = b"fake-png-data"
        mock_resp = MagicMock()
        mock_resp.generated_images = [img]
        mock_genai_client.aio.models.generate_images.return_value = mock_resp

        result = await gemini_client.generate_image("A cat")
        assert result["count"] == 1
        assert result["mime_type"] == "image/png"
        assert len(result["images"]) == 1

    async def test_generate_image_no_images(self, gemini_client, mock_genai_client):
        mock_resp = MagicMock()
        mock_resp.generated_images = []
        mock_genai_client.aio.models.generate_images.return_value = mock_resp

        result = await gemini_client.generate_image("A cat")
        assert result["count"] == 0
        assert result["images"] == []


class TestGeminiCombinedToolsMixin:
    """Tests for combined tool orchestration."""

    def test_build_combined_tools_grounding_only(self, gemini_client):
        tools = gemini_client._build_combined_tools(
            None, enable_grounding=True, enable_code_execution=False
        )
        assert len(tools) == 1

    def test_build_combined_tools_all(self, gemini_client):
        func_tools = [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        tools = gemini_client._build_combined_tools(
            func_tools, enable_grounding=True, enable_code_execution=True
        )
        assert len(tools) == 3

    def test_build_combined_tools_none(self, gemini_client):
        tools = gemini_client._build_combined_tools(
            None, enable_grounding=False, enable_code_execution=False
        )
        assert tools == []

    def test_extract_combined_tool_response(self, gemini_client, mock_response):
        # Build response with all tool types
        func_call = MagicMock()
        func_call.name = "search"
        func_call.args = {"q": "test"}

        exec_code = MagicMock()
        exec_code.code = "print(1)"

        exec_result = MagicMock()
        exec_result.output = "1"

        part1 = MagicMock()
        part1.function_call = func_call
        part1.executable_code = None
        part1.code_execution_result = None

        part2 = MagicMock()
        part2.function_call = None
        part2.executable_code = exec_code
        part2.code_execution_result = exec_result

        web = MagicMock()
        web.uri = "https://example.com"
        web.title = "Example"
        chunk = MagicMock()
        chunk.web = web

        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk]
        candidate.content.parts = [part1, part2]

        resp = mock_response()
        resp.candidates = [candidate]

        fc, sources, code, result = gemini_client._extract_combined_tool_response(resp)
        assert len(fc) == 1
        assert fc[0]["name"] == "search"
        assert len(sources) == 1
        assert code == "print(1)"
        assert result == "1"

    async def test_generate_with_all_tools(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text="Combined result")
        resp.candidates = []
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_with_all_tools(
            "Do everything",
            enable_grounding=True,
            enable_code_execution=True,
            thinking_level="high",
        )
        assert result["text"] == "Combined result"
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_generation.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_generation.py
git commit -m "test: add gemini_generation.py tests - all generation mixins with mocked genai"
```

---

### Task 10: Test `gemini_async_ops.py` - Cache, Deep Research, Video

**Files:**
- Create: `tests/test_async_ops.py`

**Step 1: Write the tests**

```python
"""Tests for fcp.gemini.gemini_async_ops - cache, deep research, video generation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from google.genai import errors as genai_errors


class TestGeminiCacheMixin:
    """Tests for context caching."""

    async def test_create_context_cache(self, gemini_client, mock_genai_client):
        cache = MagicMock()
        cache.name = "cached-content-123"
        mock_genai_client.aio.caches.create.return_value = cache

        result = await gemini_client.create_context_cache("test-cache", "content", ttl_minutes=30)
        assert result == "cached-content-123"
        mock_genai_client.aio.caches.create.assert_awaited_once()

    async def test_create_context_cache_empty_name(self, gemini_client, mock_genai_client):
        cache = MagicMock()
        cache.name = None
        mock_genai_client.aio.caches.create.return_value = cache

        result = await gemini_client.create_context_cache("test", "content")
        assert result == ""

    async def test_generate_with_cache(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text="Cached response")
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_with_cache("Query", "cache-123")
        assert result == "Cached response"

    async def test_generate_with_cache_fallback(self, gemini_client, mock_genai_client, mock_response):
        """On cache error (400/404/410), falls back to uncached generation."""
        error = genai_errors.ClientError(code=404, message="Cache not found")
        fallback_resp = mock_response(text="Uncached response")

        mock_genai_client.aio.models.generate_content.side_effect = [error, fallback_resp]

        result = await gemini_client.generate_with_cache("Query", "stale-cache")
        assert result == "Uncached response"

    async def test_generate_with_cache_no_fallback(self, gemini_client, mock_genai_client):
        error = genai_errors.ClientError(code=404, message="Cache not found")
        mock_genai_client.aio.models.generate_content.side_effect = error

        with pytest.raises(genai_errors.ClientError):
            await gemini_client.generate_with_cache("Query", "stale-cache", fallback_to_uncached=False)

    async def test_generate_with_cache_non_fallback_error(self, gemini_client, mock_genai_client):
        """Non-recoverable errors (500) are not caught even with fallback=True."""
        error = genai_errors.ClientError(code=500, message="Server error")
        mock_genai_client.aio.models.generate_content.side_effect = error

        with pytest.raises(genai_errors.ClientError):
            await gemini_client.generate_with_cache("Query", "cache-123")


class TestGeminiDeepResearchMixin:
    """Tests for deep research interactions."""

    async def test_deep_research_completed(self, gemini_client, mock_genai_client):
        interaction = MagicMock()
        interaction.id = "interaction-123"
        interaction.status = "completed"
        output = MagicMock()
        output.text = "Research report"
        interaction.outputs = [output]

        mock_genai_client.aio.interactions.create.return_value = interaction
        mock_genai_client.aio.interactions.get.return_value = interaction

        result = await gemini_client.generate_deep_research("Research query", timeout_seconds=5)
        assert result["status"] == "completed"
        assert result["report"] == "Research report"

    async def test_deep_research_failed(self, gemini_client, mock_genai_client):
        created = MagicMock()
        created.id = "interaction-123"
        created.status = "pending"

        failed = MagicMock()
        failed.id = "interaction-123"
        failed.status = "failed"
        failed.error = "Something went wrong"

        mock_genai_client.aio.interactions.create.return_value = created
        mock_genai_client.aio.interactions.get.return_value = failed

        result = await gemini_client.generate_deep_research("Query", timeout_seconds=5)
        assert result["status"] == "failed"

    async def test_deep_research_timeout(self, gemini_client, mock_genai_client):
        interaction = MagicMock()
        interaction.id = "interaction-123"
        interaction.status = "in_progress"

        mock_genai_client.aio.interactions.create.return_value = interaction
        mock_genai_client.aio.interactions.get.return_value = interaction

        with patch("fcp.gemini.gemini_async_ops.asyncio.sleep", new_callable=AsyncMock):
            result = await gemini_client.generate_deep_research("Query", timeout_seconds=0)
        assert result["status"] == "timeout"

    async def test_deep_research_poll_errors(self, gemini_client, mock_genai_client):
        interaction = MagicMock()
        interaction.id = "interaction-123"
        interaction.status = "pending"

        mock_genai_client.aio.interactions.create.return_value = interaction
        mock_genai_client.aio.interactions.get.side_effect = Exception("API error")

        with patch("fcp.gemini.gemini_async_ops.asyncio.sleep", new_callable=AsyncMock):
            result = await gemini_client.generate_deep_research("Query", timeout_seconds=120)
        assert result["status"] == "failed"
        assert "API error" in result["message"]

    async def test_deep_research_completed_no_outputs(self, gemini_client, mock_genai_client):
        interaction = MagicMock()
        interaction.id = "interaction-123"
        interaction.status = "completed"
        interaction.outputs = None

        mock_genai_client.aio.interactions.create.return_value = interaction
        mock_genai_client.aio.interactions.get.return_value = interaction

        result = await gemini_client.generate_deep_research("Query", timeout_seconds=5)
        assert result["status"] == "completed"
        assert result["report"] == ""


class TestGeminiVideoMixin:
    """Tests for video generation."""

    async def test_generate_video_completed(self, gemini_client, mock_genai_client):
        video = MagicMock()
        video.video.video_bytes = b"fake-video-data"
        operation = MagicMock()
        operation.done = True
        operation.response.generated_videos = [video]

        mock_genai_client.aio.models.generate_videos.return_value = operation

        result = await gemini_client.generate_video("A cat running")
        assert result["status"] == "completed"
        assert result["video_bytes"] == b"fake-video-data"

    async def test_generate_video_no_response(self, gemini_client, mock_genai_client):
        operation = MagicMock()
        operation.done = True
        operation.response = None

        mock_genai_client.aio.models.generate_videos.return_value = operation

        result = await gemini_client.generate_video("A cat")
        assert result["status"] == "failed"

    async def test_generate_video_timeout(self, gemini_client, mock_genai_client):
        operation = MagicMock()
        operation.done = False
        operation.name = "operation-123"

        mock_genai_client.aio.models.generate_videos.return_value = operation
        mock_genai_client.aio.operations.get.return_value = operation

        with patch("fcp.gemini.gemini_async_ops.asyncio.sleep", new_callable=AsyncMock):
            result = await gemini_client.generate_video("A cat", timeout_seconds=0)
        assert result["status"] == "timeout"
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_async_ops.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_async_ops.py
git commit -m "test: add gemini_async_ops.py tests - cache, deep research, video generation"
```

---

### Task 11: Test `gemini_live.py` - Live API

**Files:**
- Create: `tests/test_live.py`

**Step 1: Write the tests**

```python
"""Tests for fcp.gemini.gemini_live - Live voice session support."""

from unittest.mock import AsyncMock, MagicMock, patch


class TestGeminiLiveMixin:
    """Tests for live session creation and audio processing."""

    def test_create_live_session_default(self, gemini_client, mock_genai_client):
        mock_genai_client.aio.live.connect.return_value = MagicMock()
        session = gemini_client.create_live_session()
        mock_genai_client.aio.live.connect.assert_called_once()

    def test_create_live_session_with_instruction(self, gemini_client, mock_genai_client):
        mock_genai_client.aio.live.connect.return_value = MagicMock()
        session = gemini_client.create_live_session(system_instruction="Be helpful")
        mock_genai_client.aio.live.connect.assert_called_once()

    def test_create_live_session_no_food_tools(self, gemini_client, mock_genai_client):
        mock_genai_client.aio.live.connect.return_value = MagicMock()
        session = gemini_client.create_live_session(enable_food_tools=False)
        mock_genai_client.aio.live.connect.assert_called_once()

    async def test_process_live_audio(self, gemini_client, mock_genai_client):
        """Test live audio processing with mocked session."""
        # Create mock session context manager
        mock_session = AsyncMock()

        # Mock response with text
        response = MagicMock()
        response.server_content = MagicMock()
        response.server_content.model_turn = MagicMock()
        text_part = MagicMock()
        text_part.text = "I heard you"
        text_part.inline_data = None
        response.server_content.model_turn.parts = [text_part]
        response.tool_call = None

        # Mock receive as async iterator
        async def mock_receive():
            yield response

        mock_session.receive = mock_receive
        mock_session.send = AsyncMock()

        # Create async context manager
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        gemini_client.create_live_session = MagicMock(return_value=mock_ctx)

        result = await gemini_client.process_live_audio(b"audio-data")
        assert result["response_text"] == "I heard you"

    async def test_process_live_audio_with_function_call(self, gemini_client, mock_genai_client):
        """Test live audio processing that triggers a function call."""
        mock_session = AsyncMock()

        # Response with function call
        fc = MagicMock()
        fc.name = "log_meal"
        fc.args = {"dish_name": "pizza"}

        response = MagicMock()
        response.server_content = None
        response.tool_call = MagicMock()
        response.tool_call.function_calls = [fc]

        async def mock_receive():
            yield response

        mock_session.receive = mock_receive
        mock_session.send = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        gemini_client.create_live_session = MagicMock(return_value=mock_ctx)

        result = await gemini_client.process_live_audio(b"audio-data")
        assert len(result["function_calls"]) == 1
        assert result["function_calls"][0]["name"] == "log_meal"

    async def test_process_live_audio_with_audio_response(self, gemini_client, mock_genai_client):
        """Test live audio processing that returns audio."""
        mock_session = AsyncMock()

        response = MagicMock()
        response.server_content = MagicMock()
        response.server_content.model_turn = MagicMock()
        audio_part = MagicMock()
        audio_part.text = None
        audio_part.inline_data = MagicMock()
        audio_part.inline_data.data = b"response-audio"
        response.server_content.model_turn.parts = [audio_part]
        response.tool_call = None

        async def mock_receive():
            yield response

        mock_session.receive = mock_receive
        mock_session.send = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        gemini_client.create_live_session = MagicMock(return_value=mock_ctx)

        result = await gemini_client.process_live_audio(b"audio-data")
        assert result["response_audio"] == b"response-audio"
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_live.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_live.py
git commit -m "test: add gemini_live.py tests - live session and audio processing"
```

---

### Task 12: Test `__init__.py` - Package Entry Point

**Files:**
- Create: `tests/test_init.py`

**Step 1: Write the tests**

```python
"""Tests for fcp.gemini.__init__ - package entry point."""


class TestPackageInit:
    """Tests for package-level exports."""

    def test_version(self):
        from fcp.gemini import __version__

        assert __version__ == "0.1.0"

    def test_gemini_client_importable(self):
        from fcp.gemini import GeminiClient

        assert GeminiClient is not None

    def test_all_exports(self):
        from fcp.gemini import __all__

        assert "GeminiClient" in __all__
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_init.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_init.py
git commit -m "test: add __init__.py tests - package exports"
```

---

### Task 13: Integration Tests with SDK Replay System

**Files:**
- Create: `tests/test_integration.py`
- Create: `tests/replays/` (populated by recording)

**Step 1: Write the integration test framework**

```python
"""Integration tests using google-genai's DebugConfig replay system.

Usage:
  Record: GEMINI_API_KEY=<key> GOOGLE_GENAI_CLIENT_MODE=record pytest tests/test_integration.py -m integration -v
  Replay: GOOGLE_GENAI_CLIENT_MODE=replay pytest tests/test_integration.py -m integration -v

Replay files are stored in tests/replays/ and committed to the repo.
CI runs in replay mode (no API key needed).
"""

import os
from pathlib import Path

import pytest

from google import genai
from google.genai.client import DebugConfig

REPLAYS_DIR = str(Path(__file__).parent / "replays")


def _get_client() -> genai.Client:
    """Create a genai client with debug config for replay support."""
    mode = os.getenv("GOOGLE_GENAI_CLIENT_MODE", "replay")
    api_key = os.getenv("GEMINI_API_KEY", "test-key-for-replay")
    return genai.Client(
        api_key=api_key,
        debug_config=DebugConfig(
            client_mode=mode,
            replays_directory=REPLAYS_DIR,
        ),
    )


@pytest.mark.integration
class TestGenerateContent:
    """Integration tests for basic content generation."""

    async def test_simple_text_generation(self):
        client = _get_client()
        response = await client.aio.models.generate_content(
            model="gemini-3-flash-preview",
            contents="What is 2+2? Reply with just the number.",
        )
        assert response.text is not None
        assert "4" in response.text

    async def test_json_generation(self):
        from google.genai import types

        client = _get_client()
        response = await client.aio.models.generate_content(
            model="gemini-3-flash-preview",
            contents="Return a JSON object with a key 'answer' set to 42.",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        assert response.text is not None
        import json

        data = json.loads(response.text)
        assert data["answer"] == 42


@pytest.mark.integration
class TestThinkingMode:
    """Integration tests for thinking/reasoning mode."""

    async def test_thinking_generation(self):
        from google.genai import types

        client = _get_client()
        response = await client.aio.models.generate_content(
            model="gemini-3-flash-preview",
            contents="What is the square root of 144? Think step by step.",
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=1024,
                ),
            ),
        )
        assert response.text is not None
        assert "12" in response.text


@pytest.mark.integration
class TestGrounding:
    """Integration tests for Google Search grounding."""

    async def test_grounded_search(self):
        from google.genai import types

        client = _get_client()
        response = await client.aio.models.generate_content(
            model="gemini-3-flash-preview",
            contents="What is the current population of Tokyo?",
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        assert response.text is not None
```

**Step 2: Run in replay mode (will skip if no replays exist yet)**

Run: `uv run pytest tests/test_integration.py -m integration -v --no-header`
Expected: Tests SKIP or PASS (depending on whether replays exist)

**Step 3: Record replays (requires API key - manual step)**

```bash
GEMINI_API_KEY=<your-key> GOOGLE_GENAI_CLIENT_MODE=record uv run pytest tests/test_integration.py -m integration -v
```

**Step 4: Commit**

```bash
git add tests/test_integration.py tests/replays/
git commit -m "test: add integration tests with SDK replay system"
```

---

### Task 14: Coverage Verification and Gap Filling

**Step 1: Run full coverage report**

Run: `uv run pytest tests/ -m "not integration" --cov --cov-report=term-missing`
Expected: Identify any uncovered lines

**Step 2: Fill coverage gaps**

Based on the coverage report, add targeted tests for any missed branches. Common gaps:
- Exception handlers in `_fetch_media` (stale import at `gemini_base.py:92`)
- Edge cases in streaming generators
- The `_GeminiProxy.__call__` path

**Step 3: Verify 100% coverage**

Run: `uv run pytest tests/ -m "not integration" --cov --cov-report=term-missing`
Expected: 100% coverage, `fail_under = 100` passes

**Step 4: Commit**

```bash
git add tests/
git commit -m "test: achieve 100% coverage - fill remaining gaps"
```

---

### Task 15: Final Cleanup and CI Readiness

**Step 1: Run full test suite**

```bash
uv run pytest tests/ -m "not integration" -v --cov --cov-report=term-missing
```
Expected: All PASS, 100% coverage

**Step 2: Run linting on test files**

```bash
uv run ruff check tests/ --fix
uv run ruff format tests/
```

**Step 3: Update Makefile with all targets**

Verify `make test`, `make test-integration`, `make coverage` all work.

**Step 4: Final commit**

```bash
git add .
git commit -m "test: finalize test suite - lint, format, CI-ready"
```

---

## Recommendations

### Mocking Strategy Decision Tree

```
Is the code under test a pure function?
├── YES → Test directly, no mocking needed
│   (utils.py, security.py, config.py, constants.py)
└── NO → Does it call genai.Client methods?
    ├── YES → Use unittest.mock.AsyncMock on client.aio.*
    │   (generation.py, async_ops.py, live.py)
    └── NO → Does it make HTTP calls?
        ├── YES → Use respx or mock httpx.AsyncClient
        │   (base.py _fetch_media)
        └── NO → Likely testable with minimal setup
            (gemini.py singleton/proxy)
```

### SDK Replay System Usage

1. **Recording** (developer machine with API key):
   ```bash
   GEMINI_API_KEY=<key> GOOGLE_GENAI_CLIENT_MODE=record pytest tests/test_integration.py -m integration
   ```

2. **Replaying** (CI, no API key):
   ```bash
   GOOGLE_GENAI_CLIENT_MODE=replay pytest tests/test_integration.py -m integration
   ```

3. **Re-recording** (when API surface changes):
   ```bash
   rm -rf tests/replays/*
   GEMINI_API_KEY=<key> GOOGLE_GENAI_CLIENT_MODE=record pytest tests/test_integration.py -m integration
   git add tests/replays/
   ```

### Why Not Typer?

Typer is a CLI framework for building command-line interfaces. It has nothing to do with test separation or test classification. The right tool for separating unit vs. integration tests is **pytest markers** (`@pytest.mark.integration`) combined with `-m "not integration"` filtering. This is the standard pytest pattern used across the Python ecosystem.
