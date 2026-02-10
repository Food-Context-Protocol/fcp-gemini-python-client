"""Tests for fcp.gemini.security - SSRF prevention and URL validation."""

import importlib
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

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

    def test_urlparse_exception(self):
        """Cover lines 123-124: urlparse raising an exception."""
        with patch("fcp.gemini.security.urlparse", side_effect=ValueError("bad url")):
            with pytest.raises(ImageURLError, match="Invalid URL format"):
                validate_image_url("https://example.com")


class TestValidateImageUrlProduction:
    """Tests for validate_image_url() in production mode."""

    def test_http_blocked_in_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        importlib.reload(security_module)
        try:
            with pytest.raises(security_module.ImageURLError, match="HTTP URLs are not allowed in production"):
                security_module.validate_image_url("http://storage.googleapis.com/img.png")
        finally:
            monkeypatch.delenv("ENVIRONMENT", raising=False)
            importlib.reload(security_module)

    def test_localhost_not_allowed_in_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        importlib.reload(security_module)
        try:
            with pytest.raises(security_module.ImageURLError, match="not in the allowed list"):
                security_module.validate_image_url("https://localhost/img.png")
        finally:
            monkeypatch.delenv("ENVIRONMENT", raising=False)
            importlib.reload(security_module)

    def test_production_ports_restricted(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        importlib.reload(security_module)
        try:
            with pytest.raises(security_module.ImageURLError, match="Non-standard port"):
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

    async def test_head_returns_200_valid_content_type(self):
        """HEAD returns 200 with valid image content type -> True (line 204)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "image/jpeg"}

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(return_value=mock_response)

        mock_client_cm = AsyncMock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("fcp.gemini.security.httpx.AsyncClient", return_value=mock_client_cm):
            result = await verify_url_reachability("https://storage.googleapis.com/img.jpg")
        assert result is True

    async def test_head_returns_200_invalid_content_type(self):
        """HEAD returns 200 with non-image content type -> False (line 204)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(return_value=mock_response)

        mock_client_cm = AsyncMock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("fcp.gemini.security.httpx.AsyncClient", return_value=mock_client_cm):
            result = await verify_url_reachability("https://storage.googleapis.com/page.html")
        assert result is False

    async def test_head_returns_500_returns_false(self):
        """HEAD returns 500 (not in fallback list) -> False (line 202)."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(return_value=mock_response)

        mock_client_cm = AsyncMock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("fcp.gemini.security.httpx.AsyncClient", return_value=mock_client_cm):
            result = await verify_url_reachability("https://storage.googleapis.com/img.jpg")
        assert result is False

    async def test_head_returns_405_get_fallback_succeeds(self):
        """HEAD returns 405, GET fallback returns 200 with valid type -> True (lines 195-199)."""
        mock_head_response = MagicMock()
        mock_head_response.status_code = 405

        mock_stream_response = AsyncMock()
        mock_stream_response.status_code = 200
        mock_stream_response.headers = {"Content-Type": "image/png"}

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            yield mock_stream_response

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(return_value=mock_head_response)
        mock_client.stream = mock_stream

        mock_client_cm = AsyncMock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("fcp.gemini.security.httpx.AsyncClient", return_value=mock_client_cm):
            result = await verify_url_reachability("https://storage.googleapis.com/img.png")
        assert result is True

    async def test_head_returns_403_get_fallback_non_200(self):
        """HEAD returns 403, GET fallback returns non-200 -> False (lines 195-202)."""
        mock_head_response = MagicMock()
        mock_head_response.status_code = 403

        mock_stream_response = AsyncMock()
        mock_stream_response.status_code = 403
        mock_stream_response.headers = {}

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            yield mock_stream_response

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(return_value=mock_head_response)
        mock_client.stream = mock_stream

        mock_client_cm = AsyncMock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("fcp.gemini.security.httpx.AsyncClient", return_value=mock_client_cm):
            result = await verify_url_reachability("https://storage.googleapis.com/img.png")
        assert result is False

    async def test_head_returns_404_get_fallback_raises_exception(self):
        """HEAD returns 404, GET stream raises exception -> False (lines 200-201)."""
        mock_head_response = MagicMock()
        mock_head_response.status_code = 404

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            raise ConnectionError("stream failed")
            yield  # pragma: no cover

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(return_value=mock_head_response)
        mock_client.stream = mock_stream

        mock_client_cm = AsyncMock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("fcp.gemini.security.httpx.AsyncClient", return_value=mock_client_cm):
            result = await verify_url_reachability("https://storage.googleapis.com/img.png")
        assert result is False
