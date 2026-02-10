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
        GeminiBase._get_http_client()
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
        parts = await base._prepare_parts("describe", image_bytes=b"fake-image", image_mime_type="image/png")
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


class TestFetchMedia:
    """Tests for _fetch_media() - stale import at line 92 causes ModuleNotFoundError."""

    def setup_method(self):
        GeminiBase.reset_http_client()

    def teardown_method(self):
        GeminiBase.reset_http_client()

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key")
    @patch("fcp.gemini.gemini_base.genai.Client")
    async def test_fetch_media_stale_import_raises(self, mock_genai_client):
        """Line 92 imports fcp.services.gemini which doesn't exist -> ModuleNotFoundError."""
        base = GeminiBase()
        with pytest.raises(ModuleNotFoundError):
            await base._fetch_media("https://example.com/img.jpg", expected_type="image")

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key")
    @patch("fcp.gemini.gemini_base.genai.Client")
    async def test_fetch_media_with_mocked_import_success(self, mock_genai_client):
        """Mock the importlib.import_module call to simulate correct behavior."""
        base = GeminiBase()

        mock_security_module = MagicMock()
        mock_security_module.validate_image_url.return_value = "https://example.com/img.jpg"

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "image/jpeg", "content-length": "1024"}
        mock_response.content = b"fake-image-data"
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)

        with patch("importlib.import_module", return_value=mock_security_module):
            GeminiBase._http_client = mock_http
            data, mime = await base._fetch_media("https://example.com/img.jpg", expected_type="image")
            assert data == b"fake-image-data"
            assert mime == "image/jpeg"
            mock_security_module.validate_image_url.assert_called_once_with("https://example.com/img.jpg")

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key")
    @patch("fcp.gemini.gemini_base.genai.Client")
    async def test_fetch_media_invalid_url_raises_valueerror(self, mock_genai_client):
        """When validate_image_url raises ImageURLError, _fetch_media wraps it in ValueError."""
        from fcp.gemini.security import ImageURLError

        base = GeminiBase()

        mock_security_module = MagicMock()
        mock_security_module.validate_image_url.side_effect = ImageURLError("bad url")

        # ImageURLError must be importable from the patched module context too,
        # but the except clause imports it from fcp.gemini.security directly (top-level).
        with patch("importlib.import_module", return_value=mock_security_module):
            with pytest.raises(ValueError, match="Invalid media URL"):
                await base._fetch_media("https://evil.com/bad.jpg")

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key")
    @patch("fcp.gemini.gemini_base.genai.Client")
    async def test_fetch_media_invalid_content_type(self, mock_genai_client):
        """Should raise ValueError for non-image content type when expected_type='image'."""
        base = GeminiBase()

        mock_security_module = MagicMock()
        mock_security_module.validate_image_url.return_value = "https://example.com/file.txt"

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/plain", "content-length": "100"}
        mock_response.content = b"not an image"
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)

        with patch("importlib.import_module", return_value=mock_security_module):
            GeminiBase._http_client = mock_http
            with pytest.raises(ValueError, match="Invalid content type"):
                await base._fetch_media("https://example.com/file.txt", expected_type="image")

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key")
    @patch("fcp.gemini.gemini_base.genai.Client")
    async def test_fetch_media_too_large(self, mock_genai_client):
        """Should raise ValueError when content-length exceeds MAX_IMAGE_SIZE."""
        base = GeminiBase()

        mock_security_module = MagicMock()
        mock_security_module.validate_image_url.return_value = "https://example.com/huge.jpg"

        mock_response = MagicMock()
        mock_response.headers = {
            "content-type": "image/jpeg",
            "content-length": str(999_999_999),
        }
        mock_response.content = b"huge"
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)

        with patch("importlib.import_module", return_value=mock_security_module):
            GeminiBase._http_client = mock_http
            with pytest.raises(ValueError, match="Media too large"):
                await base._fetch_media("https://example.com/huge.jpg", expected_type="image")

    @patch("fcp.gemini.gemini.GEMINI_API_KEY", "test-key")
    @patch("fcp.gemini.gemini_base.genai.Client")
    async def test_fetch_media_non_image_expected_type_skips_content_type_check(self, mock_genai_client):
        """When expected_type != 'image', content-type validation is skipped."""
        base = GeminiBase()

        mock_security_module = MagicMock()
        mock_security_module.validate_image_url.return_value = "https://example.com/vid.mp4"

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "video/mp4", "content-length": "1024"}
        mock_response.content = b"video-data"
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)

        with patch("importlib.import_module", return_value=mock_security_module):
            GeminiBase._http_client = mock_http
            data, mime = await base._fetch_media("https://example.com/vid.mp4", expected_type="media")
            assert data == b"video-data"
            assert mime == "video/mp4"


class TestStreamingMixin:
    """Tests for GeminiStreamingMixin."""

    def test_async_text_stream_type(self):
        """Verify the type alias exists."""
        from collections.abc import AsyncIterator

        assert GeminiStreamingMixin.AsyncTextStream == AsyncIterator[str]
