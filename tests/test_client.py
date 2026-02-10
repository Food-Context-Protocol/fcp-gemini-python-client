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
