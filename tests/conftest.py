"""Shared test fixtures for fcp-gemini-python-client."""

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
