"""Tests for fcp.gemini.gemini_async_ops - cache, deep research, video generation."""

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
        error = genai_errors.ClientError(code=404, response_json={"error": {"message": "Cache not found"}})
        fallback_resp = mock_response(text="Uncached response")

        mock_genai_client.aio.models.generate_content.side_effect = [error, fallback_resp]

        result = await gemini_client.generate_with_cache("Query", "stale-cache")
        assert result == "Uncached response"

    async def test_generate_with_cache_no_fallback(self, gemini_client, mock_genai_client):
        error = genai_errors.ClientError(code=404, response_json={"error": {"message": "Cache not found"}})
        mock_genai_client.aio.models.generate_content.side_effect = error

        with pytest.raises(genai_errors.ClientError):
            await gemini_client.generate_with_cache("Query", "stale-cache", fallback_to_uncached=False)

    async def test_generate_with_cache_non_fallback_error(self, gemini_client, mock_genai_client):
        """Non-recoverable errors (500) are not caught even with fallback=True."""
        error = genai_errors.ClientError(code=500, response_json={"error": {"message": "Server error"}})
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

    async def test_deep_research_polls_then_completes(self, gemini_client, mock_genai_client):
        """Status transitions from in_progress to completed after polling."""
        in_progress = MagicMock()
        in_progress.id = "interaction-123"
        in_progress.status = "in_progress"

        completed = MagicMock()
        completed.id = "interaction-123"
        completed.status = "completed"
        output = MagicMock()
        output.text = "Final report"
        completed.outputs = [output]

        mock_genai_client.aio.interactions.create.return_value = in_progress
        mock_genai_client.aio.interactions.get.side_effect = [in_progress, completed]

        with patch("fcp.gemini.gemini_async_ops.asyncio.sleep", new_callable=AsyncMock):
            result = await gemini_client.generate_deep_research("Query", timeout_seconds=120)
        assert result["status"] == "completed"
        assert result["report"] == "Final report"

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

    async def test_generate_video_polls_then_completes(self, gemini_client, mock_genai_client):
        """Operation is not done on first check, then completes after polling."""
        # First iteration: not done
        pending_op = MagicMock()
        pending_op.done = False

        # Second iteration: done with video
        video = MagicMock()
        video.video.video_bytes = b"polled-video-data"
        done_op = MagicMock()
        done_op.done = True
        done_op.response.generated_videos = [video]

        mock_genai_client.aio.models.generate_videos.return_value = pending_op
        mock_genai_client.aio.operations.get.return_value = done_op

        with patch("fcp.gemini.gemini_async_ops.asyncio.sleep", new_callable=AsyncMock):
            result = await gemini_client.generate_video("A cat running", timeout_seconds=120)
        assert result["status"] == "completed"
        assert result["video_bytes"] == b"polled-video-data"

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
