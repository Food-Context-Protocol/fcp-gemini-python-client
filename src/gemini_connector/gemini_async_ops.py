"""Async and long-running Gemini operations (cache, deep research, video)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from google.genai import errors as genai_errors
from google.genai import types

from fcp.config import Config
from fcp.services.gemini_constants import MODEL_NAME
from fcp.services.gemini_helpers import _log_token_usage, gemini_retry

logger = logging.getLogger(__name__)


class GeminiCacheMixin:
    """Context caching support."""

    @gemini_retry
    async def create_context_cache(
        self,
        name: str,
        content: str,
        ttl_minutes: int = 60,
    ) -> str:
        client = self._require_client()

        cache = await client.aio.caches.create(
            model=MODEL_NAME,
            config=types.CreateCachedContentConfig(
                display_name=name,
                contents=[types.Content(parts=[types.Part(text=content)], role="user")],
                ttl=f"{ttl_minutes * 60}s",
            ),
        )
        return cache.name or ""

    @gemini_retry
    async def generate_with_cache(
        self,
        prompt: str,
        cache_name: str,
        fallback_to_uncached: bool = True,
    ) -> str:
        client = self._require_client()

        config = types.GenerateContentConfig(
            cached_content=cache_name,
        )

        try:
            response = await client.aio.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )
            _log_token_usage(response, "generate_with_cache")
            return response.text or ""
        except genai_errors.ClientError as e:
            if e.code in (400, 404, 410) and fallback_to_uncached:
                logger.warning(
                    "Cache error (code=%s): %s. Falling back to uncached generation.",
                    e.code,
                    e.message or "Unknown error",
                )
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                )
                _log_token_usage(response, "generate_with_cache_fallback")
                return response.text or ""
            raise


class GeminiDeepResearchMixin:
    """Deep research interactions."""

    async def generate_deep_research(
        self,
        query: str,
        timeout_seconds: int = Config.DEEP_RESEARCH_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        client = self._require_client()

        logger.debug(
            "[generate_deep_research] START query=%r timeout=%ds",
            f"{query[:100]}..." if len(query) > 100 else query,
            timeout_seconds,
        )
        interaction = await client.aio.interactions.create(
            input=query,
            agent=Config.DEEP_RESEARCH_AGENT,
            background=True,
        )
        logger.debug("[generate_deep_research] interaction_id=%s", interaction.id)

        start = time.monotonic()
        deadline = start + timeout_seconds
        consecutive_errors = 0
        max_consecutive_errors = 3
        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            try:
                interaction = await client.aio.interactions.get(interaction.id)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                logger.warning(
                    "Deep research poll error [interaction_id=%s] (%d/%d): %s",
                    interaction.id,
                    consecutive_errors,
                    max_consecutive_errors,
                    str(e)[:100],
                )
                if consecutive_errors >= max_consecutive_errors:
                    return {
                        "interaction_id": interaction.id,
                        "status": "failed",
                        "message": f"API error after {consecutive_errors} retries: {str(e)[:200]}",
                    }
                await asyncio.sleep(10)
                continue

            if interaction.status == "completed":
                report = ""
                if interaction.outputs:
                    report = getattr(interaction.outputs[-1], "text", "")
                logger.info(
                    "Deep research completed [interaction_id=%s]",
                    interaction.id,
                )
                return {
                    "report": report,
                    "interaction_id": interaction.id,
                    "status": "completed",
                }
            if interaction.status == "failed":
                error_msg = getattr(interaction, "error", "Unknown error")
                logger.error(
                    "Deep research failed [interaction_id=%s]: %s",
                    interaction.id,
                    error_msg,
                )
                return {
                    "interaction_id": interaction.id,
                    "status": "failed",
                    "message": str(error_msg),
                }

            await asyncio.sleep(10)

        logger.warning(
            "Deep research timeout [interaction_id=%s] after %ds",
            interaction.id,
            timeout_seconds,
        )
        return {
            "interaction_id": interaction.id,
            "status": "timeout",
            "message": f"Research still in progress after {timeout_seconds}s. "
            "Use interaction_id to check status later.",
        }


class GeminiVideoMixin:
    """Video generation using Veo."""

    @gemini_retry
    async def generate_video(
        self,
        prompt: str,
        duration_seconds: int = 8,
        aspect_ratio: str = "16:9",
        timeout_seconds: int = Config.VIDEO_GENERATION_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        client = self._require_client()

        logger.debug(
            "[generate_video] START prompt=%r duration=%ds aspect_ratio=%s timeout=%ds",
            f"{prompt[:100]}..." if len(prompt) > 100 else prompt,
            duration_seconds,
            aspect_ratio,
            timeout_seconds,
        )
        operation = await client.aio.models.generate_videos(
            model=Config.VEO_MODEL_NAME,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                duration_seconds=duration_seconds,
            ),
        )
        logger.debug("[generate_video] operation started")

        start = time.monotonic()
        deadline = start + timeout_seconds
        while True:
            now = time.monotonic()
            if now >= deadline:
                break

            if operation.done:
                if (
                    hasattr(operation, "response")
                    and operation.response
                    and (videos := operation.response.generated_videos)
                ):
                    video = videos[0]
                    video_bytes = None
                    if hasattr(video, "video") and hasattr(video.video, "video_bytes"):
                        video_bytes = video.video.video_bytes
                    logger.info("Video generation completed")
                    return {
                        "status": "completed",
                        "video_bytes": video_bytes,
                        "duration": duration_seconds,
                    }
                logger.error("Video generation failed - no response")
                return {
                    "status": "failed",
                    "message": "Video generation completed but no video returned",
                }

            await asyncio.sleep(10)
            operation = await client.aio.operations.get(operation)

        logger.warning(
            "Video generation timeout after %ds, operation: %s",
            timeout_seconds,
            getattr(operation, "name", "unknown"),
        )
        return {
            "status": "timeout",
            "operation_name": getattr(operation, "name", None),
            "message": f"Video still generating after {timeout_seconds}s. Use operation_name to check status.",
        }
