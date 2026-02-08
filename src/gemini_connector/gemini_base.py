"""Base utilities for Gemini service."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import httpx
from google import genai
from google.genai import types

from fcp.config import Config
from fcp.security import ImageURLError
from fcp.security.url_validator import validate_content_type
from fcp.services.gemini_constants import MAX_IMAGE_SIZE
from fcp.services.gemini_helpers import gemini_retry

logger = logging.getLogger(__name__)


class GeminiBase:
    """Base Gemini client utilities shared across feature modules."""

    _http_client: httpx.AsyncClient | None = None

    def __init__(self):
        import importlib

        gemini_module = importlib.import_module("fcp.services.gemini")
        api_key = gemini_module.GEMINI_API_KEY
        self.client = genai.Client(api_key=api_key) if api_key else None

    def _require_client(self) -> genai.Client:
        if not self.client:
            raise RuntimeError("GEMINI_API_KEY not configured")
        return self.client

    @classmethod
    def _get_http_client(cls) -> httpx.AsyncClient:
        """Get or create the shared HTTP client with connection pooling."""
        if cls._http_client is None:
            cls._http_client = httpx.AsyncClient(
                timeout=Config.HTTP_TIMEOUT_SECONDS,
                limits=httpx.Limits(
                    max_connections=Config.HTTP_MAX_CONNECTIONS,
                    max_keepalive_connections=Config.HTTP_MAX_KEEPALIVE_CONNECTIONS,
                ),
                headers={"User-Agent": f"{Config.SERVICE_NAME}/{Config.API_VERSION}"},
            )
        return cls._http_client

    @classmethod
    async def close_http_client(cls) -> None:
        """Close the shared HTTP client. Call on shutdown."""
        if cls._http_client is not None:
            await cls._http_client.aclose()
            cls._http_client = None

    @classmethod
    def reset_http_client(cls) -> None:
        """Reset the HTTP client for testing purposes."""
        cls._http_client = None

    async def _prepare_parts(
        self,
        prompt: str,
        image_url: str | None = None,
        media_url: str | None = None,
        image_bytes: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> list[types.Part]:
        """Prepare content parts for Gemini API."""
        parts = [types.Part(text=prompt)]

        if image_bytes and image_mime_type:
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type))
        elif image_url:
            image_data, mime_type = await self._fetch_media(image_url, expected_type="image")
            parts.append(types.Part.from_bytes(data=image_data, mime_type=mime_type))

        if media_url:
            media_data, mime_type = await self._fetch_media(media_url, expected_type="media")
            parts.append(types.Part.from_bytes(data=media_data, mime_type=mime_type))

        return parts

    @gemini_retry
    async def _fetch_media(self, url: str, expected_type: str = "image") -> tuple[bytes, str]:
        """Fetch media data from URL with security checks."""
        try:
            import importlib

            gemini_module = importlib.import_module("fcp.services.gemini")
            validated_url = gemini_module.validate_image_url(url)
        except ImageURLError as e:
            raise ValueError(f"Invalid media URL: {e}") from e

        http_client = self._get_http_client()
        response = await http_client.get(
            validated_url,
            follow_redirects=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if expected_type == "image" and not validate_content_type(content_type):
            raise ValueError(f"Invalid content type: {content_type}. Expected an image.")

        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > MAX_IMAGE_SIZE:
            raise ValueError(f"Media too large: {content_length} bytes.")

        return response.content, content_type.split(";")[0].strip()


class GeminiStreamingMixin:
    """Marker for streaming return types."""

    AsyncTextStream = AsyncIterator[str]
