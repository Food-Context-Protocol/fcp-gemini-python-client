"""Core generation methods for Gemini service."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from google.genai import types

from fcp.services.gemini_constants import MODEL_NAME
from fcp.services.gemini_helpers import _log_token_usage, _parse_json_response, gemini_retry

logger = logging.getLogger(__name__)


class GeminiGenerationMixin:
    """Basic text and JSON generation methods."""

    @gemini_retry
    async def generate_content(self, prompt: str, image_url: str | None = None, media_url: str | None = None) -> str:
        client = self._require_client()

        logger.debug(
            "[generate_content] START prompt=%r image_url=%s media_url=%s",
            f"{prompt[:100]}..." if len(prompt) > 100 else prompt,
            image_url,
            media_url,
        )
        parts = await self._prepare_parts(prompt, image_url, media_url)
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=parts,
        )
        _log_token_usage(response, "generate_content")
        result = response.text or ""
        logger.debug("[generate_content] END response_len=%d", len(result))
        return result

    @gemini_retry
    async def generate_content_stream(self, prompt: str, image_url: str | None = None) -> AsyncIterator[str]:
        client = self._require_client()

        logger.debug(
            "[generate_content_stream] START prompt=%r",
            f"{prompt[:100]}..." if len(prompt) > 100 else prompt,
        )
        parts = await self._prepare_parts(prompt, image_url)
        stream = await client.aio.models.generate_content_stream(
            model=MODEL_NAME,
            contents=parts,
        )
        chunk_count = 0
        async for chunk in stream:
            if chunk.text:
                chunk_count += 1
                yield chunk.text
        logger.debug("[generate_content_stream] END chunks=%d", chunk_count)

    @gemini_retry
    async def generate_json(
        self,
        prompt: str,
        image_url: str | None = None,
        media_url: str | None = None,
        image_bytes: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> dict[str, Any]:
        client = self._require_client()

        logger.debug(
            "[generate_json] START prompt=%r image_url=%s",
            f"{prompt[:100]}..." if len(prompt) > 100 else prompt,
            image_url,
        )
        parts = await self._prepare_parts(prompt, image_url, media_url, image_bytes, image_mime_type)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
        )
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=parts,
            config=config,
        )
        _log_token_usage(response, "generate_json")

        response_text = response.text or ""
        text = response_text.strip()
        result = _parse_json_response(text)
        logger.debug("[generate_json] END keys=%s", list(result.keys()) if isinstance(result, dict) else "list")
        return {"items": result} if isinstance(result, list) else result

    @gemini_retry
    async def generate_json_stream(self, prompt: str, image_url: str | None = None) -> AsyncIterator[str]:
        client = self._require_client()

        parts = await self._prepare_parts(prompt, image_url)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
        )
        stream = cast(
            AsyncIterator[types.GenerateContentResponse],
            client.aio.models.generate_content_stream(
                model=MODEL_NAME,
                contents=parts,
                config=config,
            ),
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text


class GeminiToolingMixin:
    """Function calling support."""

    @gemini_retry
    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        image_url: str | None = None,
        media_url: str | None = None,
    ) -> dict[str, Any]:
        client = self._require_client()

        logger.debug(
            "[generate_with_tools] START prompt=%r tools=%s",
            f"{prompt[:100]}..." if len(prompt) > 100 else prompt,
            [t.get("name") for t in tools],
        )
        parts = await self._prepare_parts(prompt, image_url, media_url)

        function_declarations = [
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=types.Schema.model_validate(tool.get("parameters", {})),
            )
            for tool in tools
        ]

        config = types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=function_declarations)],
        )

        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=parts,
            config=config,
        )
        _log_token_usage(response, "generate_with_tools")

        function_calls: list[dict[str, Any]] = []

        if candidates := getattr(response, "candidates", []):
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", []) if content else []
                if parts:
                    for part in parts:
                        if function_call := getattr(part, "function_call", None):
                            args = getattr(function_call, "args", {})
                            function_calls.append(
                                {
                                    "name": getattr(function_call, "name", ""),
                                    "args": dict(args) if args else {},
                                }
                            )

        logger.debug(
            "[generate_with_tools] END function_calls=%d",
            len(function_calls),
        )
        return {"text": response.text, "function_calls": function_calls}


class GeminiGroundingMixin:
    """Google Search grounding support."""

    @gemini_retry
    async def generate_with_grounding(self, prompt: str) -> dict[str, Any]:
        client = self._require_client()

        logger.debug(
            "[generate_with_grounding] START prompt=%r",
            f"{prompt[:100]}..." if len(prompt) > 100 else prompt,
        )
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )

        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config,
        )
        _log_token_usage(response, "generate_with_grounding")
        sources = _extract_grounding_sources(response)
        logger.debug(
            "[generate_with_grounding] END response_len=%d sources=%d",
            len(response.text or ""),
            len(sources),
        )
        return {"text": response.text, "sources": sources}

    @gemini_retry
    async def generate_json_with_grounding(self, prompt: str) -> dict[str, Any]:
        client = self._require_client()

        logger.debug(
            "[generate_json_with_grounding] START prompt=%r",
            f"{prompt[:100]}..." if len(prompt) > 100 else prompt,
        )
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            response_mime_type="application/json",
        )

        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config,
        )
        _log_token_usage(response, "generate_json_with_grounding")
        sources = _extract_grounding_sources(response)
        if response.text is None:
            raise ValueError("Gemini returned empty response")
        data = _parse_json_response(response.text)
        logger.debug(
            "[generate_json_with_grounding] END keys=%s sources=%d",
            list(data.keys()) if isinstance(data, dict) else "list",
            len(sources),
        )
        return {"data": data, "sources": sources}


class GeminiThinkingMixin:
    """Thinking and large-context operations."""

    @gemini_retry
    async def generate_with_thinking(
        self,
        prompt: str,
        thinking_level: str = "high",
        image_url: str | None = None,
        media_url: str | None = None,
    ) -> str:
        client = self._require_client()

        parts = await self._prepare_parts(prompt, image_url, media_url)
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=_get_thinking_budget(thinking_level),
            ),
        )

        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=parts,
            config=config,
        )
        _log_token_usage(response, "generate_with_thinking")

        return response.text or "" or ""

    @gemini_retry
    async def generate_json_with_thinking(
        self,
        prompt: str,
        thinking_level: str = "high",
        image_url: str | None = None,
        media_url: str | None = None,
        include_thinking_output: bool = False,
    ) -> dict[str, Any] | list[Any] | GeminiThinkingResult:
        client = self._require_client()

        parts = await self._prepare_parts(prompt, image_url, media_url)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=_get_thinking_budget(thinking_level),
            ),
        )

        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=parts,
            config=config,
        )
        _log_token_usage(response, "generate_json_with_thinking")

        response_text = response.text or ""
        analysis = _parse_json_response(response_text.strip())

        if include_thinking_output:
            thinking = _extract_thinking_content(response)
            return {"analysis": analysis, "thinking": thinking}

        return analysis

    @gemini_retry
    async def generate_with_large_context(
        self,
        prompt: str,
        thinking_level: str = "high",
    ) -> str:
        client = self._require_client()

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=_get_thinking_budget(thinking_level),
            ),
        )

        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config,
        )
        _log_token_usage(response, "generate_with_large_context")

        return response.text or "" or ""

    @gemini_retry
    async def generate_json_with_large_context(
        self,
        prompt: str,
        thinking_level: str = "high",
        image_url: str | None = None,
        media_url: str | None = None,
    ) -> dict[str, Any]:
        client = self._require_client()

        parts = await self._prepare_parts(prompt, image_url, media_url)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=_get_thinking_budget(thinking_level),
            ),
        )

        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=parts,
            config=config,
        )
        _log_token_usage(response, "generate_json_with_large_context")

        if response.text is None:
            raise ValueError("Gemini returned empty response")
        parsed = _parse_json_response(response.text.strip())
        return {"items": parsed} if isinstance(parsed, list) else parsed


class GeminiCodeExecutionMixin:
    """Code execution features."""

    @gemini_retry
    async def generate_with_code_execution(self, prompt: str) -> dict[str, Any]:
        client = self._require_client()

        logger.debug(
            "[generate_with_code_execution] START prompt=%r",
            f"{prompt[:100]}..." if len(prompt) > 100 else prompt,
        )
        config = types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())],
        )

        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config,
        )
        _log_token_usage(response, "generate_with_code_execution")

        result = {"text": response.text, "code": None, "execution_result": None}

        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "executable_code") and part.executable_code:
                            result["code"] = part.executable_code.code
                        if hasattr(part, "code_execution_result") and part.code_execution_result:
                            result["execution_result"] = part.code_execution_result.output

        logger.debug(
            "[generate_with_code_execution] END has_code=%s has_result=%s",
            result["code"] is not None,
            result["execution_result"] is not None,
        )
        return result

    @gemini_retry
    async def generate_json_with_agentic_vision(
        self,
        prompt: str,
        image_url: str,
    ) -> dict[str, Any]:
        client = self._require_client()

        parts = await self._prepare_parts(prompt, image_url=image_url)
        config = types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())],
            response_mime_type="application/json",
        )

        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=parts,
            config=config,
        )
        _log_token_usage(response, "generate_json_with_agentic_vision")

        code = None
        execution_result = None
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if code is None and hasattr(part, "executable_code") and part.executable_code:
                            code = part.executable_code.code
                        if (
                            execution_result is None
                            and hasattr(part, "code_execution_result")
                            and part.code_execution_result
                        ):
                            execution_result = part.code_execution_result.output
                if code is not None and execution_result is not None:
                    break

        if response.text is None:
            raise ValueError("Gemini returned empty response")
        analysis = _parse_json_response(response.text.strip())

        return {
            "analysis": analysis,
            "code": code,
            "execution_result": execution_result,
        }


class GeminiMediaMixin:
    """Media resolution and URL context features."""

    @gemini_retry
    async def generate_json_with_media_resolution(
        self,
        prompt: str,
        image_url: str,
        resolution: str = "high",
    ) -> dict[str, Any]:
        client = self._require_client()

        logger.debug(
            "[generate_json_with_media_resolution] START resolution=%s image_url=%s",
            resolution,
            image_url,
        )
        resolution_map = {
            "low": "MEDIA_RESOLUTION_LOW",
            "medium": "MEDIA_RESOLUTION_MEDIUM",
            "high": "MEDIA_RESOLUTION_HIGH",
        }
        media_res = resolution_map.get(resolution.lower(), "MEDIA_RESOLUTION_HIGH")

        parts = await self._prepare_parts(prompt, image_url)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            media_resolution=media_res,
        )

        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=parts,
            config=config,
        )
        _log_token_usage(response, f"generate_json_with_media_resolution({resolution})")

        response_text = response.text or ""
        text = response_text.strip()
        result = _parse_json_response(text)
        logger.debug(
            "[generate_json_with_media_resolution] END keys=%s",
            list(result.keys()) if isinstance(result, dict) else "list",
        )
        return {"items": result} if isinstance(result, list) else result

    @gemini_retry
    async def generate_json_with_url_context(
        self,
        prompt: str,
        urls: list[str],
    ) -> dict[str, Any]:
        client = self._require_client()

        parts: list[types.Part] = [types.Part(text=prompt)]
        parts.extend(types.Part(file_data=types.FileData(file_uri=url)) for url in urls)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
        )

        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=parts,
            config=config,
        )
        _log_token_usage(response, "generate_json_with_url_context")

        response_text = response.text or ""
        text = response_text.strip()
        data = _parse_json_response(text)

        return {
            "data": data,
            "sources": urls,
        }


class GeminiImageMixin:
    """Image generation using Imagen."""

    @gemini_retry
    async def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        number_of_images: int = 1,
    ) -> dict[str, Any]:
        client = self._require_client()

        import base64

        logger.debug(
            "[generate_image] START prompt=%r aspect_ratio=%s count=%d",
            f"{prompt[:100]}..." if len(prompt) > 100 else prompt,
            aspect_ratio,
            number_of_images,
        )
        response = await client.aio.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=types.GenerateImagesConfig(
                aspect_ratio=aspect_ratio,
                number_of_images=number_of_images,
            ),
        )

        images = []
        if response.generated_images:
            images.extend(
                base64.b64encode(img.image.image_bytes).decode("utf-8")
                for img in response.generated_images
                if hasattr(img, "image") and hasattr(img.image, "image_bytes")
            )
        logger.info("Generated %d images with Imagen 3", len(images))
        logger.debug("[generate_image] END images=%d", len(images))
        return {
            "images": images,
            "mime_type": "image/png",
            "count": len(images),
        }


class GeminiCombinedToolsMixin:
    """Combined tool orchestration for agents."""

    def _build_combined_tools(
        self,
        function_tools: list[dict] | None,
        *,
        enable_grounding: bool,
        enable_code_execution: bool,
    ) -> list[types.Tool]:
        tools: list[types.Tool] = []

        if enable_grounding:
            tools.append(types.Tool(google_search=types.GoogleSearch()))

        if function_tools:
            function_declarations = [
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=types.Schema.model_validate(tool.get("parameters", {})),
                )
                for tool in function_tools
            ]
            tools.append(types.Tool(function_declarations=function_declarations))

        if enable_code_execution:
            tools.append(types.Tool(code_execution=types.ToolCodeExecution()))

        return tools

    def _extract_combined_tool_response(
        self,
        response: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]], str | None, str | None]:
        function_calls: list[dict[str, Any]] = []
        sources: list[dict[str, str]] = []
        code: str | None = None
        execution_result: str | None = None

        for candidate in getattr(response, "candidates", []) or []:
            grounding_metadata = getattr(candidate, "grounding_metadata", None)
            for chunk in getattr(grounding_metadata, "grounding_chunks", []) or []:
                web = getattr(chunk, "web", None)
                if web:
                    sources.append(
                        {
                            "uri": getattr(web, "uri", ""),
                            "title": getattr(web, "title", ""),
                        }
                    )

            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) if content else []:
                function_call = getattr(part, "function_call", None)
                if function_call:
                    args = getattr(function_call, "args", {})
                    function_calls.append(
                        {
                            "name": getattr(function_call, "name", ""),
                            "args": dict(args) if args else {},
                        }
                    )

                executable_code = getattr(part, "executable_code", None)
                if executable_code:
                    code = getattr(executable_code, "code", None)

                code_execution_result = getattr(part, "code_execution_result", None)
                if code_execution_result:
                    execution_result = getattr(code_execution_result, "output", None)

        return function_calls, sources, code, execution_result

    @gemini_retry
    async def generate_with_all_tools(
        self,
        prompt: str,
        function_tools: list[dict] | None = None,
        enable_grounding: bool = False,
        enable_code_execution: bool = False,
        thinking_level: str = "high",
        image_url: str | None = None,
        media_url: str | None = None,
    ) -> dict[str, Any]:
        client = self._require_client()

        logger.debug(
            "[generate_with_all_tools] START prompt=%r grounding=%s code_exec=%s thinking=%s",
            f"{prompt[:100]}..." if len(prompt) > 100 else prompt,
            enable_grounding,
            enable_code_execution,
            thinking_level,
        )
        parts = await self._prepare_parts(prompt, image_url, media_url)

        tools = self._build_combined_tools(
            function_tools,
            enable_grounding=enable_grounding,
            enable_code_execution=enable_code_execution,
        )

        config = types.GenerateContentConfig(
            tools=tools or None,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=_get_thinking_budget(thinking_level),
            ),
        )

        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=parts,
            config=config,
        )
        _log_token_usage(response, "generate_with_all_tools")

        function_calls, sources, code, execution_result = self._extract_combined_tool_response(response)

        logger.debug(
            "[generate_with_all_tools] END response_len=%d sources=%d function_calls=%d",
            len(response.text or ""),
            len(sources),
            len(function_calls),
        )
        return {
            "text": response.text,
            "function_calls": function_calls,
            "sources": sources,
            "code": code,
            "execution_result": execution_result,
        }


from fcp.services.gemini_helpers import (  # noqa: E402
    GeminiThinkingResult,
    _extract_grounding_sources,
    _extract_thinking_content,
    _get_thinking_budget,
)
