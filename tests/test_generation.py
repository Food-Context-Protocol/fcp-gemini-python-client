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

    async def test_generate_content_with_image_url(self, gemini_client, mock_genai_client, mock_response):
        gemini_client._fetch_media = AsyncMock(return_value=(b"img", "image/png"))
        resp = mock_response(text="Image description")
        mock_genai_client.aio.models.generate_content.return_value = resp
        result = await gemini_client.generate_content("Describe", image_url="https://example.com/img.png")
        assert result == "Image description"

    async def test_generate_content_with_media_url(self, gemini_client, mock_genai_client, mock_response):
        gemini_client._fetch_media = AsyncMock(return_value=(b"media", "video/mp4"))
        resp = mock_response(text="Media description")
        mock_genai_client.aio.models.generate_content.return_value = resp
        result = await gemini_client.generate_content("Describe", media_url="https://example.com/vid.mp4")
        assert result == "Media description"

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

    async def test_generate_json_empty_response(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text=None)
        mock_genai_client.aio.models.generate_content.return_value = resp
        # When text is None, text becomes "", stripped is "", _parse_json_response("") should raise or return empty
        # Let's check what happens - _parse_json_response with empty string
        # Looking at the code: response_text = response.text or "" => ""
        # text = "".strip() => ""
        # _parse_json_response("") will try json.loads("") which raises ValueError
        with pytest.raises(Exception):
            await gemini_client.generate_json("Get JSON")

    async def test_generate_content_stream(self, gemini_client, mock_genai_client):
        chunk1 = MagicMock()
        chunk1.text = "Hello "
        chunk2 = MagicMock()
        chunk2.text = "world"
        chunk3 = MagicMock()
        chunk3.text = None  # Chunk with no text should be skipped

        async def _async_iter():
            for chunk in [chunk1, chunk2, chunk3]:
                yield chunk

        # generate_content_stream does: stream = await client.aio.models.generate_content_stream(...)
        # So the AsyncMock needs to return an async iterable when awaited
        mock_genai_client.aio.models.generate_content_stream.return_value = _async_iter()

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

        async def _async_iter(*args, **kwargs):
            for chunk in [chunk1, chunk2, chunk3]:
                yield chunk

        # generate_json_stream does NOT await the stream call; it uses cast() directly.
        # So the mock itself (not its return value) must be the async iterable.
        mock_genai_client.aio.models.generate_content_stream = _async_iter
        gemini_client.client.aio.models.generate_content_stream = _async_iter

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

    async def test_generate_with_tools_no_args(self, gemini_client, mock_genai_client, mock_response):
        """Test function call with None args."""
        func_call = MagicMock()
        func_call.name = "list_items"
        func_call.args = None

        part = MagicMock()
        part.function_call = func_call

        candidate = MagicMock()
        candidate.content.parts = [part]

        resp = mock_response(text="Listing")
        resp.candidates = [candidate]
        mock_genai_client.aio.models.generate_content.return_value = resp

        tools = [{"name": "list_items", "description": "List items"}]
        result = await gemini_client.generate_with_tools("List", tools=tools)
        assert result["function_calls"][0]["args"] == {}


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

    async def test_generate_with_grounding_no_sources(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text="No sources")
        resp.candidates = []
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_with_grounding("Search query")
        assert result["text"] == "No sources"
        assert result["sources"] == []

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

    async def test_generate_with_thinking_empty(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text=None)
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_with_thinking("Think about this")
        assert result == ""

    async def test_generate_json_with_thinking(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text='{"analysis": "deep"}')
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_json_with_thinking("Analyze this")
        assert result == {"analysis": "deep"}

    async def test_generate_json_with_thinking_include_output(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text='{"result": "yes"}')
        resp.candidates = []
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_json_with_thinking("Analyze", include_thinking_output=True)
        assert "analysis" in result
        assert "thinking" in result

    async def test_generate_with_large_context(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text="Large context answer")
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_with_large_context("Big prompt")
        assert result == "Large context answer"

    async def test_generate_with_large_context_empty(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text=None)
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_with_large_context("Big prompt")
        assert result == ""

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

        result = await gemini_client.generate_json_with_agentic_vision("Describe", "https://example.com/img.png")
        assert result["analysis"] == {"objects": ["cat"]}

    async def test_generate_json_with_agentic_vision_with_code(self, gemini_client, mock_genai_client, mock_response):
        gemini_client._fetch_media = AsyncMock(return_value=(b"img", "image/png"))

        code_part = MagicMock()
        code_part.executable_code = MagicMock()
        code_part.executable_code.code = "analyze()"
        code_part.code_execution_result = MagicMock()
        code_part.code_execution_result.output = "done"

        candidate = MagicMock()
        candidate.content.parts = [code_part]

        resp = mock_response(text='{"result": "analyzed"}')
        resp.candidates = [candidate]
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_json_with_agentic_vision("Analyze", "https://example.com/img.png")
        assert result["analysis"] == {"result": "analyzed"}
        assert result["code"] == "analyze()"
        assert result["execution_result"] == "done"

    async def test_generate_json_with_agentic_vision_empty(self, gemini_client, mock_genai_client, mock_response):
        gemini_client._fetch_media = AsyncMock(return_value=(b"img", "image/png"))
        resp = mock_response(text=None)
        resp.candidates = []
        mock_genai_client.aio.models.generate_content.return_value = resp

        with pytest.raises(ValueError, match="empty response"):
            await gemini_client.generate_json_with_agentic_vision("Describe", "https://example.com/img.png")


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

    async def test_generate_json_with_media_resolution_low(self, gemini_client, mock_genai_client, mock_response):
        gemini_client._fetch_media = AsyncMock(return_value=(b"img", "image/png"))
        resp = mock_response(text='{"detail": "coarse"}')
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_json_with_media_resolution(
            "Analyze", "https://example.com/img.png", resolution="low"
        )
        assert result == {"detail": "coarse"}

    async def test_generate_json_with_media_resolution_unknown(self, gemini_client, mock_genai_client, mock_response):
        """Unknown resolution defaults to high."""
        gemini_client._fetch_media = AsyncMock(return_value=(b"img", "image/png"))
        resp = mock_response(text='{"detail": "fine"}')
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_json_with_media_resolution(
            "Analyze", "https://example.com/img.png", resolution="ultra"
        )
        assert result == {"detail": "fine"}

    async def test_generate_json_with_media_resolution_list(self, gemini_client, mock_genai_client, mock_response):
        gemini_client._fetch_media = AsyncMock(return_value=(b"img", "image/png"))
        resp = mock_response(text="[1, 2]")
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_json_with_media_resolution("Analyze", "https://example.com/img.png")
        assert result == {"items": [1, 2]}

    async def test_generate_json_with_url_context(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text='{"summary": "content"}')
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_json_with_url_context("Summarize", urls=["https://example.com/page1"])
        assert result["data"] == {"summary": "content"}
        assert result["sources"] == ["https://example.com/page1"]

    async def test_generate_json_with_url_context_multiple_urls(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text='{"summary": "combined"}')
        mock_genai_client.aio.models.generate_content.return_value = resp

        urls = ["https://example.com/page1", "https://example.com/page2"]
        result = await gemini_client.generate_json_with_url_context("Summarize", urls=urls)
        assert result["data"] == {"summary": "combined"}
        assert result["sources"] == urls


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

    async def test_generate_image_none_generated(self, gemini_client, mock_genai_client):
        mock_resp = MagicMock()
        mock_resp.generated_images = None
        mock_genai_client.aio.models.generate_images.return_value = mock_resp

        result = await gemini_client.generate_image("A cat")
        assert result["count"] == 0
        assert result["images"] == []

    async def test_generate_image_multiple(self, gemini_client, mock_genai_client):
        img1 = MagicMock()
        img1.image.image_bytes = b"png1"
        img2 = MagicMock()
        img2.image.image_bytes = b"png2"
        mock_resp = MagicMock()
        mock_resp.generated_images = [img1, img2]
        mock_genai_client.aio.models.generate_images.return_value = mock_resp

        result = await gemini_client.generate_image("Two cats", number_of_images=2)
        assert result["count"] == 2
        assert len(result["images"]) == 2

    async def test_generate_image_base64_encoding(self, gemini_client, mock_genai_client):
        import base64

        raw_bytes = b"test-image-data"
        img = MagicMock()
        img.image.image_bytes = raw_bytes
        mock_resp = MagicMock()
        mock_resp.generated_images = [img]
        mock_genai_client.aio.models.generate_images.return_value = mock_resp

        result = await gemini_client.generate_image("A dog")
        expected_b64 = base64.b64encode(raw_bytes).decode("utf-8")
        assert result["images"][0] == expected_b64


class TestGeminiCombinedToolsMixin:
    """Tests for combined tool orchestration."""

    def test_build_combined_tools_grounding_only(self, gemini_client):
        tools = gemini_client._build_combined_tools(None, enable_grounding=True, enable_code_execution=False)
        assert len(tools) == 1

    def test_build_combined_tools_all(self, gemini_client):
        func_tools = [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        tools = gemini_client._build_combined_tools(func_tools, enable_grounding=True, enable_code_execution=True)
        assert len(tools) == 3

    def test_build_combined_tools_none(self, gemini_client):
        tools = gemini_client._build_combined_tools(None, enable_grounding=False, enable_code_execution=False)
        assert tools == []

    def test_build_combined_tools_code_execution_only(self, gemini_client):
        tools = gemini_client._build_combined_tools(None, enable_grounding=False, enable_code_execution=True)
        assert len(tools) == 1

    def test_build_combined_tools_function_tools_only(self, gemini_client):
        func_tools = [{"name": "calc", "description": "Calculate"}]
        tools = gemini_client._build_combined_tools(func_tools, enable_grounding=False, enable_code_execution=False)
        assert len(tools) == 1

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

    def test_extract_combined_tool_response_empty(self, gemini_client, mock_response):
        resp = mock_response()
        resp.candidates = []

        fc, sources, code, result = gemini_client._extract_combined_tool_response(resp)
        assert fc == []
        assert sources == []
        assert code is None
        assert result is None

    def test_extract_combined_tool_response_no_grounding(self, gemini_client, mock_response):
        candidate = MagicMock()
        candidate.grounding_metadata = None
        candidate.content.parts = []

        resp = mock_response()
        resp.candidates = [candidate]

        fc, sources, code, result = gemini_client._extract_combined_tool_response(resp)
        assert sources == []

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

    async def test_generate_with_all_tools_with_function_tools(self, gemini_client, mock_genai_client, mock_response):
        func_call = MagicMock()
        func_call.name = "search"
        func_call.args = {"q": "test"}

        part = MagicMock()
        part.function_call = func_call
        part.executable_code = None
        part.code_execution_result = None

        candidate = MagicMock()
        candidate.grounding_metadata = None
        candidate.content.parts = [part]

        resp = mock_response(text="Tool result")
        resp.candidates = [candidate]
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_with_all_tools(
            "Search for something",
            function_tools=[{"name": "search", "description": "Search", "parameters": {"type": "object"}}],
            enable_grounding=False,
            enable_code_execution=False,
        )
        assert result["text"] == "Tool result"
        assert len(result["function_calls"]) == 1
        assert result["function_calls"][0]["name"] == "search"

    async def test_generate_with_all_tools_no_tools(self, gemini_client, mock_genai_client, mock_response):
        resp = mock_response(text="Simple result")
        resp.candidates = []
        mock_genai_client.aio.models.generate_content.return_value = resp

        result = await gemini_client.generate_with_all_tools("Just think")
        assert result["text"] == "Simple result"
        assert result["function_calls"] == []
        assert result["sources"] == []
        assert result["code"] is None
        assert result["execution_result"] is None
