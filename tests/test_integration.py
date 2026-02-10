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
