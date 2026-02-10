"""Tests for fcp.gemini.gemini_live - Live voice session support."""

from unittest.mock import AsyncMock, MagicMock

from google.genai import types

from fcp.gemini.config import Config


class TestGeminiLiveMixin:
    """Tests for live session creation and audio processing."""

    def test_create_live_session_default(self, gemini_client, mock_genai_client):
        mock_genai_client.aio.live.connect.return_value = MagicMock()
        gemini_client.create_live_session()
        mock_genai_client.aio.live.connect.assert_called_once()

        # Verify call arguments
        call_kwargs = mock_genai_client.aio.live.connect.call_args
        assert call_kwargs.kwargs["model"] == Config.GEMINI_LIVE_MODEL_NAME
        config = call_kwargs.kwargs["config"]
        assert isinstance(config, types.LiveConnectConfig)

    def test_create_live_session_with_instruction(self, gemini_client, mock_genai_client):
        mock_genai_client.aio.live.connect.return_value = MagicMock()
        gemini_client.create_live_session(system_instruction="Be helpful")
        mock_genai_client.aio.live.connect.assert_called_once()

    def test_create_live_session_no_food_tools(self, gemini_client, mock_genai_client):
        mock_genai_client.aio.live.connect.return_value = MagicMock()
        gemini_client.create_live_session(enable_food_tools=False)
        mock_genai_client.aio.live.connect.assert_called_once()

    def test_create_live_session_with_food_tools_has_declarations(self, gemini_client, mock_genai_client):
        """Verify that enabling food tools includes log_meal and search_food_history."""
        mock_genai_client.aio.live.connect.return_value = MagicMock()
        gemini_client.create_live_session(enable_food_tools=True)

        call_kwargs = mock_genai_client.aio.live.connect.call_args
        config = call_kwargs.kwargs["config"]
        # The config should have tools with function declarations
        assert config.tools is not None
        assert len(config.tools) == 1
        tool = config.tools[0]
        func_names = [fd.name for fd in tool.function_declarations]
        assert "log_meal" in func_names
        assert "search_food_history" in func_names

    def test_create_live_session_no_tools_when_disabled(self, gemini_client, mock_genai_client):
        """Verify no tools when enable_food_tools=False."""
        mock_genai_client.aio.live.connect.return_value = MagicMock()
        gemini_client.create_live_session(enable_food_tools=False)

        call_kwargs = mock_genai_client.aio.live.connect.call_args
        config = call_kwargs.kwargs["config"]
        # When food tools are disabled, tools should not be set
        assert config.tools is None or config.tools == []

    def test_create_live_session_response_modalities(self, gemini_client, mock_genai_client):
        """Verify response modalities include AUDIO and TEXT."""
        mock_genai_client.aio.live.connect.return_value = MagicMock()
        gemini_client.create_live_session()

        call_kwargs = mock_genai_client.aio.live.connect.call_args
        config = call_kwargs.kwargs["config"]
        assert config.response_modalities == ["AUDIO", "TEXT"]

    async def test_process_live_audio(self, gemini_client, mock_genai_client):
        """Test live audio processing with mocked session."""
        # Create mock session context manager
        mock_session = AsyncMock()

        # Mock response with text
        response = MagicMock()
        response.server_content = MagicMock()
        response.server_content.model_turn = MagicMock()
        text_part = MagicMock()
        text_part.text = "I heard you"
        text_part.inline_data = None
        response.server_content.model_turn.parts = [text_part]
        response.tool_call = None

        # Mock receive as async iterator
        async def mock_receive():
            yield response

        mock_session.receive = mock_receive
        mock_session.send = AsyncMock()

        # Create async context manager
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        gemini_client.create_live_session = MagicMock(return_value=mock_ctx)

        result = await gemini_client.process_live_audio(b"audio-data")
        assert result["response_text"] == "I heard you"
        assert result["response_audio"] is None
        assert result["function_calls"] == []

    async def test_process_live_audio_with_function_call(self, gemini_client, mock_genai_client):
        """Test live audio processing that triggers a function call."""
        mock_session = AsyncMock()

        # Response with function call
        fc = MagicMock()
        fc.name = "log_meal"
        fc.args = {"dish_name": "pizza"}

        response = MagicMock()
        response.server_content = None
        response.tool_call = MagicMock()
        response.tool_call.function_calls = [fc]

        async def mock_receive():
            yield response

        mock_session.receive = mock_receive
        mock_session.send = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        gemini_client.create_live_session = MagicMock(return_value=mock_ctx)

        result = await gemini_client.process_live_audio(b"audio-data")
        assert len(result["function_calls"]) == 1
        assert result["function_calls"][0]["name"] == "log_meal"
        assert result["function_calls"][0]["args"] == {"dish_name": "pizza"}

    async def test_process_live_audio_with_audio_response(self, gemini_client, mock_genai_client):
        """Test live audio processing that returns audio."""
        mock_session = AsyncMock()

        response = MagicMock()
        response.server_content = MagicMock()
        response.server_content.model_turn = MagicMock()
        audio_part = MagicMock()
        audio_part.text = None
        audio_part.inline_data = MagicMock()
        audio_part.inline_data.data = b"response-audio"
        response.server_content.model_turn.parts = [audio_part]
        response.tool_call = None

        async def mock_receive():
            yield response

        mock_session.receive = mock_receive
        mock_session.send = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        gemini_client.create_live_session = MagicMock(return_value=mock_ctx)

        result = await gemini_client.process_live_audio(b"audio-data")
        assert result["response_audio"] == b"response-audio"

    async def test_process_live_audio_sends_correct_data(self, gemini_client, mock_genai_client):
        """Test that process_live_audio sends audio with correct parameters."""
        mock_session = AsyncMock()

        # Empty response - no server_content, no tool_call
        response = MagicMock()
        response.server_content = None
        response.tool_call = None

        async def mock_receive():
            yield response

        mock_session.receive = mock_receive
        mock_session.send = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        gemini_client.create_live_session = MagicMock(return_value=mock_ctx)

        await gemini_client.process_live_audio(b"audio-data", mime_type="audio/wav", sample_rate=44100)

        # Verify send was called
        mock_session.send.assert_called_once()
        call_kwargs = mock_session.send.call_args
        assert call_kwargs.kwargs["end_of_turn"] is True

        # Check the input
        sent_input = call_kwargs.kwargs["input"]
        assert isinstance(sent_input, types.LiveClientRealtimeInput)
        assert len(sent_input.media_chunks) == 1
        blob = sent_input.media_chunks[0]
        assert blob.data == b"audio-data"
        assert blob.mime_type == "audio/wav;rate=44100"

    async def test_process_live_audio_default_mime_and_rate(self, gemini_client, mock_genai_client):
        """Test that defaults are audio/pcm at 16000 Hz."""
        mock_session = AsyncMock()

        response = MagicMock()
        response.server_content = None
        response.tool_call = None

        async def mock_receive():
            yield response

        mock_session.receive = mock_receive
        mock_session.send = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        gemini_client.create_live_session = MagicMock(return_value=mock_ctx)

        await gemini_client.process_live_audio(b"audio-data")

        call_kwargs = mock_session.send.call_args
        blob = call_kwargs.kwargs["input"].media_chunks[0]
        assert blob.mime_type == "audio/pcm;rate=16000"

    async def test_process_live_audio_empty_response(self, gemini_client, mock_genai_client):
        """Test processing when no responses are received."""
        mock_session = AsyncMock()

        async def mock_receive():
            return
            yield  # Make it an async generator that yields nothing

        mock_session.receive = mock_receive
        mock_session.send = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        gemini_client.create_live_session = MagicMock(return_value=mock_ctx)

        result = await gemini_client.process_live_audio(b"audio-data")
        assert result["response_text"] is None
        assert result["response_audio"] is None
        assert result["transcription"] is None
        assert result["function_calls"] == []

    async def test_process_live_audio_multiple_text_parts(self, gemini_client, mock_genai_client):
        """Test that multiple text parts are concatenated."""
        mock_session = AsyncMock()

        # First response
        response1 = MagicMock()
        response1.server_content = MagicMock()
        response1.server_content.model_turn = MagicMock()
        part1 = MagicMock()
        part1.text = "Hello "
        part1.inline_data = None
        response1.server_content.model_turn.parts = [part1]
        response1.tool_call = None

        # Second response
        response2 = MagicMock()
        response2.server_content = MagicMock()
        response2.server_content.model_turn = MagicMock()
        part2 = MagicMock()
        part2.text = "world!"
        part2.inline_data = None
        response2.server_content.model_turn.parts = [part2]
        response2.tool_call = None

        async def mock_receive():
            yield response1
            yield response2

        mock_session.receive = mock_receive
        mock_session.send = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        gemini_client.create_live_session = MagicMock(return_value=mock_ctx)

        result = await gemini_client.process_live_audio(b"audio-data")
        assert result["response_text"] == "Hello world!"

    async def test_process_live_audio_function_call_with_no_args(self, gemini_client, mock_genai_client):
        """Test function call with None args returns empty dict."""
        mock_session = AsyncMock()

        fc = MagicMock()
        fc.name = "search_food_history"
        fc.args = None

        response = MagicMock()
        response.server_content = None
        response.tool_call = MagicMock()
        response.tool_call.function_calls = [fc]

        async def mock_receive():
            yield response

        mock_session.receive = mock_receive
        mock_session.send = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        gemini_client.create_live_session = MagicMock(return_value=mock_ctx)

        result = await gemini_client.process_live_audio(b"audio-data")
        assert len(result["function_calls"]) == 1
        assert result["function_calls"][0]["name"] == "search_food_history"
        assert result["function_calls"][0]["args"] == {}

    async def test_process_live_audio_uses_system_instruction(self, gemini_client, mock_genai_client):
        """Test that process_live_audio creates session with system instruction."""
        mock_session = AsyncMock()

        response = MagicMock()
        response.server_content = None
        response.tool_call = None

        async def mock_receive():
            yield response

        mock_session.receive = mock_receive
        mock_session.send = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        captured_kwargs = {}

        def capture_create(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_ctx

        gemini_client.create_live_session = capture_create

        await gemini_client.process_live_audio(b"audio-data")
        assert "system_instruction" in captured_kwargs
        assert captured_kwargs["system_instruction"] is not None
        assert captured_kwargs["enable_food_tools"] is True

    async def test_process_live_audio_response_no_model_turn(self, gemini_client, mock_genai_client):
        """Test response with server_content but no model_turn."""
        mock_session = AsyncMock()

        response = MagicMock()
        response.server_content = MagicMock()
        response.server_content.model_turn = None
        response.tool_call = None

        async def mock_receive():
            yield response

        mock_session.receive = mock_receive
        mock_session.send = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        gemini_client.create_live_session = MagicMock(return_value=mock_ctx)

        result = await gemini_client.process_live_audio(b"audio-data")
        # Should not crash; response_text should be None since no text parts collected
        assert result["response_text"] is None
