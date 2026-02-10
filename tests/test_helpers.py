"""Tests for fcp.gemini.gemini_helpers - retry, parsing, extraction helpers."""

from unittest.mock import MagicMock

import pytest

from fcp.gemini.gemini_helpers import (
    GeminiThinkingResult,
    _extract_grounding_sources,
    _extract_thinking_content,
    _get_thinking_budget,
    _log_token_usage,
    _parse_json_response,
    gemini_retry,
)


class TestGetThinkingBudget:
    """Tests for _get_thinking_budget()."""

    def test_minimal(self):
        assert _get_thinking_budget("minimal") == 512

    def test_low(self):
        assert _get_thinking_budget("low") == 1024

    def test_medium(self):
        assert _get_thinking_budget("medium") == 2048

    def test_high(self):
        assert _get_thinking_budget("high") == 4096

    def test_case_insensitive(self):
        assert _get_thinking_budget("HIGH") == 4096
        assert _get_thinking_budget("Medium") == 2048

    def test_unknown_returns_default(self):
        result = _get_thinking_budget("unknown")
        assert result == 16384  # Default fallback


class TestExtractThinkingContent:
    """Tests for _extract_thinking_content()."""

    def test_no_candidates(self):
        response = MagicMock()
        response.candidates = None
        assert _extract_thinking_content(response) is None

    def test_empty_candidates(self):
        response = MagicMock()
        response.candidates = []
        assert _extract_thinking_content(response) is None

    def test_candidate_with_no_content(self):
        candidate = MagicMock()
        candidate.content = None
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_thinking_content(response) is None

    def test_candidate_with_no_parts(self):
        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = None
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_thinking_content(response) is None

    def test_thinking_part_extracted(self):
        part = MagicMock()
        part.thought = True
        part.text = "I'm thinking about this..."
        candidate = MagicMock()
        candidate.content.parts = [part]
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_thinking_content(response) == "I'm thinking about this..."

    def test_non_thinking_parts_skipped(self):
        regular_part = MagicMock()
        regular_part.thought = False
        thinking_part = MagicMock()
        thinking_part.thought = True
        thinking_part.text = "thinking"
        candidate = MagicMock()
        candidate.content.parts = [regular_part, thinking_part]
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_thinking_content(response) == "thinking"

    def test_multiple_thinking_parts_joined(self):
        part1 = MagicMock()
        part1.thought = True
        part1.text = "first thought"
        part2 = MagicMock()
        part2.thought = True
        part2.text = "second thought"
        candidate = MagicMock()
        candidate.content.parts = [part1, part2]
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_thinking_content(response) == "first thought\nsecond thought"

    def test_thinking_part_with_no_text(self):
        part = MagicMock()
        part.thought = True
        part.text = None
        candidate = MagicMock()
        candidate.content.parts = [part]
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_thinking_content(response) is None

    def test_no_candidates_attr(self):
        """Response object without candidates attribute."""
        response = MagicMock(spec=[])
        assert _extract_thinking_content(response) is None


class TestParseJsonResponse:
    """Tests for _parse_json_response()."""

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Empty response"):
            _parse_json_response("")

    def test_none_text(self):
        with pytest.raises(ValueError, match="Empty response"):
            _parse_json_response(None)

    def test_valid_json_object(self):
        assert _parse_json_response('{"key": "value"}') == {"key": "value"}

    def test_valid_json_array(self):
        assert _parse_json_response("[1, 2, 3]") == [1, 2, 3]

    def test_whitespace_stripped(self):
        assert _parse_json_response('  {"key": "value"}  ') == {"key": "value"}

    def test_bom_stripped(self):
        assert _parse_json_response('\ufeff{"key": "value"}') == {"key": "value"}

    def test_markdown_wrapped(self):
        text = '```json\n{"key": "value"}\n```'
        assert _parse_json_response(text) == {"key": "value"}

    def test_completely_invalid(self):
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            _parse_json_response("not json at all")


class TestExtractGroundingSources:
    """Tests for _extract_grounding_sources()."""

    def test_no_candidates(self):
        response = MagicMock()
        response.candidates = []
        assert _extract_grounding_sources(response) == []

    def test_no_grounding_metadata(self):
        candidate = MagicMock()
        candidate.grounding_metadata = None
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_grounding_sources(response) == []

    def test_no_grounding_chunks_attr(self):
        candidate = MagicMock(spec=["grounding_metadata"])
        candidate.grounding_metadata = MagicMock(spec=[])
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_grounding_sources(response) == []

    def test_empty_grounding_chunks(self):
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = []
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_grounding_sources(response) == []

    def test_extracts_sources(self):
        chunk = MagicMock()
        chunk.web.uri = "https://example.com"
        chunk.web.title = "Example"
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk]
        response = MagicMock()
        response.candidates = [candidate]
        result = _extract_grounding_sources(response)
        assert result == [{"uri": "https://example.com", "title": "Example"}]

    def test_chunk_without_web_skipped(self):
        chunk = MagicMock(spec=[])  # No 'web' attribute
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk]
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_grounding_sources(response) == []


class TestLogTokenUsage:
    """Tests for _log_token_usage()."""

    def test_with_usage_metadata(self):
        response = MagicMock()
        response.usage_metadata.prompt_token_count = 100
        response.usage_metadata.candidates_token_count = 200
        result = _log_token_usage(response, "test_method")
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 200
        assert result["total_tokens"] == 300
        assert result["cost_usd"] > 0

    def test_without_usage_metadata(self):
        response = MagicMock()
        response.usage_metadata = None
        result = _log_token_usage(response, "test_method")
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["total_tokens"] == 0
        assert result["cost_usd"] == 0.0

    def test_no_usage_metadata_attr(self):
        response = MagicMock(spec=[])
        result = _log_token_usage(response, "test_method")
        assert result["input_tokens"] == 0

    def test_non_numeric_token_counts(self):
        """Handles mock objects that return non-int values."""
        response = MagicMock()
        response.usage_metadata.prompt_token_count = "not a number"
        response.usage_metadata.candidates_token_count = "not a number"
        result = _log_token_usage(response, "test_method")
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0

    def test_cost_calculation(self):
        response = MagicMock()
        response.usage_metadata.prompt_token_count = 1000
        response.usage_metadata.candidates_token_count = 500
        result = _log_token_usage(response, "test_method")
        expected_cost = (1000 * 0.0001) + (500 * 0.0003)
        assert result["cost_usd"] == round(expected_cost, 6)

    def test_latency_and_success_passed(self):
        """Verify latency/success args are accepted (metrics recording)."""
        response = MagicMock()
        response.usage_metadata.prompt_token_count = 10
        response.usage_metadata.candidates_token_count = 20
        result = _log_token_usage(response, "test_method", latency_seconds=1.5, success=False)
        assert result["input_tokens"] == 10


class TestGeminiRetryDecorator:
    """Tests for the retry decorator."""

    def test_gemini_retry_exists(self):
        """Verify the retry decorator is created."""
        assert gemini_retry is not None

    def test_thinking_result_type(self):
        """Verify GeminiThinkingResult TypedDict structure."""
        result: GeminiThinkingResult = {"analysis": {"key": "value"}, "thinking": "thoughts"}
        assert result["analysis"] == {"key": "value"}
        assert result["thinking"] == "thoughts"
