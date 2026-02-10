"""Tests for fcp.gemini.utils - JSON extraction utilities."""

from fcp.gemini.utils import (
    _extract_balanced_json,
    extract_json,
    extract_json_with_key,
    record_gemini_usage,
)


class TestExtractJson:
    """Tests for extract_json()."""

    def test_none_input(self):
        assert extract_json(None) is None

    def test_empty_string(self):
        assert extract_json("") is None

    def test_non_string_input(self):
        assert extract_json(123) is None

    def test_pure_json_object(self):
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_pure_json_array(self):
        result = extract_json("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_json_in_markdown_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_json_in_generic_code_block(self):
        text = '```\n{"key": "value"}\n```'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_json_embedded_in_prose(self):
        text = 'Here is the result: {"key": "value"} and more text'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_array_before_object(self):
        text = 'Result: [{"a": 1}] then {"b": 2}'
        result = extract_json(text)
        assert result == [{"a": 1}]

    def test_object_before_array(self):
        text = 'Result: {"a": 1} then [2, 3]'
        result = extract_json(text)
        assert result == {"a": 1}

    def test_array_after_failed_object(self):
        """Strategy 6: array tried after object was attempted first."""
        text = "{invalid json} then [1, 2, 3]"
        result = extract_json(text)
        assert result == [1, 2, 3]

    def test_last_resort_object_pattern(self):
        text = 'blah blah {"name": "test", "value": 42} blah'
        result = extract_json(text)
        assert result == {"name": "test", "value": 42}

    def test_last_resort_array_of_objects(self):
        text = 'data: [{"id": 1}, {"id": 2}] end'
        result = extract_json(text)
        assert result == [{"id": 1}, {"id": 2}]

    def test_completely_invalid(self):
        assert extract_json("no json here at all") is None

    def test_whitespace_handling(self):
        result = extract_json('  \n  {"key": "value"}  \n  ')
        assert result == {"key": "value"}

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = extract_json(text)
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_json_with_escaped_quotes(self):
        text = '{"key": "value with \\"quotes\\""}'
        result = extract_json(text)
        assert result == {"key": 'value with "quotes"'}

    def test_json_code_block_with_invalid_json(self):
        """Strategy 2 JSONDecodeError: ```json block contains invalid JSON."""
        text = "```json\n{not valid json}\n```"
        result = extract_json(text)
        # Falls through to later strategies; balanced extraction finds {not valid json}
        # which also fails, so returns None
        assert result is None

    def test_generic_code_block_with_invalid_json(self):
        """Strategy 3 JSONDecodeError: generic ``` block contains invalid JSON."""
        # Need text that does NOT match ```json but DOES match ```, with invalid JSON.
        # Also must not have been caught by strategy 2.
        text = "```\nnot valid json at all\n```"
        result = extract_json(text)
        assert result is None

    def test_balanced_array_extraction_json_decode_error(self):
        """Strategy 4 JSONDecodeError: balanced array found but invalid JSON."""
        # Array appears before any object, balanced extraction succeeds but json.loads fails.
        text = "[not, valid, json] and nothing else"
        result = extract_json(text)
        assert result is None

    def test_strategy6_array_after_object_json_decode_error(self):
        """Strategy 6 JSONDecodeError: array after object, balanced but invalid JSON."""
        # Object index < array index, object extraction tried first (strategy 5) and fails,
        # then strategy 6 tries the array which also has invalid JSON.
        text = '{"valid": "obj"} then [not valid json]'
        result = extract_json(text)
        assert result == {"valid": "obj"}

    def test_strategy6_array_after_object_invalid_both(self):
        """Strategy 6: both object and array are balanced but array has invalid JSON."""
        # Need: obj_index < arr_index, obj fails json.loads, array balanced but also fails.
        text = "{bad object content} then [also bad array content]"
        result = extract_json(text)
        assert result is None

    def test_last_resort_pattern_json_decode_error(self):
        """Strategy 7: regex pattern matches but json.loads fails, continues to next."""
        # First pattern matches an object-like string with a key, but it's not valid JSON.
        # Second pattern also fails or doesn't match, returns None.
        text = 'some {"key": value without quotes} end'
        result = extract_json(text)
        assert result is None


class TestExtractBalancedJson:
    """Tests for _extract_balanced_json()."""

    def test_simple_object(self):
        result = _extract_balanced_json('{"a": 1}', "{", "}")
        assert result == '{"a": 1}'

    def test_simple_array(self):
        result = _extract_balanced_json("[1, 2]", "[", "]")
        assert result == "[1, 2]"

    def test_nested(self):
        result = _extract_balanced_json('{"a": {"b": 1}}', "{", "}")
        assert result == '{"a": {"b": 1}}'

    def test_no_match(self):
        assert _extract_balanced_json("no braces", "{", "}") is None

    def test_with_string_containing_braces(self):
        text = '{"key": "value with { and } inside"}'
        result = _extract_balanced_json(text, "{", "}")
        assert result == text

    def test_unbalanced_returns_none(self):
        assert _extract_balanced_json('{"unclosed', "{", "}") is None

    def test_with_escaped_backslash(self):
        text = '{"key": "value\\\\"}'
        result = _extract_balanced_json(text, "{", "}")
        assert result is not None


class TestExtractJsonWithKey:
    """Tests for extract_json_with_key()."""

    def test_none_input(self):
        assert extract_json_with_key(None, "key") is None

    def test_empty_string(self):
        assert extract_json_with_key("", "key") is None

    def test_non_string_input(self):
        assert extract_json_with_key(123, "key") is None

    def test_key_found(self):
        result = extract_json_with_key('{"target": "value"}', "target")
        assert result == {"target": "value"}

    def test_key_not_found(self):
        result = extract_json_with_key('{"other": "value"}', "target")
        assert result is None

    def test_key_in_embedded_json(self):
        text = 'some text {"target": "value", "extra": 1} more text'
        result = extract_json_with_key(text, "target")
        assert result is not None
        assert result["target"] == "value"

    def test_returns_list_without_key(self):
        """extract_json returns a list, which doesn't have dict keys."""
        result = extract_json_with_key("[1, 2, 3]", "key")
        assert result is None

    def test_fallback_pattern_match(self):
        """The regex fallback pattern after extract_json fails to find key."""
        text = 'prefix {"required_key": "found"} suffix'
        result = extract_json_with_key(text, "required_key")
        assert result == {"required_key": "found"}

    def test_fallback_pattern_invalid_json(self):
        """Regex fallback finds a match containing the key but JSON is invalid."""
        # The regex will match because it finds {"target"...} but the content is not valid JSON.
        text = 'prefix {"target": unquoted_value} suffix'
        result = extract_json_with_key(text, "target")
        assert result is None

    def test_fallback_pattern_key_missing_after_parse(self):
        """Regex fallback parses valid JSON but the required key is not actually present."""
        # extract_json finds {"other": "value"} which lacks "target", so falls to regex.
        # Regex for "target" won't match text that doesn't contain "target" at all -> None.
        text = '{"other": "value"}'
        result = extract_json_with_key(text, "target")
        assert result is None

    def test_fallback_regex_succeeds_with_valid_json_and_key(self):
        """Regex fallback finds valid JSON containing the required key (lines 144-145).

        extract_json returns a list (not a dict), so the key check on line 136 fails.
        The regex fallback then finds the embedded object with the required key.
        """
        # extract_json will parse the leading array [1,2] first (array before object).
        # Since the result is a list, not a dict, line 136 fails.
        # The regex fallback then searches the raw text for {"target"...} and finds it.
        text = '[1, 2] and {"target": "found"}'
        result = extract_json_with_key(text, "target")
        assert result == {"target": "found"}


class TestRecordGeminiUsage:
    """Tests for record_gemini_usage() no-op stub."""

    def test_no_op(self):
        """Should not raise - it's a no-op stub."""
        record_gemini_usage(
            method="test",
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.001,
            latency_seconds=0.5,
            success=True,
        )
