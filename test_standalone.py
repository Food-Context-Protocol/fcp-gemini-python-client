#!/usr/bin/env python3
"""Quick smoke test to verify gemini-connector is standalone."""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from gemini_connector.config import Config
        print("✓ config.Config")

        from gemini_connector.security import validate_image_url, ImageURLError
        print("✓ security.validate_image_url")

        from gemini_connector.utils import extract_json, record_gemini_usage
        print("✓ utils.extract_json")

        from gemini_connector.gemini_constants import (
            MODEL_NAME,
            THINKING_BUDGETS,
            MAX_IMAGE_SIZE,
            RETRYABLE_EXCEPTIONS,
        )
        print("✓ gemini_constants")

        from gemini_connector import GeminiClient
        print("✓ GeminiClient")

        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test Config works with env vars."""
    print("\nTesting Config...")
    from gemini_connector.config import Config

    assert hasattr(Config, "GEMINI_API_KEY")
    assert hasattr(Config, "GEMINI_MODEL_NAME")
    assert hasattr(Config, "DEEP_RESEARCH_TIMEOUT_SECONDS")
    print(f"✓ Config has all attributes")
    print(f"  Model: {Config.GEMINI_MODEL_NAME}")
    print(f"  Timeout: {Config.HTTP_TIMEOUT_SECONDS}s")
    return True


def test_security():
    """Test security module."""
    print("\nTesting security...")
    from gemini_connector.security import validate_image_url, ImageURLError

    # Valid URL
    try:
        url = validate_image_url("https://images.unsplash.com/test.jpg")
        print(f"✓ Valid URL accepted: {url[:50]}...")
    except ImageURLError as e:
        print(f"✗ Should accept valid URL: {e}")
        return False

    # Invalid URL
    try:
        validate_image_url("file:///etc/passwd")
        print("✗ Should reject file:// URLs")
        return False
    except ImageURLError:
        print("✓ Rejected file:// URL")

    return True


def test_utils():
    """Test utils module."""
    print("\nTesting utils...")
    from gemini_connector.utils import extract_json

    # Test JSON extraction
    text = '```json\n{"key": "value"}\n```'
    result = extract_json(text)
    if result == {"key": "value"}:
        print("✓ JSON extraction works")
        return True
    else:
        print(f"✗ JSON extraction failed: {result}")
        return False


def test_constants():
    """Test constants from fcp-gemini-core."""
    print("\nTesting constants...")
    from gemini_connector.gemini_constants import (
        MODEL_NAME,
        THINKING_BUDGETS,
        MAX_IMAGE_SIZE,
    )

    assert MODEL_NAME == "gemini-3-flash-preview"
    assert isinstance(THINKING_BUDGETS, dict)
    assert "low" in THINKING_BUDGETS
    assert MAX_IMAGE_SIZE > 0

    print(f"✓ Constants loaded correctly")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Thinking budgets: {list(THINKING_BUDGETS.keys())}")
    return True


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("gemini-connector Standalone Smoke Test")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Security", test_security),
        ("Utils", test_utils),
        ("Constants", test_constants),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} raised exception: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print("=" * 60)
    print(f"Total: {passed}/{total} passed")

    if passed == total:
        print("\n🎉 All smoke tests passed!")
        print("gemini-connector is standalone and working!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
