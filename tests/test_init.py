"""Tests for fcp.gemini.__init__ - package entry point."""


class TestPackageInit:
    """Tests for package-level exports."""

    def test_version(self):
        from fcp.gemini import __version__

        assert __version__ == "0.1.0"

    def test_gemini_client_importable(self):
        from fcp.gemini import GeminiClient

        assert GeminiClient is not None

    def test_all_exports(self):
        from fcp.gemini import __all__

        assert "GeminiClient" in __all__
