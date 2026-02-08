"""Gemini Connector - Standalone Gemini 3 Python Client

A comprehensive Python client for Google's Gemini 3 API with support for:
- Multimodal vision (images, video, audio)
- Google Search grounding with citations
- Extended thinking mode
- Code execution
- Live API (audio/video streaming)
- Context caching
- Function calling
"""

from .gemini import GeminiClient

__version__ = "0.1.0"
__all__ = ["GeminiClient"]
