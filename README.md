# Gemini Connector

Standalone Python client for Google's Gemini 3 API with comprehensive support for advanced features.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)

## Features

- ✅ **Multimodal Vision** - Images, video, audio analysis
- ✅ **Google Search Grounding** - Real-time search with citations
- ✅ **Extended Thinking** - Deep reasoning mode
- ✅ **Code Execution** - Run Python code in Gemini sandbox
- ✅ **Live API** - Real-time audio/video streaming
- ✅ **Context Caching** - Reduce costs for repeated context
- ✅ **Function Calling** - Structured tool integration
- ✅ **Type-Safe** - Full Pydantic schema validation

## Installation

```bash
pip install gemini-connector
```

## Quick Start

```python
from gemini_connector import GeminiClient

# Initialize client
client = GeminiClient(api_key="your-api-key")

# Analyze an image
result = await client.analyze_image(
    image_url="https://example.com/food.jpg",
    prompt="What food is in this image?"
)

# Use Google Search grounding
response = await client.generate_content(
    prompt="Latest FDA food recalls",
    use_search_grounding=True
)

# Extended thinking mode
deep_result = await client.generate_content(
    prompt="Complex reasoning task",
    thinking_mode=True
)
```

## Gemini 3 Features

### 1. Multimodal Vision
```python
# Image analysis
result = await client.analyze_image(
    image_url="https://example.com/image.jpg",
    prompt="Describe this image"
)

# Video analysis
result = await client.analyze_video(
    video_url="https://example.com/video.mp4",
    prompt="Summarize this video"
)
```

### 2. Google Search Grounding
```python
# Real-time search with citations
result = await client.generate_content(
    prompt="Current weather in San Francisco",
    use_search_grounding=True
)
# Access grounding metadata
for chunk in result.grounding_metadata.grounding_chunks:
    print(chunk.web.uri)
```

### 3. Extended Thinking
```python
# Deep reasoning mode
result = await client.generate_content(
    prompt="Solve this complex problem...",
    thinking_mode=True
)
```

### 4. Code Execution
```python
# Run Python code
result = await client.generate_content(
    prompt="Calculate fibonacci(100)",
    code_execution=True
)
```

### 5. Live API
```python
# Real-time streaming
async for chunk in client.live_stream(audio_data):
    print(chunk.text)
```

### 6. Context Caching
```python
# Cache large context
cached_result = await client.generate_content_with_cache(
    system_instruction="Long system prompt...",
    prompt="User query",
    ttl_seconds=3600
)
```

## Configuration

```python
from gemini_connector import GeminiClient

client = GeminiClient(
    api_key="your-api-key",
    model="gemini-2.0-flash-exp",  # or gemini-2.0-flash-thinking-exp
    temperature=0.7,
    max_tokens=8192
)
```

## Requirements

- Python 3.11+
- Google Gemini API key (get from [ai.google.dev](https://ai.google.dev))

## Part of Food Context Protocol

This library was extracted from the [Food Context Protocol](https://github.com/Food-Context-Protocol/fcp) project to provide a standalone, reusable Gemini 3 client.

## License

Apache-2.0 - See [LICENSE](LICENSE)

## Contributing

See [CONTRIBUTING.md](https://github.com/Food-Context-Protocol/fcp/blob/main/CONTRIBUTING.md)
