"""Live API interactions for Gemini."""

from __future__ import annotations

from typing import Any

from google.genai import types

from fcp.config import Config


class GeminiLiveMixin:
    """Live voice session support."""

    def create_live_session(
        self,
        system_instruction: str | None = None,
        enable_food_tools: bool = True,
    ) -> Any:
        client = self._require_client()

        config_options: dict[str, Any] = {
            "response_modalities": ["AUDIO", "TEXT"],
        }

        if system_instruction:
            config_options["system_instruction"] = system_instruction

        if enable_food_tools:
            log_meal_params = types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "dish_name": types.Schema(type=types.Type.STRING, description="Name of the dish"),
                    "description": types.Schema(type=types.Type.STRING, description="Description of the meal"),
                    "venue": types.Schema(type=types.Type.STRING, description="Where the meal was consumed"),
                    "meal_type": types.Schema(
                        type=types.Type.STRING,
                        enum=["breakfast", "lunch", "dinner", "snack"],
                        description="Type of meal",
                    ),
                },
                required=["dish_name"],
            )
            search_params = types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="Search query"),
                },
                required=["query"],
            )
            config_options["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name="log_meal",
                            description="Log a meal the user described",
                            parameters=log_meal_params,
                        ),
                        types.FunctionDeclaration(
                            name="search_food_history",
                            description="Search the user's food history",
                            parameters=search_params,
                        ),
                    ]
                )
            ]

        config = types.LiveConnectConfig(**config_options)

        return client.aio.live.connect(
            model=Config.GEMINI_LIVE_MODEL_NAME,
            config=config,
        )

    async def process_live_audio(
        self,
        audio_data: bytes,
        mime_type: str = "audio/pcm",
        sample_rate: int = 16000,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "transcription": None,
            "response_text": None,
            "response_audio": None,
            "function_calls": [],
        }

        system_instruction = """You are a friendly food logging assistant.
Help users log their meals by asking clarifying questions about what they ate.
When you have enough information, call the log_meal function.
Be conversational and helpful."""

        async with self.create_live_session(
            system_instruction=system_instruction,
            enable_food_tools=True,
        ) as session:
            await session.send(
                input=types.LiveClientRealtimeInput(
                    media_chunks=[
                        types.Blob(
                            data=audio_data,
                            mime_type=f"{mime_type};rate={sample_rate}",
                        )
                    ]
                ),
                end_of_turn=True,
            )

            response_parts = []
            async for response in session.receive():
                if hasattr(response, "server_content") and response.server_content:
                    content = response.server_content
                    if hasattr(content, "model_turn") and content.model_turn:
                        for part in content.model_turn.parts:
                            if hasattr(part, "text") and part.text:
                                response_parts.append(part.text)
                            if hasattr(part, "inline_data") and part.inline_data:
                                result["response_audio"] = part.inline_data.data

                if hasattr(response, "tool_call") and response.tool_call:
                    for fc in response.tool_call.function_calls:
                        result["function_calls"].append(
                            {
                                "name": fc.name,
                                "args": dict(fc.args) if fc.args else {},
                            }
                        )

            result["response_text"] = "".join(response_parts) if response_parts else None

        return result
