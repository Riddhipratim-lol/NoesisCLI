"""
Direct LLM Route Execution.
Streams the response for general programming queries using Gemini 3.1 Flash-Lite.
"""

from noesiscli.models.client import GeminiClient
from noesiscli.config import GEMINI_3_1_FLASH_LITE
from typing import Generator

class DirectResponder:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or GeminiClient(primary_model=GEMINI_3_1_FLASH_LITE)

    def execute(self, query: str) -> Generator[str, None, None]:
        system_instruction = (
            "You are NoesisCLI, a professional AI coding assistant and codebase architect.\n"
            "Answer the user's general programming question clearly and concisely."
        )
        return self.llm_client.stream(query, system_instruction=system_instruction)
