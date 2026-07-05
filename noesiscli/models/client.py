"""
Fail-safe Gemini Client.
Orchestrates API calls to Gemini 3.5 Flash and implements automatic fallback
routing to Gemini 3.1 Flash-Lite in case of failures or rate limits.
"""

import os
import sys
from typing import Generator
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

def _extract_text(content) -> str:
    """
    Safely extract string content from LangChain response message or chunk content,
    handling cases where content is a list of blocks/parts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts)
    return str(content) if content is not None else ""

class GeminiClient:
    """
    Fail-safe Gemini Client that calls primary model (Gemini 3.5 Flash)
    and falls back to secondary model (Gemini 3.1 Flash-Lite) on failure.
    """
    def __init__(self, primary_model: str = None, fallback_model: str = None):
        from noesiscli.config import GEMINI_3_5_FLASH, GEMINI_3_1_FLASH_LITE
        self.primary_model_name = primary_model or GEMINI_3_5_FLASH
        self.fallback_model_name = fallback_model or GEMINI_3_1_FLASH_LITE
        
        self.api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            os.environ["GOOGLE_API_KEY"] = self.api_key

        # lazy initialization (Takes time to delay until the first request)    
        self._primary_llm = None
        self._fallback_llm = None

    @property
    def primary_llm(self):
        if self._primary_llm is None:
            if not self.api_key and not os.environ.get("PYTEST_CURRENT_TEST"):
                raise ValueError("Google API key is not configured. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
            self._primary_llm = ChatGoogleGenerativeAI(
                model=self.primary_model_name,
                temperature=0.2,
                google_api_key=self.api_key
            )
        return self._primary_llm

    @property
    def fallback_llm(self):
        if self._fallback_llm is None:
            if not self.api_key and not os.environ.get("PYTEST_CURRENT_TEST"):
                raise ValueError("Google API key is not configured. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
            self._fallback_llm = ChatGoogleGenerativeAI(
                model=self.fallback_model_name,
                temperature=0.2,
                google_api_key=self.api_key
            )
        return self._fallback_llm

    def generate(self, prompt: str, system_instruction: str = None) -> str:
        """
        Generates text using the primary model. If it fails, falls back to the fallback model.
        """
        if not self.api_key and not os.environ.get("PYTEST_CURRENT_TEST"):
            raise ValueError("Google API key is not configured. Set GOOGLE_API_KEY or GEMINI_API_KEY.")

        messages = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))
        messages.append(HumanMessage(content=prompt))

        try:
            response = self.primary_llm.invoke(messages)
            return _extract_text(response.content)
        except Exception as e:
            print(f"\n[Warning] Primary model {self.primary_model_name} failed: {e}. Falling back to {self.fallback_model_name}...", file=sys.stderr)
            try:
                response = self.fallback_llm.invoke(messages)
                return _extract_text(response.content)
            except Exception as fe:
                raise RuntimeError(f"Both primary and fallback models failed. Fallback error: {fe}") from e

    def stream(self, prompt: str, system_instruction: str = None) -> Generator[str, None, None]:
        """
        Streams text response using the primary model. If it fails, falls back to the fallback model.
        """
        if not self.api_key and not os.environ.get("PYTEST_CURRENT_TEST"):
            raise ValueError("Google API key is not configured. Set GOOGLE_API_KEY or GEMINI_API_KEY.")

        messages = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))
        messages.append(HumanMessage(content=prompt))

        try:
            # We try to obtain the stream. If obtaining or reading the first token fails,
            # we switch to fallback.
            token_stream = self.primary_llm.stream(messages)
            iterator = iter(token_stream)
            try:
                first_chunk = next(iterator)
                yield _extract_text(first_chunk.content)
            except StopIteration:
                return
                
            for chunk in iterator:
                yield _extract_text(chunk.content)
                
        except Exception as e:
            print(f"\n[Warning] Primary model stream failed: {e}. Falling back to {self.fallback_model_name}...", file=sys.stderr)
            try:
                for chunk in self.fallback_llm.stream(messages):
                    yield _extract_text(chunk.content)
            except Exception as fe:
                raise RuntimeError(f"Both primary and fallback streaming failed. Fallback error: {fe}") from e
