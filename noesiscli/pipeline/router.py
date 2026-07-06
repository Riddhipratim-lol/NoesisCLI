"""
Intelligent Query Router.
Uses Gemini 3.1 Flash-Lite to route queries to either Direct LLM
or Repository-specific RAG.
"""

from noesiscli.models.client import GeminiClient
from noesiscli.config import GEMINI_3_1_FLASH_LITE

class QueryRouter:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or GeminiClient(primary_model=GEMINI_3_1_FLASH_LITE)

    def route(self, query: str) -> str:
        system_instruction = (
            "You are a routing assistant for a coding AI agent. "
            "Your job is to classify a user's coding-related query into one of two categories:\n"
            "1. 'direct_llm': For general programming questions, concepts, syntax, explanations, or algorithms that do "
            "not refer to or depend on the contents of the current repository (e.g. 'What is a decorator?', 'Explain recursion').\n"
            "2. 'repository_rag': For questions that specifically ask about the contents, structure, configuration, "
            "behavior, or location of files/code in the user's uploaded repository (e.g. 'Explain the authentication flow in this repo', "
            "'Where is the DB configured?', 'How does the payment module work?').\n\n"
            "Respond with EXACTLY either 'direct_llm' or 'repository_rag'. Do not include any other text or punctuation."
        )
        response = self.llm_client.generate(query, system_instruction=system_instruction)
        clean_res = response.strip().lower()
        if "repository_rag" in clean_res:
            return "repository_rag"
        return "direct_llm"
