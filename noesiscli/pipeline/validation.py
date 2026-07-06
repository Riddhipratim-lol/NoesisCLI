"""
Query Validation Layer.
Uses Gemini 3.1 Flash-Lite to validate whether the incoming query
is related to programming or software development.
"""

from noesiscli.models.client import GeminiClient
from noesiscli.config import GEMINI_3_1_FLASH_LITE

class QueryValidator:
    def __init__(self, llm_client=None):
        # By default, use Gemini 3.1 Flash-Lite for validation node
        self.llm_client = llm_client or GeminiClient(primary_model=GEMINI_3_1_FLASH_LITE)

    def validate(self, query: str) -> bool:
        system_instruction = (
            "You are a validation assistant for a coding AI agent. "
            "Your job is to determine whether the user's input is a query related to programming, software development, "
            "computer science, or repository analysis. "
            "Respond with EXACTLY the word 'True' (if it is related to programming/code/software/repo) or 'False' "
            "(if it is not related, e.g. asking about weather, movies, jokes, or general off-topic conversation). "
            "Do not include any other text, explanation, or punctuation."
        )
        response = self.llm_client.generate(query, system_instruction=system_instruction)
        return response.strip().lower() == "true"

    def validate_and_route(self, query: str) -> tuple[bool, str]:
        system_instruction = (
            "You are an assistant for a coding AI agent. "
            "Your job is to classify the user's input query into one of three categories:\n"
            "1. 'invalid': If the query is NOT related to programming, software development, computer science, "
            "or repository analysis (e.g., asking about weather, movies, jokes, or general off-topic conversation).\n"
            "2. 'direct_llm': For valid programming/coding queries, concepts, syntax, explanations, or algorithms "
            "that do not refer to or depend on the contents of the current repository (e.g., 'What is a decorator?', 'Explain recursion').\n"
            "3. 'repository_rag': For questions that specifically ask about the contents, structure, configuration, "
            "behavior, or location of files/code in the user's uploaded repository (e.g., 'Explain the authentication flow in this repo', "
            "'Where is the DB configured?', 'How does the payment module work?').\n\n"
            "Respond with EXACTLY one of: 'invalid', 'direct_llm', or 'repository_rag'. "
            "Do not include any other text, explanation, or punctuation."
        )
        response = self.llm_client.generate(query, system_instruction=system_instruction)
        clean_res = response.strip().lower()
        if "repository_rag" in clean_res:
            return True, "repository_rag"
        elif "direct_llm" in clean_res:
            return True, "direct_llm"
        elif clean_res == "true":
            from noesiscli.pipeline.router import QueryRouter
            router = QueryRouter(llm_client=self.llm_client)
            return True, router.route(query)
        else:
            return False, "invalid"
