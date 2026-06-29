"""
Fail-safe Gemini Client.
Orchestrates API calls to Gemini 3.5 Flash and implements automatic fallback
routing to Gemini 3.1 Flash-Lite in case of failures or rate limits.
"""
