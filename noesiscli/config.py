"""
Configuration settings for NoesisCLI.
Defines:
  - Model definitions (Gemini 3.5 Flash, Gemini 3.1 Flash-Lite)
  - Embedding model (Voyage AI voyage-code-3)
  - Storage paths (e.g., .noesis/ folder inside analyzed repository)
  - Supported programming languages and file extensions
"""

# Gemini Models
GEMINI_3_5_FLASH = "gemini-3.5-flash"
GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite"

# Embedding Models
VOYAGE_CODE_3 = "voyage-code-3"

# Storage Configuration
DEFAULT_STORAGE_DIR = ".noesis"

# Supported Programming Languages & extensions
SUPPORTED_LANGUAGES = {
    "python": [".py"],
    "javascript": [".js", ".jsx"],
    "typescript": [".ts", ".tsx"],
    "go": [".go"],
    "java": [".java"],
    "cpp": [".cpp", ".cc", ".cxx", ".h", ".hpp"]
}
