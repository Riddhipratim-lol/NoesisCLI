import os

class RepositoryScanner:
    """
    Repository Scanner.
    Recursively scans local repositories to find source code files matching supported extensions,
    excluding ignored folders like .git, build, dist, .venv, node_modules, etc.
    """
    def __init__(self, ignore_dirs=None):
        if ignore_dirs is None:
            self.ignore_dirs = {
                ".git",
                ".venv",
                "venv",
                "__pycache__",
                ".noesis",
                ".pytest_cache",
                "node_modules",
                "build",
                "dist",
            }
        else:
            self.ignore_dirs = set(ignore_dirs)

        # Supported file extensions for Python, JS, TS, Go, Java, C++
        self.supported_extensions = {
            ".py",
            ".js",
            ".jsx",
            ".mjs",
            ".cjs",
            ".ts",
            ".tsx",
            ".go",
            ".java",
            ".cpp",
            ".cc",
            ".cxx",
            ".c",
            ".h",
            ".hpp",
            ".hxx",
        }

    def scan(self, repo_path: str) -> list[str]:
        """
        Recursively scans repo_path and returns absolute paths of all supported files.
        """
        found_files = []
        # Ensure we work with absolute path of the repository
        repo_abs_path = os.path.abspath(repo_path)
        
        for root, dirs, files in os.walk(repo_abs_path):
            # Prune ignored directories in-place so os.walk doesn't traverse them
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]

            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in self.supported_extensions:
                    full_path = os.path.join(root, file)
                    found_files.append(os.path.abspath(full_path))

        return sorted(found_files)
