import pytest
import os
import shutil
import tempfile
from unittest.mock import MagicMock

@pytest.fixture
def temp_repo():
    """Creates a temporary directory with some mock source files for testing the scanner and parser."""
    temp_dir = tempfile.mkdtemp()
    
    # Create structure
    os.makedirs(os.path.join(temp_dir, "src"))
    os.makedirs(os.path.join(temp_dir, "tests"))
    os.makedirs(os.path.join(temp_dir, ".venv"))
    os.makedirs(os.path.join(temp_dir, ".git"))
    
    # Write Python file
    python_content = """
class UserService:
    def __init__(self, db):
        self.db = db
        
    def authenticate(self, username, password):
        user = self.db.find_user(username)
        if user and user.check_password(password):
            return user
        return None
        
def root_function():
    return "Hello world"
"""
    with open(os.path.join(temp_dir, "src", "user_service.py"), "w") as f:
        f.write(python_content.strip())
        
    # Write JS file
    js_content = """
function calculateSum(a, b) {
    return a + b;
}
export { calculateSum };
"""
    with open(os.path.join(temp_dir, "src", "math.js"), "w") as f:
        f.write(js_content.strip())
        
    # Write Ignored file
    with open(os.path.join(temp_dir, ".venv", "ignored.py"), "w") as f:
        f.write("def ignored_func(): pass")
        
    yield temp_dir
    
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_code_chunks():
    """Returns a list of mock code chunk dictionaries as expected by indexing/retrieval systems."""
    return [
        {
            "code_content": "class UserService:\n    def authenticate(self, username, password): pass",
            "file_path": "/mock/project/src/user_service.py",
            "node_type": "class",
            "start_line": 1,
            "end_line": 2,
            "signature": "class UserService",
            "parent_class": None,
            "imports": [],
            "docstring": "Service to handle user authentication",
            "summary": "Handles user credentials verification"
        },
        {
            "code_content": "def find_user(username):\n    return db.query(username)",
            "file_path": "/mock/project/src/db.py",
            "node_type": "function",
            "start_line": 5,
            "end_line": 6,
            "signature": "def find_user(username)",
            "parent_class": None,
            "imports": ["db"],
            "docstring": "Retrieves user from database",
            "summary": "Queries user database"
        }
    ]
