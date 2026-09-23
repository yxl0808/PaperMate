import sys
import traceback
from datetime import datetime


class PaperMateError(Exception):
    """Base exception for PaperMate."""
    pass


class LLMError(PaperMateError):
    """LLM API errors after retries exhausted."""
    pass


class EmbeddingError(PaperMateError):
    """Embedding API errors after retries exhausted."""
    pass


class DatabaseError(PaperMateError):
    """SQLite or ChromaDB integrity errors."""
    pass


class InputValidationError(PaperMateError):
    """Invalid user input."""
    pass


def log_error(error: Exception, context: str = "") -> str:
    """Log error to stderr and return a user-safe message."""
    timestamp = datetime.now().isoformat()
    tb = traceback.format_exc()
    print(f"\n[{timestamp}] ERROR [{context}]: {error}", file=sys.stderr)
    print(tb, file=sys.stderr)
    return f"[发生错误: {error}] 请重试或输入 /menu 返回主菜单。"


def safe_state() -> dict:
    """Return a safe empty state for graph recovery."""
    return {"messages": [], "error": None}
