import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from config import DB_PATH
from core.error_handler import DatabaseError


_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,
    abstract TEXT DEFAULT '',
    keywords TEXT DEFAULT '',
    method TEXT DEFAULT '',
    dataset TEXT DEFAULT '',
    imported_at TEXT NOT NULL,
    reading_progress REAL DEFAULT 0.0,
    last_read_at TEXT
);

CREATE TABLE IF NOT EXISTS reading_sessions (
    session_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    num_questions INTEGER DEFAULT 0,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

CREATE TABLE IF NOT EXISTS notes (
    note_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    labels TEXT DEFAULT '',
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

CREATE TABLE IF NOT EXISTS user_profile (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

#创建connect返回对象
def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
#     #mkdir(parents=True, exist_ok=True) 确保父目录存在：
# parents=True — 如果 data/ 的上层目录也不存在，会递归创建
# exist_ok=True — 如果目录已存在也不报错（不加这个参数的话，目录已存在会抛 FileExistsError）
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    #WAL 模式（line 53）：PRAGMA journal_mode=WAL，允许多读一写并发，不用锁
    return conn

#创建数据库表
def init_db():
    """Initialize database schema. Safe to call multiple times."""
    try:
        conn = _get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to initialize database: {e}")


def check_db_integrity() -> bool:
    """Check database integrity on startup. Returns True if healthy."""
    try:
        conn = _get_conn()
        result = conn.execute("PRAGMA integrity_check").fetchone()
        conn.close()
        return result[0] == "ok"
    except sqlite3.Error:
        return False


# --- Papers CRUD ---

def add_paper(title: str, authors: list[str], abstract: str = "",
              keywords: list[str] = None, method: str = "",
              dataset: str = "") -> str:
    paper_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    conn.execute(
        """INSERT INTO papers (paper_id, title, authors, abstract, keywords,
           method, dataset, imported_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (paper_id, title, json.dumps(authors, ensure_ascii=False), abstract,
         json.dumps(keywords or [], ensure_ascii=False), method, dataset, now)
    )
    # 数据库不能直接存列表
    # 所以用
    # json.dumps()
    # 转成字符串存进去
    conn.commit()
    conn.close()
    return paper_id


def get_paper(paper_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    d["authors"] = json.loads(d["authors"])
    d["keywords"] = json.loads(d.get("keywords", "[]"))
    return d


def list_papers(limit: int = 20, offset: int = 0) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM papers ORDER BY imported_at DESC LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    conn.close()
    papers = []
    for row in rows:
        d = dict(row)
        d["authors"] = json.loads(d["authors"])
        d["keywords"] = json.loads(d.get("keywords", "[]"))
        papers.append(d)
    return papers


def delete_paper(paper_id: str):
    conn = _get_conn()
    conn.execute("DELETE FROM papers WHERE paper_id=?", (paper_id,))
    conn.execute("DELETE FROM reading_sessions WHERE paper_id=?", (paper_id,))
    conn.execute("DELETE FROM notes WHERE paper_id=?", (paper_id,))
    conn.commit()
    conn.close()


def paper_exists_by_title_author(title: str, first_author: str) -> str | None:
    """Return paper_id if duplicate exists, else None."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT paper_id, authors FROM papers WHERE title=?",
        (title,)
    ).fetchall()
    conn.close()
    for row in rows:
        authors = json.loads(row["authors"])
        if authors and authors[0] == first_author:
            return row["paper_id"]
    return None


def update_reading_progress(paper_id: str, progress: float):
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    conn.execute(
        "UPDATE papers SET reading_progress=?, last_read_at=? WHERE paper_id=?",
        (min(progress, 1.0), now, paper_id)
    )
    conn.commit()
    conn.close()


# --- Notes ---

def add_note(paper_id: str, content: str, labels: list[str] = None) -> str:
    note_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO notes (note_id, paper_id, content, created_at, labels) VALUES (?,?,?,?,?)",
        (note_id, paper_id, content, now, json.dumps(labels or [], ensure_ascii=False))
    )
    conn.commit()
    conn.close()
    return note_id


def list_notes(paper_id: str = None, limit: int = 50) -> list[dict]:
    conn = _get_conn()
    if paper_id:
        rows = conn.execute(
            "SELECT * FROM notes WHERE paper_id=? ORDER BY created_at DESC LIMIT ?",
            (paper_id, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM notes ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()
    notes = []
    for row in rows:
        d = dict(row)
        d["labels"] = json.loads(d.get("labels", "[]"))
        notes.append(d)
    return notes


def export_notes_markdown(paper_id: str = None) -> str:
    """Export notes as markdown string."""
    notes = list_notes(paper_id=paper_id, limit=500)
    if not notes:
        return "暂无笔记。"
    lines = ["# PaperMate 笔记导出\n"]
    for note in notes:
        paper = get_paper(note["paper_id"])
        paper_title = paper["title"] if paper else "未知论文"
        lines.append(f"## [{paper_title}] — {note['created_at'][:10]}")
        lines.append(note["content"])
        lines.append("")
    return "\n".join(lines)


# --- User Profile ---

def set_profile(key: str, value: str):
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO user_profile (key, value) VALUES (?, ?)",
        (key, value)
    )
    conn.commit()
    conn.close()


def get_profile(key: str) -> str | None:
    conn = _get_conn()
    row = conn.execute("SELECT value FROM user_profile WHERE key=?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else None


def get_all_profile() -> dict[str, str]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM user_profile").fetchall()
    conn.close()
    return {row["key"]: row["value"] for row in rows}


def count_papers() -> int:
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) as cnt FROM papers").fetchone()
    conn.close()
    return row["cnt"]


def is_db_empty() -> bool:
    return count_papers() == 0


# --- Session ---

def start_session(paper_id: str) -> str:
    session_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO reading_sessions (session_id, paper_id, started_at) VALUES (?,?,?)",
        (session_id, paper_id, now)
    )
    conn.commit()
    conn.close()
    return session_id


def end_session(session_id: str, num_questions: int = 0):
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    conn.execute(
        "UPDATE reading_sessions SET ended_at=?, num_questions=? WHERE session_id=?",
        (now, num_questions, session_id)
    )
    conn.commit()
    conn.close()
