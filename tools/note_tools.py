from langchain_core.tools import tool
from core.memory import add_note, list_notes, export_notes_markdown


@tool
def take_note(content: str, paper_id: str, labels: str = "") -> str:
    """Take a note associated with a paper.

    Args:
        content: The note content
        paper_id: The paper ID to associate the note with
        labels: Comma-separated labels/tags
    """
    label_list = [l.strip() for l in labels.split(",") if l.strip()] if labels else []
    note_id = add_note(paper_id, content, label_list)
    return f"笔记已保存 (ID: {note_id})"


@tool
def view_notes(paper_id: str = "") -> str:
    """View notes, optionally filtered by paper.

    Args:
        paper_id: Optional paper ID to filter notes
    """
    pid = paper_id if paper_id else None
    notes = list_notes(paper_id=pid, limit=20)
    if not notes:
        return "暂无笔记。"
    lines = ["笔记列表:"]
    for note in notes:
        lines.append(f"  [{note['note_id']}] {note['content'][:100]} — {note['created_at'][:10]}")
    return "\n".join(lines)


@tool
def export_notes(paper_id: str = "") -> str:
    """Export notes as markdown.

    Args:
        paper_id: Optional paper ID to limit export scope
    """
    pid = paper_id if paper_id else None
    return export_notes_markdown(pid)
