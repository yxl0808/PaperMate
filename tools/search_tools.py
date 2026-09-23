from langchain_core.tools import tool
from core.rag import search
from core.memory import list_papers


@tool
def search_paper(query: str, paper_id: str = "") -> str:
    """Search within a paper's content. If paper_id is empty, searches all papers.
    Returns formatted search results.

    Args:
        query: The search query
        paper_id: Optional paper ID to limit search scope
    """
    pid = paper_id if paper_id else None
    chunks = search(query, paper_id=pid, top_k=3)
    if not chunks:
        return "未找到相关内容。"
    lines = []
    for i, chunk in enumerate(chunks):
        lines.append(f"[片段 {i+1}] (相关度 {1-chunk['distance']:.2f})")
        lines.append(chunk["document"][:500])
        lines.append("")
    return "\n".join(lines)


@tool
def list_my_papers() -> str:
    """List all papers in the user's library."""
    papers = list_papers(limit=30)
    if not papers:
        return "论文库为空。"
    lines = ["已导入论文列表:"]
    for i, p in enumerate(papers):
        authors_str = ", ".join(p["authors"][:3])
        progress = p["reading_progress"]
        lines.append(f"  [{i+1}] {p['title']} — {authors_str} (进度: {progress:.0%})")
    return "\n".join(lines)
