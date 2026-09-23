"""Knowledge management agent: list, filter, delete, export."""
from pathlib import Path
from core.memory import (
    list_papers, delete_paper, get_paper, export_notes_markdown,
    list_notes, update_reading_progress
)
from core.rag import delete_paper_chunks


def show_paper_list(keyword: str = ""):
    """Display paper list, optionally filtered by keyword."""
    papers = list_papers(limit=50)
    if keyword:
        kw = keyword.lower()
        papers = [p for p in papers
                  if kw in p["title"].lower()
                  or any(kw in k.lower() for k in p.get("keywords", []))
                  or kw in p.get("method", "").lower()]

    if not papers:
        print("没有找到匹配的论文。")
        return

    print(f"\n===== 论文库 ({len(papers)} 篇) =====")
    for i, p in enumerate(papers):
        authors_str = ", ".join(p["authors"][:2])
        progress = p["reading_progress"]
        last = p.get("last_read_at", "从未")[:10] if p.get("last_read_at") else "从未"
        print(f"  [{i+1}] {p['title']}")
        print(f"      作者: {authors_str} | 进度: {progress:.0%} | 最近: {last}")
        print(f"      ID: {p['paper_id']}")
    print()


def delete_paper_interactive(paper_id: str):
    """Delete a paper and all its data."""
    paper = get_paper(paper_id)
    if not paper:
        print("论文不存在。")
        return
    title = paper["title"]
    confirm = input(f"确认删除论文 [{title}] 及其所有笔记？(yes/no): ").strip().lower()
    if confirm == "yes":
        delete_paper_chunks(paper_id)
        delete_paper(paper_id)
        print(f"已删除: {title}")
    else:
        print("已取消。")


def show_notes(paper_id: str = ""):
    """Display notes."""
    notes = list_notes(paper_id=paper_id if paper_id else None, limit=50)
    if not notes:
        print("暂无笔记。")
        return
    print(f"\n===== 笔记 ({len(notes)} 条) =====")
    for note in notes:
        paper = get_paper(note["paper_id"])
        paper_title = paper["title"] if paper else "未知论文"
        print(f"  [{note['note_id']}] [{paper_title}] — {note['created_at'][:10]}")
        print(f"       {note['content'][:200]}")
        print()


def manage_knowledge_loop():
    """Interactive knowledge management loop."""
    print("\n===== 知识库管理 =====")
    print("命令: list [关键词], delete <paper_id>, notes [paper_id], export [paper_id], /menu")

    while True:
        cmd = input("\n📚 管理 > ").strip()
        if not cmd:
            continue
        if cmd == "/menu":
            break
        if cmd == "list" or cmd.startswith("list "):
            keyword = cmd[5:].strip() if len(cmd) > 4 else ""
            show_paper_list(keyword)
        elif cmd.startswith("delete "):
            paper_id = cmd[7:].strip()
            delete_paper_interactive(paper_id)
        elif cmd == "notes" or cmd.startswith("notes "):
            paper_id = cmd[6:].strip() if len(cmd) > 5 else ""
            show_notes(paper_id)
        elif cmd == "export" or cmd.startswith("export "):
            paper_id = cmd[7:].strip() if len(cmd) > 6 else ""
            md = export_notes_markdown(paper_id if paper_id else None)
            print(md)
            output_path = f"notes_export_{paper_id or 'all'}.md"
            Path(output_path).write_text(md, encoding="utf-8")
            print(f"\n已导出到: {output_path}")
        else:
            print("未知命令。可用: list, delete <id>, notes, export, /menu")
