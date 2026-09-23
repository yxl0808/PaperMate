"""PaperMate CLI entry point."""
import asyncio
import sys
from pathlib import Path

from config import DEEPSEEK_API_KEY
from core.memory import init_db, check_db_integrity, is_db_empty, list_papers, get_paper
from core.rag import check_chroma_integrity, is_chroma_empty
from core.error_handler import log_error
from onboarding import run_onboarding
from supervisor import classify_intent_rule, classify_intent
from agents.paper_parser import build_parser_graph
from agents.paper_qa import run_qa_loop
from agents.cross_paper import run_compare_loop
from agents.knowledge_mgr import manage_knowledge_loop, show_paper_list

MENU = """
===== PaperMate =====
1. 导入新论文
2. 继续阅读（选择已导入论文）
3. 跨论文搜索
4. 知识库管理
5. 智能模式
6. 退出

选择 > """


def check_system_health() -> bool:
    """Startup integrity checks."""
    try:
        init_db()
        if not check_db_integrity():
            print("⚠ 数据库完整性检查失败，将重新初始化。")
    except Exception as e:
        log_error(e, "db_init")
        print("⚠ 数据库初始化失败，部分功能可能不可用。")

    if not check_chroma_integrity():
        print("⚠ 向量库不可用，论文检索功能可能受限。")
    return True


async def import_paper():
    """Interactive paper import flow."""
    pdf_path = input("输入论文PDF路径: ").strip()
    if not pdf_path:
        print("路径不能为空。")
        return

    graph = build_parser_graph()
    state = {"messages": [], "pdf_path": pdf_path, "paper_id": "", "status": "pending"}
    result = await graph.ainvoke(state)

    for msg in result.get("messages", []):
        print(msg.content)


async def continue_reading():
    """Select a paper and enter Q&A mode."""
    papers = list_papers()
    if not papers:
        print("论文库为空，请先导入论文。")
        return

    show_paper_list()
    choice = input("输入论文编号或 paper_id (输入 0 取消): ").strip()
    if choice == "0":
        return
#两种输入方式：序号输入、标题输入
    paper_id = None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(papers):
            paper_id = papers[idx]["paper_id"]
    else:
        paper_id = choice

    if not paper_id:
        print("无效选择。")
        return

    await run_qa_loop(paper_id)


async def smart_mode():
    """Smart mode: auto-route user queries."""
    papers = list_papers()
    paper_id = papers[0]["paper_id"] if papers else ""
    if not paper_id:
        print("请先导入论文。")
        return

    print("\n===== 智能模式 =====")
    print("直接说出你的需求，我会自动判断并执行。")
    print("/menu 返回主菜单\n")

    while True:
        user_input = input("💡 ").strip()
        if not user_input:
            continue
        if user_input == "/menu":
            break

        intent = await classify_intent(user_input)
        print(f"  → 路由到: {intent}")

        if intent == "qa":
            await run_qa_loop(paper_id)
        elif intent == "compare":
            await run_compare_loop(paper_id)
        elif intent == "manage":
            manage_knowledge_loop()
        elif intent == "parser":
            await import_paper()
        else:
            await run_qa_loop(paper_id)


async def main():
    """Main CLI loop."""
    check_system_health()

    if not run_onboarding():
        return

    while True:
        print(MENU)
        choice = input().strip()

        if choice == "1":
            await import_paper()
        elif choice == "2":
            await continue_reading()
        elif choice == "3":
            papers = list_papers()
            paper_id = papers[0]["paper_id"] if papers else ""
            await run_compare_loop(paper_id)
        elif choice == "4":
            manage_knowledge_loop()
        elif choice == "5":
            await smart_mode()
        elif choice == "6":
            print("再见！")
            break
        else:
            print("无效选项，请输入 1-6。")


if __name__ == "__main__":
    asyncio.run(main())
