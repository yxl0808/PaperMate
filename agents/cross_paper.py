"""Cross-paper comparison agent."""
import asyncio
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

from core.rag import search
from core.context_mgr import build_context
from core.llm_client import stream_chat
from core.memory import list_papers, get_paper


class CompareState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_paper_id: str


def _pick_candidates(current_id: str, limit: int = 3) -> list[dict]:
    all_papers = list_papers(limit=20)
    return [p for p in all_papers if p["paper_id"] != current_id][:limit]


async def compare_node(state: CompareState) -> CompareState:
    user_msg = state["messages"][-1].content if state["messages"] else ""
    current_id = state.get("current_paper_id", "")

    current_paper = get_paper(current_id)
    candidates = _pick_candidates(current_id)

    # Search current paper
    current_chunks = search(user_msg, paper_id=current_id, top_k=3)

    # Search candidate papers
    cross_chunks = []
    for c in candidates:
        chunks = search(user_msg, paper_id=c["paper_id"], top_k=2)
        cross_chunks.extend(chunks)

    # Current paper context
    current_title = current_paper["title"] if current_paper else "当前论文"

    messages = build_context(
        system_prompt_name="cross_paper",
        user_query=f"当前论文 [{current_title}]: {user_msg}\n\n请与以下论文进行对比。",
        rag_chunks=current_chunks,
        cross_paper_chunks=cross_chunks,
        conversation_history=None,
    )

    print()
    full_response = ""
    async for token in stream_chat(messages):
        print(token, end="", flush=True)
        full_response += token
    print()

    return {
        **state,
        "messages": list(state["messages"]) + [AIMessage(content=full_response)],
    }


def build_compare_graph() -> StateGraph:
    graph = StateGraph(CompareState)
    graph.add_node("compare", compare_node)
    graph.set_entry_point("compare")
    graph.add_edge("compare", END)
    return graph.compile()


async def run_compare_loop(current_paper_id: str = ""):
    """Interactive cross-paper comparison loop."""
    current_paper = get_paper(current_paper_id) if current_paper_id else None
    title = current_paper["title"] if current_paper else "论文库"

    print(f"===== 跨论文对比: {title} =====")
    if not current_paper_id:
        papers = list_papers(limit=30)
        if len(papers) < 2:
            print("需要至少导入 2 篇论文才能进行对比。")
            return
        print("已导入论文:")
        for i, p in enumerate(papers):
            print(f"  [{i+1}] {p['title']}")

    print("请描述你想对比的内容，/menu 返回主菜单\n")

    graph = build_compare_graph()

    while True:
        user_input = input("🔍 对比问题: ").strip()
        if not user_input:
            continue
        if user_input == "/menu":
            break

        state = {"messages": [HumanMessage(content=user_input)],
                 "current_paper_id": current_paper_id}
        result = await graph.ainvoke(state)
        print()
