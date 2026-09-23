"""Cross-paper comparison agent."""
import asyncio
from typing import TypedDict, Annotated, Sequence, Callable
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
    candidate_paper_ids: list[str] | None


def _pick_candidates(
    current_id: str,
    limit: int = 3,
    candidate_paper_ids: list[str] | None = None,
) -> list[dict]:
    if candidate_paper_ids is not None:
        candidates = []
        seen = set()
        for paper_id in candidate_paper_ids:
            if paper_id and paper_id != current_id and paper_id not in seen:
                candidates.append({"paper_id": paper_id})
                seen.add(paper_id)
        return candidates

    all_papers = list_papers(limit=20)
    return [p for p in all_papers if p["paper_id"] != current_id][:limit]


async def compare_node(state: CompareState) -> CompareState:
    user_msg = state["messages"][-1].content if state["messages"] else ""
    current_id = state.get("current_paper_id", "")
    candidate_paper_ids = state.get("candidate_paper_ids")

    current_paper = get_paper(current_id)
    candidates = _pick_candidates(
        current_id,
        candidate_paper_ids=candidate_paper_ids,
    )

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


def _resolve_paper_choice(choice: str, papers: list[dict]) -> str | None:
    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(papers):
            return papers[index]["paper_id"]
        return None

    for paper in papers:
        if paper["paper_id"] == choice:
            return choice
    return None


def select_compare_papers(
    papers: list[dict],
    input_fn: Callable[[str], str] = input,
) -> tuple[str, list[str]] | None:
    """Interactively select the current paper and comparison candidates."""
    if len(papers) < 2:
        print("至少需要导入 2 篇论文才能进行对比。")
        return None

    print("已导入论文：")
    for index, paper in enumerate(papers, start=1):
        print(f"  [{index}] {paper['title']} (ID: {paper['paper_id']})")

    current_choice = input_fn("当前论文编号或 paper_id (输入 0 取消): ").strip()
    if current_choice == "0":
        return None
    current_id = _resolve_paper_choice(current_choice, papers)
    if current_id is None:
        print("当前论文选择无效。")
        return None

    candidate_choice = input_fn(
        "候选论文编号或 paper_id（多个用逗号分隔，输入 0 取消）: "
    ).strip()
    if candidate_choice == "0":
        return None

    candidate_ids = []
    seen = set()
    for value in candidate_choice.split(","):
        value = value.strip()
        candidate_id = _resolve_paper_choice(value, papers)
        if candidate_id is None:
            print(f"候选论文选择无效: {value}")
            return None
        if candidate_id != current_id and candidate_id not in seen:
            candidate_ids.append(candidate_id)
            seen.add(candidate_id)

    if not candidate_ids:
        print("至少选择一篇不同于当前论文的候选论文。")
        return None
    return current_id, candidate_ids


async def run_compare_loop(
    current_paper_id: str = "",
    candidate_paper_ids: list[str] | None = None,
):
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
                 "current_paper_id": current_paper_id,
                 "candidate_paper_ids": candidate_paper_ids}
        result = await graph.ainvoke(state)
        print()
