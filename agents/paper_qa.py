"""Single-paper Q&A agent with RAG, quality check, and steering engineering."""
import asyncio
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

from core.rag import search_with_quality_check
from core.context_mgr import build_context
from core.llm_client import stream_chat
from core.memory import get_paper, update_reading_progress, start_session, end_session
from core.error_handler import log_error


QA_SYSTEM_PROMPT = "qa_system"

OFF_TOPIC_KEYWORDS = ["写代码", "写文章", "代写", "翻译", "做作业", "写论文",
                      "generate code", "write essay", "write paper", "do homework"]


class QAState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    paper_id: str
    session_id: str
    question_count: int
    mode: str  # IDLE, RETRIEVING, GENERATING, DONE
    _rag_chunks: list
    _is_low_quality: bool


def _is_off_topic(user_input: str) -> bool:
    """Simple keyword-based off-topic detection."""
    return any(kw in user_input.lower() for kw in OFF_TOPIC_KEYWORDS)
#any() 是一个内置函数，只要传入的迭代器中有一个元素为 True，它就返回 True

def qa_retrieve(state: QAState) -> QAState:
    """RAG retrieval node."""
    user_msg = state["messages"][-1].content if state["messages"] else ""
    if not user_msg:
        return {**state, "mode": "DONE"}

    chunks, is_low_quality = search_with_quality_check(
        query=user_msg,
        paper_id=state["paper_id"],
    )
    return {
        **state,
        "mode": "RETRIEVING",
        "messages": list(state["messages"]) + ([
            AIMessage(content=f"_检索到 {len(chunks)} 个相关片段_\n")
        ] if chunks else []),
        "_rag_chunks": chunks,
        "_is_low_quality": is_low_quality,
    }


async def qa_generate(state: QAState) -> QAState:
    """Generate answer with streaming."""
    user_msg = state["messages"][-1].content if state["messages"] else ""
    rag_chunks = state.get("_rag_chunks", [])
    is_low_quality = state.get("_is_low_quality", False)

    # Off-topic check
    if _is_off_topic(user_msg):
        return {
            **state,
            "mode": "DONE",
            "messages": list(state["messages"]) + [
                AIMessage(content="[偏离检测] 我是论文阅读助手，请提出与当前论文相关的问题。输入 /menu 返回主菜单。")
            ]
        }

    # Build context
    conversation = list(state["messages"])
    paper = get_paper(state["paper_id"])
    paper_context = f"当前论文: {paper['title']}" if paper else ""

    messages = build_context(
        system_prompt_name=QA_SYSTEM_PROMPT,
        user_query=f"{paper_context}\n\n用户问题: {user_msg}",
        rag_chunks=rag_chunks,
        conversation_history=conversation[:-1] if len(conversation) > 1 else None,
    )

    # Stream response
    print()  # newline before streaming
    full_response = ""
    async for token in stream_chat(messages):
        print(token, end="", flush=True)
        full_response += token
    print()  # newline after streaming

    if is_low_quality:
        full_response = "[检索结果有限，以下回答可能不完整]\n" + full_response

    new_count = state.get("question_count", 0) + 1
    return {
        **state,
        "mode": "DONE",
        "question_count": new_count,
        "messages": list(state["messages"]) + [AIMessage(content=full_response)],
    }


def should_continue(state: QAState) -> str:
    user_msg = state["messages"][-1].content if state["messages"] else ""
    if user_msg.strip() == "/menu":
        return END
    return "generate"

# entry ──→ retrieve ──→ generate ──→ END

def build_qa_graph() -> StateGraph:
    graph = StateGraph(QAState)
    graph.add_node("retrieve", qa_retrieve)
    graph.add_node("generate", qa_generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


async def run_qa_loop(paper_id: str):
    """Interactive Q&A loop for a specific paper."""
    paper = get_paper(paper_id)
    if not paper:
        print("论文不存在。")
        return

    session_id = start_session(paper_id)
    print(f"\n===== 论文问答: {paper['title']} =====")
    print("输入问题开始，/menu 返回主菜单，/note 记笔记\n")

    graph = build_qa_graph()
    state = {"messages": [], "paper_id": paper_id, "session_id": session_id,
             "question_count": 0, "mode": "IDLE"}
    question_count = 0

    while True:
        user_input = input("❓ ").strip()
        if not user_input:
            continue
        if user_input == "/menu":
            break

        state["messages"] = list(state["messages"]) + [HumanMessage(content=user_input)]
        state["mode"] = "IDLE"
        result = await graph.ainvoke(state)
        state["messages"] = result["messages"]
        question_count = result.get("question_count", question_count)

    end_session(session_id, num_questions=question_count)
    update_reading_progress(paper_id, min(question_count * 0.05, 1.0))
