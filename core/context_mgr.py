"""Token budget management and three-tier memory injection."""
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import TOKEN_BUDGET, TOKEN_SUMMARY_TRIGGER, MAX_RECENT_ROUNDS
from core.memory import get_all_profile
from core.prompt_loader import load_prompt


def estimate_tokens(messages: list) -> int:
    """Rough token estimate: ~1.5 chars per token for Chinese, ~4 chars for English."""
    total = 0
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        total += len(content) // 2
    return total


def build_context(
    system_prompt_name: str,
    user_query: str,
    rag_chunks: list[dict] = None,
    conversation_history: list = None,
    cross_paper_chunks: list[dict] = None,
) -> list:
    """Assemble the full message list respecting token budget.

    Budget allocation:
    - System prompt (role + workflow + output constraint)
    - User profile (cold memory)
    - RAG chunks (warm memory)
    - Cross-paper chunks (if applicable)
    - Recent conversation rounds (hot memory)
    - History summary (compressed cold memory)
    - Generation buffer reserved
    """
    messages = []

    # Layer 1: System prompt (~300 tokens)
    system_text = load_prompt(system_prompt_name)
    if system_text:
        messages.append(SystemMessage(content=system_text))

    # Layer 2: User profile (~800 tokens)
    profile = get_all_profile()
    if profile:
        profile_lines = ["\n[用户画像]"]
        for k, v in profile.items():
            profile_lines.append(f"- {k}: {v}")
        profile_text = "\n".join(profile_lines)
        messages.append(SystemMessage(content=profile_text))

    # Layer 3: RAG chunks (~2000 tokens)
    if rag_chunks:
        rag_lines = ["\n[参考论文片段]"]
        for i, chunk in enumerate(rag_chunks):
            rag_lines.append(f"--- 片段 {i+1} (相关度: {1-chunk['distance']:.2f}) ---")
            rag_lines.append(chunk["document"])
        messages.append(SystemMessage(content="\n".join(rag_lines)))

    # Layer 4: Cross-paper chunks (~1500 tokens)
    if cross_paper_chunks:
        cp_lines = ["\n[其他论文参考片段]"]
        for i, chunk in enumerate(cross_paper_chunks):
            cp_lines.append(f"--- 片段 {i+1} ---")
            cp_lines.append(chunk["document"])
        messages.append(SystemMessage(content="\n".join(cp_lines)))

    # Layer 5: Recent conversation rounds
    if conversation_history:
        recent = conversation_history[-(MAX_RECENT_ROUNDS * 2):]
        messages.extend(recent)

    # Layer 6: User query
    messages.append(HumanMessage(content=user_query))

    # Check budget and compress if needed
    current_tokens = estimate_tokens(messages)
    if current_tokens > TOKEN_BUDGET * TOKEN_SUMMARY_TRIGGER and conversation_history:
        old_messages = conversation_history[:-(MAX_RECENT_ROUNDS * 2)]
        if old_messages:
            summary = _summarize_history(old_messages)
            messages.insert(len(messages) - MAX_RECENT_ROUNDS * 2 - 1,
                          SystemMessage(content=f"[历史摘要]\n{summary}"))

    return messages


def _summarize_history(messages: list) -> str:
    """Compress old messages into a short summary placeholder."""
    human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    if not human_msgs:
        return "无历史对话。"
    topics = [m.content[:80] for m in human_msgs[-10:]]
    return "用户之前讨论了: " + "; ".join(topics)
