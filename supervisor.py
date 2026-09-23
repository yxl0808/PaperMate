"""Supervisor router: classify user intent and dispatch to sub-agent."""
import asyncio
from core.llm_client import chat
from core.prompt_loader import load_prompt
from langchain_core.messages import HumanMessage


INTENT_MAP = {
    "PARSE": "parser",
    "QA": "qa",
    "COMPARE": "compare",
    "MANAGE": "manage",
}


async def classify_intent(user_input: str) -> str:
    """Classify user intent using lightweight LLM call.
    Returns one of: parser, qa, compare, manage, unknown
    """
    prompt = load_prompt("supervisor", user_input=user_input)
    response = await chat([HumanMessage(content=prompt)])
    label = response.strip().upper()
    return INTENT_MAP.get(label, "unknown")


def classify_intent_rule(user_input: str) -> str:
    """Rule-based intent classification (zero token, fallback for smart mode).
    Returns one of: parser, qa, compare, manage, unknown
    """
    text = user_input.lower()
    # Parser triggers
    if any(kw in text for kw in ["导入", "上传", "添加论文", "import", "upload", "add paper"]):
        return "parser"
    # Compare triggers
    if any(kw in text for kw in ["对比", "比较", "异同", "compare", "diff", "versus", " vs "]):
        return "compare"
    # Manage triggers
    if any(kw in text for kw in ["列表", "笔记", "删除", "导出", "进度", "list", "note", "delete", "export"]):
        return "manage"
    # Default to QA
    return "qa"
