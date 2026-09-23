import time
from typing import AsyncIterator
from langchain_deepseek import ChatDeepSeek
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_CHAT_MODEL,
    MAX_RETRIES, RETRY_BASE_DELAY
)


def _create_llm(streaming: bool = True) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=DEEPSEEK_CHAT_MODEL,
        api_key=DEEPSEEK_API_KEY,
        api_base=DEEPSEEK_BASE_URL,
        streaming=streaming,
        temperature=0.7,
        max_tokens=2048,
    )


async def stream_chat(messages: list) -> AsyncIterator[str]:
    """Stream tokens from DeepSeek API with exponential backoff retry."""
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            llm = _create_llm(streaming=True)
            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield chunk.content
            return
        except Exception as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
    yield f"\n[服务暂时不可用，请稍后重试。错误: {last_exception}]"


async def chat(messages: list) -> str:
    """Non-streaming chat with retry."""
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            llm = _create_llm(streaming=False)
            response = await llm.ainvoke(messages)
            return response.content
        except Exception as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
    return f"[服务暂时不可用，请稍后重试。错误: {last_exception}]"
