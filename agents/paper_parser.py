"""Paper parsing agent: PDF -> Markdown -> chunks -> ChromaDB + SQLite."""
import uuid
import json
from pathlib import Path
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages

from config import PAPERS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from core.llm_client import chat
from core.rag import add_chunks, delete_paper_chunks
from core.memory import (
    add_paper, paper_exists_by_title_author, delete_paper, get_paper
)
from core.prompt_loader import load_prompt
from core.error_handler import log_error


class ParserState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    pdf_path: str
    paper_id: str
    status: str


def _extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using Docling."""
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        raise ImportError("请安装 docling: pip install docling")
    except Exception as e:
        raise RuntimeError(f"PDF解析失败: {e}")
# #except ImportError: — 专门捕获导入模块失败的错误（比如 docling 未安装）。
# except Exception as e: — 捕获所有其他类型的异常（Exception 是所有内置异常的基类），并将异常对象赋值给变量 e。

def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) > chunk_size and current:
            chunks.append(current.strip())
            words = current.split()
            overlap_text = " ".join(words[-overlap:]) if len(words) > overlap else current
            current = overlap_text + "\n\n" + para
        else:
            current = (current + "\n\n" + para).strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _extract_metadata_with_llm(text: str) -> dict:
    """Use DeepSeek to extract paper metadata from text."""
    prompt_text = load_prompt("parser_system")
    preview = text[:400]
    messages = [
        HumanMessage(content=f"{prompt_text}\n\n论文文本:\n{preview}")
    ]
    import asyncio
    response = asyncio.run(chat(messages))
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        pass
    return {"title": Path(getattr(_current_pdf, "path", "unknown")).stem,
            "authors": [], "abstract": "", "keywords": [], "method": "", "dataset": ""}


_current_pdf = None


def parse_pdf_node(state: ParserState) -> ParserState:
    """Main parsing node: extract text, metadata, chunk, embed, store."""
    pdf_path = state["pdf_path"]
    full_path = PAPERS_DIR / pdf_path if not Path(pdf_path).is_absolute() else Path(pdf_path)

    if not full_path.exists():
        return {
            **state,
            "status": "error",
            "messages": [AIMessage(content=f"文件不存在: {full_path}")]
        }

    try:
        # Step 1: PDF -> Markdown
        text = _extract_text_from_pdf(str(full_path))
        if not text.strip():#text.strip() 是 Python 字符串的一个方法，作用是：去除字符串首尾的空白字符（包括空格、制表符 \t、换行符 \n、回车符 \r 等），并返回一个新的字符串，原字符串不变。
            return {**state, "status": "error",
                    "messages": [AIMessage(content="PDF解析结果为空，请检查文件。")]}

        # Step 2: Extract metadata via LLM
        global _current_pdf
        _current_pdf = full_path
        metadata = _extract_metadata_with_llm(text)

        # Step 3: Duplicate check
        title = metadata.get("title", full_path.stem)
        authors = metadata.get("authors", [])
        first_author = authors[0] if authors else ""
        existing_id = paper_exists_by_title_author(title, first_author)
        if existing_id:
            return {
                **state,
                "status": "duplicate",
                "messages": [AIMessage(
                    content=f"该论文已导入 (ID: {existing_id})。\n标题: {title}\n是否要覆盖？输入 yes 覆盖，no 取消。"
                )],
                "paper_id": existing_id,
            }

        # Step 4: Chunk and embed
        chunks = _split_text(text)
        paper_id = add_paper(
            title=title,
            authors=authors,
            abstract=metadata.get("abstract", ""),
            keywords=metadata.get("keywords", []),
            method=metadata.get("method", ""),
            dataset=metadata.get("dataset", ""),
        )

        # 构建向量元数据
        metadatas = [
            {"chunk_id": f"{paper_id}_chunk_{i}",
             "section": "",
             "fig_id": "",
             "chunk_type": "text"}
            for i in range(len(chunks))
        ]
        add_chunks(texts=chunks, paper_id=paper_id, metadatas=metadatas)

        authors_str = ", ".join(authors[:3])
        return {
            **state,
            "status": "done",
            "paper_id": paper_id,
            "messages": [AIMessage(content=(
                f"论文导入成功！\n"
                f"  ID: {paper_id}\n"
                f"  标题: {title}\n"
                f"  作者: {authors_str}\n"
                f"  分块数: {len(chunks)}\n\n"
                f"输入 /menu 返回主菜单，或直接开始提问。"
            ))]
        }

    except Exception as e:
        error_msg = log_error(e, "parse_pdf_node")
        return {**state, "status": "error", "messages": [AIMessage(content=error_msg)]}


def handle_duplicate_node(state: ParserState) -> ParserState:
    """Handle duplicate paper: overwrite or cancel."""
    last_msg = state["messages"][-1].content if state["messages"] else ""
    if "yes" in last_msg.lower():
        old_paper_id = state.get("paper_id", "")
        if old_paper_id:
            delete_paper_chunks(old_paper_id)
            delete_paper(old_paper_id)
        return {**state, "status": "reparse"}
    return {**state, "status": "cancelled",
            "messages": [AIMessage(content="导入已取消。输入 /menu 返回主菜单。")]}


def route_after_parse(state: ParserState) -> str:
    if state["status"] == "duplicate":
        return "handle_duplicate"
    return END


def route_after_duplicate(state: ParserState) -> str:
    if state["status"] == "reparse":
        return "parse_pdf"
    return END


def build_parser_graph():
    graph = StateGraph(ParserState)
    graph.add_node("parse_pdf", parse_pdf_node)
    graph.add_node("handle_duplicate", handle_duplicate_node)
    graph.set_entry_point("parse_pdf")
    graph.add_conditional_edges("parse_pdf", route_after_parse, {
        "handle_duplicate": "handle_duplicate",
        END: END,
    })
    graph.add_conditional_edges("handle_duplicate", route_after_duplicate, {
        "parse_pdf": "parse_pdf",
        END: END,
    })
    return graph.compile()

 #                    ┌── "parse_pdf" ─→ END（无重复）
 # entry ─→ parse_pdf ─→ route_after_parse
 #                    └── "handle_duplicate" ─→ route_after_duplicate
 #                                        ├──→ "parse_pdf"（用户选覆盖，重新解析）
 #                                        └──→ END（取消）
