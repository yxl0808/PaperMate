import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_EMBED_MODEL = "deepseek-embed"

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"
CHROMA_COLLECTION = "papermate"
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
DB_PATH = PROJECT_ROOT / "data" / "learner.db"

# RAG 检索：返回最相似的 Top-K 个文本块，距离阈值用于过滤低质量结果（core/rag.py）
RAG_TOP_K = 5
RAG_DISTANCE_THRESHOLD = 0.5

# 对话上下文管理：token 预算上限、触发摘要的阈值（70%）、保留最近 N 轮不压缩（core/context_mgr.py）
TOKEN_BUDGET = 8192
TOKEN_SUMMARY_TRIGGER = 0.7
MAX_RECENT_ROUNDS = 5

# API 调用重试：最大重试次数和指数退避的基数秒数（core/embedder.py, core/llm_client.py）
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0

# 论文文本拆分：每块字符数、相邻块之间的重叠词数（agents/paper_parser.py）
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
# 用户输入最大长度限制（预留，尚未使用）
MAX_INPUT_LENGTH = 10000
