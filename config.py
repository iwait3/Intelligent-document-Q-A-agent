"""SmartDoc 配置管理"""
import os
from dotenv import load_dotenv

load_dotenv()

# HuggingFace 国内镜像（必须在 import huggingface 之前设置）
if os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")

# OpenRouter 配置（兼容 OpenAI 接口格式）
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# 免费模型列表（随时可能变动，可在 openrouter.ai/models 确认）：
#   google/gemma-3-27b-it:free
#   meta-llama/llama-4-scout:free
#   qwen/qwen3-32b:free
CHAT_MODEL = os.getenv("CHAT_MODEL", "google/gemma-3-27b-it:free")

# 本地 Embedding（完全免费，不消耗 API 额度）
# 使用 HuggingFace sentence-transformers，首次运行会自动下载模型约 460MB
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# 路径配置
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
FAISS_INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

# RAG 配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
