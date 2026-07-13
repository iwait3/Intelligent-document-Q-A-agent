# SmartDoc 智能文档问答系统 — 从零到一开发全记录

> 一个基于 RAG 的智能文档问答系统，上传文档即可用自然语言提问，AI 基于文档内容回答并标注来源。
> 技术栈：LangChain + OpenRouter 免费模型 + 本地 Embedding + FAISS + Streamlit

---

## 一、项目效果

用户在 Web 界面上传文档（PDF/TXT/MD/DOCX），系统自动将文档分块、向量化并存入本地向量数据库。用户提问后，系统检索相关文档片段，交给大模型生成回答，并标注信息来源。对话历史持久化存储，重启不丢失。

```
用户上传文档 → 文档分块 → 向量化(Embedding) → 存入 FAISS
                                              ↓
用户提问 → 语义检索相关片段 → 拼接 Prompt → 大模型生成回答 → 显示来源
```

---

## 二、环境准备

### 2.1 系统要求

- **操作系统**：Windows 10/11、macOS、Linux 均可
- **Python**：3.10 ~ 3.12（⚠️ 3.13 存在部分依赖兼容问题，不推荐）
- **网络**：需要访问 OpenRouter API 和 HuggingFace 模型下载

### 2.2 安装 Python

前往 [python.org](https://www.python.org/downloads/) 下载安装，安装时**勾选 "Add Python to PATH"**。

验证安装：

```bash
python --version   # 应显示 3.10+ 版本
pip --version
```

### 2.3 创建项目目录

```bash
mkdir smartdoc
cd smartdoc
mkdir -p data/docs
```

---

## 三、安装依赖

创建 `requirements.txt`：

```txt
# LLM & Agent 框架
langchain>=0.3
langchain-openai>=0.3
langchain-community>=0.3
langchain-text-splitters>=0.3
langchain-core>=0.3

# 本地向量数据库
faiss-cpu>=1.7

# 本地 Embedding 模型
sentence-transformers>=3.0

# 文档解析
pypdf>=4.0
python-docx>=1.0
docx2txt>=0.8

# Web 界面
streamlit>=1.30

# 环境变量管理
python-dotenv>=1.0
```

安装：

```bash
pip install -r requirements.txt
```

> 💡 **国内用户加速**：使用清华镜像源可大幅提升下载速度
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
> ```
> `sentence-transformers` 会同时安装 PyTorch（约 2GB），国内镜像可节省大量等待时间。

---

## 四、获取 API Key

本项目使用 **OpenRouter** 连接免费大模型，零成本运行。

1. 注册 [openrouter.ai](https://openrouter.ai/)
2. 进入 [Keys 页面](https://openrouter.ai/keys) 创建 API Key
3. 复制 Key（格式：`sk-or-v1-...`）

> OpenRouter 免费模型列表（随时变动，以官网为准）：
> - `google/gemma-3-27b-it:free`
> - `meta-llama/llama-4-scout:free`
> - `qwen/qwen3-32b:free`

---

## 五、项目结构

```
smartdoc/
├── app.py              # Streamlit 前端界面
├── agent.py            # RAG Agent 核心逻辑
├── config.py           # 配置管理
├── requirements.txt    # Python 依赖
├── .env                # 环境变量（API Key，不提交到 Git）
├── .env.example        # 环境变量模板
├── .gitignore
├── Dockerfile          # Docker 部署配置
├── docker-compose.yml
└── data/               # 运行时数据
    ├── docs/           # 上传的原始文档
    ├── faiss_index/    # FAISS 向量数据库
    ├── doc_meta.json   # 文档元数据
    └── chat_history.json  # 对话历史
```

---

## 六、逐文件解析

### 6.1 config.py — 配置管理

集中管理所有配置项，从 `.env` 文件读取敏感信息（API Key）。

```python
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
CHAT_MODEL = os.getenv("CHAT_MODEL", "google/gemma-3-27b-it:free")

# 本地 Embedding（免费，不消耗 API 额度）
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# 路径配置
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
FAISS_INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

# RAG 配置
CHUNK_SIZE = 500        # 文本分块最大字符数
CHUNK_OVERLAP = 50      # 相邻块重叠字符数
TOP_K = 4               # 检索返回的文档块数量
```

**要点**：
- `HF_ENDPOINT` 镜像设置必须在 HuggingFace 相关库 import 之前生效
- OpenRouter 兼容 OpenAI 接口格式，所以可以用 `langchain-openai` 的 `ChatOpenAI`
- `BAAI/bge-small-zh-v1.5` 是中文 Embedding 模型，体积小、速度快，适合入门

### 6.2 agent.py — RAG Agent 核心逻辑

这是整个项目的心脏，包含四个核心能力：

#### ① LLM 和 Embedding 初始化

```python
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_llm():
    """通过 OpenRouter 调用免费大模型"""
    return ChatOpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        model=config.CHAT_MODEL,
        temperature=0.3,
    )

# 全局单例，避免重复加载模型
_embeddings_instance = None

def get_embeddings():
    """本地 HuggingFace Embedding（免费，首次运行自动下载模型）"""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=config.LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_instance
```

**设计决策**：
- 使用 OpenRouter 而非直接调用 OpenAI → 零成本
- Embedding 使用本地模型而非 API → 无调用费用，无网络延迟
- Embedding 用全局单例 → 避免每次请求重新加载模型（加载需要几秒）

#### ② 文档加载与处理

```python
import pypdf
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, Docx2txtLoader

def _load_pdf(file_path: str) -> list[Document]:
    """用 pypdf 直接加载 PDF"""
    docs = []
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(page_content=text, metadata={"page": i + 1}))
    return docs

# 注意：TextLoader 必须指定 encoding="utf-8"，否则 Windows 下默认 GBK 会报错
LOADER_MAP = {
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": lambda p: TextLoader(p, encoding="utf-8"),
    ".docx": lambda p: Docx2txtLoader(p),
}

def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        docs = _load_pdf(file_path)
        if not docs:
            raise ValueError("该 PDF 无法提取文字（可能是扫描件/图片 PDF）")
        return docs
    loader_factory = LOADER_MAP.get(ext)
    if not loader_factory:
        raise ValueError(f"不支持的文件格式: {ext}")
    return loader_factory(file_path).load()
```

**踩坑记录**：
- ❌ `PyPDFLoader`（langchain-community）与 pypdf 6.x 不兼容，会报 `not enough values to unpack` → 改用 pypdf 直接读取
- ❌ `TextLoader` 在 Windows 下默认 GBK 编码，读取 UTF-8 文件会报 `UnicodeDecodeError` → 显式指定 `encoding="utf-8"`

#### ③ 文档入库（分块 + 向量化）

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def add_document(file_path: str, file_name: str) -> int:
    """加载文档 → 分块 → 向量化 → 存入 FAISS，返回入库块数"""
    docs = load_document(file_path)
    for doc in docs:
        doc.metadata["source_file"] = file_name

    # 分块：按语义边界切分，保留重叠避免断句
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # 向量化入库
    embeddings = get_embeddings()
    existing_vs = get_vectorstore()
    if existing_vs:
        existing_vs.add_documents(chunks)
        save_vectorstore(existing_vs)
    else:
        new_vs = FAISS.from_documents(chunks, embeddings)
        save_vectorstore(new_vs)

    # 记录元数据
    meta = _load_meta()
    if file_name not in meta["documents"]:
        meta["documents"].append(file_name)
        _save_meta(meta)

    return len(chunks)
```

**关键参数**：
- `chunk_size=500`：每块最大 500 字符。太大则检索不精准，太小则上下文断裂
- `chunk_overlap=50`：相邻块重叠 50 字符，避免关键信息被切断
- `separators`：优先按段落、换行、句号切分，保持语义完整

#### ④ RAG 问答

```python
def ask(question: str, chat_history: list = None) -> dict:
    """检索相关文档 → 拼接 Prompt → 调用 LLM → 返回回答+来源"""
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return {"answer": "请先上传文档。", "sources": []}

    # 1. 语义检索
    retriever = vectorstore.as_retriever(
        search_type="mmr",        # 最大边际相关性，兼顾相关性和多样性
        search_kwargs={"k": 4},
    )
    docs = retriever.invoke(question)

    # 2. 拼接上下文
    context_text = "\n\n---\n\n".join(
        f"[来源: {doc.metadata.get('source_file', '未知')}]\n{doc.page_content}"
        for doc in docs
    )

    # 3. 构建 Prompt
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(context=context_text)},
    ]
    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    # 4. 调用 LLM
    response = get_llm().invoke(messages)

    # 5. 提取来源
    sources = list({doc.metadata.get("source_file", "未知") for doc in docs})

    return {"answer": response.content, "sources": sources}
```

**设计决策**：
- 使用 `mmr`（Maximum Marginal Relevance）检索策略而非简单相似度 → 避免返回内容重复的片段
- 手动组合检索+生成链，而非使用 `langchain.chains` → 避免版本兼容问题，代码也更清晰
- Prompt 中明确要求"只根据文档内容回答"和"无法回答时明确说明" → 减少幻觉

### 6.3 app.py — Streamlit 前端

提供 Web 界面，功能包括：
- 左侧边栏：文档上传、文档列表管理、清空对话
- 主区域：对话式问答，显示来源标注
- 对话历史持久化（存入 `data/chat_history.json`）

```python
import json

CHAT_HISTORY_FILE = os.path.join(config.DATA_DIR, "chat_history.json")

def load_chat_history() -> list:
    """从磁盘加载对话历史"""
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat_history(history: list):
    """保存对话历史到磁盘"""
    os.makedirs(os.path.dirname(CHAT_HISTORY_FILE), exist_ok=True)
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# 初始化时从磁盘恢复历史
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# 每条消息后自动保存
st.session_state.chat_history.append({"role": "user", "content": question})
save_chat_history(st.session_state.chat_history)
```

---

## 七、环境变量配置

在项目根目录创建 `.env` 文件：

```bash
# OpenRouter API Key
OPENROUTER_API_KEY=sk-or-v1-你的key

# 免费大模型
CHAT_MODEL=google/gemma-3-27b-it:free

# 本地 Embedding 模型
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

# HuggingFace 国内镜像（加速模型下载）
HF_ENDPOINT=https://hf-mirror.com
```

> ⚠️ `.env` 文件包含 API Key，**绝不要提交到 Git**。已在 `.gitignore` 中排除。

---

## 八、启动项目

```bash
cd smartdoc
HF_ENDPOINT=https://hf-mirror.com streamlit run app.py --server.port 8501 --server.headless true
```

浏览器访问 `http://localhost:8501` 即可使用。

> `--server.headless true` 跳过首次启动的邮箱输入提示。
> `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像加速模型下载（首次运行需下载约 100MB）。

---

## 九、Docker 部署

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/data/docs

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### docker-compose.yml

```yaml
version: "3.8"

services:
  smartdoc:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - HF_ENDPOINT=https://hf-mirror.com
    volumes:
      - smartdoc_data:/app/data
    restart: unless-stopped

volumes:
  smartdoc_data:
```

### 一键部署

```bash
# 构建并启动
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

---

## 十、踩坑记录与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `No module named 'langchain.chains'` | LangChain 1.x 移除了 `chains` 模块 | 手动组合检索+生成链，不依赖 `langchain.chains` |
| `not enough values to unpack (expected 2, got 1)` | `PyPDFLoader` 与 pypdf 6.x 不兼容 | 用 pypdf 直接读取 PDF，绕过 PyPDFLoader |
| `UnicodeDecodeError: 'gbk' codec` | TextLoader 在 Windows 下默认 GBK 编码 | 指定 `encoding="utf-8"` |
| PDF 加载返回 0 页 | 扫描件/图片 PDF 无可提取文字 | 提示用户上传含可选文字的 PDF |
| 模型下载极慢/超时 | 国内访问 HuggingFace 受限 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| `pip install` 超时 | PyPI 官方源国内慢 | 使用清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 对话重启后丢失 | `st.session_state` 仅内存存储 | 对话历史持久化到 `chat_history.json` |
| Streamlit 首次启动卡住 | 要求输入邮箱 | 添加 `--server.headless true` 参数 |

---

## 十一、RAG 核心原理速览

```
┌──────────────────────────────────────────────────┐
│                    离线阶段                        │
│                                                    │
│  文档 ──→ 分块 ──→ Embedding ──→ 存入向量数据库     │
│  (PDF)   (500字/块)  (bge-small)    (FAISS)       │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│                    在线阶段                        │
│                                                    │
│  用户问题 ──→ Embedding ──→ 语义检索 ──→ 拼接Prompt │
│                                ↓                    │
│  回答 ←── LLM生成 ←── 带上下文的Prompt              │
└──────────────────────────────────────────────────┘
```

**为什么需要 RAG？**
- 纯 LLM 知识冻结在训练数据中，不知道你的私有文档内容
- RAG 让 LLM "阅读"你的文档后再回答，大幅减少幻觉
- 本质：检索（找到相关内容）+ 生成（基于内容回答）

**为什么分块而不是把整个文档塞给 LLM？**
- LLM 有上下文窗口限制（通常 8K-128K token）
- 整篇文档检索不精准，小块语义匹配更准确
- 减少 Token 消耗，降低成本

---

## 十二、后续优化方向

- [ ] **支持 OCR**：集成 PaddleOCR/Tesseract，处理扫描件 PDF
- [ ] **混合检索**：向量检索 + 关键词检索（BM25），提升召回率
- [ ] **重排序**：用 Reranker 模型对检索结果二次排序
- [ ] **流式输出**：LLM 逐字返回，体验更流畅
- [ ] **多用户支持**：添加用户认证，对话隔离
- [ ] **对话导出**：支持导出为 Markdown/PDF

---

## 许可

MIT License — 随意使用、修改、发布。
