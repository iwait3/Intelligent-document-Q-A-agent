"""
SmartDoc RAG Agent 核心逻辑
流程：文档加载 → 分块 → 向量化（本地Embedding） → 检索 → 生成回答（OpenRouter免费模型）
"""
import os
import json

# 必须最先加载 config，因为里面会设置 HF_ENDPOINT 镜像
# 而 HuggingFaceEmbeddings 导入时会读取这个环境变量
import config

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pypdf


def get_llm():
    """获取 OpenRouter 免费大模型"""
    return ChatOpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        model=config.CHAT_MODEL,
        temperature=0.3,
    )


# 本地 Embedding 实例（全局单例，避免重复加载模型）
_embeddings_instance = None

def get_embeddings():
    """获取本地 HuggingFace Embedding（免费，首次运行自动下载模型）"""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=config.LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_instance


def get_vectorstore() -> FAISS:
    """加载已有的 FAISS 向量数据库，如果不存在则返回 None"""
    index_path = config.FAISS_INDEX_DIR
    if os.path.exists(os.path.join(index_path, "index.faiss")):
        return FAISS.load_local(
            index_path,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return None


def save_vectorstore(vectorstore: FAISS):
    """保存向量数据库到磁盘"""
    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(config.FAISS_INDEX_DIR)


# ── 文档元数据管理 ─────────────────────────────────────────

METADATA_FILE = os.path.join(config.DATA_DIR, "doc_meta.json")


def _load_meta() -> dict:
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents": []}


def _save_meta(meta: dict):
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ── 文档加载与处理 ────────────────────────────────────────

def _load_pdf(file_path: str) -> list[Document]:
    """用 pypdf 直接加载 PDF，绕过 LangChain PyPDFLoader 的兼容性问题"""
    docs = []
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"page": i + 1},
                ))
    return docs


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
            raise ValueError("该 PDF 无法提取文字（可能是扫描件/图片 PDF），请上传含可选文字的 PDF")
        return docs
    loader_factory = LOADER_MAP.get(ext)
    if not loader_factory:
        raise ValueError(f"不支持的文件格式: {ext}")
    loader = loader_factory(file_path)
    return loader.load()


def add_document(file_path: str, file_name: str) -> int:
    """加载文档、分块、向量化并存入数据库。返回入库的文本块数量。"""
    docs = load_document(file_path)
    for doc in docs:
        doc.metadata["source_file"] = file_name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    existing_vs = get_vectorstore()
    if existing_vs:
        existing_vs.add_documents(chunks)
        save_vectorstore(existing_vs)
    else:
        new_vs = FAISS.from_documents(chunks, embeddings)
        save_vectorstore(new_vs)

    meta = _load_meta()
    if file_name not in meta["documents"]:
        meta["documents"].append(file_name)
        _save_meta(meta)

    return len(chunks)


def list_documents() -> list[str]:
    return _load_meta()["documents"]


def delete_document(file_name: str) -> bool:
    meta = _load_meta()
    if file_name in meta["documents"]:
        meta["documents"].remove(file_name)
        _save_meta(meta)
    file_path = os.path.join(config.DOCS_DIR, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    _rebuild_vectorstore()
    return True


def _rebuild_vectorstore():
    meta = _load_meta()
    all_chunks = []
    for doc_name in meta["documents"]:
        doc_path = os.path.join(config.DOCS_DIR, doc_name)
        if not os.path.exists(doc_path):
            continue
        docs = load_document(doc_path)
        for doc in docs:
            doc.metadata["source_file"] = doc_name
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
        )
        all_chunks.extend(splitter.split_documents(docs))

    import shutil
    if os.path.exists(config.FAISS_INDEX_DIR):
        shutil.rmtree(config.FAISS_INDEX_DIR)
    if all_chunks:
        new_vs = FAISS.from_documents(all_chunks, get_embeddings())
        save_vectorstore(new_vs)


# ── RAG 问答（手动组合，不依赖 langchain.chains）────────────

SYSTEM_TEMPLATE = """你是一个专业的文档问答助手。请基于以下检索到的文档内容来回答用户的问题。

要求：
1. 只根据提供的文档内容回答，不要编造信息
2. 如果文档中没有相关内容，请明确说明"根据已有文档，我无法回答这个问题"
3. 回答要清晰、有条理
4. 在回答末尾标注信息来源的文档名

检索到的文档内容：
{context}
"""


def ask(question: str, chat_history: list = None) -> dict:
    """执行 RAG 问答，返回 {"answer": ..., "sources": [...]}"""
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return {"answer": "请先上传文档，我才能回答问题。", "sources": []}

    # 1. 检索相关文档
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.TOP_K},
    )
    docs = retriever.invoke(question)

    # 2. 拼接上下文
    context_text = "\n\n---\n\n".join(
        f"[来源: {doc.metadata.get('source_file', '未知')}]\n{doc.page_content}"
        for doc in docs
    )

    # 3. 构建消息列表
    messages = []
    messages.append({"role": "system", "content": SYSTEM_TEMPLATE.format(context=context_text)})

    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    # 4. 调用 LLM
    llm = get_llm()
    response = llm.invoke(messages)

    # 5. 提取来源
    sources = []
    for doc in docs:
        source = doc.metadata.get("source_file", "未知")
        if source not in sources:
            sources.append(source)

    return {
        "answer": response.content,
        "sources": sources,
    }
