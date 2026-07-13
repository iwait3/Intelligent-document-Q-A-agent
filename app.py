"""
SmartDoc - 智能文档问答系统
启动方式：streamlit run app.py
"""
import streamlit as st
import os
import json
from agent import add_document, ask, list_documents, delete_document
import config

# ── 页面配置 ──────────────────────────────────────────────

st.set_page_config(
    page_title="SmartDoc 智能文档问答",
    page_icon="📚",
    layout="wide",
)

# ── 对话历史持久化 ────────────────────────────────────────

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


# ── 初始化 Session State ──────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

# ── 侧边栏：文档管理 ──────────────────────────────────────

with st.sidebar:
    st.header("📄 文档管理")

    # 上传文档
    uploaded_files = st.file_uploader(
        "上传文档",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="支持 PDF、TXT、Markdown、Word 文档",
    )

    if uploaded_files:
        for file in uploaded_files:
            save_path = os.path.join(config.DOCS_DIR, file.name)
            os.makedirs(config.DOCS_DIR, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

            with st.spinner(f"正在处理 {file.name}..."):
                try:
                    chunk_count = add_document(save_path, file.name)
                    st.success(f"✅ {file.name} → {chunk_count} 个文本块已入库")
                    st.session_state.doc_count += 1
                except Exception as e:
                    st.error(f"❌ 处理 {file.name} 失败：{e}")

    # 已上传文档列表
    st.divider()
    st.subheader("已上传文档")
    docs = list_documents()
    if docs:
        for doc_name in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"📎 {doc_name}")
            with col2:
                if st.button("🗑️", key=f"del_{doc_name}", help=f"删除 {doc_name}"):
                    delete_document(doc_name)
                    st.session_state.doc_count += 1
                    st.rerun()
    else:
        st.info("暂无文档，请先上传")

    # 清空对话
    st.divider()
    if st.button("🗑️ 清空对话记录"):
        st.session_state.chat_history = []
        save_chat_history([])
        st.rerun()

# ── 主区域：对话问答 ──────────────────────────────────────

st.title("📚 SmartDoc 智能文档问答")
st.caption("上传文档，用自然语言提问，AI 基于文档内容回答并标注来源")

# 显示历史对话
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📌 来源：{', '.join(msg['sources'])}")

# 输入框
if question := st.chat_input("输入你的问题..."):
    # 显示用户消息
    st.session_state.chat_history.append({"role": "user", "content": question})
    save_chat_history(st.session_state.chat_history)
    with st.chat_message("user"):
        st.markdown(question)

    # AI 回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                result = ask(question, st.session_state.chat_history[:-1])
                answer = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer = f"出错了：{e}"
                sources = []

        st.markdown(answer)
        if sources:
            st.caption(f"📌 来源：{', '.join(sources)}")

    # 记录到历史并持久化
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    save_chat_history(st.session_state.chat_history)
