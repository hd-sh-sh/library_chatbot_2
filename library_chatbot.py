# -*- coding: utf-8 -*-
import os
import sys
import hashlib
from pathlib import Path
import streamlit as st

# =====================================================
# 1. sqlite3 호환 (Chroma용)
# =====================================================
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

# =====================================================
# 2. LangChain / Chroma
# =====================================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

# =====================================================
# 3. API KEY
# =====================================================
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# =====================================================
# 4. Streamlit 기본 설정
# =====================================================
st.set_page_config(page_title="PDF RAG 챗봇", page_icon="📚")
st.title("📚 PDF 기반 RAG 챗봇 (디버그 포함)")

# =====================================================
# 5. 캐시 함수
# =====================================================
@st.cache_resource(show_spinner=False)
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

@st.cache_resource(show_spinner=False)
def build_vectorstore(docs, persist_dir):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_dir
    )

@st.cache_resource(show_spinner=False)
def load_or_create_vectorstore(docs, persist_dir):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.isdir(persist_dir) and any(os.scandir(persist_dir)):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

    return build_vectorstore(docs, persist_dir)

# =====================================================
# 6. PDF 업로드
# =====================================================
uploaded = st.file_uploader("📄 PDF 파일 업로드", type=["pdf"])

pdf_path = None
file_id = None
persist_dir = None

if uploaded:
    tmp_dir = Path(".streamlit_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    data = uploaded.getvalue()
    file_id = hashlib.sha256(data).hexdigest()[:12]

    pdf_path = str(tmp_dir / uploaded.name)
    with open(pdf_path, "wb") as f:
        f.write(data)

    persist_dir = f"./chroma_db/{fi
