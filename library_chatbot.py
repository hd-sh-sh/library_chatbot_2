# -*- coding: utf-8 -*-
import os
import sys
import hashlib
from pathlib import Path
import streamlit as st

# =========================================================
# 1. sqlite3 호환 (Chroma 오류 방지)
# =========================================================
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

# =========================================================
# 2. LangChain / Chroma
# =========================================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_chroma import Chroma

# =========================================================
# 3. API KEY
# =========================================================
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# =========================================================
# 4. Streamlit UI 기본
# =========================================================
st.set_page_config(page_title="PDF RAG 챗봇", page_ico
