# -*- coding: utf-8 -*-
import os
import sys
import shutil
import streamlit as st
from pathlib import Path

# =========================================================
# sqlite3 호환 (Chroma 안정화)
# =========================================================
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

# =========================================================
# LangChain
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
# OpenAI API KEY
# =========================================================
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="PDF 추가학습 RAG 챗봇", page_icon="📚")
st.header("📚 PDF 추가 학습 RAG 챗봇")

# =========================================================
# 사이드바: 학습 방식
# =========================================================
mode = st.sidebar.radio(
    "📘 PDF 학습 방식",
    ("추가 학습 (누적)", "새로 학습 (기존 초기화)")
)

if mode == "새로 학습 (기존 초기화)":
    if st.sidebar.button("🧹 기존 학습 데이터 삭제"):
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        st.sidebar.success("기존 학습 데이터 삭제 완료")

# =========================================================
# PDF 업로드
# =========================================================
uploaded = st.file_uploader("📄 PDF 파일 업로드", type=["pdf"])

if not uploaded:
    st.info("PDF를 업로드하면 질문 입력창이 나타납니다.")
    st.stop()

tmp_dir = Path(".streamlit_tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)

pdf_path = tmp_dir / uploaded.name
pdf_path.write_bytes(uploaded.getbuffer())

# =========================================================
# PDF 로드
# =========================================================
pages = PyPDFLoader(str(pdf_path)).load()

# =========================================================
# VectorStore (추가 학습 핵심)
# =========================================================
persist_dir = "./chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
split_docs = splitter.split_documents(pages)

if os.path.isdir(persist_dir) and any(os.scandir(persist_dir)):
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    vectorstore.add_documents(split_docs)   # ✅ 추가 학습
else:
    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_dir
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# =========================================================
# RAG Chain
# =========================================================
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "이전 대화를 참고해 독립적인 질문으로 바꿔라."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]
)

qa_system_prompt = (
    "너는 PDF 문서 기반 질의응답 도우미이다.\n"
    "반드시 아래 문서 내용(context)에 근거해서만 답변해야 한다.\n"
    "문서에 없는 내용이거나 근거가 없으면\n"
    "반드시 '해당 내용은 제공된 PDF 문서에서 찾을 수 없습니다.'라고 답하라.\n"
    "절대 추측하거나 일반 지식으로 답하지 마라.\n"
    "대답은 한국어로 하고, 존댓말을 사용하라.\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini")

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer"
)

# =========================================================
# 채팅 UI
# =========================================================
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input("질문을 입력하세요"):
    st.chat_message("human").write(prompt)

    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            response = conversational_rag_chain.invoke(
                {"input": prompt},
                {"configurable": {"session_id": "any"}}
            )

            st.write(response.get("answer", ""))

            with st.expander("📄 참고 문서"):
                for doc in response.get("context", []):
                    st.markdown(
                        doc.metadata.get("source", "source"),
                        help=doc.page_content
                    )
