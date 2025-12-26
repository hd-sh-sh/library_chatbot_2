# -*- coding: utf-8 -*-
import os
import sys
import streamlit as st
from pathlib import Path

# =========================================================
# sqlite3 호환 (Chroma 오류 방지)
# =========================================================
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

# =========================================================
# LangChain imports
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
# PDF 로드 함수 (캐시 OK)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_and_split_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# =========================================================
# VectorStore 생성/로드 (캐시 사용 ❌)
# =========================================================
def build_or_load_vectorstore(docs, persist_directory="./chroma_db"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.isdir(persist_directory) and any(os.scandir(persist_directory)):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    split_docs = splitter.split_documents(docs)

    return Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_directory
    )

# =========================================================
# RAG Chain 초기화
# =========================================================
def initialize_chain(selected_model: str, pdf_path: str):
    pages = load_and_split_pdf(pdf_path)
    vectorstore = build_or_load_vectorstore(pages)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 질문 재구성 프롬프트
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "이전 대화를 참고해 독립적인 질문으로 바꿔라."),
            MessagesPlaceholder("history"),
            ("human", "{input}")
        ]
    )

    # ✅ PDF에 없는 내용은 반드시 모른다고 답하게 강제
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

    llm = ChatOpenAI(model=selected_model)

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="국립부경대 도서관 규정 Q&A", page_icon="📚")
st.header("국립부경대 도서관 규정 Q&A 챗봇 💬📚")

# 모델 선택
option = st.selectbox(
    "GPT 모델 선택",
    ("gpt-4o-mini", "gpt-3.5-turbo-0125")
)

# =========================================================
# PDF 업로드
# =========================================================
DEFAULT_PDF = "[챗봇프로그램및실습] 부경대학교 규정집.pdf"
uploaded = st.file_uploader("PDF를 업로드하거나 기본 PDF로 실행하세요", type=["pdf"])

pdf_path = None

if uploaded is not None:
    tmp_dir = Path(".streamlit_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / uploaded.name
    pdf_path.write_bytes(uploaded.getbuffer())
else:
    if os.path.exists(DEFAULT_PDF):
        pdf_path = DEFAULT_PDF

if not pdf_path:
    st.info("먼저 PDF를 업로드해주세요.")
    st.stop()

# =========================================================
# RAG 체인 + 채팅
# =========================================================
rag_chain = initialize_chain(option, str(pdf_path))
chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer"
)

# 기존 대화 표시
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# 질문 입력
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
                    st.markdown(doc.metadata.get("source", "source"),
                                help=doc.page_content)
