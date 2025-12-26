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
st.set_page_config(page_title="PDF RAG 챗봇", page_icon="📚")
st.header("📚 PDF 기반 RAG 챗봇")

# =========================================================
# 5. 캐시 함수
# =========================================================
@st.cache_resource(show_spinner=False)
def load_pdf(path):
    return PyPDFLoader(path).load()

@st.cache_resource(show_spinner=False)
def build_or_load_vectorstore(docs, persist_dir):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.isdir(persist_dir) and any(os.scandir(persist_dir)):
        return Chroma(
            persist_directory=persist_dir,
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
        persist_directory=persist_dir
    )

# =========================================================
# 6. 모델 선택
# =========================================================
model_name = st.selectbox(
    "GPT 모델 선택",
    ("gpt-4o-mini", "gpt-3.5-turbo-0125")
)

# =========================================================
# 7. PDF 업로드
# =========================================================
uploaded = st.file_uploader("📄 PDF 업로드", type=["pdf"])

pdf_path = None
persist_dir = None

if uploaded:
    tmp_dir = Path(".streamlit_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    data = uploaded.getvalue()
    file_id = hashlib.sha256(data).hexdigest()[:12]

    pdf_path = tmp_dir / uploaded.name
    pdf_path.write_bytes(data)

    persist_dir = f"./chroma_db/{file_id}"
    st.success("PDF 업로드 완료")

if not pdf_path:
    st.info("PDF를 업로드하면 질문 입력창이 나타납니다.")
    st.stop()

# =========================================================
# 8. PDF → 벡터 DB
# =========================================================
pages = load_pdf(str(pdf_path))
vectorstore = build_or_load_vectorstore(pages, persist_dir)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# =========================================================
# 9. RAG 체인
# =========================================================
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "이전 대화를 참고해 독립적인 질문으로 바꿔라."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "너는 반드시 PDF에서 검색된 내용(context)만으로 답해야 한다.\n"
         "context에 근거가 없으면 'PDF에서 근거를 찾지 못했습니다.'라고만 답하라.\n\n"
         "{context}"
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]
)

llm = ChatOpenAI(model=model_name)

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
    output_messages_key="answer",
)

# =========================================================
# 10. 채팅 UI (질문 입력창 ✔)
# =========================================================
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input("질문을 입력하세요"):
    st.chat_message("human").write(prompt)

    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke(
                {"input": prompt},
                config
            )

            st.write(response.get("answer", ""))

            # ===============================
            # 🔍 디버그 패널
            # ===============================
            with st.expander("🔍 RAG 디버그"):
                ctx = response.get("context", [])
                st.write("검색된 문서 수:", len(ctx))
                st.write("PDF 경로:", pdf_path)
                st.write("DB 경로:", persist_dir)

                for i, doc in enumerate(ctx, 1):
                    st.markdown(f"### 문서 {i}")
                    st.code(doc.page_content[:400])
