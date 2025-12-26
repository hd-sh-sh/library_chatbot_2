# -*- coding: utf-8 -*-
import os
import sys
import hashlib
from pathlib import Path
import streamlit as st

# =========================================================
# 1. sqlite3 호환 (Chroma 안정화)
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
# 3. OpenAI API Key
# =========================================================
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# =========================================================
# 4. Streamlit 기본 설정
# =========================================================
st.set_page_config(page_title="PDF RAG 챗봇", page_icon="📚")
st.header("📚 PDF 기반 RAG 챗봇")

# =========================================================
# 5. PDF 업로드
# =========================================================
uploaded = st.file_uploader("📄 PDF 파일 업로드", type=["pdf"])

if not uploaded:
    st.info("PDF를 업로드하면 질문 입력창이 나타납니다.")
    st.stop()

# =========================================================
# 6. PDF 저장 + 고유 ID
# =========================================================
tmp_dir = Path(".streamlit_tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)

pdf_bytes = uploaded.getvalue()
file_id = hashlib.sha256(pdf_bytes).hexdigest()[:12]

pdf_path = tmp_dir / uploaded.name
pdf_path.write_bytes(pdf_bytes)

persist_dir = f"./chroma_db/{file_id}"

# =========================================================
# 7. PDF 로드
#    ❌ 캐시 사용 안 함 (Document 객체 때문)
# =========================================================
pages = PyPDFLoader(str(pdf_path)).load()

# =========================================================
# 8. VectorStore 생성 또는 로드
#    ❌ 캐시 사용 안 함 (핵심)
# =========================================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if os.path.isdir(persist_dir) and any(os.scandir(persist_dir)):
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
else:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    split_docs = splitter.split_documents(pages)

    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_dir
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# =========================================================
# 9. RAG Chain
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
         "context에 근거가 없으면 반드시 'PDF에서 근거를 찾지 못했습니다.'라고 답하라.\n\n"
         "{context}"
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini")

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    qa_chain
)

chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

# =========================================================
# 10. 채팅 UI (질문 입력창 정상)
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

            # 🔍 디버그 패널
            with st.expander("🔍 RAG 디버그"):
                ctx = response.get("context", [])
                st.write("검색된 문서 수:", len(ctx))
                st.write("PDF 경로:", pdf_path)
                st.write("DB 경로:", persist_dir)

                for i, doc in enumerate(ctx, 1):
                    st.markdown(f"### 문서 {i}")
                    st.code(doc.page_content[:400])
