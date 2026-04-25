# 📚 Multi-PDF RAG Research Assistant
**Stack:** LangChain · ChromaDB · Gemini API · Streamlit

## 📋 Project Overview
End-to-end RAG pipeline that ingests multiple PDFs, embeds chunks into ChromaDB, and answers natural language queries via the Gemini API with source citation. Features semantic search with cosine similarity reranking and session-based conversation history.

---

## 🗺️ Project Lifecycle — A to Z

### Phase 1: Environment Setup (Day 1)

```bash
pip install langchain langchain-google-genai chromadb \
            pypdf streamlit python-dotenv sentence-transformers \
            langchain-community
```

**Folder Structure:**
```
04_MultiPDF_RAG_Assistant/
├── data/
│   └── pdfs/                   # Upload your PDFs here
├── vectorstore/                # ChromaDB persistent storage
├── src/
│   ├── pdf_loader.py           # Load & chunk PDFs
│   ├── embedder.py             # Embed chunks → ChromaDB
│   ├── retriever.py            # Semantic search + reranking
│   └── qa_chain.py             # LangChain + Gemini QA chain
├── app.py                      # Streamlit UI
├── .env                        # GOOGLE_API_KEY=...
└── README.md
```

---

### Phase 2: PDF Loading & Chunking (Day 1–2)

```python
# src/pdf_loader.py
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_chunk_pdfs(pdf_dir: str, chunk_size=1000, chunk_overlap=200):
    all_docs = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, filename))
            pages = loader.load()
            print(f"Loaded {filename}: {len(pages)} pages")
            all_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks: {len(chunks)}")
    return chunks

# Test
chunks = load_and_chunk_pdfs("data/pdfs/")
print(chunks[0].page_content[:200])
print(chunks[0].metadata)          # source, page number
```

---

### Phase 3: Embedding & ChromaDB Vector Store (Day 2)

```python
# src/embedder.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "vectorstore/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # Free, fast

def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="pdf_rag"
    )
    vectorstore.persist()
    print(f"✅ ChromaDB built with {vectorstore._collection.count()} vectors")
    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="pdf_rag"
    )
```

---

### Phase 4: Retriever with Reranking (Day 2–3)

```python
# src/retriever.py
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def build_retriever(vectorstore, k=10, rerank_top_n=4):
    # Base retriever: cosine similarity search
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    # Cross-encoder reranker (optional but improves quality)
    reranker_model = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    compressor = CrossEncoderReranker(model=reranker_model, top_n=rerank_top_n)

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    return retriever
```

---

### Phase 5: QA Chain with Gemini API (Day 3)

```python
# src/qa_chain.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()  # Loads GOOGLE_API_KEY from .env

SYSTEM_PROMPT = """You are a precise research assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I don't know based on the provided documents."
Always cite the source document and page number.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer (with source citation):"""

def build_qa_chain(retriever):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate.from_template(SYSTEM_PROMPT)
        }
    )
    return qa_chain

def ask(qa_chain, question: str):
    result = qa_chain({"question": question})
    answer = result["answer"]
    sources = [
        f"📄 {doc.metadata.get('source','?')} — Page {doc.metadata.get('page','?')}"
        for doc in result["source_documents"]
    ]
    return answer, list(set(sources))
```

---

### Phase 6: Streamlit UI (Day 4)

```python
# app.py
import streamlit as st
import tempfile, os
from src.pdf_loader import load_and_chunk_pdfs
from src.embedder import build_vectorstore, load_vectorstore
from src.retriever import build_retriever
from src.qa_chain import build_qa_chain, ask

st.set_page_config(page_title="📚 Multi-PDF RAG Assistant", layout="wide")
st.title("📚 Multi-PDF RAG Research Assistant")
st.caption("Powered by LangChain · ChromaDB · Gemini API")

# Sidebar: Upload PDFs
with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type="pdf", accept_multiple_files=True
    )
    if st.button("🔧 Build Knowledge Base") and uploaded_files:
        with st.spinner("Processing PDFs..."):
            tmp_dir = "data/pdfs/"
            os.makedirs(tmp_dir, exist_ok=True)
            for f in uploaded_files:
                with open(os.path.join(tmp_dir, f.name), "wb") as out:
                    out.write(f.read())
            chunks = load_and_chunk_pdfs(tmp_dir)
            vectorstore = build_vectorstore(chunks)
            st.session_state.vectorstore = vectorstore
            st.success(f"✅ Indexed {len(chunks)} chunks from {len(uploaded_files)} PDFs!")

# Main chat
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state and "vectorstore" in st.session_state:
    retriever = build_retriever(st.session_state.vectorstore)
    st.session_state.qa_chain = build_qa_chain(retriever)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if "qa_chain" not in st.session_state:
            st.warning("Please upload and process PDFs first!")
        else:
            with st.spinner("Thinking..."):
                answer, sources = ask(st.session_state.qa_chain, prompt)
            st.markdown(answer)
            with st.expander("📌 Sources"):
                for src in sources:
                    st.write(src)
            full_response = answer + "\n\n**Sources:**\n" + "\n".join(sources)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
```

---

### Phase 7: Run Locally & Deploy (Day 4–5)

```bash
# Create .env file
echo "GOOGLE_API_KEY=your_gemini_api_key_here" > .env

# Run Streamlit app
streamlit run app.py

# Deploy to Streamlit Cloud
# 1. Push to GitHub
# 2. Go to share.streamlit.io
# 3. Connect repo → set GOOGLE_API_KEY in secrets
```

---

## ⏱️ Timeline Summary
| Phase | Task | Duration |
|-------|------|----------|
| 1 | Setup | 1 hr |
| 2 | PDF Loader + Chunking | 2 hrs |
| 3 | ChromaDB Vectorstore | 2 hrs |
| 4 | Retriever + Reranking | 2 hrs |
| 5 | Gemini QA Chain | 3 hrs |
| 6 | Streamlit UI | 4 hrs |
| 7 | Deploy | 2 hrs |
| **Total** | | **~16 hrs** |

---

## 🔗 Resources
- LangChain RAG: [python.langchain.com](https://python.langchain.com/docs/use_cases/question_answering/)
- Gemini API: [ai.google.dev](https://ai.google.dev)
- ChromaDB: [docs.trychroma.com](https://docs.trychroma.com)
- Streamlit: [docs.streamlit.io](https://docs.streamlit.io)
