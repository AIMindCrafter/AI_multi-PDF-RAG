"""
Multi-PDF RAG Assistant — Deployable on Hugging Face Spaces
Stack: LangChain · ChromaDB · LaMini-Flan-T5-248M · Gradio
"""

import os
import re
import shutil
import datetime
import torch
from typing import Any, List, Optional
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gradio as gr

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
UPLOAD_DIR = Path("uploaded_pdfs")
DB_DIR = Path("chroma_db")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_ID = "MBZUAI/LaMini-Flan-T5-248M"

UPLOAD_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS AT STARTUP
# ═══════════════════════════════════════════════════════════════════════════════
print("⏳ Loading Embedding Model...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

print("⏳ Loading LLM (LaMini-Flan-T5-248M)...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_ID)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_model = llm_model.to(device)
print(f"✅ LLM loaded on: {device}")


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM LLM WRAPPER (LangChain-compatible)
# ═══════════════════════════════════════════════════════════════════════════════
class FlanT5LLM(LLM):
    model: Any
    tokenizer: Any
    device: Any
    max_new_tokens: int = 256

    @property
    def _llm_type(self) -> str:
        return "flan-t5"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


llm = FlanT5LLM(model=llm_model, tokenizer=tokenizer, device=device)

# RAG Prompt Template
RAG_TEMPLATE = """Use the context below to answer the question.
If the answer is not in the context, say "I don't have enough information from the uploaded documents."
Keep the answer concise and accurate.

Context:
{context}

Question: {question}

Answer:"""

prompt_template = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════════
vector_db = None
rag_chain = None
retriever = None


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    """Remove noise from extracted PDF text."""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.replace('\x00', '')
    return text.strip()


def format_docs(docs):
    """Format retrieved documents into a single context string."""
    return "\n\n".join([d.page_content for d in docs])


def get_uploaded_files() -> List[str]:
    """List currently uploaded PDF files."""
    if UPLOAD_DIR.exists():
        return [f.name for f in UPLOAD_DIR.glob("*.pdf")]
    return []


def get_status_message() -> str:
    """Generate a status message for the upload panel."""
    files = get_uploaded_files()
    if files:
        file_list = "\n".join([f"- 📄 {f}" for f in files])
        return f"### ✅ Knowledge Base Active\n\n**{len(files)} PDF(s) loaded:**\n{file_list}"
    return "### 📭 No Documents Uploaded\n\nUpload PDFs and click **Process** to build your knowledge base."


# ═══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def ingest_pdfs(files) -> str:
    """
    Process uploaded PDFs: load → clean → chunk → embed → store in ChromaDB.
    """
    global vector_db, rag_chain, retriever

    if not files:
        return "⚠️ No files uploaded. Please upload at least one PDF."

    try:
        # Save uploaded files
        saved_files = []
        for file in files:
            filename = Path(file.name).name
            dest = UPLOAD_DIR / filename
            shutil.copy(file.name, dest)
            saved_files.append(filename)

        # Load all PDFs from upload directory
        all_documents = []
        for pdf_path in UPLOAD_DIR.glob("*.pdf"):
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            all_documents.extend(pages)

        if not all_documents:
            return "⚠️ No readable content found in the uploaded PDFs."

        # Clean text
        for doc in all_documents:
            doc.page_content = clean_text(doc.page_content)

        # Chunk
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(all_documents)

        # Enhance metadata
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        for chunk in chunks:
            chunk.metadata["ingestion_date"] = current_date
            chunk.metadata["category"] = "custom_knowledge_base"
            source = chunk.metadata.get('source', '')
            chunk.metadata["file_name"] = Path(source).name if source else "Unknown"

        # Remove old DB and rebuild
        if DB_DIR.exists():
            shutil.rmtree(DB_DIR)
        DB_DIR.mkdir(exist_ok=True)

        # Create vector store
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=str(DB_DIR)
        )

        # Build RAG chain
        retriever = vector_db.as_retriever(search_kwargs={"k": 6})
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        file_list = "\n".join([f"  📄 {f}" for f in saved_files])
        all_files = "\n".join([f"  📄 {f}" for f in get_uploaded_files()])

        return (
            f"✅ **Ingestion Complete!**\n\n"
            f"**New files added:**\n{file_list}\n\n"
            f"**Total pages loaded:** {len(all_documents)}\n"
            f"**Chunks created:** {len(chunks)}\n\n"
            f"**All files in knowledge base:**\n{all_files}\n\n"
            f"💬 You can now ask questions in the **Chat** tab!"
        )

    except Exception as e:
        return f"❌ **Error during ingestion:** {str(e)}"


def clear_knowledge_base() -> str:
    """Remove all uploaded PDFs and the vector database."""
    global vector_db, rag_chain, retriever

    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir(exist_ok=True)

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    DB_DIR.mkdir(exist_ok=True)

    vector_db = None
    rag_chain = None
    retriever = None

    return "🗑️ Knowledge base cleared. Upload new PDFs to get started."


def chat_with_rag(user_message: str, history: list) -> str:
    """Main chat function — handles greetings and RAG queries."""
    global rag_chain, retriever

    # Router: handle greetings
    greetings = ["hi", "hello", "hey", "how are you", "good morning",
                 "good evening", "who are you", "what can you do"]
    cleaned = user_message.lower().strip().replace("!", "").replace("?", "")

    if cleaned in greetings:
        return (
            "👋 Hello! I'm your **Multi-PDF RAG Assistant**.\n\n"
            "Upload your PDFs in the **📂 Upload & Manage** tab, then come back here to ask questions!\n\n"
            "I'll answer based on your documents and cite the source pages."
        )

    # Check if knowledge base is ready
    if rag_chain is None:
        return (
            "⚠️ **No knowledge base found.**\n\n"
            "Please go to the **📂 Upload & Manage** tab, upload your PDFs, "
            "and click **🚀 Process & Build Knowledge Base** first."
        )

    # RAG Execution
    try:
        answer = rag_chain.invoke(user_message)

        # Post-processing: citations
        source_docs = retriever.invoke(user_message)
        unique_sources = set()
        for doc in source_docs:
            file_name = doc.metadata.get('file_name', 'Unknown')
            page = doc.metadata.get('page', '?')
            display_page = int(page) + 1 if isinstance(page, (int, float)) else page
            unique_sources.add(f"📄 {file_name} — Page {display_page}")

        citations = ""
        if unique_sources:
            citations = "\n\n---\n**📌 Sources:**\n" + "\n".join(sorted(unique_sources))

        return answer + citations

    except Exception as e:
        return f"❌ Error processing query: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# TRY TO LOAD EXISTING DB ON STARTUP
# ═══════════════════════════════════════════════════════════════════════════════
try:
    if DB_DIR.exists() and any(DB_DIR.iterdir()):
        print("⏳ Loading existing vector database...")
        vector_db = Chroma(persist_directory=str(DB_DIR), embedding_function=embedding_model)
        retriever = vector_db.as_retriever(search_kwargs={"k": 6})
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )
        print("✅ Existing knowledge base loaded!")
except Exception as e:
    print(f"⚠️ No existing DB found, starting fresh: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO UI — PREMIUM DARK THEME
# ═══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
/* Global */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

/* Header banner */
#header-banner {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px;
    padding: 32px 24px;
    margin-bottom: 16px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

#header-banner h1 {
    color: #ffffff;
    font-size: 2em;
    margin-bottom: 4px;
    background: linear-gradient(90deg, #a78bfa, #818cf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

#header-banner p {
    color: rgba(255, 255, 255, 0.7);
    font-size: 1.05em;
    margin: 0;
}

/* Status card */
.status-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.05));
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 16px;
}

/* Upload area */
.upload-area {
    border: 2px dashed rgba(99, 102, 241, 0.4) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease;
}

.upload-area:hover {
    border-color: rgba(99, 102, 241, 0.8) !important;
    background: rgba(99, 102, 241, 0.05) !important;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}

.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
}

.danger-btn {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.2) !important;
}

/* Chat styling */
.chatbot {
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Tab styling */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 1em !important;
}

/* Footer */
#footer-info {
    text-align: center;
    padding: 12px;
    color: rgba(255, 255, 255, 0.4);
    font-size: 0.85em;
}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD THE GRADIO APP
# ═══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    title="Multi-PDF RAG Assistant",
) as demo:

    # Header
    gr.HTML("""
        <div id="header-banner">
            <h1>📚 Multi-PDF RAG Assistant</h1>
            <p>Upload your PDFs, build a knowledge base, and ask questions with AI-powered answers & citations</p>
            <p style="font-size: 0.85em; margin-top: 8px; color: rgba(255,255,255,0.45);">
                Powered by LangChain · ChromaDB · LaMini-Flan-T5-248M · BGE Embeddings
            </p>
        </div>
    """)

    with gr.Tabs():
        # ─── TAB 1: CHAT ─────────────────────────────────────────────────
        with gr.TabItem("💬 Chat", id="chat-tab"):
            chatbot_display = gr.Chatbot(
                height=520,
                placeholder="<div style='text-align:center; color: rgba(255,255,255,0.4); padding: 40px;'>"
                            "<h3>📚 Upload PDFs first, then ask me anything!</h3>"
                            "<p>Go to the Upload & Manage tab to get started.</p></div>",
                elem_classes=["chatbot"],
                label="Chat",
                type="messages",
            )
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ask a question about your documents...",
                    container=False,
                    scale=7,
                    show_label=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, elem_classes=["primary-btn"])
            clear_chat_btn = gr.Button("🗑️ Clear Chat", size="sm")

            # Chat state
            chat_history_state = gr.State([])

            def respond(user_message, history):
                """Handle user message, get RAG response, update chat."""
                if not user_message.strip():
                    return "", history, history

                # Add user message
                history = history + [{"role": "user", "content": user_message}]

                # Get bot response
                # Build simple history pairs for chat_with_rag
                simple_history = []
                for msg in history:
                    if msg["role"] == "user":
                        simple_history.append([msg["content"], None])
                    elif msg["role"] == "assistant" and simple_history:
                        simple_history[-1][1] = msg["content"]

                bot_response = chat_with_rag(user_message, simple_history)
                history = history + [{"role": "assistant", "content": bot_response}]

                return "", history, history

            def clear_chat():
                return [], []

            # Wire up chat
            chat_input.submit(
                fn=respond,
                inputs=[chat_input, chat_history_state],
                outputs=[chat_input, chatbot_display, chat_history_state],
            )
            send_btn.click(
                fn=respond,
                inputs=[chat_input, chat_history_state],
                outputs=[chat_input, chatbot_display, chat_history_state],
            )
            clear_chat_btn.click(
                fn=clear_chat,
                outputs=[chatbot_display, chat_history_state],
            )

        # ─── TAB 2: UPLOAD & MANAGE ──────────────────────────────────────
        with gr.TabItem("📂 Upload & Manage", id="upload-tab"):
            gr.Markdown("### 📤 Upload Your PDF Documents")
            gr.Markdown(
                "Upload one or more PDF files. The system will extract text, "
                "split into chunks, generate embeddings, and build a searchable knowledge base."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    file_upload = gr.File(
                        label="Drop your PDFs here",
                        file_count="multiple",
                        file_types=[".pdf"],
                        elem_classes=["upload-area"],
                    )
                    with gr.Row():
                        process_btn = gr.Button(
                            "🚀 Process & Build Knowledge Base",
                            variant="primary",
                            elem_classes=["primary-btn"],
                            scale=3,
                        )
                        clear_kb_btn = gr.Button(
                            "🗑️ Clear All",
                            variant="stop",
                            elem_classes=["danger-btn"],
                            scale=1,
                        )

                with gr.Column(scale=2):
                    status_output = gr.Markdown(
                        value=get_status_message(),
                        label="Status",
                        elem_classes=["status-card"],
                    )

            # Wire up buttons
            process_btn.click(
                fn=ingest_pdfs,
                inputs=[file_upload],
                outputs=[status_output],
            )
            clear_kb_btn.click(
                fn=clear_knowledge_base,
                inputs=[],
                outputs=[status_output],
            )

        # ─── TAB 3: ABOUT ────────────────────────────────────────────────
        with gr.TabItem("ℹ️ About", id="about-tab"):
            gr.Markdown("""
### 🏗️ Architecture

This application implements a complete **Retrieval-Augmented Generation (RAG)** pipeline:

```
PDF Upload → Text Extraction → Chunking → Embedding → ChromaDB Vector Store
                                                              ↓
User Query → Embedding → Semantic Search → Context Retrieval → LLM → Answer + Citations
```

### 🧩 Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Document Loader** | PyPDFLoader | Extract text from PDF pages |
| **Text Splitter** | RecursiveCharacterTextSplitter | Chunk text (1000 chars, 200 overlap) |
| **Embeddings** | BGE-small-en-v1.5 | Dense vector representations |
| **Vector Store** | ChromaDB | Persistent similarity search index |
| **LLM** | LaMini-Flan-T5-248M | Answer generation (runs locally, no API key needed) |
| **Framework** | LangChain + Gradio | RAG chain orchestration + Web UI |

### 📊 Key Features
- 🔒 **100% Local & Free** — No API keys required. LLM runs entirely on-device.
- 📄 **Multi-PDF Support** — Upload and query across multiple documents simultaneously.
- 📌 **Source Citations** — Every answer includes the source file name and page number.
- ♻️ **Persistent Storage** — ChromaDB persists your knowledge base between sessions.
- ⚡ **Lightweight** — The 248M parameter model runs smoothly on CPU.

### 🧪 Evaluation
This pipeline was evaluated using the **RAGAS** framework with GPT-4o-mini as the judge,
measuring Faithfulness, Response Relevancy, and Context Precision.
            """)

    # Footer
    gr.HTML('<div id="footer-info">Built with ❤️ using LangChain, ChromaDB & Gradio</div>')


# ═══════════════════════════════════════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
