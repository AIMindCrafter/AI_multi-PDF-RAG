---
title: Multi-PDF RAG Assistant
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 📚 Multi-PDF RAG Assistant

A fully local, API-key-free **Retrieval-Augmented Generation** system that lets you upload PDFs and ask questions about them with source citations.

## 🏗️ Architecture

```
PDF Upload → Text Extraction → Chunking → Embedding → ChromaDB Vector Store
                                                              ↓
User Query → Embedding → Semantic Search → Context Retrieval → LLM → Answer + Citations
```

## 🧩 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Embeddings** | BGE-small-en-v1.5 |
| **Vector Store** | ChromaDB |
| **LLM** | LaMini-Flan-T5-248M (248M params, runs on CPU) |
| **Framework** | LangChain + Gradio |

## 📊 Features

- 🔒 **100% Local & Free** — No API keys needed
- 📄 **Multi-PDF Support** — Query across multiple documents
- 📌 **Source Citations** — Answers include file name + page number
- ⚡ **Lightweight** — Runs on CPU with a 248M parameter model

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:7860

## 🧪 Evaluation

Evaluated using the **RAGAS** framework with GPT-4o-mini as judge, measuring:
- Faithfulness
- Response Relevancy
- Context Precision
