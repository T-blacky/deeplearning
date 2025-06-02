# Retrieval-Augmented QA with Local T5 and LangChain

This project implements a lightweight **retrieval-augmented question answering (RAG)** system using:
- Local models (T5 for generation, MiniLM for embeddings)
- FAISS vector store for efficient retrieval
- LangChain utilities for document loading and preprocessing
- A **custom QA function** (without using LangChain’s built-in `RetrievalQA`)

It loads a PDF file, indexes its contents into a vector database, and answers natural language questions based on the document content — all running locally on GPU.

---

## Features

- ✅ Load and parse PDF documents using LangChain
- ✅ Split documents into overlapping chunks
- ✅ Compute semantic embeddings using Sentence Transformers (MiniLM)
- ✅ Store and search documents using FAISS
- ✅ Answer natural language questions using local T5 model
- ✅ Fully GPU-enabled (both embedding and generation)

---

## Key Design Choice

Unlike many LangChain projects, this implementation does **not** use the built-in `RetrievalQA` chain.  
Instead, it defines a custom function to:
1. Retrieve relevant chunks
2. Construct a context-aware prompt
3. Run inference with Hugging Face’s `pipeline`

This gives full control over the prompt format and model behavior.

---