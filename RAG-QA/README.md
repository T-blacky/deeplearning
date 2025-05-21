# Retrieval-Augmented Generation QA with FAISS + BM25 Hybrid Retriever

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using multiple retrieval strategies that combines:
- **FAISS** for fast, semantic vector-based retrieval
- **BM25** for precise, lexical keyword-based re-ranking

A **T5-based generator** is then used to produce answers from the retrieved context.

---

## Features

✅ FAISS dense vector retrieval  
✅ BM25 reranking within FAISS shortlist  
✅ T5 model for question answering  
✅ Supports both ROUGE-L and BLEU evaluation metrics
✅ Reinforcement-style training 
✅ Easy-to-switch modular codebase

---
## Retrieval Modes

The system supports three retrieval strategies:
- `BM25`: purely keyword-based retrieval
- `FAISS`: semantic vector search using Sentence-BERT
- `Hybrid`: FAISS shortlist → BM25 reranking

---

## Training Strategy

The training loop uses a **reinforcement-style reward shaping approach**:
- The model generates answers using teacher-forcing.
- ROUGE-L or BLEU scores are computed between the generated answer and the ground truth.
- The loss is scaled by the normalized reward, aligning model behavior with ROUGE-based or BLEU objectives.

---

## Project Structure
rag_qa/
├── rag_qa.py # contains the full RAG-QA pipeline in one file: data loading, FAISS+BM25 hybrid retrieval, T5 training and evaluation
├── main.py # Main pipeline: loads data, trains model, runs eval
├── retriever.py # FAISS & BM25 index building
├── dataset.py # Dataset classes for training and hybrid retrieval
├── model.py # Model loading, training loop, and evaluation
├── corpus.csv # Context corpus for retrieval
├── train.csv # Training QA pairs with context_id or context
├── dev.csv # Validation set
├── test.csv # Test set
├── best_model.pth # Checkpoint (auto-saved if better ROUGE)
└── README.md # This file