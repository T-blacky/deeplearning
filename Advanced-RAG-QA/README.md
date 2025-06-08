# Advanced RAG-QA: Retrieval-Augmented Question Answering System

This project implements a modular, trainable Retrieval-Augmented Generation (RAG) pipeline combining:

- **Hybrid Retrieval** (BM25 + FAISS) with a learnable fusion gate
- **Optional Query Rewriting** (T5-small)
- **Answer Generation** using T5-base
- **Evaluation** with BLEU & ROUGE-L
- **Trainable Components**: neural fusion gate in HybridRetriever

---

## Features

- **Trainable Fusion**: Learn how to balance BM25 vs FAISS dynamically per document
- **Modularized Components**: Easily plug or swap out retrievers and generators
- **Query Rewriting**: Rewrite user queries using T5-small for improved retrieval
- **Evaluation Ready**: Includes ROUGE and BLEU scoring for automated testing

---