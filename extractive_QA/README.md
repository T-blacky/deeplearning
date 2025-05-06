# 🧠 Extractive Question Answering with BERT

This project implements an **Extractive QA system** using `BERT` and `PyTorch`, trained on a SQuAD-style dataset.

---

## 📚 What Is Extractive QA?

Extractive QA is the task of **finding a span of text** in a given passage (context) that answers a given question.

- Input: `(question, context)`
- Output: start and end **token positions** that represent the answer span within the context

Example:
```text
Context:  J.K. Rowling wrote the Harry Potter series.
Question: Who wrote Harry Potter?
Answer:   J.K. Rowling
```
---

## 🧠 Core Understanding

Here’s the conceptual foundation this project stands on:

### 1. BERT Encodes Everything
Input: `(question, context)` → `BERT` encodes each token into a vector.

### 2. Model Predicts a Span, Not a Sentence
don't generate answers — we locate the **start and end tokens** of the correct span.

### 3. Offset Mapping is Everything
link the char-level answer back to token positions using Hugging Face's `offset_mapping`. 

### 4. Loss = Location Accuracy
use `CrossEntropyLoss` on start and end logits (no softmax needed) — this is a span classification task, not generative.

---

## ⚠️ Note on Large Dataset

Due to GitHub’s file size limit, the file `extractive_QA/data_train.csv` is **not included** in this repository.
