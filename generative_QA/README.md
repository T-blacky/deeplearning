# Generative Question Answering with T5

This project implements a **generative question answering (QA)** system using the **T5 model** from Hugging Face Transformers. Unlike extractive QA (which selects a span from a passage), this model generates free-form answers in natural language, making it suitable for tasks like closed-book QA or abstractive summarization-based QA.

---

##  Features

- Based on `T5ForConditionalGeneration`
- Fully custom training loop (no Trainer API)
- Trains on user-provided `question + context → answer` data
- Supports early stopping and test set evaluation
- Loads and saves model from local path
- PyTorch implementation with standard `Dataset` and `DataLoader`

---