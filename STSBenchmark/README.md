# Semantic Textual Similarity (STS) with BERT + Attention Pooling

This project implements a semantic textual similarity (STS) system using BERT embeddings and a custom attention pooling mechanism.  
The goal is to predict a real-valued similarity score between pairs of sentences.

---

## Project Overview

- **Task**: Predict the semantic similarity score between two sentences (regression task).
- **Dataset**: STS-Benchmark (train/dev/test splits).
- **Model**:
  - Base Encoder: `bert-base-uncased`
  - Multiple pooling strategies over token embeddings:
    - CLS pooling
    - Mean pooling
    - Attention pooling
  - Regression head (fully connected layer)
- **Training Techniques**:
  - Early stopping based on dev set MSE
  - Save best model checkpoint during training
- **Evaluation**:
  - Final performance reported on test set.

---

## Main Features

- Fine-tuning a pretrained BERT encoder.
- Implemented and compared three pooling strategies:
  - **CLS pooling**: Use the [CLS] token's hidden state.
  - **Mean pooling**: Average the hidden states of all valid tokens (ignoring PAD).
  - **Attention pooling**: Learn attention scores to weight token importance dynamically.
- Masked attention scores to avoid counting padding tokens.
- Full training workflow:
  - Train on train set
  - Validate on dev set after each epoch
  - Early stop if no improvement
  - Save best model automatically
  - Evaluate final test performance
- Clear train/dev/test separation to prevent data leakage.

---