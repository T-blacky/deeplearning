# Generative Question Answering with T5

This project implements a **generative question answering (QA)** system using the **T5 model** from Hugging Face Transformers. Unlike extractive QA (which selects a span from a passage), this model generates free-form answers in natural language, making it suitable for tasks like closed-book QA or abstractive summarization-based QA.

---

## Features

- Built on `T5ForConditionalGeneration` (Hugging Face)
- Fully custom PyTorch training loop (no Trainer API)
- Trains on user-provided `question + context → answer` data
- Includes test set evaluation and early stopping
- Local model checkpointing
- **Supports reinforcement learning (REINFORCE) with ROUGE-L and BLEU reward**
- Optionally switch between BLEU or ROUGE as reward signal
- Masked padding tokens and correctly computed log-probabilities
- Logits gathered using `decoder_input_ids` for proper alignment

---

## Reinforcement Learning Extension (REINFORCE)

After supervised training, this project fine-tunes the model using the **REINFORCE algorithm**:

- Model generates answers with `do_sample=True`
- Computes **ROUGE-L F1 score** against reference answers
- Calculates log-probabilities of generated tokens
- Uses the REINFORCE loss: `-log_prob × reward`
- Rewards are normalized per batch for training stability

This allows the model to **optimize directly for sequence-level evaluation metrics** like ROUGE or BLEU, even though these metrics are non-differentiable.

---