# BERT-Based Named Entity Recognition (NER)

This project implements a Named Entity Recognition (NER) system using a pre-trained BERT model and PyTorch. It demonstrates a full pipeline from raw token-level data to model training, label alignment, and evaluation.

## 📌 Features

- Token classification using Hugging Face's `BertTokenizerFast` and `BertModel`
- Proper label alignment with subword tokenization
- PyTorch `Dataset` and `DataLoader` implementation
- Training and validation with `CrossEntropyLoss`, ignoring subword and special tokens
- Early stopping based on validation loss

## 📁 Dataset Structure

The original dataset is a token-level CSV with columns:

- `sentence_id`: identifier for each sentence
- `word`: each token (word)
- `ner_tag`: integer label (already encoded)

Grouped into sentence-level samples like:
```json
{
  "words": ["Barack", "Obama", "visited", "China"],
  "ner_tags": [1, 2, 0, 3]
}

tag2id = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-MISC": 7,
    "I-MISC": 8
}
