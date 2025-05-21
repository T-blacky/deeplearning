from retriever import build_faiss_index, build_bm25_index
from dataset import TrainData, TestData
from model import train, test, load_model
from transformers import T5Tokenizer
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# === Config ===
CORPUS_PATH = 'corpus.csv'
TRAIN_PATH = 'train.csv'
DEV_PATH = 'dev.csv'
TEST_PATH = 'test.csv'
MODEL_NAME = 't5-base'
TOP_K = 20
TOP_N = 3

# === Load corpus and build retrievers ===
corpus_df = pd.read_csv(CORPUS_PATH)
contexts = corpus_df['context'].tolist()

encoder = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index, _ = build_faiss_index(contexts, encoder)
bm25, tokenized_corpus = build_bm25_index(contexts)

# === Load tokenizer ===
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# === Load data ===
train_df = pd.read_csv(TRAIN_PATH)
dev_df = pd.read_csv(DEV_PATH)
test_df = pd.read_csv(TEST_PATH)

train_data = TrainData(train_df, tokenizer)
dev_data = TestData(dev_df, tokenizer, encoder, faiss_index, contexts, bm25, tokenized_corpus, top_k=TOP_K, top_n=TOP_N)
test_data = TestData(test_df, tokenizer, encoder, faiss_index, contexts, bm25, tokenized_corpus, top_k=TOP_K, top_n=TOP_N)

# === Train and evaluate ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(MODEL_NAME, device)

train(model, train_data, dev_data, tokenizer, device)
rouge_score = test(model, test_data, tokenizer, device)
print("Test ROUGE-L:", rouge_score)
