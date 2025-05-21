import faiss
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

def build_faiss_index(contexts, encoder):
    embeddings = encoder.encode(contexts, convert_to_numpy=True, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def build_bm25_index(contexts):
    tokenized = [word_tokenize(c.lower()) for c in contexts]
    return BM25Okapi(tokenized), tokenized