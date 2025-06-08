import faiss
import numpy as np
import json
from typing import List,Tuple
from sentence_transformers import SentenceTransformer
import os 

class FAISSRetriever:
    def __init__(self,corpus_path:str,index_path:str=None,model:str='./all-MiniLM-L6-v2'):
        """
        Args:
            corpus_path (str): Path to JSONL file with "id" and "content"
            index_path (str): Optional path to prebuilt FAISS index
            model (str): SentenceTransformer model name
        """
        self.model=SentenceTransformer(model)
        self.corpus=self._load_corpus(corpus_path)
        self.doc_texts=[doc['context'] for doc in self.corpus]
        self.doc_ids=[doc['id'] for doc in self.corpus]

        self.embedding=self.model.encode(self.doc_texts,convert_to_numpy=True,show_progress_bar=True)
        self.index=faiss.IndexFlatIP(self.embedding.shape[1])
        self.index.add(self.embedding)
    
    def _load_corpus(self,path:str)->List[dict]:
        with open(path,'r',encoding='utf-8') as f:
            return [json.load(line) for line in f]
    
    def retrieve(self,query:str,top_k:int=5)->List[Tuple[str,str,float]]:
        """
        Args:
            query (str): input question
            top_k (int): number of docs to retrieve

        Returns:
            List of tuples: (doc_id, doc_content, score)
        """
        query_embedding=self.model.encode([query],convert_to_numpy=True)
        query_embedding=query_embedding/np.linalg.norm(query_embedding)

        D,I=self.index.search(query_embedding,top_k)

        return[
            (self.doc_ids[i],self.doc_texts[i],float(D[0][j]))
            for j,i in enumerate(I[0])
        ]