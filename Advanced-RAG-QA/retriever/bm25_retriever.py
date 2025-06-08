from rank_bm25 import BM25Okapi
import json
import os
from typing import List,Tuple
from utils.helpers import tokenize

class BM25Retriever:
    def __init__(self,corpus_path:str):
        """
        Args:
            corpus_path (str): Path to JSONL file with "id" and "context"
        """
        self.corpus=self._load_corpus(corpus_path)
        self.tokenized_corpus=[tokenize(doc['context']) for doc in self.corpus]
        self.bm25=BM25Okapi(self.tokenized_corpus)
    
    def _load_corpus(self,path:str)->List[dict]:
        with open(path,'r',encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    
    def retrieve(self,query:str,top_k:int=5)->List[Tuple[str,str,float]]:
        """
        Args:
            query (str): input question
            top_k (int): number of docs to retrieve

        Returns:
            List of tuples: (doc_id, doc_context, bm25_score)
        """
        tokenized_query=tokenize(query)
        scores=self.bm25.get_scores(tokenized_query)
        top_k_indicies=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:top_k]
        
        return[
            (self.corpus[i]['id'],self.corpus[i]['context'],scores[i])
            for i in top_k_indicies
        ]