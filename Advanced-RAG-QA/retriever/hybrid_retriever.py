import torch
import torch.nn as nn
import numpy as np
from retriever.bm25_retriever import BM25Retriever
from retriever.faiss_retriever import FAISSRetriever
from typing import List,Tuple

class HybridRetriever(nn.Module):
    def __init__(self,corpus_path:str,hidden_dim:int=8):
        super().__init__()
        self.bm25=BM25Retriever(corpus_path)
        self.faiss=FAISSRetriever(corpus_path)

        # MLP for computing gate: input = [bm25_score, faiss_score]
        self.gate_net=nn.Sequential(
            nn.Linear(2,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),
            nn.Sigmoid()
        )
    
    def retrieve(self,query:str,top_k:int=5)->List[Tuple[str,str,float]]:
        bm25_result=self.bm25.retrieve(query,top_k)
        faiss_result=self.faiss.retrieve(query,top_k)

        # score dict
        bm25_dict={doc_id:score for doc_id,_,score in bm25_result}
        faiss_dict={doc_id:score for doc_id,_,score in faiss_result}
        all_ids=list(set(bm25_dict.keys())|set(faiss_dict.keys()))

        # normalize
        bm25_norm=self._normalize_scores(bm25_dict,all_ids)
        faiss_norm=self._normalize_scores(faiss_dict,all_ids)

        fused_scores={}
        for doc_id in all_ids:
            bm25_score=torch.tensor(bm25_norm[doc_id],dtype=torch.float32)
            faiss_score=torch.tensor(faiss_norm[doc_id],dtype=torch.float32)

            combined = torch.tensor([bm25_score.item(), faiss_score.item()]).unsqueeze(0)
            gate=self.gate_net(combined).squeeze(0).item()
            fused_score=gate*bm25_score.item()+(1-gate)*faiss_score.item()

            fused_scores[doc_id]=fused_score
        
        top_k_ids=sorted(fused_scores.keys(),key=lambda k:fused_scores[k],reverse=True)[:top_k]

        return[
            (doc_id,self._get_doc_context(doc_id),fused_scores[doc_id])
            for doc_id in top_k_ids
        ]
    
    def _normalize_scores(self,score_dict,all_ids):
        scores=np.array([score_dict.get(i,0.0) for i in all_ids])
        if scores.max()==scores.min():
            normed=np.zeros_like(scores)
        else:
            normd=(scores-scores.min())/(scores.max()-scores.min())
        
        return {
            doc_id:float(s)
            for doc_id,s in zip(all_ids,normed)
        }
    
    def _get_doc_content(self, doc_id: str) -> str:
        for doc in self.faiss.corpus:
            if doc['id'] == doc_id:
                return doc['context']
        return ""
