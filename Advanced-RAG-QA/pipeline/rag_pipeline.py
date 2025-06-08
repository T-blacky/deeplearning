from retriever.hybrid_retriever import HybridRetriever
from retriever.query_rewrite import QueryRewriter
from generator.generator import Generator

class RAGPipeline:
    def __init__(self,corpus_path:str,use_query_rewriting:bool=True,model:str='../t5'):
        self.retriever=HybridRetriever(corpus_path)
        self.generator=Generator(model)
        self.using_query_rewriting=use_query_rewriting
        self.query_rewriter=QueryRewriter(model) if use_query_rewriting else None
    
    def answer_question(self,question:str,top_k:int=5)->str:
        original_question=question

        # Step 1: Rewrite query if enabled
        if self.using_query_rewriting:
            question=self.query_rewriter.rewrite(original_question)

        # Step 2: Retrieve documents
        retrieved=self.retriever.retrieve(question,top_k)
        context=' '.join([doc for _,doc,_ in retrieved])

        # Step 3: Generate answer
        answer = self.generator.generate(question, context)

        return answer