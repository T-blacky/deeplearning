from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import torch

class QueryRewriter:
    def __init__(self,model='../t5'):
        self.device=('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer=AutoTokenizer.from_pretrained(model)
        self.model=AutoModelForSeq2SeqLM.from_pretrained(model).to(self.device)
    
    def rewrite(self,query:str)->str:
        prompt=f'rewrite the question: {query}'
        inputs=self.tokenizer(prompt,return_tensors='pt',truncation=True).to(self.device)
        outputs=self.model.generate(**inputs,max_new_tokens=32)
        return self.tokenizer.decode(outputs[0],skip_special_tokens=True)