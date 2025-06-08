from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import torch
from typing import List

class Generator:
    def __init__(self,model:str='../t5'):
        self.device=('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer=AutoTokenizer.from_pretrained(model)
        self.model=AutoModelForSeq2SeqLM.from_pretrained(model).to(self.device)

    def generate(self,question:str,context:str,max_new_tokens:int=64)->str:
        """
        Concatenate context and question, and generate an answer.

        Args:
            question (str): The user query
            context (str): The retrieved passage(s)
            max_new_tokens (int): Output max length

        Returns:
            str: Generated answer
        """
        input_text=f'question: {question} context: {context}'
        inputs=self.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True
        ).to(self.device)

        outputs=self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4
        )

        return self.tokenizer.decode(outputs[0],skip_special_tokens=True)
