import re
from typing import List

def tokenize(text:str)->List[str]:
    # Basic whitespace + punctuation tokenizer using regular expression
    return re.findall(r'\bw+\b',text.lower())