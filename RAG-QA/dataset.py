from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize

class TrainData(Dataset):
    def __init__(self, df, tokenizer, max_in_len=512, max_out_len=64):
        self.data = df
        self.tokenizer = tokenizer
        self.max_in_len = max_in_len
        self.max_out_len = max_out_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q = self.data.iloc[idx]['question']
        a = self.data.iloc[idx]['answer']
        c = self.data.iloc[idx]['context']
        
        x = self.tokenizer(f"question: {q} context: {c}", padding='max_length', max_length=self.max_in_len, truncation=True, return_tensors='pt')
        y = self.tokenizer(a, padding='max_length', max_length=self.max_out_len, truncation=True, return_tensors='pt')

        return {
            'input_ids': x['input_ids'].squeeze(0),
            'attention_mask': x['attention_mask'].squeeze(0),
            'labels': y['input_ids'].squeeze(0)
        }

class TestData(Dataset):
    def __init__(self, df, tokenizer, encoder, faiss_index, contexts, bm25, tokenized_contexts, top_k=20, top_n=3, max_in_len=512, max_out_len=64):
        self.data = df
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.faiss_index = faiss_index
        self.contexts = contexts
        self.bm25 = bm25
        self.tokenized_contexts = tokenized_contexts
        self.top_k = top_k
        self.top_n = top_n
        self.max_in_len = max_in_len
        self.max_out_len = max_out_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q = self.data.iloc[idx]['question']
        a = self.data.iloc[idx]['answer']
        
        q_embed = self.encoder.encode([q], convert_to_numpy=True)
        _, faiss_indices = self.faiss_index.search(q_embed, self.top_k)
        
        tokenized_q = word_tokenize(q.lower())
        scores = self.bm25.get_scores(tokenized_q)
        ranked = sorted([(scores[i], i) for i in faiss_indices[0]], reverse=True)[:self.top_n]

        context = " ".join([self.contexts[i] for _, i in ranked])
        x = self.tokenizer(f"question: {q} context: {context}", padding='max_length', max_length=self.max_in_len, truncation=True, return_tensors='pt')
        y = self.tokenizer(a, padding='max_length', max_length=self.max_out_len, truncation=True, return_tensors='pt')

        return {
            'input_ids': x['input_ids'].squeeze(0),
            'attention_mask': x['attention_mask'].squeeze(0),
            'labels': y['input_ids'].squeeze(0)
        }