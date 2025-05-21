import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from transformers import T5Tokenizer,T5ForConditionalGeneration,AdamW
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

corpus_df=pd.read_csv('corpus.csv')
contexts=corpus_df['context'].to_list()

# FAISS
encoder=SentenceTransformer('all-MiniLM-L6-v2')
contexts_embedding=encoder.encode(contexts,show_progress_bar=True,convert_to_numpy=True)# embedding corpus, contexts_embedding' shape: (number of contexts, dimension(typically 384))
dimension=contexts_embedding.shape[1]
faiss_index=faiss.IndexFlatL2(dimension)
faiss_index.add(contexts_embedding)

# BM25
tokenized_corpus=[word_tokenize(doc.lower()) for doc in corpus_df['context']]
bm25=BM25Okapi(tokenized_corpus)

n_faiss=20
top_n=3

class TrainData(Dataset):
    def __init__(self,df,corpus_df,tokenizer,max_in_len=512,max_out_len=64):
        super().__init__()
        self.data=df
        self.corpus=corpus_df
        self.tokenizer=tokenizer
        self.max_in_len=max_in_len
        self.max_out_len=max_out_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data=self.data.iloc[index]
        question=data['question']
        context_id=data['context_id']
        context=self.corpus.iloc[context_id]['context']

        input_text=f'question:{question} context:{context}'
        output_text=data['answer']

        inputs=self.tokenizer(
            input_text,
            padding='max_length',
            max_length=self.max_in_len,
            truncation=True,
            return_tensors='pt'
        )

        outputs=self.tokenizer(
            output_text,
            padding='max_length',
            max_length=self.max_out_len,
            truncation=True,
            return_tensors='pt'
        )

        return{
            'input_ids':inputs['input_ids'].squeeze(0), # shape before squeeze: (1, seq_len)
            'attention_mask':inputs['attention_mask'].squeeze(0),
            'labels':outputs['input_ids'].squeeze(0)
        }

class TestData(Dataset):
    def __init__(self,df,corpus_df,tokenizer,contexts,max_in_len=512,max_out_len=64):
        super().__init__()
        self.data=df
        self.corpus=corpus_df
        self.tokenizer=tokenizer
        self.max_in_len=max_in_len
        self.max_out_len=max_out_len
        self.contexts=contexts
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data=self.data.iloc[index]
        question=data['question']

        # using  only FAISS as retriever
        #question_embedding=encoder.encode([question],convert_to_numpy=True)
        #distance,indices=faiss_index.search(question_embedding,top_n)
        # shape of distance and indices:(queries_num, top_n)
        #context=' '.join(self.contexts[i] for i in indices[0])

        # using only BM25 as retriever
        #tokenized_question=word_tokenize(question.lower())
        #score=bm25.get_scores(tokenized_question)
        #context_id=sorted(range(len(score)), key=lambda i:score[i],reverse=True)[:top_n]
        #context=' '.join(self.corpus.iloc[i]['context'] for i in context_id)

        # using hybrid retrieval strategy, semantic-first, lexical-second retrieval

        # Step 1: FAISS → get top-K semantic candidates
        question_embedding=encoder.encode([question],convert_to_numpy=True)
        _,indices=faiss_index.search(question_embedding,n_faiss)

        # Step 2: BM25 → re-rank those K based on keyword match
        tokenized_question=word_tokenize(question.lower())
        bm25_score=[]
        scores=bm25.get_scores(tokenized_question)# return a list of score, shape:(context_num,)
        for idx in indices[0]:
            score=scores[idx]
            bm25_score.append((score,idx))
        bm25_score.sort(reverse=True)

        # Step 3: Get top-n contexts
        top_context_ids=[idx for _,idx in bm25_score[:top_n]]
        context=' '.join(self.corpus.iloc[i]['context'] for i in top_context_ids)

        input_text=f'question:{question} context:{context}'
        output_text=data['answer']

        inputs=self.tokenizer(
            input_text,
            padding='max_length',
            max_length=self.max_in_len,
            truncation=True,
            return_tensors='pt'
        )

        outputs=self.tokenizer(
            output_text,
            padding='max_length',
            max_length=self.max_out_len,
            truncation=True,
            return_tensors='pt'
        )

        return{
            'input_ids':inputs['input_ids'].squeeze(0), # shape before squeeze: (1, seq_len)
            'attention_mask':inputs['attention_mask'].squeeze(0),
            'labels':outputs['input_ids'].squeeze(0)
        }

train_df=pd.read_csv('train.csv')
dev_df=pd.read_csv('dev.csv')
test_df=pd.read_csv('test.csv')

tokenizer=T5Tokenizer.from_pretrained('../t5')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=T5ForConditionalGeneration.from_pretrained('../t5').to(device)
optimizer=AdamW(model.parameters(),lr=2e-5)

train_data=TrainData(train_df,corpus_df,tokenizer)
dev_data=TestData(dev_df,corpus_df,tokenizer,contexts)
test_data=TestData(test_df,corpus_df,tokenizer,contexts)

train_loader=DataLoader(train_data,batch_size=8,shuffle=True)
dev_loader=DataLoader(dev_data,batch_size=8,shuffle=False)
test_loader=DataLoader(test_data,batch_size=8,shuffle=False)

def train(loader,model,optimizer,device):
    model.train()
    total_loss=0

    for batch in loader:
        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        labels=batch['labels'].to(device)

        generated_ids=model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=64,
            do_sample=True
        )

        decoded_pred=tokenizer.batch_decode(generated_ids,skip_special_tokens=True)
        decoded_ref=tokenizer.batch_decode(labels,skip_special_tokens=True)

        scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
        rewards=[]

        for pred,ref in zip(decoded_pred,decoded_ref):
            # BLEU
            #bleu=sentence_bleu([ref.split()],pred.split())
            #rewards.append(bleu)

            # ROUGE
            rouge=scorer.score(ref,pred)['rougeL'].fmeasure
            rewards.append(rouge)

        generated_out=model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=generated_ids)
        logits=generated_out.logits

        log_probs=nn.functional.log_softmax(logits,dim=-1)
        gen_log_probs=torch.gather(
            log_probs,2,generated_ids.unsqueeze(-1)
        ).squeeze(-1)

        pad_mask=(generated_ids!=tokenizer.pad_token_id).float()
        log_probs_sum=(pad_mask*gen_log_probs).sum(dim=-1)

        reward_tensor=torch.tensor(rewards,device=device)
        reward_tensor=(reward_tensor-reward_tensor.mean())/(reward_tensor.std()+1e-8)

        loss=-(log_probs_sum*reward_tensor).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
    
    return total_loss

@torch.no_grad()
def test(loader,model,device):
    model.eval()
    total_reward=0
    total_count=0

    for batch in loader:
        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        labels=batch['labels'].to(device)

        generated_ids=model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=64,
            do_sample=True
        )

        decoded_pred=tokenizer.batch_decode(generated_ids,skip_special_tokens=True)
        decoded_ref=tokenizer.batch_decode(labels,skip_special_tokens=True)

        scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)

        for ref,pred in zip(decoded_pred,decoded_ref):
            # BLEU
            #bleu=sentence_bleu([ref.split()],pred.split())
            #reward.append(bleu)

            # ROUGE
            rouge=scorer.score(ref,pred)['rougeL'].fmeasure
            total_count+=1
            total_reward+=rouge
        
    reward=total_reward/total_count
    return reward

best_score=0
patience=3
counter=0
for i in range(5):
    train_loss=train(train_loader,model,optimizer,device)
    dev_score=test(dev_loader,model,device)
    print(f'train loss:{train_loss:.2f}, validation rouge reward:{dev_score:.2f}')

    if best_score<dev_score:
        best_score=dev_score
        counter=0
        torch.save(model.state_dict(),'best_model.pth')
    else:
        counter+=1
        if counter==patience:
            print('activate early stop!')
            break

model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
test_reward=test(test_loader,model,device)
print(f'test ROUGE score:{test_reward:.2f}')