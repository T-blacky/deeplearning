import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from transformers import T5Tokenizer,T5ForConditionalGeneration,AdamW
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

class MyData(Dataset):
    def __init__(self,df,tokenizer,max_input_len=512,max_output_len=64):
        super().__init__()
        self.data=df
        self.tokenizer=tokenizer
        self.max_input_len=max_input_len
        self.max_output_len=max_output_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data=self.data.iloc[index]
        input_text=f"question:{data['question']} context:{data['context']}"
        output_text=data['answer']

        inputs=self.tokenizer(
            input_text,
            padding='max_length',
            max_length=self.max_input_len,
            truncation=True,
            return_tensors='pt'
        )

        targets=self.tokenizer(
            output_text,
            padding='max_length',
            max_length=self.max_output_len,
            truncation=True,
            return_tensors='pt'
        )

        return{
            'input_ids':inputs['input_ids'].squeeze(0),
            'attention_mask':inputs['attention_mask'].squeeze(0),
            'labels':targets['input_ids'].squeeze(0)
        }

model_path = "./t5"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
optimizer=AdamW(model.parameters(),lr=2e-5)

df_train = pd.read_csv("train.csv")
df_dev = pd.read_csv("dev.csv")
df_test = pd.read_csv('test.csv')

train_data = MyData(df_train, tokenizer)
dev_data = MyData(df_dev, tokenizer)
test_data = MyData(df_test, tokenizer)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=8, shuffle=False)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

def train(loader,model,device,optimizer):
    model.train()
    total_loss=0

    for batch in loader:
        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        labels=batch['labels'].to(device)

        generated_ids = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_length=64,
                                do_sample=True
                            )

        decoded_pred=tokenizer.batch_decode(generated_ids,skip_special_tokens=True)
        decoded_labels=tokenizer.batch_decode(labels,skip_special_tokens=True)

        scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
        rewards=[]

        for pred,ref in zip(decoded_pred,decoded_labels):
            # BLEU
            #bleu=sentence_bleu([ref.split()],pred.split())
            #rewards.append(bleu)

            # ROUGE
            rouge=scorer.score(ref,pred)['rougeL'].fmeasure
            rewards.append(rouge)
        generate_out = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=generated_ids)
        logits=generate_out.logits
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
    total_loss=0

    for batch in loader:
        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        labels=batch['labels'].to(device)

        out=model(input_ids,attention_mask,labels)
        loss=out.loss

        total_loss+=loss.item()
    
    return total_loss / len(loader)

best_loss=float('inf')
patience=3
counter=0
for i in range(5):
    train_loss=train(train_loader,model,device,optimizer)
    dev_loss=test(dev_loader,model,device)

    if dev_loss<best_loss:
        best_loss=dev_loss
        counter=0
        torch.save(model.state_dict(),'best_model.pth')
    else:
        counter+=1 
        if counter==patience:
            break

model.load_state_dict(torch.load('best_model.pth'))
test_loss=test(test_loader,model,device)
