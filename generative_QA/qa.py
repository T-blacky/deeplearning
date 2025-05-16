import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from transformers import T5Tokenizer,T5ForConditionalGeneration,AdamW

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

        out=model(input_ids,attention_mask,labels)
        loss=out.loss

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
