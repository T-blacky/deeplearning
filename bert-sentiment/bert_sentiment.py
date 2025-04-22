import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import pandas as pd

class MyData(Dataset):
    def __init__(self,df,tokenizer,max_len=256):
        self.text=df['review']
        self.label=df['sentiment']
        self.tokenizer=tokenizer
        self.max_len=max_len
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        text=self.text.iloc[index]
        label=torch.tensor(self.label.iloc[index],dtype=torch.long)

        encoded=self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids':encoded['input_ids'].squeeze(0),
            'attn_mask':encoded['attention_mask'].squeeze(0),
            'label':label
        }

class Classifier(nn.Module):
    def __init__(self,hidden_size=768,class_num=2):
        super().__init__()
        self.model=BertModel.from_pretrained('./bert-base-uncased')
        self.linear=nn.Linear(hidden_size,class_num)
    
    def forward(self,input_ids,attn_mask):
        x=self.model(input_ids,attn_mask)
        cls=x.last_hidden_state[:,0,:] # cls
        logit=self.linear(cls)

        return logit

def train(model,dataloader,criterion,optimizer,device):
    model.train()
    total_loss=0

    for batch in dataloader:
        input_ids=batch['input_ids'].to(device)
        attn_mask=batch['attn_mask'].to(device)
        label=batch['label'].to(device)

        optimizer.zero_grad()
        logit=model(input_ids,attn_mask)
        loss=criterion(logit,label)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def test(model,dataloader,device):
    model.eval()
    total,correct=0,0

    for batch in dataloader:
        input_ids=batch['input_ids'].to(device)
        attn_mask=batch['attn_mask'].to(device)
        label=batch['label'].to(device)

        out=model(input_ids,attn_mask)
        pred=torch.argmax(out,dim=1)
        
        correct+=(pred==label).sum().item()
        total+=label.size(0)
    
    return correct/total

df=pd.read_csv('IMDB Dataset_1.csv')
df_train,df_test=train_test_split(df,test_size=0.2,random_state=42,shuffle=True)
tokenizer=BertTokenizer.from_pretrained('./bert-base-uncased')

train_data=MyData(df_train,tokenizer,max_len=256)
test_data=MyData(df_test,tokenizer,max_len=256)
train_data_loader=DataLoader(train_data,16,shuffle=True)
test_data_loader=DataLoader(test_data,16,shuffle=True)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=Classifier().to(device)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(10):
    train_loss=train(model,train_data_loader,criterion,optimizer,device)
    test_loss=test(model,test_data_loader,device)

    print(f'epoch{epoch+1}, train loss:{train_loss:.4f}; test accuracy:{test_loss:.4f}')
