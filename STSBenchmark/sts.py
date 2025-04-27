import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from transformers import BertModel,BertTokenizer,AdamW

class MyData(Dataset):
    def __init__(self,df,tokenizer,max_len=256):
        super().__init__()

        self.sentence1=df['sentence1']
        self.sentence2=df['sentence2']
        self.score=df['score']
        self.tokenizer=tokenizer
        self.max_len=max_len
    
    def __len__(self):
        return len(self.score)
    
    def __getitem__(self, index):
        sentence1=self.sentence1.iloc[index]
        sentence2=self.sentence2.iloc[index]
        score=torch.tensor(self.score.iloc[index],dtype=torch.float)
        encode=self.tokenizer(
            sentence1,
            sentence2,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return{
            'input_ids':encode['input_ids'].squeeze(0),
            'attn_mask':encode['attention_mask'].squeeze(0),
            'score':score
        }

class MyModel(nn.Module):
    def __init__(self,hidden_size=768,dropout=0.1):
        super().__init__()
        self.bert=BertModel.from_pretrained('../bert-base-uncased')
        self.fc=nn.Linear(hidden_size,1)
        self.dropout=nn.Dropout(dropout)
        self.attention_pooling=nn.Linear(hidden_size,1)
    
    def forward(self,input_ids,attn_mask):
        x=self.bert(input_ids,attn_mask)
        x=self.dropout(x)

        cls=x.last_hidden_state[:,0,:]

        mask=attn_mask.unsqueeze(-1).expand(x.last_hidden_state.size()).float()# shape:(batch_size,seq_len)->(batch_size,seq_len,1)
                                                                               #       ->(batch_size,seq_len,hidden_size)
        sum_hidden=(mask*x.last_hidden_state).sum(1)#shape:(batch_size,hidden_size)
        sum_mask=mask.sum(1)#shape:(batch_size,hidden_size)
        mean_pooled=sum_hidden/sum_mask

        score=self.attention_pooling(x.last_hidden_state).squeeze(-1)# shape:(batch_size,seq_len)
        score=score.masked_fill(attn_mask==0,-1e9)# avoid the disturbance of padding
        weight=torch.softmax(score,dim=1)
        attn_pooled=torch.bmm(weight.unsqueeze(1),x.last_hidden_state)# shape:(batch_size,1,seq_len)@(batch_size,seq_len,hidden_size)
                                                                      #       =(batch_size,1,hidden_size)
        attn_pooled=attn_pooled.squeeze(1)
        
        out=self.fc(attn_pooled)

        return out

def train(loader,model,criterion,optimizer,device):
    model.train()
    total_loss=0

    for batch in loader:
        input_ids=batch['input_ids'].to(device)
        attn_mask=batch['attn_mask'].to(device)
        score=batch['score'].to(device)

        pred=model(input_ids,attn_mask)

        optimizer.zero_grad()
        loss=criterion(pred,score)
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
        attn_mask=batch['attn_mask'].to(device)
        score=batch['score'].to(device)
        pred=model(input_ids,attn_mask)

        loss=nn.functional.mse_loss(pred,score)
        total_loss+=loss.item()
        
    return total_loss


train_df=pd.read_csv('sts-train.csv')
dev_df=pd.read_csv('sts-dev.csv')
test_df=pd.read_csv('sts-test.csv')

tokenizer=BertTokenizer.from_pretrained('../bert-base-uncased')

train_data=MyData(train_df,tokenizer,max_len=256)
dev_data=MyData(dev_df,tokenizer,max_len=256)
test_data=MyData(test_df,tokenizer,max_len=256)

train_data_loader=DataLoader(train_data,batch_size=16,shuffle=True)
dev_data_loader=DataLoader(dev_data,batch_size=16,shuffle=True)
test_data_loader=DataLoader(test_data,batch_size=16,shuffle=False)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=MyModel()
model=model.to(device)
optimizer=AdamW(model.parameters(),lr=2e-5)
criterion=nn.MSELoss()

best_dev_mse=float('inf') # start with infinity
patience=5 # epoch allowed that without improvement
patience_counter=0 # the number of bad epoch so far
for epoch in range(5):
    train_loss=train(train_data_loader,model,criterion,optimizer,device)
    dev_mse=test(dev_data_loader,model,device)
    print(f'epoch{epoch+1}: train loss={train_loss:.2f}, dev mse={dev_mse:.2f}')

    if dev_mse<best_dev_mse:
        best_dev_mse=dev_mse
        torch.save(model.state_dict(),'best_model.pth')# save best model's weight
    else:
        patience_counter+=1

    if patience_counter==patience:
        print('activate early stop!')
        break

model.load_state_dict(torch.load('best_model.pth'))
test_mse = test(test_data_loader, model, device)
print(f"Final Test MSE: {test_mse:.2f}")