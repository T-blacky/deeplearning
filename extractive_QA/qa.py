import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from transformers import BertModel,BertTokenizerFast,AdamW

class MyData(Dataset):
    def __init__(self,df,tokenizer,max_len=256):
        super().__init__()
        self.df=df
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.encoded=[]

        for _,row in self.df.iterrows():
            question=row['question']
            context=row['context']
            is_imppossible=row['is_impossible']
            ans_start=row['ans_start']
            ans_end=row['ans_end']

            encode=self.tokenizer(
                question,
                context,
                truncation='only_second',
                padding='max_length',
                max_length=self.max_len,
                return_offset_mapping=True,
                return_token_type_ids=True
            )

            input_ids=encode['input_ids']
            attn_mask=encode['attention_mask']
            offset_mapping=encode['offset_mapping']
            token_type_ids=encode['token_type_ids']

            start_position=end_position=0
            if not is_imppossible and ans_start!=-1:
                for idx,(start,end) in enumerate(offset_mapping):
                    if token_type_ids[idx]!=1:
                        continue   # only need to focus on context
                    if start<=ans_start<end:
                        start_position=idx
                    if start<ans_end<=end:
                        end_position=idx
            
            self.encoded.append(
                {
                    'input_ids':input_ids,
                    'attn_mask':attn_mask,
                    'token_type_ids':token_type_ids,
                    'start_position':start_position,
                    'end_position':end_position
                }
            )
    
    def __len__(self):
        return len(self.encoded)
    
    def __getitem__(self, index):
        item=self.encoded[index]

        return {k:torch.tensor(v) for k,v in item.items()}
        

class MyModel(nn.Module):
    def __init__(self,hidden_size=768,dropout=0.1):
        super().__init__()
        self.bert=BertModel.from_pretrained('../bert-base-uncased')
        self.dropout=nn.Dropout(p=dropout)
        self.fc=nn.Linear(hidden_size,2)
    
    def forward(self,input_ids,attn_mask):
        x=self.bert(input_ids,attn_mask)
        x=self.dropout(x)
        x=self.fc(x)
        start,end=x.split(1,dim=-1)
        # the CrossEntrophy Loss need the input shape in 2D, in this case:(batch_size,seq_len)
        # each num in seq_len represents the possibility for this token being the beginning or ending
        start=start.squeeze(-1)
        end=end.squeeze(-1)

        return start,end

def train(loaders,model,criterion,optimizer,device):
    model.train()
    total_loss=0

    for batch in loaders:
        input_ids=batch['input_ids'].to(device)
        attn_mask=batch['attn_mask'].to(device)
        start_position=batch['start_position'].to(device)
        end_position=batch['end_position'].to(device)

        pred_start,pred_end=model(input_ids,attn_mask)
        
        optimizer.zero_grad()
        loss_start=criterion(pred_start,start_position)
        loss_end=criterion(pred_end,end_position)
        loss=(loss_start+loss_end)/2
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
    
    return total_loss

@torch.no_grad()
def test(loaders,model,criterion,device):
    model.eval()
    total_loss=0

    for batch in loaders:
        input_ids=batch['input_ids'].to(device)
        attn_mask=batch['attn_mask'].to(device)
        start_position=batch['start_position'].to(device)
        end_position=batch['end_position'].to(device)

        pred_start,pred_end=model(input_ids,attn_mask)
        
        loss_start=criterion(pred_start,start_position)
        loss_end=criterion(pred_end,end_position)
        loss=(loss_start+loss_end)/2

        total_loss+=loss.item()
    
    return total_loss

tokenizer=BertTokenizerFast.from_pretrained('../bert-base-uncased')

df_train=pd.read_csv('data_train.csv')
df_dev=pd.read_csv('data_dev.csv')
train_data=MyData(df_train,tokenizer,max_len=256)
dev_data=MyData(df_dev,tokenizer,max_len=256)
train_loader=DataLoader(train_data,batch_size=16,shuffle=True)
dev_loader=DataLoader(dev_data,batch_size=16,shuffle=False)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=MyModel().to(device)
optimizer=AdamW(model.parameters,lr=2e-5)
criterion=nn.CrossEntropyLoss()

patience=3
counter=0
best_loss=float('inf')
for i in range(5):
    train_loss=train(train_loader,model,criterion,optimizer,device)
    dev_loss=test(dev_loader,model,criterion,device)
    print(f'epoch{i+1}, train loss is {train_loss:.2f}, validation loss is {dev_loss:.2f}')

    if dev_loss<best_loss:
        best_loss=dev_loss
        counter=0
        torch.save(model.state_dict(),'best_model.pth')
    elif counter==patience:
        print('Activate early stop!')
        break
    else:
        counter+=1
    