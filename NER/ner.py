import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from transformers import BertModel,BertTokenizerFast,AdamW

class MyData(Dataset):
    def __init__(self,data,tokenizer,max_len=256):
        super().__init__()
        self.data=data
        self.tokenizer=tokenizer
        self.max_len=max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        words=self.data[index]['words']
        tags=self.data[index]['ner_tags']
        encode=self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors=None,
        )

        word_ids=encode.word_ids(batch_index=0)

        aligned_label=[]
        previous_word_id=None
        for word_id in word_ids:
            if word_id is None:
                aligned_label.append(-100)
            elif word_id ==previous_word_id:
                aligned_label.append(-100)
            else:
                aligned_label.append(tags[word_id])
            
            previous_word_id=word_id

        return{
            'input_ids': torch.tensor(encode['input_ids']),         
            'attn_mask': torch.tensor(encode['attention_mask']),
            'ner_tags':torch.tensor(aligned_label)
        }

class MyModel(nn.Module):
    def __init__(self,num_class=9,hidden_size=768,dropout=0.1):
        super().__init__()
        self.bert=BertModel.from_pretrained('../bert-base-uncased')
        self.classifier=nn.Linear(hidden_size,num_class)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,input_ids,attn_mask):
        x=self.bert(input_ids,attn_mask)
        x=self.dropout(x.last_hidden_state)
        out=self.classifier(x) # focus on individual token, don't need to pooling using cls ect like sentence-focused task, for example stsmark

        return out

def train(loaders,model,criterion,optimizer,device):
    model.train()
    total_loss=0

    for batch in loaders:
        input_ids=batch['input_ids'].to(device)
        attn_mask=batch['attn_mask'].to(device)
        ner_tags=batch['ner_tags'].to(device)

        pred=model(input_ids,attn_mask)

        optimizer.zero_grad()
        loss=criterion(pred.view(-1, pred.shape[-1]), ner_tags.view(-1))
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
        ner_tags=batch['ner_tags'].to(device)

        pred=model(input_ids,attn_mask)
        
        loss=criterion(pred.view(-1, pred.shape[-1]), ner_tags.view(-1))
        total_loss+=loss.item()
    
    return total_loss


df_train=pd.read_csv('ner_train.csv')
df_train=df_train.groupby('sentence_id')
df_dev=pd.read_csv('ner_dev.csv')
df_dev=df_dev.groupby('sentence_id')
df_test=pd.read_csv('ner_test.csv')
df_test=df_test.groupby('sentence_id')

tokenizer=BertTokenizerFast.from_pretrained('../bert-base-uncased')

set_train=[]
for _,group in df_train:
    words=group['word'].tolist()
    labels=group['ner_tag'].tolist()
    set_train.append({'words':words,'ner_tags':labels})
set_dev=[]
for _,group in df_dev:
    words=group['word'].tolist()
    labels=group['ner_tag'].tolist()
    set_dev.append({'words':words,'ner_tags':labels})
set_test=[]
for _,group in df_test:
    words=group['word'].tolist()
    labels=group['ner_tag'].tolist()
    set_test.append({'words':words,'ner_tags':labels})

train_data=MyData(set_train,tokenizer,max_len=256)
dev_data=MyData(set_dev,tokenizer,max_len=256)
test_data=MyData(set_test,tokenizer,max_len=256)

train_data_loader=DataLoader(train_data,batch_size=16,shuffle=True)
dev_data_loader=DataLoader(dev_data,batch_size=16,shuffle=False)
test_data_loader=DataLoader(test_data,batch_size=16,shuffle=False)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=MyModel().to(device)
criterion=nn.CrossEntropyLoss(ignore_index=-100)
optimizer=AdamW(model.parameters(),lr=2e-5)

best_loss=float('inf')
patience=1
counter=0
for i in range(2):
    train_loss=train(train_data_loader,model,criterion,optimizer,device)
    dev_loss=test(dev_data_loader,model,criterion,device)
    print(f'epoch{i+1}, train loss is {train_loss:.2f}, validation loss is {dev_loss:.2f}')

    if dev_loss<best_loss:
        best_loss=dev_loss
        counter=0
        torch.save(model.state_dict(),'best_model.pth')
    else:
        counter+=1

    if counter==patience:
        print('activate early stop!')
        break

model.load_state_dict(torch.load('best_model.pth'))
test_loss=test(test_data_loader,model,criterion,device)
print(f'test loss is {test_loss:.2f}')