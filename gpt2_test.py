import torch
import torch.nn as nn
import torch.nn.functional as F
import math

vocab_size = 100
embedding_dim = 64
n_heads = 4
seq_len = 16
n_layers = 2

class MyAttention(nn.Module):
    def __init__(self,d_model,d_k,d_v):
        super().__init__()
        self.q=nn.Linear(d_model,d_k,bias=False)
        self.k=nn.Linear(d_model,d_k,bias=False)
        self.v=nn.Linear(d_model,d_v,bias=False)
    
    def forward(self,input):
        Q=self.q(input)
        K=self.k(input)
        V=self.v(input)

        score=F.softmax(Q@K.transpose(-1,-2)/math.sqrt(K.shape[-1]),dim=-1)

        output=score@V

        return output

class MyMultiheadAttention(nn.Module):
    def __init__(self,head_num,d_model):
        super().__init__()
        self.q=nn.Linear(d_model,d_model,bias=False)
        self.k=nn.Linear(d_model,d_model,bias=False)
        self.v=nn.Linear(d_model,d_model,bias=False)
        self.out=nn.Linear(d_model,d_model,bias=False)

        assert d_model%head_num==0
        self.d_k=d_model//head_num

        self.d_model=d_model
        self.head_num=head_num

        self.drop=nn.Dropout(p=0.1)

    def forward(self,Q,K,V,mask=None):
        Q=self.q(Q)
        K=self.k(K)
        V=self.v(V)

        batch_size=Q.shape[0]

        Q=Q.view(batch_size,-1,self.head_num,self.d_k).transpose(1,2)
        K=K.view(batch_size,-1,self.head_num,self.d_k).transpose(1,2)
        V=V.view(batch_size,-1,self.head_num,self.d_k).transpose(1,2)

        score=torch.matmul(Q,K.transpose(-1,-2))/math.sqrt(self.d_k)

        if mask is not None:
            score=score.masked_fill(mask==0,1e-9)
        
        score=F.softmax(score,dim=-1)
        score=self.drop(score)

        output=torch.matmul(score,V).transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        output=self.out(output)

        return output

class MyAddNorm(nn.Module):
    def __init__(self,d_model):
        super().__init__()

        self.addnorm=nn.LayerNorm(d_model)
    
    def forward(self,input):
        return self.addnorm(input)

class MyFC(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.fc1=nn.Linear(d_model,d_ff)
        self.fc2=nn.Linear(d_ff,d_model)
        self.drop=nn.Dropout(p=dropout)

    def forward(self,input):
        x=self.fc1(input)
        x=self.drop(x)
        x=self.fc2(x)
        output=self.drop(x)

        return output
    
class MyPositionalEmbedding(nn.Module):
    def __init__(self,seq_len,d_model):
        super().__init__()
        self.pos_embedding=nn.Parameter(torch.randn(1,seq_len,d_model))
    
    def forward(self,x):
        return x+self.pos_embedding[:,:x.shape[1],:]

def GenerateMask(seq_len):
    mask=torch.triu(torch.ones(seq_len,seq_len),diagonal=1).bool()

    return ~mask

class MyDecoderBlock(nn.Module):
    def __init__(self,head_num,d_model,d_ff,dropout=0.1):
        super().__init__()

        self.attn=MyMultiheadAttention(head_num,d_model)
        self.fc=MyFC(d_model,d_ff)
        self.addnorm1=MyAddNorm(d_model)
        self.addnorm2=MyAddNorm(d_model)
        self.drop=nn.Dropout(p=dropout)
    
    def forward(self,x):
        mask=GenerateMask(x.shape[1])
        x=self.addnorm1(x)
        x_attn=self.attn(x,x,x,mask)
        x=x+self.drop(x_attn)

        x=self.addnorm2(x)
        x_fc=self.fc(x)
        output=x+self.drop(x_fc)

        return output

class MyDecoder(nn.Module):
    def __init__(self,vac_size,d_model,seq_len,head_num,d_ff,layer_num,dropout=0.1):
        super().__init__()

        self.word_embedding=nn.Embedding(vac_size,d_model)
        self.pos_embedding=MyPositionalEmbedding(seq_len,d_model)
        self.drop=nn.Dropout(p=dropout)

        self.decoders=nn.ModuleList([MyDecoderBlock(head_num,d_model,d_ff) for _ in range(layer_num)])
    
    def forward(self,input):
        x_word=self.word_embedding(input)
        x=self.pos_embedding(x_word)
        x=self.drop(x)

        for decoder in self.decoders:
            x=decoder(x)
        
        return x

class MyGPT(nn.Module):
    def __init__(self,vac_size,d_model,seq_len,head_num,d_ff,layer_num):
        super().__init__()

        self.decoder=MyDecoder(vac_size,d_model,seq_len,head_num,d_ff,layer_num)
        self.final_layer=nn.Linear(d_model,vac_size)
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self,x):
        x=self.decoder(x)
        x = self.final_norm(x)
        logits=self.final_layer(x)

        return logits

model = MyGPT(
    vac_size=100,
    d_model=768,
    seq_len=16,
    head_num=12,
    d_ff=256,
    layer_num=12
)

def generate(model, start_tokens, max_new_tokens):
    model.eval()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            input = start_tokens[:, -seq_len:]
            logits = model(input)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            start_tokens = torch.cat([start_tokens, next_token], dim=1)
    return start_tokens
