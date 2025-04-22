import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        score=F.softmax((Q@K.transpose(-2,-1))/math.sqrt(K.shape[-1]),dim=-1)
        output=score@V

        return output

def MyGenerateMask(seq_len):
    mask=torch.triu(torch.ones(seq_len,seq_len),diagonal=1).bool()
    return ~mask

def MyGeneratePaddingMask(input,pad=0):
    mask=(input!=pad) # mask.shape=(batch_size,seq_len)
    mask=mask.unsqueeze(1).unsqueeze(2) #mask.shape=(batch,1,1,seq_len)
    return mask

class MyMultiHeadAttention(nn.Module):
    def __init__(self,head_num,d_model):
        super().__init__()
        self.q=nn.Linear(d_model,d_model,bias=False)
        self.k=nn.Linear(d_model,d_model,bias=False)
        self.v=nn.Linear(d_model,d_model,bias=False)
        self.out=nn.Linear(d_model,d_model,bias=False)

        self.head_num=head_num
        self.d_model=d_model

        assert d_model%head_num==0
        self.d_k=d_model//head_num

        self.dropout=nn.Dropout(p=0.1)
    
    def forward(self,Q,K,V,mask=None):
        batch_size=Q.shape[0]
        Q=self.q(Q)
        K=self.k(K)
        V=self.v(V)

        Q=Q.view(batch_size,-1,self.head_num,self.d_k).transpose(1,2)
        K=K.view(batch_size,-1,self.head_num,self.d_k).transpose(1,2)
        V=V.view(batch_size,-1,self.head_num,self.d_k).transpose(1,2)

        attn_score=torch.matmul(Q,K.transpose(-1,-2))/math.sqrt(self.d_k)
        if mask is not None:
            attn_score=attn_score.masked_fill(mask==0,-1e9)
        attn_score=F.softmax(attn_score,dim=-1)
        attn_score=self.dropout(attn_score)
        context=torch.matmul(attn_score,V)

        context=context.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        output=self.out(context)

        return output

class MyPositionWiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.linear1=nn.Linear(d_model,d_ff)
        self.linear2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,input):
        x=self.linear1(input)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.linear2(x)

        return x
    
class MyAddNorm(nn.Module):
    def __init__(self,d_model,dropout=0.1):
        super().__init__()
        self.norm=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,input,sublayer_input):
        return self.norm(input+self.dropout(sublayer_input))

class MyTransformerEncoderBlock(nn.Module):
    def __init__(self,d_model,head_num,d_ff,dropout=0.1):
        super().__init__()
        self.attn=MyMultiHeadAttention(head_num,d_model)
        self.ffn=MyPositionWiseFeedForward(d_model,d_ff,dropout)
        self.addnorm1=MyAddNorm(d_model,dropout)
        self.addnorm2=MyAddNorm(d_model,dropout)
    
    def forward(self,x):
        attn_out=self.attn(x,x,x)
        x=self.addnorm1(x,attn_out)

        ffn_out=self.ffn(x)
        output=self.addnorm2(x,ffn_out)

        return output

class MyPositionalEncoding(nn.Module):
    def __init__(self,max_len,d_model):
        super().__init__()

        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len).unsqueeze(1).float() #shape: (max_len,1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000)/d_model)) #shape: (d_model//2,)
        
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        pe=pe.unsqueeze(0) #shape: (1,max_len,d_model)
        self.register_buffer('pe',pe)

        self.dropout=nn.Dropout(p=0.1)
    
    def forward(self,x):
        #x.shape: (batch_size,seq_len,d_model)
        x=x+self.pe[:,:x.shape[1],:]
        x=self.dropout(x)

        return x

class MyTransformerEncoder(nn.Module):
    def __init__(self,vocab_len,d_model,head_num,layer_num,d_ff,dropout=0.1,max_len=5000):
        super().__init__()

        self.embedding=nn.Embedding(vocab_len,d_model)
        self.positional_encoding=MyPositionalEncoding(max_len,d_model)
        self.dropout=nn.Dropout(dropout)

        self.blocks=nn.ModuleList([MyTransformerEncoderBlock(d_model,head_num,d_ff,dropout) for _ in range(layer_num)])

    def forward(self,x): # x.shape: (batch_size,seq_len)
        x=self.embedding(x) # x.shape: (batch_size,seq_len,d_model)
        x=self.positional_encoding(x)
        x=self.dropout(x)

        for block in self.blocks:
            x=block(x)
        
        return x

class MyTransformerDecoderBlock(nn.Module):
    def __init__(self,head_num,d_model,d_ff,dropout=0.1):
        super().__init__()

        self.d_model=d_model

        self.attn_mask=MyMultiHeadAttention(head_num,d_model)
        self.attn_cross=MyMultiHeadAttention(head_num,d_model)

        self.dropout=nn.Dropout(p=dropout)

        self.addnorm1=MyAddNorm(d_model,dropout)
        self.addnorm2=MyAddNorm(d_model,dropout)
        self.addnorm3=MyAddNorm(d_model,dropout)

        self.fc=MyPositionWiseFeedForward(d_model,d_ff,dropout)

    def forward(self,x,en_out,enc_dec_mask):
        mask=MyGenerateMask(x.shape[1]).unsqueeze(0).unsqueeze(1)
        x_attn_mask=self.attn_mask(x,x,x,mask) # self mask attention
        x_attn_mask=self.dropout(x_attn_mask)
        x=self.addnorm1(x,x_attn_mask)

        x_attn_cross=self.attn_cross(x,en_out,en_out,enc_dec_mask) # encoder decoder cross attnetion
        x_attn_cross=self.dropout(x_attn_cross)
        x=self.addnorm2(x,x_attn_cross)

        x_fc=self.fc(x)
        out=self.addnorm3(x,x_fc)

        return out

class MyTransformerDecoder(nn.Module):
    def __init__(self,vocab_size,d_model,head_num,d_ff,layer_num,max_len,dropout=0.1):
        super().__init__()

        self.embedding=nn.Embedding(vocab_size,d_model)
        self.position_encoding=MyPositionalEncoding(max_len,d_model)
        self.dropout=nn.Dropout(p=dropout)

        self.blocks=nn.ModuleList([MyTransformerDecoderBlock(head_num,d_model,d_ff,dropout) for _ in range(layer_num)])

    def forward(self,tgt_input,enc_out,enc_dec_mask):
        tgt_input=self.embedding(tgt_input)
        x=self.position_encoding(tgt_input)
        x=self.dropout(x)

        for block in self.blocks:
            x=block(x,enc_out,enc_dec_mask)
        
        return x

class MyTransformer(nn.Module):
    def __init__(self,vocab_size,d_model,head_num,d_ff,layer_num,max_len,dropout=0.1):
        super().__init__() 

        self.encoder = MyTransformerEncoder(vocab_size,d_model,head_num,layer_num,d_ff,dropout,max_len)
        self.decoder = MyTransformerDecoder(vocab_size,d_model,head_num,d_ff,layer_num,max_len,dropout)
        self.final_linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt, src_mask, tgt_mask, enc_dec_mask):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, enc_dec_mask, tgt_mask)
        logits = self.final_linear(dec_out)
        
        return logits
    