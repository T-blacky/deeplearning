import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self,in_feature,out_feature,r=4,alpha=16):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(out_feature,in_feature))
        self.weight.requires_grad=False # freeze the weight of the original model

        self.A=nn.parameter(torch.randn(r,in_feature))
        self.B=nn.Parameter(torch.randn(out_feature,r))
        self.scaling=alpha//r
    
    def forward(self,x):
        x_W=F.linear(x,self.weight) # shape: (…, out_feature)
        x_A=F.linear(x,self.A) # shape: (…, r)
        x_AB=F.linear(x_A,self.B) # shape:(…, out_feature)

        return x_W+self.scaling*x_AB

class MyAttention(nn.Module):
    def __init__(self,d_model,r=4,alpha=16):
        super().__init__()

        self.Q=LoRALinear(d_model,d_model,r,alpha)
        self.K=nn.Linear(d_model,d_model) # don't need LoRA for K since K is shared across all incoming queries,
                                          # Changing K means changing the structure of the attention space for all tokens,
                                          # which will destabilize learned attention patterns from the pre-trained model
        self.V=LoRALinear(d_model,d_model,r,alpha)
        self.out=nn.Linear(d_model,d_model)

    def forward(self,x):
        Q=self.Q(x)
        K=self.K(x)
        V=self.V(x)

        d_k=Q.shape[-1]
        attn_score=torch.matmul(Q,K.transpose(-1,-2))/math.sqrt(d_k)
        attn_prob=torch.softmax(attn_score,dim=-1)

        context=torch.matmul(attn_prob,V)

        return self.out(context)
