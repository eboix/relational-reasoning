import torch
import torchvision
import torch.nn as nn
import math
import numpy as np
import time
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import argparse
import logging
import os

####### MLP model

class TemplateMatchingMLP(nn.Module):
    def __init__(self,  vocab_size, template_length, hidden_dim, output_dim, num_layers):
        super().__init__()
        
        input_dim = vocab_size * template_length
        self.vocab_size = vocab_size
        self.template_length = template_length
        
        lin_list = [nn.Linear(input_dim, hidden_dim)]
        for i in range(num_layers-2):
            lin_list.append(nn.Linear(hidden_dim,hidden_dim))
        self.linears = nn.ModuleList(lin_list)
        
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, seq):
        x = F.one_hot(seq.view(-1).long(),num_classes=self.vocab_size).view(-1,self.template_length*self.vocab_size).float()
        # Convert x to one-hot
        for l in self.linears:
            x = l(x)
            x = F.relu(x)
        return self.output(x)
    
    ####### MLP model

class TemplateMatchingMLPWithXXT(nn.Module):
    def __init__(self,  vocab_size, template_length, hidden_dim, output_dim, num_layers, xxt_scaling=1.0):
        super().__init__()
        
        self.xxt_scaling = xxt_scaling
        input_dim = vocab_size * template_length
        self.vocab_size = vocab_size
        self.template_length = template_length
        
        lin_list = [nn.Linear(input_dim+template_length**2, hidden_dim)]
        for i in range(num_layers-2):
            lin_list.append(nn.Linear(hidden_dim,hidden_dim))
        self.linears = nn.ModuleList(lin_list)
        
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, seq):
        XXT = seq.unsqueeze(1) == seq.unsqueeze(-1)
        XXT = XXT.view(seq.shape[0], -1) * self.xxt_scaling
        x = F.one_hot(seq.view(-1).long(),num_classes=self.vocab_size).view(-1,self.template_length*self.vocab_size).float()
        x = torch.cat((x, XXT), dim=1)
        # Convert x to one-hot
        for l in self.linears:
            x = l(x)
            x = F.relu(x)
        return self.output(x)
    
######## MLP model with shared embedding layer
    
class TemplateMatchingMLPSharedEmbedding(nn.Module):
    def __init__(self,  vocab_size, template_length, hidden_dim, output_dim, num_layers):
        super().__init__()
        
        self.to_embedding = nn.Embedding(vocab_size,hidden_dim)
        input_dim = hidden_dim * template_length
        
        lin_list = [nn.Linear(input_dim, hidden_dim)]
        for i in range(num_layers-2):
            lin_list.append(nn.Linear(hidden_dim,hidden_dim))
        self.linears = nn.ModuleList(lin_list)
        
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.to_embedding(x) # Embed x
        x = x.view(x.shape[0],-1) # Concatenate embeddings
        for l in self.linears:
            x = l(x)
            x = F.relu(x)
        return self.output(x)
    
    
######## MLP with shared embedding and XX^T
    
class TemplateMatchingMLPSharedEmbeddingWithXXT(nn.Module):
    def __init__(self,  vocab_size, template_length, hidden_dim, output_dim, num_layers, xxt_scaling=1.0):
        super().__init__()
        
        self.xxt_scaling = xxt_scaling
        self.to_embedding = nn.Embedding(vocab_size,hidden_dim)
        input_dim = hidden_dim * template_length
        
        lin_list = [nn.Linear(input_dim+template_length**2, hidden_dim)]
        for i in range(num_layers-2):
            lin_list.append(nn.Linear(hidden_dim,hidden_dim))
        self.linears = nn.ModuleList(lin_list)
        
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
    
        XXT = x.unsqueeze(1) == x.unsqueeze(-1)
        XXT = XXT.view(x.shape[0], -1) * self.xxt_scaling
        
        x = self.to_embedding(x) # Embed x
        x = x.view(x.shape[0],-1) # Concatenate embeddings
        x = torch.cat((x, XXT), dim=1)
        for l in self.linears:
            x = l(x)
            x = F.relu(x)
        return self.output(x)

####### RNN adapted from Pytorch tutorial https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
class TemplateMatchingRNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim):
        super().__init__()
        
        self.to_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.i2h = nn.Linear(2*hidden_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # Shape of input is batch_size x seq_length
        assert(len(x.shape) == 2)
        seq_len = x.shape[1]
        
        hidden = torch.zeros(x.shape[0], self.hidden_dim).to(x.device)
        for i in range(seq_len):
            # Get current embedding
            curr_emb = self.to_embedding(x[:,i])
            combined = torch.cat((curr_emb, hidden), 1)
            hidden = F.relu(self.i2h(combined))
        
        output = self.h2o(hidden)
        return output

####### LSTM adapted from Pytorch tutorial https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
class TemplateMatchingLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim):
        super().__init__()
        
        self.to_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # Shape of input is batch_size x seq_length
        assert(len(x.shape) == 2)
        seq_len = x.shape[1]
        
        hidden = (torch.randn(1, x.shape[0], self.hidden_dim).to(x.device),
                  torch.randn(1, x.shape[0], self.hidden_dim).to(x.device))
        for i in range(seq_len):
            # Get current embedding
            curr_emb = self.to_embedding(x[:,i])
            curr_emb = curr_emb.view(1, curr_emb.shape[0], curr_emb.shape[1])
            out, hidden = self.lstm(curr_emb, hidden)
        
        output = self.fc(out.view(out.shape[1], out.shape[2]))
        return output
        
####### ESBN wrapping the lucid-rains ESBN repository
class TemplateMatchingESBN(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Shape of input is batch_size x seq_length
        assert(False), "Not yet implemented"

    
####### Transformer implementation adapted from vit_pytorch

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.,attn_mult=1.0, trainable_wvwo_identity=False, wvwo_id_mult=1.0, trainable_wkwq_identity=False, wkwq_id_mult=1.0):
        super().__init__()
        inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)
        project_out = True

        self.heads = heads
        self.scale = attn_mult * dim_head ** -0.5
        
        self.trainable_wvwo_identity = trainable_wvwo_identity
        if trainable_wvwo_identity:
            self.wvwo_identity_scalings = torch.nn.Parameter(torch.zeros(heads)) # Initialize at 0
            self.wvwo_id_mult = wvwo_id_mult
            
        self.trainable_wkwq_identity = trainable_wkwq_identity
        if trainable_wkwq_identity:
            self.wkwq_identity_scalings = torch.nn.Parameter(torch.zeros(heads)) # Initialize at 0
            self.wkwq_id_mult = wkwq_id_mult

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        qk = self.to_qk(x).chunk(2, dim = -1)
        v = self.to_v(x)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qk)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if self.trainable_wkwq_identity:
            # Add XX^T to each head, weighted appropriately
            xxt = torch.matmul(x, x.transpose(-1,-2))
            xxt = xxt.unsqueeze(1)
            xxt = xxt * self.wkwq_identity_scalings.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            # print('dots',torch.norm(dots))
            # print('wkwq_id', torch.norm(xxt * self.wkwq_id_mult))
            dots = dots + xxt * self.wkwq_id_mult
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
        

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        if self.trainable_wvwo_identity:
            # Also compute identity for each head
            
            # Attn is batch_size x num_heads x seqlen x seqlen
            # v is batch_size x num_heads x seqlen x head_dim
            # x is batch_size x seqlen x embed_dim
            # Compute b x seqlen x embed_dim matrix, which is
            # M[b, i, s] = \sum_h \sum_j idscaling[h] * attn[b, h, i, j] * x[b, j, s]
            
            # Rescale attention by id
            attn = attn * self.wvwo_identity_scalings.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            attn = torch.sum(attn, dim=1)
            M = torch.matmul(attn, x)
            out = out + M * self.wvwo_id_mult
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.,attn_mult=1.0, trainable_wvwo_identity=False,wvwo_id_mult=1.0, trainable_wkwq_identity=False, wkwq_id_mult=1.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout,attn_mult=attn_mult, trainable_wvwo_identity=trainable_wvwo_identity, wvwo_id_mult=wvwo_id_mult, trainable_wkwq_identity=trainable_wkwq_identity, wkwq_id_mult=wkwq_id_mult)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

    
    

    
class TemplateMatchingTransformer(nn.Module):
    def __init__(self, *, context_length, num_token_types, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0., attn_mult=1.0, trainable_wvwo_identity=False, wvwo_id_mult=1.0, trainable_wkwq_identity=False, wkwq_id_mult=1.0):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Data is of the shape minibatch_size x context_length x num_token_types
        # dim is the residual dimension
        # num_classes is the output dimension

        # W_E matrix (operates on one-hot embedding)
        self.to_embedding = nn.Embedding(num_token_types,dim)
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, context_length+1, dim))
        
        # Add CLS token, and classify based on it
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        # Transformer with residual dimension dim, a certain depth and number of heads and dimension of each head and dimension of each mlp
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, attn_mult=attn_mult, trainable_wvwo_identity=trainable_wvwo_identity, wvwo_id_mult=wvwo_id_mult, trainable_wkwq_identity=trainable_wkwq_identity, wkwq_id_mult=wkwq_id_mult)

        self.pool = pool
        assert(pool == 'cls') # classify based on the embedding from the last token
        self.to_latent = nn.Identity()

        # classify into a certain number of classes
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim,elementwise_affine = False),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_embedding(x)
        x = torch.cat([x, self.cls_token.repeat(x.shape[0], 1,1)], dim=1)
        
        # n is the context length
        b, n, _ = x.shape
        
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        assert(n == x.shape[1])
        # Using last token if in CLS mode
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, n-1]

        x = self.to_latent(x)
        return self.mlp_head(x)
    

##### Template-Matching transformer with shared embeddding and unembedding, for multiclass classification

class TemplateMatchingTransformer_TiedEmbeddingUnembedding(nn.Module):
    def __init__(self, *, context_length, num_token_types, num_classes, dim, depth, heads, mlp_dim, pool = 'final', dim_head = 64, dropout = 0., emb_dropout = 0., attn_mult=1.0, trainable_wvwo_identity=False, wvwo_id_mult, trainable_wkwq_identity=False, wkwq_id_mult=1.0):
        super().__init__()

        assert pool in {'cls', 'mean', 'final'}, 'pool type must be either cls (cls token) or mean (mean pooling), or final (final token)'
        self.pool = pool
        assert(pool == 'final' or pool == 'cls'), "Other versions not yet implemented here"
        
        assert(num_token_types == num_classes), "Input dim has to equal output dim for transformer with tied embedding and unembedding"

        # Data is of the shape minibatch_size x context_length x num_token_types
        # dim is the residual dimension
        # num_classes is the output dimension

        # W_E matrix (operates on one-hot embedding)
        self.to_embedding = nn.Embedding(num_token_types,dim)
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, context_length, dim))
        
        if self.pool == 'cls':
            # Add CLS token, and classify based on it
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        # Transformer with residual dimension dim, a certain depth and number of heads and dimension of each head and dimension of each mlp
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, attn_mult=attn_mult, trainable_wvwo_identity=trainable_wvwo_identity, wvwo_id_mult=wvwo_id_mult, trainable_wkwq_identity=trainable_wkwq_identity, wkwq_id_mult=wkwq_id_mult)

        self.to_latent = nn.Identity()

        # classify into a certain number of classes
        self.final_ln = nn.LayerNorm(dim,elementwise_affine = False)


    def forward(self, x):
        x = self.to_embedding(x)
        
        if self.pool == 'cls':
            x = torch.cat([x, self.cls_token.repeat(x.shape[0], 1,1)], dim=1)
    
        # n is the context length
        b, n, _ = x.shape
        
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        assert(n == x.shape[1])
        
        # Using last token if in CLS mode or FINAL mode
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, n-1]

        x = self.to_latent(x)
        x = self.final_ln(x)
        
        x = x @ self.to_embedding.weight.T
        return x