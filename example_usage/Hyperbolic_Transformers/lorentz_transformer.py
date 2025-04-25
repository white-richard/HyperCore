'''
Example of building a Lorentz BERT model
'''
from tqdm import tqdm
import torch
import torch.nn as nn
import hypercore.nn as hnn
import torch.nn.functional as F
from hypercore.manifolds import Lorentz
from torch.optim.lr_scheduler import MultiStepLR
from hypercore.optimizers import Optimizer, LR_Scheduler
import numpy as np
import math
from geoopt import ManifoldParameter

class LorentzPositionwiseFeedForward(nn.Module):
    def __init__(self, manifold, d_model, d_ff):
        super(LorentzPositionwiseFeedForward, self).__init__()
        self.manifold = manifold
        self.w_1 = hnn.LorentzLinear(manifold, d_model, d_ff)
        self.w_2 = hnn.LorentzLinear(manifold, d_ff + 1, d_model - 1)
        self.act = hnn.LorentzActivation(manifold, F.relu)

    def forward(self, x):
        return self.w_2(self.act(self.w_1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, manifold, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.manifold = manifold
        self.self_attn = hnn.LorentzMultiheadAttention(manifold, d_model, d_model - 1, num_heads, attention_type='full', trans_heads_concat=True) # -1 because lorentz lienar adds time dimension
        self.feed_forward = LorentzPositionwiseFeedForward(self.manifold, d_model, d_ff)
        self.dropout = hnn.LorentzDropout(self.manifold, dropout)
        self.residual = hnn.LResNet(manifold, weight=1.0, use_scale=True, scale=3.0)
        self.norm1 = hnn.LorentzLayerNorm(manifold, d_model - 1) #layer norm is only on space dimension
        self.norm2 = hnn.LorentzLayerNorm(manifold, d_model - 1) #layer norm is only on space dimension

    def forward(self, inputs, mask):
        context = self.dropout(self.self_attn(inputs, inputs, mask=mask))
        context = self.norm1(self.residual(context, inputs))
        ff_out= self.feed_forward(context)
        output = self.norm2(self.residual(context, self.dropout(ff_out)))
        return output
    
class DecoderLayer(nn.Module):
    def __init__(self, manifold, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.manifold = manifold
        self.self_attn = hnn.LorentzMultiheadAttention(manifold, d_model, d_model - 1, num_heads, attention_type='full', trans_heads_concat=True)
        self.cross_attn = hnn.LorentzMultiheadAttention(manifold, d_model, d_model - 1, num_heads, attention_type='full', trans_heads_concat=True)
        self.feed_forward = LorentzPositionwiseFeedForward(manifold, d_model, d_ff)
        self.norm1 = hnn.LorentzLayerNorm(manifold, d_model - 1)
        self.norm2 = hnn.LorentzLayerNorm(manifold, d_model - 1)
        self.norm3 = hnn.LorentzLayerNorm(manifold, d_model - 1)
        self.dropout = hnn.LorentzDropout(dropout)
        self.residual = hnn.LResNet(manifold, weight=1.0, use_scale=True, scale=3.0)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, tgt_mask)
        x = self.norm1(self.residual(x, self.dropout(attn_output)))
        attn_output = self.cross_attn(x, enc_output, src_mask)
        x = self.norm2(self.residual(x, self.dropout(attn_output)))
        ff_output = self.feed_forward(x)
        x = self.norm3(self.residual(x, self.dropout(ff_output)))
        return x
    
class Encoder(torch.nn.Module):
    def __init__(self, manifold_in, manifold_hidden, vocab_size, seq_len, d_model=256,  d_ff=512, num_head=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.encoder_blocks = torch.nn.ModuleList()
        for i in range(num_layers):
            self.encoder_blocks.append(EncoderLayer(manifold_hidden, d_model + 1, num_head, d_ff, dropout))
        
        # word embedding + positional encoding
        self.embedding = hnn.LorentzEmbeddings(manifold_in, vocab_size, d_model, padding_idx=0)
    def sequence_mask(self, lengths, max_len=None):
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return (torch.arange(0, max_len, device=lengths.device)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))
    def forward(self, x, lengths):
        emb = self.embedding(x).transpose(0, 1)
        mask = ~self.sequence_mask(lengths).unsqueeze(1)
        for block in self.encoder_blocks:
            emb = block(emb, mask)
        return emb
    

