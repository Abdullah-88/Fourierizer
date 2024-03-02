import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
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


class FourierFFTLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, hidden_states):
        return torch.fft.fft(torch.fft.fft(hidden_states.float(), dim=-1), dim=-2).real


class FNetLayer(nn.Module):
    def __init__(self,dim,hidden_dim, dropout ):
        super().__init__()
        self.fft =  FourierFFTLayer()
        self.mixing_layer_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Linear(dim, hidden_dim)
        self.output_dense = nn.Linear(hidden_dim, dim)
        self.output_layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        fft_output = self.fft(hidden_states)
        fft_output = self.mixing_layer_norm(fft_output + hidden_states)
        intermediate_output = self.feed_forward(fft_output)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)
        output = self.dropout(output)
        output = self.output_layer_norm(output + fft_output)
        return output




class FourierGatingUnit(nn.Module):
    def __init__(self,dim, hidden_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(dim,dim)      
        self.Fnet = FNetLayer(dim, hidden_dim, dropout)
       

    def forward(self, x):
        u, v = x, x 
        u = self.proj(u)   
        v = self.Fnet(v)
        out = u * v
        return out


class FourierizerBlock(nn.Module):
    def __init__(self, d_model, d_ffn,dropout):
        super().__init__()
       
        self.norm = nn.LayerNorm(d_model)       
        self.mgu = FourierGatingUnit(d_model,d_ffn,dropout)
        self.ffn = FeedForward(d_model,d_ffn,dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mgu(x)   
        x = x + residual      
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        out = x + residual
        return out


class Fourierizer(nn.Module):
    def __init__(self, d_model, d_ffn,num_layers,dropout):
        super().__init__()
        
        self.model = nn.Sequential(
            *[FourierizerBlock(d_model, d_ffn,dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)








