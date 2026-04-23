import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

    
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)

        positions = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
        return x + self.pos_embed(positions)
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        H = self.num_heads
        Hd = self.head_dim

        # -------------------------
        # project
        # -------------------------
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # -------------------------
        # split into heads
        # (B, H, T, Hd)
        # -------------------------
        Q = Q.view(B, T, H, Hd).transpose(1, 2)
        K = K.view(B, T, H, Hd).transpose(1, 2)
        V = V.view(B, T, H, Hd).transpose(1, 2)

        # -------------------------
        # attention scores
        # -------------------------
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, T, T)
        scores = scores / math.sqrt(Hd)

        if mask is not None:
            # mask: (B, T)
            attn_mask = mask[:, None, None, :]  # (B,1,1,T)
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        scores = scores - scores.max(dim=-1, keepdim=True).values
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # apply attention
        out = torch.matmul(attn, V)

        # -------------------------
        # merge heads
        # -------------------------
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(out)

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff_layer = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):

        x = x + self.dropout(self.self_attention(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ff_layer(self.norm2(x)))

        return x
    
class Encoder(nn.Module):
    def __init__(self, embed_dim, n_layers, num_heads=8):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
    
class DynamicsModel(nn.Module):
    def __init__(self, n_in, embed_dim, n_layers, max_len, out_dim, num_heads=8):
        super().__init__()

        self.embed = nn.Linear(n_in, embed_dim)
        self.pos_enc = PositionalEncoding(max_len, embed_dim)

        self.encoder = Encoder(embed_dim, n_layers, num_heads)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.register_buffer("x_mean", torch.zeros(n_in))
        self.register_buffer("x_std", torch.ones(n_in))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Small init on output projection to keep early predictions near zero
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, mask=None):

        x = (x - self.x_mean) / self.x_std

        # x: (B, T, D)
        x = self.embed(x)

        x = self.pos_enc(x)

        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)

        x = torch.cat([cls, x], dim=1)
        
        if mask is not None:
            cls_mask = torch.ones(B, 1, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
        
        x = self.encoder(x, mask=mask)
        
        cls_out = x[:, 0, :]
        return self.head(cls_out)