"""A small GPT in plain PyTorch with modern enhancements:
- Weight Tying
- Rotary Position Embeddings (RoPE)
- RMSNorm
- SwiGLU MLP
- Bias-free linear layers
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 4096      # updated for BPE tokenizer
    block_size = 128
    n_layer = 5
    n_head = 4
    n_embd = 160
    dropout = 0.0
    tie_weights = True     # enable weight tying to save parameters


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        
        # Bias-free projections
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj._is_residual = True  # flag for scaled initialization
        self.drop = nn.Dropout(cfg.dropout)
        
        # Precompute RoPE inverse frequencies
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_half(self, x):
        d = x.shape[-1]
        return torch.cat((-x[..., d // 2:], x[..., :d // 2]), dim=-1)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        
        # Reshape to [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Compute RoPE cos & sin on the fly based on sequence length T
        t = torch.arange(T, dtype=torch.float32, device=x.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]  # [1, 1, T, head_dim]
        sin = emb.sin()[None, None, :, :]  # [1, 1, T, head_dim]
        
        # Apply RoPE to queries and keys
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Intermediate dimension of SwiGLU is 8/3 of n_embd
        self.d_ff = int(8 / 3 * cfg.n_embd)
        self.w1 = nn.Linear(cfg.n_embd, self.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.n_embd, self.d_ff, bias=False)
        self.w3 = nn.Linear(self.d_ff, cfg.n_embd, bias=False)
        self.w3._is_residual = True  # flag for scaled initialization
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # Note: no pos_emb needed because we use RoPE!
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
            
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02
            # residual connections initialized with std scaled by depth
            if getattr(m, "_is_residual", False):
                std = std / math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        # Count tied parameters only once
        if self.cfg.tie_weights:
            # exclude the head parameter because it is tied
            other_params = sum(p.numel() for name, p in self.named_parameters() if "head.weight" not in name)
            return other_params
        else:
            return sum(p.numel() for p in self.parameters())

    def load_state_dict(self, state_dict, strict=True):
        if self.cfg.tie_weights and "head.weight" not in state_dict:
            state_dict = state_dict.copy()
            state_dict["head.weight"] = state_dict["tok_emb.weight"]
        return super().load_state_dict(state_dict, strict=strict)

