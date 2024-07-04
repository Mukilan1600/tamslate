import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import Config, generate_masks

class SelfAttentionHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(Config.n_embd, head_size, bias=False)
        self.query = nn.Linear(Config.n_embd, head_size, bias=False)
        self.value = nn.Linear(Config.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x, mask):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        wei = wei.masked_fill(mask == 0, float(-1e9))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class CrossAttentionHead(nn.Module):
    """ one head of cross-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(Config.n_embd, head_size, bias=False)
        self.query = nn.Linear(Config.n_embd, head_size, bias=False)
        self.value = nn.Linear(Config.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x, y, mask):
        B, T, C = x.shape
        k = self.key(y)
        q = self.query(x)
        v = self.value(y)
        wei = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        if mask is not None:
            wei = wei.masked_fill(mask == 0, float(-1e9))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadSelfAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(Config.n_embd, Config.n_embd)
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x, mask):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class MultiHeadCrossAttention(nn.Module):
    """ multiple heads of cross-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(Config.n_embd, Config.n_embd)
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x, y, mask):
        out = torch.cat([h(x, y, mask) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(Config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = Config.n_embd // Config.n_head
        self.self_att = MultiHeadSelfAttention(Config.n_head, head_size)
        self.ln1 = nn.LayerNorm(Config.n_embd)
        self.ln2 = nn.LayerNorm(Config.n_embd)
        self.ffwd = FeedForward(Config.n_embd)

    def forward(self, x, mask):
        x = x + self.self_att(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = Config.n_embd // Config.n_head
        self.self_att = MultiHeadSelfAttention(Config.n_head, head_size)
        self.cross_att = MultiHeadCrossAttention(Config.n_head, head_size)
        self.ln1 = nn.LayerNorm(Config.n_embd)
        self.ln2 = nn.LayerNorm(Config.n_embd)
        self.ln3 = nn.LayerNorm(Config.n_embd)
        self.ffwd = FeedForward(Config.n_embd)

        self.register_buffer("tril", torch.tril(torch.ones(Config.block_size, Config.block_size, dtype=torch.bool, device=Config.device)))

    def forward(self, x, x_mask, ca, ca_mask):
        x = x + self.self_att(self.ln1(x), torch.logical_and(x_mask, self.tril))
        x = x + self.cross_att(self.ln2(x), ca, ca_mask)
        x = x + self.ffwd(self.ln3(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model, device=Config.device)
        position = torch.arange(0, max_len, dtype=torch.float, device=Config.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=Config.device).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.input_emb = nn.Embedding(vocab_size, Config.n_embd)
        self.positional_encoding = PositionalEncoding(Config.n_embd)
        self.blocks = nn.Sequential(*[EncoderBlock() for _ in range(Config.n_layer)])

    def forward(self, x, mask):
        B, T = x.shape
        x = self.input_emb(x) * math.sqrt(Config.n_embd)
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        for block in self.blocks:
            x = block(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.input_emb = nn.Embedding(vocab_size, Config.n_embd)
        self.positional_encoding = PositionalEncoding(Config.n_embd)
        self.blocks = nn.Sequential(*[DecoderBlock() for _ in range(Config.n_layer)])
        self.ln_f = nn.LayerNorm(Config.n_embd)
        self.lm_head = nn.Linear(Config.n_embd, vocab_size)

    def forward(self, x, x_mask, ca, ca_mask):
        B, T = x.shape
        x = self.input_emb(x) * math.sqrt(Config.n_embd)
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        for block in self.blocks:
            x = block(x, x_mask, ca, ca_mask)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(Config.vocab_size)
        self.decoder = Decoder(Config.vocab_size)

    def forward(self, x, x_mask, y, y_mask, ca_mask, targets=None):
        ca_x = self.encoder(y, y_mask)
        logits = self.decoder(x, x_mask, ca_x, ca_mask)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets, ignore_index=Config.PAD_TOKEN)
        return logits, loss

    def generate(self, eng, eng_l, idx, max_new_tokens):
        # Move idx to the correct device
        idx = idx.to(Config.device)
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -Config.block_size:]
            # Use torch.nn.functional.pad for padding
            idx_pad = F.pad(idx_cond, (0, Config.block_size - idx_cond.size(1)), value=Config.PAD_TOKEN)
            
            # Ensure masks are generated correctly
            eng_m, out_m, ca_m = generate_masks(eng, idx_pad)
            
            # Get the predictions
            logits, _ = self(idx_pad, out_m, eng, eng_m, ca_m)
            logits = logits[:, idx_cond.size(1) - 1, :]  # Focus on the last time step
            probs = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Check for end token
            if (idx_next == Config.END_TOK).all():
                break

        return idx