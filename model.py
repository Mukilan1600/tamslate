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

class PositionalEmbedding(nn.Module):
    def __init__(self, block_size, d_model):
        super().__init__()
        
        # Create a long enough position encoding table
        pe = torch.zeros(block_size, d_model)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is expected to have shape (batch_size, seq_len, d_model)
        return self.pe[:x.size(1), :]

class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.input_emb = nn.Embedding(vocab_size, Config.n_embd)
        self.positional_encoding = PositionalEmbedding(Config.block_size, Config.n_embd)
        self.blocks = nn.Sequential(*[EncoderBlock() for _ in range(Config.n_layer)])

    def forward(self, x, mask):
        B, T = x.shape
        x = self.input_emb(x) * math.sqrt(Config.n_embd) # B, T, C
        x = x + self.positional_encoding(x) # B, T, C + B, T
        for block in self.blocks:
            x = block(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.input_emb = nn.Embedding(vocab_size, Config.n_embd)
        self.positional_encoding = PositionalEmbedding(Config.block_size, Config.n_embd)
        self.blocks = nn.Sequential(*[DecoderBlock() for _ in range(Config.n_layer)])
        self.ln_f = nn.LayerNorm(Config.n_embd)
        self.lm_head = nn.Linear(Config.n_embd, vocab_size)

    def forward(self, x, x_mask, ca, ca_mask):
        B, T = x.shape
        x = self.input_emb(x) * math.sqrt(Config.n_embd)
        x = x + self.positional_encoding(x) # B, T, C + B, T
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
    
    def _sample_top_p_(self, probs: torch.tensor) -> torch.tensor:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = (probs_sum - probs_sort) > Config.p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        idx_next = torch.multinomial(probs_sort, num_samples=1)
        idx_next = torch.gather(probs_idx, -1, idx_next)

        return idx_next

    def generate(self, eng, eng_l, idx, max_new_tokens):
        # Move idx to the correct device
        idx = idx.to(Config.device)
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -Config.block_size:]
            # Use torch.nn.functional.pad for padding
            idx_pad = F.pad(idx_cond, (0, Config.block_size - idx_cond.size(1)), value=Config.PAD_TOKEN)

            print(eng)
            
            # Ensure masks are generated correctly
            eng_m, out_m, ca_m = generate_masks(eng, idx_pad)
            
            # Get the predictions
            logits, _ = self(idx_pad, out_m, eng, eng_m, ca_m)
            logits = logits[:, idx_cond.size(1) - 1, :]  # Focus on the last time step
            probs = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities
            
            # Sample from the distribution
            idx_next = self._sample_top_p_(probs)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Check for end token
            if (idx_next == Config.END_TOK).all():
                break

        return idx