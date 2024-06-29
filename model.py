import torch
import torch.nn as nn
from torch.nn import functional as F
from train import Config, generate_masks

class SelfAttentionHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(Config.n_embd, head_size, bias=False)
        self.query = nn.Linear(Config.n_embd, head_size, bias=False)
        self.value = nn.Linear(Config.n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x, mask):

        B,T,C = x.shape

        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(x) # (B,T,C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)

        wei = wei.masked_fill(mask == 0, float(-1e9)) # (B, T, T)

        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out



class CrossAttentionHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(Config.n_embd, head_size, bias=False)
        self.query = nn.Linear(Config.n_embd, head_size, bias=False)
        self.value = nn.Linear(Config.n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x, y, mask):
        B,T,C = x.shape

        k = self.key(y)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(y) # (B,T,C)


        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)


        wei = wei.masked_fill(mask == 0, float(-1e9)) # (B, T, T)

        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
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
        out = self.dropout(self.proj(out))
        return out

class MultiHeadCrossAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(Config.n_embd, Config.n_embd)
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x, y, mask):
        out = torch.cat([h(x, y, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(Config.dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = Config.n_embd // Config.n_head
        self.self_att= MultiHeadSelfAttention(Config.n_head, head_size)
        self.ln1 = nn.LayerNorm(Config.n_embd)
        self.ln2 = nn.LayerNorm(Config.n_embd)
        self.ffwd = FeedFoward(Config.n_embd)

    def forward(self, o):
        x, m = o["x"], o["m"]

        x = x + self.self_att(self.ln1(x), m)
        x = x + self.ffwd(self.ln2(x))

        return {"x":x, "m":m}


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = Config.n_embd // Config.n_head
        self.self_att = MultiHeadSelfAttention(Config.n_head, head_size)
        self.cross_att = MultiHeadCrossAttention(Config.n_head, head_size)
        self.ln1 = nn.LayerNorm(Config.n_embd)
        self.ln2 = nn.LayerNorm(Config.n_embd)
        self.ln3 = nn.LayerNorm(Config.n_embd)
        self.ffwd = FeedFoward(Config.n_embd)

        self.register_buffer("tril", torch.tril(torch.ones(Config.block_size, Config.block_size, dtype=torch.int, device=Config.device)))


    def forward(self, o):

        x, x_m, ca, ca_m = o["x"], o["x_m"], o["ca"], o["ca_m"]

        x = x + self.self_att(self.ln1(x), x_m & self.tril)
        x = x + self.cross_att(self.ln2(x), ca, ca_m)
        x = x + self.ffwd(self.ln3(x))

        return {"x":x, "x_m":x_m, "ca":ca, "ca_m":ca_m}

class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.input_emb = nn.Embedding(vocab_size, Config.n_embd)
        self.position_emb = nn.Embedding(Config.block_size, Config.n_embd)
        self.blocks = nn.Sequential(*[EncoderBlock() for _ in range(Config.n_layer)])

    def forward(self, x, mask):
        B, T = x.shape

        tok_emb = self.input_emb(x) # B, T, C
        pos_emb = self.position_emb(torch.arange(T, device=Config.device)) # T, C

        x = tok_emb + pos_emb # B, T, C

        o = {"x": x, "m": mask}
        x = self.blocks(o)["x"]

        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.input_embedding = nn.Embedding(vocab_size, Config.n_embd)
        self.position_embedding = nn.Embedding(Config.block_size, Config.n_embd)
        self.blocks = nn.Sequential(*[DecoderBlock() for _ in range(Config.n_layer)])
        self.ln_f = nn.LayerNorm(Config.n_embd)
        self.lm_head = nn.Linear(Config.n_embd, vocab_size)

    def forward(self, x, x_mask, ca, ca_mask):
        B, T = x.shape

        tok_emb = self.input_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=Config.device))

        x = tok_emb + pos_emb
        o = {"x": x, "x_m": x_mask, "ca": ca, "ca_m": ca_mask}
        x = self.blocks(o)["x"]

        x = self.ln_f(x)
        x = self.lm_head(x)

        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.encoder = Encoder(Config.vocab_Size)
        self.decoder = Decoder(Config.vocab_Size)

    def forward(self, x, x_mask, y, y_mask, ca_mask, targets=None):

        # Encoder
        ca_x = self.encoder(y, y_mask)

        logits = self.decoder(x, x_mask, ca_x, ca_mask)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=Config.PAD_TOKEN)

        return logits, loss
    
    def generate(self, eng, eng_l, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # crop idx to the last block_size tokens
            idx_cond = idx[:, -Config.block_size:]
            idx_pad = torch.tensor([[*x, *[Config.PAD_TOKEN for _ in range(Config.block_size-len(x))]] for x in idx_cond], dtype=torch.long, device=Config.device)

            out_m, eng_m, ca_m = generate_masks(eng, idx_cond)




            # get the predictions
            logits, loss = self(idx_pad, out_m, eng, eng_m, ca_m)
            # focus only on the last time step
            logits = logits[:, len(idx_cond[0])-1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)


            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

            if idx_next == Config.END_TOK:
              break
        return idx