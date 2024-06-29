import tiktoken
from datasets import load_dataset
from collections import namedtuple
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)



gpt_4o_enc = tiktoken.encoding_for_model("gpt-4o")
# In production, load the arguments directly instead of accessing private attributes
# See openai_public.py for examples of arguments for specific encodings
enc = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="cl100k_im",
    pat_str=gpt_4o_enc._pat_str,
    mergeable_ranks=gpt_4o_enc._mergeable_ranks,
    special_tokens={
        **gpt_4o_enc._special_tokens,
        "<|PAD|>": gpt_4o_enc.n_vocab,
        "<|START|>": gpt_4o_enc.n_vocab+1,
        "<|END|>": gpt_4o_enc.n_vocab+2,
    }
)

PAD_TOKEN = enc.encode('<|PAD|>', allowed_special='all')[0]

encode = lambda x: enc.encode(x, allowed_special='all')
decode = lambda x: enc.decode(x)

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 256
vocab_Size = enc.n_vocab
max_iters = 200000
eval_interval = 1000
learning_rate = 3e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 512
n_head = 8
n_layer = 10
dropout = 0.2
# ------------

# Load dataset from Huggingface
ds = load_dataset("Hemanth-thunder/en_ta")

def generate_masks(en, ta):
  '''Generate look ahead and cross-attention masks'''
  size = len(en)
  en_m = torch.zeros(size, block_size, block_size, dtype=torch.int, device=device)
  ta_m = torch.zeros(size, block_size, block_size, dtype=torch.int, device=device)
  ca_m = torch.zeros(size, block_size, block_size, dtype=torch.int, device=device)

  en = (en != PAD_TOKEN).sum(1)
  ta = (ta != PAD_TOKEN).sum(1)

  for i in range(size):
    en_m[i, :en[i], :en[i]] = 1
    ta_m[i, :ta[i], :ta[i]] = 1
    ca_m[i, :ta[i], :en[i]] = 1


  return en_m, ta_m, ca_m

def pad(x):
  '''Pad a sequence to the desired block size'''
  if len(x) < block_size:
        x = x + [PAD_TOKEN for _ in range(block_size - len(x))]
  return x


def prepare_dataset(x):
  '''Encode and pad each sentence in the dataset'''
  del x['Unnamed: 0']
  en = encode(x['en'])
  ta = encode('<|START|>' + x['ta'] + '<|PAD|>')

  x['en'] = pad(en)
  x['ta'] = pad(ta)
  x['en_len'] = len(en)
  x['ta_len'] = len(ta)

  return x

encoded_ds = ds.map(prepare_dataset)
encoded_ds = encoded_ds.filter(lambda x: len(x['en']) == 256 and len(x['ta']) == 256)
encoded_ds = encoded_ds.with_format('torch')


class DataLoader():
  '''Lite dataloader class to load batch of sentences'''
  DataBatch = namedtuple('DataBatch', ['output', 'out_mask', 'input', 'in_mask', 'ca_mask', 'targets'])
  def __init__(self, split, current_batch=0):
    self.split = split
    self.dataset = encoded_ds[split]
    self.current_batch = (current_batch) % (self.dataset.num_rows//batch_size)

  def get_next_batch(self):
    idx = self.current_batch * batch_size
    xi = self.dataset[idx : idx + batch_size]['en']
    xo = self.dataset[idx : idx + batch_size]['ta']

    t_m, e_m, ca_m = generate_masks(xi, xo)

    ta_padding = torch.full((batch_size, 1), PAD_TOKEN)
    y = torch.cat((xo[:, 1:], ta_padding), dim=1)

    xi, xo, y = xi.to(device), xo.to(device), y.to(device)

    self.current_batch = (self.current_batch + 1) % (self.dataset.num_rows//batch_size)

    if self.current_batch == 0:
      encoded_ds.shuffle()
      self.dataset = encoded_ds[self.split]

    return DataLoader.DataBatch(xo, t_m, xi, e_m, ca_m, y)
  
'''Model'''
class SelfAttentionHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

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

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

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
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class MultiHeadCrossAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

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
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.self_att= MultiHeadSelfAttention(n_head, head_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedFoward(n_embd)

    def forward(self, o):
        x, m = o["x"], o["m"]

        x = x + self.self_att(self.ln1(x), m)
        x = x + self.ffwd(self.ln2(x))

        return {"x":x, "m":m}


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.self_att = MultiHeadSelfAttention(n_head, head_size)
        self.cross_att = MultiHeadCrossAttention(n_head, head_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.ffwd = FeedFoward(n_embd)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size, dtype=torch.int, device=device)))


    def forward(self, o):

        x, x_m, ca, ca_m = o["x"], o["x_m"], o["ca"], o["ca_m"]

        x = x + self.self_att(self.ln1(x), x_m & self.tril)
        x = x + self.cross_att(self.ln2(x), ca, ca_m)
        x = x + self.ffwd(self.ln3(x))

        return {"x":x, "x_m":x_m, "ca":ca, "ca_m":ca_m}

class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.input_emb = nn.Embedding(vocab_size, n_embd)
        self.position_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[EncoderBlock() for _ in range(n_layer)])

    def forward(self, x, mask):
        B, T = x.shape

        tok_emb = self.input_emb(x) # B, T, C
        pos_emb = self.position_emb(torch.arange(T, device=device)) # T, C

        x = tok_emb + pos_emb # B, T, C

        o = {"x": x, "m": mask}
        x = self.blocks(o)["x"]

        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.input_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[DecoderBlock() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, x_mask, ca, ca_mask):
        B, T = x.shape

        tok_emb = self.input_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=device))

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
        self.encoder = Encoder(enc.n_vocab)
        self.decoder = Decoder(enc.n_vocab)

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
            loss = F.cross_entropy(logits, targets, ignore_index=PAD_TOKEN)

        return logits, loss

    def generate(self, eng, eng_l, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        END_TOK = encode('<|END|>')[0]
        for _ in range(max_new_tokens):

            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            idx_pad = torch.tensor([[*x, *[PAD_TOKEN for _ in range(block_size-len(x))]] for x in idx_cond], dtype=torch.long, device=device)

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

            if idx_next == END_TOK:
              break
        return idx
"""--------------------"""
  
torch.set_float32_matmul_precision("high")

model = BigramLanguageModel()
model = model.to(device)
# model = torch.compile(model)

# Print the model size and parameter size
param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2

print(f'Model size: {size_all_mb:.3f}MB, Params: {sum(p.numel() for p in model.parameters())/1e6}M ')

train_dl = DataLoader('train')
val_dl = DataLoader('validation')

max_lr = 3e-5
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = warmup_steps * 10

def get_lr(step):
    if step < warmup_steps:
        return max_lr * ((step + 1)/warmup_steps)
    
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps)/(max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
    

@torch.no_grad()
def estimate_loss():
    '''Estimate average loss over eval_iters steps'''
    out = {}
    model.eval()
    for split in ['train', 'validation']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split=='train':
              Xo, XoM, Xi, XiM, CaM, Y = train_dl.get_next_batch()
            else:
              Xo, XoM, Xi, XiM, CaM, Y = val_dl.get_next_batch()
            logits, loss = model(Xo, XoM, Xi, XiM, CaM, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

from copy import deepcopy
import time
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

min_loss = 12
Xo, XoM, Xi, XiM, CaM, Y = train_dl.get_next_batch()
for step in range(40):
    t0 = time.time()
    # sample a batch of data
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
      # evaluate the loss
      logits, loss = model(Xo, XoM, Xi, XiM, CaM, Y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm(model.parameters() , 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000
    print(f"Step {step:4d} | norm: {norm:.4f} | lr: {lr:.6f} | loss: {loss.item():.2f} | dt: {dt:.2f}")