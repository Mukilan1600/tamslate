from model import BigramLanguageModel
import tiktoken
from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import namedtuple

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
batch_size = 32 # how many independent sequences will we process in parallel?
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
  
torch.set_float32_matmul_precision("high")

model = BigramLanguageModel()
model = model.to(device)
model = torch.compile(model)

# Print the model size and parameter size
param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2

print(f'Model size: {size_all_mb:.3f}MB, Params: {sum(p.numel() for p in model.parameters())/1e6, 'M'}')

train_dl = DataLoader('train')
val_dl = DataLoader('validation')

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
for iter in range(40):
    t0 = time.time()
    # sample a batch of data
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
      # evaluate the loss
      logits, loss = model(Xo, XoM, Xi, XiM, CaM, Y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000
    print(f"Step {iter}, loss: {loss.item():.2f}, dt: {dt:.2f}")