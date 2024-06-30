import torch
import string

# Define the Unicode range for Tamil characters
tamil_range_start = 0x0B80
tamil_range_end = 0x0BFF

# Set of punctuation characters for quick lookup
punctuation_set = set(string.punctuation)
special_tokens = ["<|PAD|>", "<|START|>", "<|END|>"]

# Combined filter set for Characters, punctuation, and space characters
vocab = [*special_tokens, *sorted(list(set(string.ascii_letters) | {chr(i) for i in range(tamil_range_start, tamil_range_end + 1)} | set(string.digits) | punctuation_set | {' '}))]

kv = {v:k for k,v in enumerate(vocab)}
vk = {k:v for k,v in enumerate(vocab)}

encode = lambda s: [kv[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda s: [vk[c] for c in s] # encoder: take a string, output a list of integers

# hyperparameters
class Config:
    batch_size = 64 # how many independent sequences will we process in parallel?
    block_size = 256
    vocab_size = len(vocab)
    max_steps = 20000
    eval_interval = 1000
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.05
    decay_steps = max_steps * 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 100
    n_embd = 512
    n_head = 8
    n_layer = 3
    dropout = 0.2   
    PAD_TOKEN = 0
    START_TOKEN = 1
    END_TOK = 2
    CURRENT_ITER = 1
# ------------


def generate_masks(en, ta):
  '''Generate look ahead and cross-attention masks'''
  size = len(en)
  en_m = torch.zeros(size, Config.block_size, Config.block_size, dtype=torch.int, device=Config.device)
  ta_m = torch.zeros(size, Config.block_size, Config.block_size, dtype=torch.int, device=Config.device)
  ca_m = torch.zeros(size, Config.block_size, Config.block_size, dtype=torch.int, device=Config.device)

  en = (en != Config.PAD_TOKEN).sum(1)
  ta = (ta != Config.PAD_TOKEN).sum(1)

  for i in range(size):
    en_m[i, :en[i], :en[i]] = 1
    ta_m[i, :ta[i], :ta[i]] = 1
    ca_m[i, :ta[i], :en[i]] = 1


  return en_m, ta_m, ca_m