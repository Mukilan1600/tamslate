import torch
import string
import shutil

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
decode = lambda s: "".join([vk[c] for c in s]) # decoder: takes a list of encodings, output a string

# hyperparameters
class Config:
    batch_size = 128 # how many independent sequences will we process in parallel?
    block_size = 256
    vocab_size = len(vocab)
    data_rows = 50000
    epochs = 15
    max_steps = (data_rows//batch_size) * epochs
    eval_interval = 500
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.05
    decay_steps = max_steps * 0.65
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 50
    test_iters = 10
    n_embd = 512
    n_head = 8
    n_layer = 6
    dropout = 0.1
    PAD_TOKEN = 0
    START_TOKEN = 1
    END_TOK = 2
    CURRENT_ITER = 0
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


def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer=None):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    if not optimizer is None:
      optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']