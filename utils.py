import torch

# hyperparameters
class Config:
    batch_size = 24 # how many independent sequences will we process in parallel?
    block_size = 256
    vocab_Size = 200022
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
    n_layer = 10
    dropout = 0.2   
    PAD_TOKEN = 200019
    END_TOK = 200021
    CURRENT_ITER = 14171
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