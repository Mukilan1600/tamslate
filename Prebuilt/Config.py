import torch

from Vocabulary import vocab

# B-32: 55
# B-64: 117.5
# B-128: 240
class Config:
    batch_size = 128 # how many independent sequences will we process in parallel?
    block_size = 256
    vocab_size = len(vocab)
    data_rows = 200000
    epochs = 400
    max_steps = (data_rows//batch_size) * epochs
    eval_interval = 500
    save_iter = 1500
    step_interval = 50
    max_lr = 6e-4
    min_lr = 0.1 * max_lr
    warmup_steps = max_steps * 0.01
    decay_steps = max_steps * 0.95
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 50
    test_iters = 10
    n_embd = 512
    n_head = 8
    n_layer = 2
    dropout = 0.1
    PAD_TOKEN = 0
    START_TOKEN = 1
    END_TOK = 2
    p = 0.9
    CONTIUE = False
# ------------