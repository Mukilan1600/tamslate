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

encode = lambda s: [kv[c] for c in s if c in kv] # encoder: take a string, output a list of integers
decode = lambda s: "".join([vk[c] for c in s]) # decoder: takes a list of encodings, output a string

# hyperparameters
class Config:
    batch_size = 32 # how many independent sequences will we process in parallel?
    block_size = 256
    vocab_size = len(vocab)
    data_rows = 200000
    epochs = 400
    max_steps = (data_rows//batch_size) * epochs
    eval_interval = 10000
    step_interval = 100
    max_lr = 6e-4
    min_lr = 0.1 * max_lr
    warmup_steps = max_steps * 0.01
    decay_steps = max_steps * 0.95
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 100
    test_iters = 10
    n_embd = 512
    n_head = 4
    n_layer = 2
    dropout = 0.1
    PAD_TOKEN = 0
    START_TOKEN = 1
    END_TOK = 2
    CURRENT_ITER = 0
    p = 0.9
# ------------


def generate_masks(en, ta):
    '''Generate look ahead and cross-attention masks'''
    size = len(en)
    block_size = Config.block_size
    device = Config.device

    # Ensure input tensors are on the correct device
    en = en.to(device)
    ta = ta.to(device)

    # Calculate sequence lengths
    en_lens = (en != Config.PAD_TOKEN).sum(1)
    ta_lens = (ta != Config.PAD_TOKEN).sum(1)

    # Create base masks
    base_mask = torch.ones(block_size, block_size, dtype=torch.bool, device=device).unsqueeze(0).expand(size, -1, -1)
    
    # Generate masks
    en_m = base_mask.clone()
    ta_m = base_mask.clone()
    ca_m = base_mask.clone()

    # Create indices for masking
    seq_indices = torch.arange(block_size, device=device)

    # Apply sequence length masks
    en_mask = seq_indices[None, :] >= en_lens[:, None, None]
    ta_mask = seq_indices[None, :] >= ta_lens[:, None, None]

    en_m = en_m & ~en_mask & ~en_mask.transpose(-1, -2)
    ta_m = ta_m & ~ta_mask & ~ta_mask.transpose(-1, -2)
    ca_m = ca_m & ~en_mask & ~ta_mask.transpose(-1, -2)

    return en_m.int(), ta_m.int(), ca_m.int()


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