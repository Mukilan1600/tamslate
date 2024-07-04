from pathlib import Path
import time
from model import BigramLanguageModel
from torchtext.data.metrics import bleu_score
from datasets import load_dataset
from collections import namedtuple
import torch
import math

from utils import Config, generate_masks, encode, decode, load_ckp, save_ckp, vocab

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

def pad(x):
  '''Pad a sequence to the desired block size'''

  if len(x) < Config.block_size:
        x = x + [Config.PAD_TOKEN for _ in range(Config.block_size-len(x))]
  return x


def prepare_dataset(x):
  '''Encode and pad each sentence in the dataset'''
  del x['Unnamed: 0']

  en = encode(x['en'].strip().lower())
  ta = [Config.START_TOKEN, *encode(x['ta'].strip().lower()), Config.END_TOK]

  x['en'] = pad(en)
  x['ta'] = pad(ta)
  x['en_len'] = len(en)
  x['ta_len'] = len(ta)

  return x

if __name__=='__main__':
   
    # Load dataset from Huggingface
    ds = load_dataset("Hemanth-thunder/en_ta")
    
    print("Original size: ", ds['train'].num_rows)
    encoded_ds = ds.filter(lambda x: sum([0 if i in vocab else 1 for i in x['en'].strip()])==0 and sum([0 if i in vocab else 1 for i in x['ta'].strip()]) == 0)
    print("Filtered for valid characters: ", encoded_ds['train'].num_rows)
    encoded_ds = encoded_ds.map(prepare_dataset)
    encoded_ds = encoded_ds.filter(lambda x: len(x['en'])==Config.block_size and len(x['ta']) == Config.block_size)
    print("Filtered for block size: ", encoded_ds['train'].num_rows)
    encoded_ds = encoded_ds.filter(lambda x,t: t <= Config.data_rows, with_indices=True)
    print("Filtered for total size: ", encoded_ds['train'].num_rows)
    
    encoded_ds = encoded_ds.with_format('torch')


    class DataLoader():
        '''Lite dataloader class to load batch of sentences'''
        DataBatch = namedtuple('DataBatch', ['output', 'out_mask', 'input', 'in_mask', 'ca_mask', 'targets'])
        def __init__(self, split, current_batch=0):
            self.split = split
            self.dataset = encoded_ds[split]
            self.current_batch = (current_batch) % (self.dataset.num_rows//Config.batch_size)

        def get_next_batch(self):
            idx = self.current_batch * Config.batch_size
            xi = self.dataset[idx : idx + Config.batch_size]['en']
            xo = self.dataset[idx : idx + Config.batch_size]['ta']

            e_m, t_m, ca_m = generate_masks(xi, xo)

            ta_padding = torch.full((Config.batch_size, 1), Config.PAD_TOKEN)
            y = torch.cat((xo[:, 1:], ta_padding), dim=1)

            xi, xo, y = xi.to(Config.device), xo.to(Config.device), y.to(Config.device)
            self.current_batch = (self.current_batch + 1) % (self.dataset.num_rows//Config.batch_size)

            if self.current_batch == 0:
                shuffled_ds = encoded_ds.shuffle()
                self.dataset = shuffled_ds[self.split]

            return DataLoader.DataBatch(xo, t_m, xi, e_m, ca_m, y)
        
        def get_test_batch(self):
            idx = self.current_batch
            xi = self.dataset[idx]['en'].tolist()
            xo = self.dataset[idx]['ta'].tolist()
            
            self.current_batch = (self.current_batch + 1) % (self.dataset.num_rows)
            
            if self.current_batch == 0:
                shuffled_ds = encoded_ds.shuffle()
                self.dataset = shuffled_ds[self.split]

            return (xi, xo)

    torch.set_float32_matmul_precision("high")

    model = BigramLanguageModel()
    model = model.to(Config.device)
    model = torch.compile(model)

    print("---------------------------")
    # Print the model size and parameter size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    print(f'Model size: {size_all_mb:.3f}MB, Params: {sum(p.numel() for p in model.parameters())/1e6}M ')
    print("---------------------------")
    print("".join([f"{v:18s}: {m}\n" for v, m in vars(Config).items() if not (v.startswith('_')  or callable(m))]))
    print("---------------------------")


        # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.min_lr)
    start_epoch = 0

    best_model_path = "./checkpoint/best_model.pt"
    checkpoint_file = Path(best_model_path)
    if checkpoint_file.is_file():
        model, optimizer, start_epoch = load_ckp(best_model_path, model, optimizer)
        print(f"Resuming training from iteration: {start_epoch}...")

    train_dl = DataLoader('train', start_epoch)
    val_dl = DataLoader('validation', start_epoch)
    test_dl = DataLoader('test', start_epoch)

    def get_lr(step):
        if step < Config.warmup_steps:
            return Config.max_lr * ((step + 1)/Config.warmup_steps)
        
        if step > Config.decay_steps:
            return Config.min_lr
        
        decay_ratio = (step - Config.warmup_steps)/(Config.decay_steps - Config.warmup_steps)
        assert 0 <= decay_ratio <= 1

        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return Config.min_lr + coeff * (Config.max_lr - Config.min_lr)


    @torch.no_grad()
    def estimate_loss():
        '''Estimate average loss over eval_iters steps'''
        out = {}
        for split in ['train', 'validation']:
            losses = torch.zeros(Config.eval_iters)
            for k in range(Config.eval_iters):
                if split=='train':
                    batch = train_dl.get_next_batch()
                else:
                    batch = val_dl.get_next_batch()
                
                logits, loss = model(batch.output, batch.out_mask, batch.input, batch.in_mask, batch.ca_mask, batch.targets)
                losses[k] = loss.item()
            out[split] = losses.mean()
        return out

    from copy import deepcopy

    @torch.no_grad()
    def generate(str):
        eng_ctx = encode(str.lower())
        eng_l = torch.tensor([min(len(eng_ctx), Config.block_size)], dtype=torch.long, device=Config.device)
        eng_ctx =  [*eng_ctx, *[Config.PAD_TOKEN for _ in range(Config.block_size-len(eng_ctx))]]

        out = [[Config.START_TOKEN]]

        out_txt = model.generate(eng = torch.tensor([eng_ctx], dtype=torch.long, device=Config.device), eng_l = eng_l, idx = torch.tensor(out, dtype=torch.long, device=Config.device), max_new_tokens=256)[0].tolist()
        return out_txt

    @torch.no_grad()
    def evaluate_bleu():
        model.eval()
        original = []
        predictions = []

        with torch.no_grad():
            for i in range(Config.test_iters):
                # Extract inputs from batch
                input, out = test_dl.get_test_batch()
                en = decode([x for x in input if x != Config.PAD_TOKEN])
                ta = decode([x for x in out[1:-2] if x != Config.PAD_TOKEN])
                original.append(ta.split())
                # Generate predictions
                preds = generate(en)

                predictions.append(decode(preds[1:-2]).split())
        # Calculate BLEU score
        bleu_score_val = bleu_score(predictions, original, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])

        return bleu_score_val

    min_loss = 12
    for iter in range(start_epoch, Config.max_steps):

        t0 = time.time()

        # every once in a while evaluate the loss on train and val sets
        if iter % Config.eval_interval == 0 or iter == Config.max_steps - 1:
            model.eval()
            with torch.no_grad():
                losses = estimate_loss()
                bleu = 0
                print(f"Eval step {iter:4d} | train loss {losses['train']:.4f} | val loss {losses['validation']:.4f} | BLUE: {bleu:.4f}")
                with open("loss_log.txt", 'a') as f:
                    f.write(f"Eval step {iter:4d} | train loss {losses['train']:.4f} | val loss {losses['validation']:.4f} | BLUE: {bleu:.4f}\n")

                checkpoint = {
                    'epoch': iter + 1,
                    'state_dict': deepcopy(model.state_dict()),
                    'optimizer': optimizer.state_dict()
                }
                if losses['validation'] <= min_loss:
                    save_ckp(checkpoint, True, f'./checkpoint/iter_{iter}_model.pt', best_model_path)
                else:
                    save_ckp(checkpoint, False, f'./checkpoint/iter_{iter}_model.pt', best_model_path)

                min_loss = losses['validation']
                out_1 = decode(generate("mma vice president qazi hussain ahmad declared last month: 'we are not extremists."))
                out_2 = decode(generate("Hello, how are you?"))

                with open("gen_log.txt", 'a') as f:
                    f.write(f"Eval step {iter:4d} | Sample 1: {out_1}<|eos|>\n")
                    f.write(f"Eval step {iter:4d} | Sample 2: {out_2}<|eos|>\n")
            model.train()

        optimizer.zero_grad(set_to_none=True)

        # sample a batch of data
        batch = train_dl.get_next_batch()

        with torch.autocast(device_type=Config.device, dtype=torch.bfloat16):
            # evaluate the loss
            logits, loss = model(batch.output, batch.out_mask, batch.input, batch.in_mask, batch.ca_mask, batch.targets)

        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(iter)
        for params in optimizer.param_groups:
            params['lr'] = lr

        optimizer.step()

        t1 = time.time()
        dt = (t1 - t0) * 1000
        print(f"Step {iter:4d} | norm: {norm:.4f} | lr: {lr:.6f} | loss: {loss.item():.4f} | dt: {dt:.2f}")
        with open("step_log.txt", 'a') as f:
            f.write(f"Step {iter:4d} | norm: {norm:.4f} | lr: {lr:.6f} | loss: {loss.item():.4f} | dt: {dt:.2f}\n")