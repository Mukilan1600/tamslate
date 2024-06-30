from pathlib import Path
from model import BigramLanguageModel
import torch
from utils import Config, encode, decode
import tiktoken

model = BigramLanguageModel()
model = model.to(Config.device)
model = torch.compile(model)

model_save_path = "./checkpoint/model.pt"
checkpoint_file = Path(model_save_path)
if checkpoint_file.is_file():
    print("Loading checkpoint...")
    print(model.load_state_dict(torch.load(model_save_path)))

model.eval()


def generate(str):
    eng_ctx = encode(str)
    eng_l = torch.tensor([min(len(eng_ctx), Config.block_size)], dtype=torch.long, device=Config.device)
    eng_ctx =  [*eng_ctx, *[Config.PAD_TOKEN for _ in range(Config.block_size-len(eng_ctx))]]

    out = [[Config.START_TOKEN]]

    out_txt = decode(model.generate(eng = torch.tensor([eng_ctx], dtype=torch.long, device=Config.device), eng_l = eng_l, idx = torch.tensor(out, dtype=torch.long, device=Config.device), max_new_tokens=256)[0].tolist())
    with open("infer.txt", 'a') as f:
            f.write(f"{out_txt}\n")
    print(out_txt)

generate("My name is Abdul")
generate("The weather is nice today")
generate("Information has surfaced in recent years suggesting that Julius Rosenberg was involved in passing some form of intelligence to Soviet officials during the Second World War.")