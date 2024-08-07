import torch

from Model import MTModel
from pathlib import Path
from Utils import load_ckp
from Config import Config
from Vocabulary import decode, encode


model = MTModel(Config)
model = model.to(Config.device)


best_model_path = "./checkpoint/best_model.pt"
checkpoint_file = Path(best_model_path)
if checkpoint_file.is_file():
    model, optimizer, start_epoch = load_ckp(best_model_path, model, None)
    print(f"Module loaded")

def generate(str):
    src = encode(str.lower())
    src =  [Config.START_TOKEN, *src, Config.END_TOK, *[Config.PAD_TOKEN for _ in range(Config.block_size-len(src)-2)]]

    out = [[Config.START_TOKEN]]

    out_txt = model.generate(torch.tensor([src], dtype=torch.long, device=Config.device), torch.tensor(out, dtype=torch.long, device=Config.device), max_new_tokens=256)[0].tolist()
    return out_txt

out_1 = decode(generate("mma vice president qazi hussain ahmad declared last month: 'we are not extremists."))
out_2 = ""#decode(generate("Information has surfaced in recent years suggesting that Julius Rosenberg was involved in passing some form of intelligence to Soviet officials during the Second World War."))

with open("./logs/inf_log.txt", 'a') as f:
    f.write(f"Sample 1: {out_1}<|eos|> | Sample 2: {out_2}<|eos|>\n")