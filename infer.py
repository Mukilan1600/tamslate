from pathlib import Path
from model import BigramLanguageModel
import torch
from utils import Config, encode, decode, load_ckp
import tiktoken

model = BigramLanguageModel()
model = model.to(Config.device)
model = torch.compile(model)

model_save_path = "./checkpoint/best_model.pt"
checkpoint_file = Path(model_save_path)
if checkpoint_file.is_file():
    print("Loading checkpoint...")
    model, optimizer, start_epoch = load_ckp(model_save_path, model, None)

model.eval()


def generate(str):
    eng_ctx = encode(str.strip().lower())
    eng_l = torch.tensor([min(len(eng_ctx), Config.block_size)], dtype=torch.long, device=Config.device)
    eng_ctx =  [Config.START_TOKEN ,*eng_ctx, Config.END_TOK, *[Config.PAD_TOKEN for _ in range(Config.block_size-len(eng_ctx)-2)]]

    out = [[Config.START_TOKEN]]

    out_txt = decode(model.generate(eng = torch.tensor([eng_ctx], dtype=torch.long, device=Config.device), eng_l = eng_l, idx = torch.tensor(out, dtype=torch.long, device=Config.device), max_new_tokens=256)[0].tolist())
    with open("./logs/infer.txt", 'a') as f:
            f.write(f"{out_txt}\n")
    # print(out_txt)


if __name__=="__main__":
    generate("mma vice president qazi hussain ahmad declared last month: 'we are not extremists.")
    generate("This is india's industry")
    generate("United states of America bombed japan")
    generate("You look ugly")