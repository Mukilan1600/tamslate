from pathlib import Path
from model import BigramLanguageModel
import torch
from utils import Config
import tiktoken

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


encode = lambda x: enc.encode(x, allowed_special='all')
decode = lambda x: enc.decode(x)

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

    out = [[encode('<|START|>')[0]]]
    out_txt = decode(model.generate(eng = torch.tensor([eng_ctx], dtype=torch.long, device=Config.device), eng_l = eng_l, idx = torch.tensor(out, dtype=torch.long, device=Config.device), max_new_tokens=256)[0].tolist())
    with open("infer.txt", 'w') as f:
            f.write(f"{out_txt}\n")
    print(out_txt)

generate("My name is Abdul")
generate("The weather is nice today")
generate("Information has surfaced in recent years suggesting that Julius Rosenberg was involved in passing some form of intelligence to Soviet officials during the Second World War.")