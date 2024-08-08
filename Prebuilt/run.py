from Train import generate

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
model.eval()
while True:
    s = input("Enter english: ").strip()
    print("Tamil: ", end=" ")
    out = generate(s, model)
    with open("run.txt", "a") as f:
        f.write(f"{decode(out)}\n")
    print()