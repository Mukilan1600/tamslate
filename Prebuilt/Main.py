import torch

from Model import MTModel

a = torch.randint(0, 20, (3, 256))
b = torch.randint(0, 20, (3, 256))

x,y,z = MTModel._generate__masks_(None, a, b)