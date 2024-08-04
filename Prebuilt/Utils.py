import torch
import shutil

from Config import Config

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


def generate_masks(src, tgt):
        '''Generate look ahead and cross-attention masks'''

        src_m = src == Config.PAD_TOKEN
        tgt_m = tgt == Config.PAD_TOKEN

        return src_m, tgt_m, src_m