import string
import numpy as np
from tqdm import tqdm
import itertools
import multiprocessing as mp
import sys

en_file = "dataset/NLLB.en-ta.en"
ta_file = "dataset/NLLB.en-ta.ta"

BLOCK_SIZE=256

# Define the Unicode range for Tamil characters
tamil_range_start = 0x0B80
tamil_range_end = 0x0BFF

# Set of punctuation characters for quick lookup
punctuation_set = set(string.punctuation)
special_tokens = {"<|START|>", "<|END|>", "<|PAD|>"}

# Combined filter set for Characters, punctuation, and space characters
ta_vocab = sorted(list({chr(i) for i in range(tamil_range_start, tamil_range_end + 1)} | set(string.digits) | punctuation_set | {' '} | special_tokens))
en_vocab = sorted(list(set(string.ascii_letters) | set(string.digits) | punctuation_set | {' '} | special_tokens))

ta_kv = {v:k for k,v in enumerate(ta_vocab)}
ta_vk = {k:v for k,v in enumerate(ta_vocab)}

en_kv = {v:k for k,v in enumerate(en_vocab)}
en_vk = {k:v for k,v in enumerate(en_vocab)}

ta_encode = lambda s: [ta_kv[c] for c in s] # encoder: take a string, output a list of integers
en_encode = lambda s: [en_kv[c] for c in s] # encoder: take a string, output a list of integers

ta_decode = lambda s: "".join([ta_vk[c] for c in s]) # decoder: take a list, output a decoded string
en_decode = lambda s: "".join([en_vk[c] for c in s]) # decoder: take a list, output a decoded string

def encode(str, enc, kv, se_tok=False):
    encodings = enc(str)
    if se_tok:
        encodings = [kv['<|START|>'], *encodings, kv['<|END|>']]
    n = len(encodings)
    encodings = [*encodings, *[kv['<|PAD|>'] for _ in range(BLOCK_SIZE-n)]]
    return encodings


def is_valid_sentence(str, vocab, max_size):
    if len(str)>max_size:
        return False
    for c in str:
        if c not in vocab:
            return False
    return True

def process_and_save_shards(eng_file, tam_file, output_prefix, en_vocab, ta_vocab, start_idx=0, max_shard_size=1_000_000, batch_size=8_000_000):
    current = mp.current_process()
    pos = current._identity[0]-1

    en_sentences = []
    ta_sentences = []
    shard_index = start_idx//max_shard_size
    total_sentences = 0

    def save_shard(shard_index, sentences, output_suffix):
        np.save(f"{output_prefix}_{shard_index}_{output_suffix}.npy", np.array(sentences))

    with open(eng_file, 'r', encoding='utf-8') as en_f, open(tam_file, 'r', encoding='utf-8') as ta_f:
        with tqdm(total=max_shard_size, desc=f"Pid #{pos}, Shard #{shard_index}", position=pos) as pbar:
            for en_line, ta_line in zip(itertools.islice(en_f, start_idx, start_idx+batch_size), itertools.islice(ta_f, start_idx, start_idx+batch_size)):
                en_line, ta_line = en_line.strip(), ta_line.strip()
                
                # Skip invalid sentences
                if not is_valid_sentence(en_line, en_vocab, BLOCK_SIZE) or not is_valid_sentence(ta_line, ta_vocab, BLOCK_SIZE-2):
                    pbar.update(1)
                    continue
                
                en_sentences.append(encode(en_line, en_encode, en_kv))
                ta_sentences.append(encode(ta_line, ta_encode, ta_kv, se_tok=True))
                total_sentences += 1

                if len(en_sentences) == max_shard_size:
                    save_shard(shard_index, en_sentences, "en")
                    save_shard(shard_index, ta_sentences, "ta")
                    shard_index += 1
                    en_sentences = []
                    ta_sentences = []
                pbar.update(1)

    if en_sentences or ta_sentences:
        save_shard(shard_index, en_sentences, "en")
        save_shard(shard_index, ta_sentences, "ta")


if __name__ == '__main__':
    mp.freeze_support()
    n_workers = mp.cpu_count() // 2

    output_prefix = "data_encodings/shard"

    idx = [(en_file, ta_file, output_prefix, en_vocab, ta_vocab, i, 1_000_000, 1_000_000) for i in range(0, 42_000_000, 1_000_000)]

    print("No. of workers: ", n_workers)
    with mp.Pool(processes=n_workers, initializer=tqdm.set_lock, initargs=(mp.RLock(),)) as p:
        p.starmap(process_and_save_shards, idx)