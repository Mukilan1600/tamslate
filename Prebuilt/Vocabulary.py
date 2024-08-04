import string

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