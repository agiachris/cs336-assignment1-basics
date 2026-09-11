import regex as re
from collections import defaultdict

# GPT-2 regex.
# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# text = "some text that i'll pre-tokenize"
# for m in re.finditer(PAT, text):
#     print(m.group(0))

# Construct training corpus.
data = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

# Initialize vocabulary.
idxs_to_bytes = {i: bytes([i]) for i in range(256)}
idxs_to_bytes[256] = "<|endoftext|>".encode('utf-8')
bytes_to_idxs = {v: k for k, v in idxs_to_bytes.items()}

# Step 1. Count pretoken occurrences.
pretokens = data.split(' ')
pretoken_dict = {}
for t in pretokens:
    pretoken_dict[t] = pretoken_dict.get(t, 0) + 1
print(pretoken_dict)

# Step 2. Convert pretokens to byte sequences.
pretoken_dict = {tuple(bytes([b]) for b in k.encode('utf-8')): v for k, v in pretoken_dict.items()}
print(pretoken_dict)

# Step 3. Count byte pair occurrences. 
byte_pairs = defaultdict(int)
for byte_seq, count, in pretoken_dict.items():
    for i in range(1, len(byte_seq)):
        b_prev = byte_seq[i-1]
        b_curr = byte_seq[i]
        byte_pairs[(b_prev, b_curr)] += count

print(byte_pairs)
breakpoint()

byte_pairs_list = sorted([(k, v) for k, v in byte_pairs.items()], key=lambda x: -x[1])
byte_pair_ties = [k for k, v in byte_pairs_list if v == max(v for _, v in byte_pairs_list)]
merge_pair = max(byte_pair_ties)

