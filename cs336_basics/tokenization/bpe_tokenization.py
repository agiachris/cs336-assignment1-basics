from typing import List, Dict, Tuple, BinaryIO, Union, Iterable, Iterator

import os
import time
import pickle
import regex as re
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool


CURR_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = CURR_DIR.parent.parent


GPT2_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _worker(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return pretokenize(chunk, special_tokens)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> List[int]:
    """Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes.
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced.
    # Chunks start on previous index, don't include last index.
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time.

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess.
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk.

            # If EOF, this boundary should be at the end of the file.
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk.
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks.
    return sorted(set(chunk_boundaries))


def pretokenize(
    text: str, 
    special_tokens: List[str]
) -> Dict[Tuple[bytes, ...], int]:
    """Split on special tokens, then according to GPT-2 regex. Returns {byte-tuple: count}."""
    if special_tokens:
        chunks = re.split("|".join(re.escape(t) for t in special_tokens), text)
    else:
        chunks = [text]
    
    pretoken_dict = defaultdict(int)
    for chunk in chunks:
        for m in re.finditer(GPT2_REGEX, chunk):
            pretoken_dict[tuple(bytes([t]) for t in m.group(0).encode('utf-8'))] += 1

    return dict(pretoken_dict)


def pretokenize_iter(
    text: str, 
    special_tokens: List[str]
) -> List[Tuple[bytes, ...]]:
    """Split on special tokens, then according to GPT-2 regex. Returns {byte-tuple: count}."""
    if special_tokens:
        chunks = re.split("|".join(re.escape(t) for t in special_tokens), text)
    else:
        chunks = [text]
    
    pretokens = []
    for chunk in chunks:
        for m in re.finditer(GPT2_REGEX, chunk):
            pretokens.append(tuple(bytes([t]) for t in m.group(0).encode('utf-8')))

    return pretokens


def count_byte_pairs(
    pretoken_dict: Dict[Tuple[bytes, ...], int]
):
    """Compute byte pair counts."""
    byte_pair_dict = defaultdict(int)
    for byte_seq, byte_count in pretoken_dict.items():
        for i in range(1, len(byte_seq)):
            b1 = byte_seq[i-1]
            b2 = byte_seq[i]
            byte_pair_dict[(b1, b2)] += byte_count
    return dict(byte_pair_dict)


def apply_merge(key, merge_tokens):
    new_key = []
    skip = False
    for i in range(len(key)):
        if skip:
            skip = False
            continue
        if i < len(key) - 1 and (key[i], key[i + 1]) == merge_tokens:
            new_key.append(key[i] + key[i + 1])
            skip = True
        else:
            new_key.append(key[i])
    return tuple(new_key)
    

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[int],
    parallel: bool = True
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Implement BPE tokenizer training.
    
    Args:
        input_path: str  Path to a text file with BPE tokenizer training data.
        vocab_size: int  A positive integer that defines the maximum final vocabulary size 
                (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: list[str]  A list of strings to add to the vocabulary. During training, treat
                them as hard boundaries that prevent merges across their spans, but do not include them when
                computing merge statistics

    Returns:
        vocab: dict[int, bytes]  The tokenizer vocabulary, a mapping from int (token ID in the
                vocabulary) to bytes (token bytes).
        merges: list[tuple[bytes, bytes]]  A list of BPE merges produced from training. Each list
                item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
                <token2>. The merges should be ordered by order of creation.
    """
    start_time = time.time()
    # Chunking data for parallel processing.
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # Instantiate stores.
        vocab = {i: bytes([i]) for i in range(256)}
        for token in special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")

        # Serial pretokenization implementation.
        pretoken_dict = defaultdict(int)
        if parallel:
            tasks = [(input_path, s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]
            with Pool(processes=num_processes) as pool:
                for chunk_counts in pool.imap_unordered(_worker, tasks):
                    for k, v in chunk_counts.items():
                        pretoken_dict[k] += v
        else:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")

                # Step 1. Remove special tokens from chunk.
                # Step 2. Pretokenize chunk according to GPT2_REGEX.
                chunk_pretoken_dict = pretokenize(
                    text=chunk,
                    special_tokens=special_tokens
                )
                for k, v in chunk_pretoken_dict.items():
                    pretoken_dict[k] += v

        merges = []
        while len(vocab) < vocab_size:
            # Step 3. Compute byte pair counts.
            byte_pair_dict = count_byte_pairs(pretoken_dict)
            if not byte_pair_dict:
                break

            # Step 4. Identify the maximum occurring byte pair and add to the vocabulary.
            best = max(byte_pair_dict, key=lambda p: (byte_pair_dict[p], p))
            merges.append(best)
            vocab[len(vocab)] = best[0] + best[1]

            # Step 5. Perform byte pair merge and repeat.
            pretoken_dict = {
                apply_merge(k, best): v for k, v in pretoken_dict.items()
            }

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
    return vocab, merges


class Tokenizer:

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        for t in self.special_tokens:
            if t not in self.vocab.values():
                self.vocab[len(self.vocab)] = t.encode('utf-8')

        # Utility attributes.
        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: Union[str, Path],
        merges_filepath: Union[str, Path],
        special_tokens: List[str] = None
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens=special_tokens)

    def encode(self, text: str) -> List[int]:
        """Encode string to token ids."""
        # Step 1. Pretokenize; pre-token keys are important, presumably not their counts anymore.
        pretokens = pretokenize_iter(text, self.special_tokens)

        # Step 2. Apply merges in order of their creation.
        for m in self.merges:
            pretokens = [apply_merge(k, m) for k in pretokens]

        # Step 3. Encode using the vocabulary.
        encoding = [self.byte_to_id[t] for seq in pretokens for t in seq]
        return encoding

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode a stream, yielding token IDs without loading it all."""
        buf = ""
        for chunk in iterable:
            buf += chunk
            if len(buf) < 30:
                continue

            # Scan backwards for a safe cut: a position whose preceding char is
            # non-whitespace and whose own char is whitespace. No pre-token can
            # span that junction, so splitting there cannot change the result.
            cut = 0
            for p in range(len(buf) - 1, 0, -1):
                if buf[p].isspace() and not buf[p - 1].isspace():
                    cut = p
                    break

            # cut == 0 means no safe boundary yet (all whitespace, or one long
            # unbroken run). Keep buffering rather than risk a bad split.
            if cut:
                yield from self.encode(buf[:cut])
                buf = buf[cut:]

        if buf:
            yield from self.encode(buf)

    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text string."""
        # Step 1. Convert ids to bytes using vocabulary.
        # Step 2. Concatenate the entire byte sequence.
        byte_seq = b''.join([self.vocab[i] for i in ids])
        
        # Step 3. Decode using utf-8. 
        string_seq = byte_seq.decode('utf-8', errors='replace')
        return string_seq


if __name__ == "__main__":
    train_bpe_flag = False
    load_tokenizer_flag = True

    if train_bpe_flag:
        input_path = ROOT_DIR / "data" / "TinyStoriesV2-GPT4-train.txt"
        vocab, merges = train_bpe(
            input_path=str(input_path),
            vocab_size=1000 + 1 + 12,
            special_tokens=["<|endoftext|>"],
            parallel=True
        )
        with open(f"{CURR_DIR}/output/vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
        with open(f"{CURR_DIR}/output/merges.pkl", "wb") as f:
            pickle.dump(merges, f)
        
    if load_tokenizer_flag:
        vocab_filepath = CURR_DIR / "output" / "vocab.pkl"
        merges_filepath = CURR_DIR / "output" / "merges.pkl"
        tokenizer = Tokenizer.from_files(
            vocab_filepath=vocab_filepath,
            merges_filepath=merges_filepath,
            special_tokens=["<|endoftext|>"]
        )

        test_string = "Hello world, this is a special test designed to test the loading of a BPE tokenizer."
        token_ids = tokenizer.encode(test_string)
        test_string_rec = tokenizer.decode(token_ids)
        breakpoint()
        print(f"Test string: {test_string}")
        print(f"Token ids: {token_ids}")
        print(f"Test string rec: {test_string_rec}")

