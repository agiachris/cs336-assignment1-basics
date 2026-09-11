from typing import Union, Tuple, Any, Optional

import os
import shutil
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from functools import partial

import torch
import numpy as np
from multiprocessing import Pool


from transformer.model import (
    TransformerLLM, 
    AdamW,
    gradient_clipping,
    cross_entropy_loss,
    perplexity
)
from tokenization.bpe_tokenization import (
    Tokenizer,
    find_chunk_boundaries,
)


CURR_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = CURR_DIR.parent
VOCAB_PATH = CURR_DIR / "tokenization" / "output"
DATA_PATH = {
    "train": ROOT_DIR / "data" / "TinyStoriesV2-GPT4-valid.txt",
    "valid": ROOT_DIR / "data" / "TinyStoriesV2-GPT4-valid.txt"
}
TOKEN_PATH = {
    "train": CURR_DIR / "data" / "train.bin",
    "valid": CURR_DIR / "data" / "valid.bin",
}


BATCH_SIZE = 64
CONTEXT_LENGTH = 8
TRAIN_KWARGS = {
    "train_steps": 100000,
    "valid_steps": 10,
    "batch_size": BATCH_SIZE,
    "context_length": CONTEXT_LENGTH,
    "log_freq": 50,
    "eval_freq": 10000,
    "grad_clip": 1.0
}
OPTIM_KWARGS = {} # Use defaults.
MODEL_KWARGS = {
    "vocab_size": 1000 + 256 + 1,
    "context_length": CONTEXT_LENGTH,
    "num_layers": 8,
    "d_model": 128,
    "num_heads": 8,
    "theta": 10000,
    "d_ff": 256,
    "device": "cpu",
    "dtype": torch.float,
}


def _worker(args: Tuple[Any, ...]) -> str:
    tokenizer, input_path, output_path, dtype, flush_every, s, e = args

    with open(input_path, "rb") as f, open(output_path, "wb") as out_file:
        f.seek(s)
        text = f.read(e - s).decode("utf-8", errors="ignore")

        buf = []
        for doc in text.split("<|endoftext|>"):
            if not doc:
                continue
            buf.extend(tokenizer.encode(doc + "<|endoftext|>"))
            if len(buf) >= flush_every:
                np.array(buf, dtype=dtype).tofile(out_file)
                buf = []

        if buf:
            np.array(buf, dtype=dtype).tofile(out_file)

    return output_path


def pretokenize(
    tokenizer: Tokenizer,
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    dtype: np.dtype = np.uint16, 
    flush_every: int = 1 << 22,
    num_processes: int = 8,
) -> None:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            file=f,
            desired_num_chunks=num_processes,
            split_special_token=b"<|endoftext|>"
        )
    
    _tasks = [
        (
            tokenizer,
            str(input_path),
            str(Path(output_path).parent / f"shard{i}.bin"),
            dtype,
            flush_every,
            s,
            e
        )
        for i, (s, e) in enumerate(zip(boundaries, boundaries[1:]))
    ]

    # Always import Pool. Pool takes num processes
    with Pool(processes=len(_tasks)) as pool:
        shard_paths = pool.map(_worker, _tasks)

    with open(output_path, "wb") as out_file:
        for shard_path in shard_paths:
            with open(shard_path, "rb") as shard_file:
                shutil.copyfileobj(shard_file, out_file)
            os.remove(shard_path)


def load_dataset(split: str) -> np.memmap:
    return np.memmap(TOKEN_PATH[split], dtype=np.uint16, mode="r")
    

def sample_batch(
    x: np.memmap,
    batch_size: int,
    context_length: int,
    device: str = "cpu"
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Returns batch of token id sequences.
    
    Args:
        x: np.memmap (N,) uint16
        batch_size: Batch size
        context_length: Context length
        device: Device
    
    Returns:
        (B, S) inputs and (B, S) targets of token ids.
    """
    start_B = np.random.randint(0, len(x) - context_length, size=batch_size)
    idx_BS1 = start_B[:, None] + np.arange(context_length + 1)[None, :]
    chunk_BS1 = torch.from_numpy(x[idx_BS1].astype(np.int64))
    inputs_BS = chunk_BS1[:, :-1]
    targets_BS = chunk_BS1[:, 1:]
    if device.startswith("cuda"):
        return (inputs_BS.pin_memory().to(device, non_blocking=True),
                targets_BS.pin_memory().to(device, non_blocking=True))
    return inputs_BS.to(device), targets_BS.to(device)


def train(
    tokenizer: Tokenizer,
    model: TransformerLLM,
    optim: AdamW,
    train_dataset: np.memmap,
    valid_dataset: Optional[np.memmap] = None,
    train_steps: int = 100000,
    valid_steps: int = 100,
    batch_size: int = BATCH_SIZE,
    context_length: int = CONTEXT_LENGTH,
    log_freq: int = 1000,
    eval_freq: int = 10000,
    grad_clip: float = 1.0,
    device: str = "cpu",
) -> TransformerLLM:
    model.to(device)
    model.train()
    sample_from = partial(
        sample_batch,
        batch_size=batch_size,
        context_length=context_length,
        device=device   
    )

    # Commence training.
    train_log = []
    valid_log = []
    train_metrics = defaultdict(list)

    pbar = tqdm(range(train_steps), total=train_steps, desc=f"Training steps")
    for train_step in pbar:
        # Compute loss.
        inputs_BS, targets_BS = sample_from(train_dataset)
        logits_BSV = model(inputs_BS)
        loss = cross_entropy_loss(logits_BSV, targets_BS)

        # Update params.
        optim.zero_grad()
        loss.backward()
        gradient_clipping(list(model.parameters()), grad_clip)
        optim.step()

        # Log train metrics.
        train_metrics["loss"].append(loss.item())
        if train_step % log_freq == 0:
            log_dict = {m: np.array(v).mean() for m, v in train_metrics.items()}
            train_log.append(log_dict)
            train_metrics = defaultdict(list)
            pbar.set_postfix(train_loss=f"{train_log[-1]["loss"]:.4f}")

        # Log valid metrics.
        if train_step % eval_freq == 0 and valid_dataset is not None:
            model.eval()
            valid_metrics = defaultdict(list)
            with torch.no_grad():
                for _ in range(valid_steps):
                    inputs_BS, targets_BS = sample_from(valid_dataset)
                    logits_BSV = model(inputs_BS)
                    valid_metrics["loss"].append(cross_entropy_loss(logits_BSV, targets_BS).item())
                    valid_metrics["perplexity"].append(perplexity(logits_BSV, targets_BS).mean().item())
            log_dict = {m: np.array(v).mean() for m, v in valid_metrics.items()}
            valid_log.append(log_dict)
            model.train()

    return model


if __name__ == "__main__":
    tokenize_datasets = False
    train_transformer = True
 
    # Instantiate tokenizer.
    tokenizer = Tokenizer.from_files(
        vocab_filepath=VOCAB_PATH / "vocab.pkl",
        merges_filepath=VOCAB_PATH / "merges.pkl",
        special_tokens=["<|endoftext|>"]
    )

    # Optionally pretokenize.
    if tokenize_datasets:
        pretokenized = False
        train_filepath = DATA_PATH["train"]
        valid_filepath = DATA_PATH["valid"]
        for split in TOKEN_PATH.keys():
            if not TOKEN_PATH[split].exists():
                print(f"Pretokenizing the {split} split")
                pretokenize(
                    tokenizer=tokenizer,
                    input_path=DATA_PATH[split], 
                    output_path=TOKEN_PATH[split],
                    num_processes=8
                )

    if train_transformer:
        model = TransformerLLM(**MODEL_KWARGS)
        optim = AdamW(model.parameters(), **OPTIM_KWARGS)
        train_dataset = load_dataset(split="train")
        valid_dataset = load_dataset(split="valid")
        model = train(
            tokenizer=tokenizer,
            model=model,
            optim=optim,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            **TRAIN_KWARGS
        )
        