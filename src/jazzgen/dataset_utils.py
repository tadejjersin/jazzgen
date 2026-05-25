import os
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

def get_midi_paths(max_jazz_only=True, min_jazz_p=0.3):
    """
    Returns paths to MIDI files from the root. Returns only files where the classifier assigned the 
    largest score to jazz, if max_jazz_only=True, and only files where the jazz score is larger than jazz_p.
    """
    res_path = os.path.join(root_dir, "data", "genre_scores.json")
    with open(res_path, "r", encoding="utf-8") as f:
        res_dict = json.load(f)
    # filter by max
    if max_jazz_only:
        res_dict = {k: v for k, v in res_dict.items() if v["jazz"] == max(v.values())}
    # filter by jazz scores
    res_dict = {k: v for k, v in res_dict.items() if v["jazz"] > min_jazz_p}
    return list(res_dict.keys())

def create_split(midi_paths, train_ratio=0.85, val_ratio=0.1, seed=42):
    """
    Creates a train/val/test split of the MIDI files.
    """
    midi_paths = np.array(sorted(midi_paths))
    rng = np.random.default_rng(seed)

    indices = np.arange(len(midi_paths))
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    split = {
        "train": midi_paths[train_indices].tolist(),
        "val": midi_paths[val_indices].tolist(),
        "test": midi_paths[test_indices].tolist(),
    }

    return split


def tokenize_midi_paths(midi_paths, out_dir, model_name="loubb/aria-medium-base"):
    """
    Tokenizes MIDI files with Aria tokenizer and saves each token sequence as .npy.
    Returns a list of dicts with paths to the saved token files.
    """

    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    token_items = []

    for i, midi_path in enumerate(midi_paths):
        try:
            token_path = os.path.join(out_dir, f"{i:06d}.npy")
            if os.path.exists(token_path):
                token_ids = np.load(token_path)
            else:
                # encode MIDI
                encoded = tokenizer.encode_from_file(
                    os.path.join(root_dir, midi_path),
                    return_tensors="pt",
                )
                token_ids = encoded.input_ids[0].cpu().numpy().astype(np.int32)

                # save tokens to file (to not keep everything in memory)
                np.save(token_path, token_ids)

            token_items.append({
                "midi_path": midi_path,
                "token_path": token_path,
                "num_tokens": len(token_ids),
            })

        except Exception as e:
            print(f"Skipping {midi_path}: {e}")

    return token_items

class AriaChunkDataset(Dataset):
    """
    Simple dataset for decoder-only LM training.

    Each item returns:
        input_ids
        attention_mask
        labels

    labels are the same as input_ids because Hugging Face causal LM models
    usually shift internally during loss computation.
    """

    def __init__(self, token_items, block_size=2048, stride=1024):
        self.token_items = token_items
        self.block_size = block_size
        self.stride = stride

        self.chunks = []
        # create chunks
        for item_idx, item in enumerate(token_items):
            n_tokens = item["num_tokens"]

            if n_tokens < block_size:
                continue

            for start in range(0, n_tokens - block_size + 1, stride):
                self.chunks.append((item_idx, start))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        item_idx, start = self.chunks[idx]
        
        # load tokens from saved encoded MIDI
        token_path = self.token_items[item_idx]["token_path"]
        tokens = np.load(token_path)
        # get the chunk of tokens
        chunk = tokens[start:start + self.block_size]

        input_ids = torch.tensor(chunk, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }