"""
src/models/dataset.py

PyTorch Dataset with sliding-window chunking for long texts.
Each long document is split into overlapping chunks (size=200 tokens,
stride=50 tokens) before tokenisation.  Short texts produce a single chunk.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def sliding_window_chunks(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int = 200,
    stride: int = 50,
) -> list[dict[str, torch.Tensor]]:
    """Tokenise *text* and split into overlapping chunks.

    Parameters
    ----------
    text:
        Raw input string.
    tokenizer:
        HuggingFace tokenizer (must support ``encode_plus``).
    chunk_size:
        Number of tokens per chunk (excluding special tokens).
    stride:
        Step size between consecutive chunks.

    Returns
    -------
    list[dict]
        Each element is a dict with ``input_ids`` and ``attention_mask``
        tensors of shape ``(chunk_size,)``.
    """
    # Tokenise without truncation to get the full sequence
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids: torch.Tensor = encoding["input_ids"].squeeze(0)   # (L,)
    attention_mask: torch.Tensor = encoding["attention_mask"].squeeze(0)

    total_len = input_ids.size(0)
    chunks: list[dict[str, torch.Tensor]] = []

    start = 0
    while start < max(total_len, 1):
        end = start + chunk_size
        chunk_ids = input_ids[start:end]
        chunk_mask = attention_mask[start:end]

        # Pad to chunk_size if the last chunk is shorter
        pad_len = chunk_size - chunk_ids.size(0)
        if pad_len > 0:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            chunk_ids = torch.cat(
                [chunk_ids, torch.full((pad_len,), pad_id, dtype=torch.long)]
            )
            chunk_mask = torch.cat(
                [chunk_mask, torch.zeros(pad_len, dtype=torch.long)]
            )

        chunks.append({"input_ids": chunk_ids, "attention_mask": chunk_mask})

        if end >= total_len:
            break
        start += stride

    # Guarantee at least one chunk even for empty text
    if not chunks:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        chunks.append(
            {
                "input_ids": torch.full((chunk_size,), pad_id, dtype=torch.long),
                "attention_mask": torch.zeros(chunk_size, dtype=torch.long),
            }
        )

    return chunks


class StressDataset(Dataset):
    """Dataset that yields one (chunk_input_ids, chunk_attention_mask, label)
    tuple per sliding-window chunk.

    Parameters
    ----------
    texts:
        List of raw text strings.
    labels:
        List / 1-D tensor of float labels in [0, 1].
    tokenizer:
        HuggingFace tokenizer.
    chunk_size:
        Token count per window (default 200).
    stride:
        Sliding-window step (default 50).
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[float] | Any,
        tokenizer: PreTrainedTokenizerBase,
        chunk_size: int = 200,
        stride: int = 50,
    ) -> None:
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.stride = stride

        # Pre-compute all chunks and store flat lists
        self._input_ids: list[torch.Tensor] = []
        self._attention_masks: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

        for text, label in zip(texts, labels):
            chunks = sliding_window_chunks(
                str(text), tokenizer, chunk_size=chunk_size, stride=stride
            )
            for chunk in chunks:
                self._input_ids.append(chunk["input_ids"])
                self._attention_masks.append(chunk["attention_mask"])
                self._labels.append(torch.tensor(float(label), dtype=torch.float32))

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self._input_ids[idx],
            "attention_mask": self._attention_masks[idx],
            "label": self._labels[idx],
        }
