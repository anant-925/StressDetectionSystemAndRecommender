"""
src/models/inference.py

Helper class that loads a trained MultichannelCNNWithAttention checkpoint
and runs inference on raw text, returning:
  - stress probability
  - per-token attention weights (average across branches, for heatmap)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from src.models.dataset import sliding_window_chunks
from src.models.model import MultichannelCNNWithAttention

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "checkpoints" / "best_model.pt"


class StressInferenceEngine:
    """Load a trained checkpoint and score new texts.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``best_model.pt`` saved by ``src/training/train.py``.
    device:
        ``'cpu'``, ``'cuda'``, or ``None`` (auto-detect).
    """

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt["tokenizer_name"])
        self.chunk_size: int = ckpt.get("chunk_size", 200)
        self.stride: int = ckpt.get("stride", 50)

        self.model = MultichannelCNNWithAttention(
            vocab_size=ckpt["vocab_size"],
            embed_dim=ckpt["embed_dim"],
            num_filters=ckpt["num_filters"],
            kernel_sizes=tuple(ckpt["kernel_sizes"]),
            dropout=ckpt.get("dropout", 0.5),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def predict(self, text: str) -> dict:
        """Run inference on a single text.

        Returns
        -------
        dict
            ``stress_score`` (float in [0,1]),
            ``tokens``       (list[str]),
            ``attn_weights`` (list[float] aligned with ``tokens``).
        """
        chunks = sliding_window_chunks(
            text, self.tokenizer, self.chunk_size, self.stride
        )

        all_probs: list[float] = []
        # We'll aggregate attention across chunks for visualisation.
        # For the heatmap we use only the first chunk (most representative).
        first_chunk_attn: list[float] | None = None
        first_chunk_tokens: list[str] = []

        with torch.no_grad():
            for i, chunk in enumerate(chunks):
                input_ids = chunk["input_ids"].unsqueeze(0).to(self.device)   # (1, S)
                attention_mask = chunk["attention_mask"].unsqueeze(0).to(self.device)

                output = self.model(input_ids, attention_mask)
                prob = output["probs"].item()
                all_probs.append(prob)

                if i == 0:
                    # Average attention across all branches → (S,)
                    branch_attns = [w.squeeze(0).cpu().numpy() for w in output["attn_weights"]]
                    import numpy as np
                    avg_attn = np.mean(branch_attns, axis=0).tolist()

                    # Decode token ids for heatmap labels
                    ids = chunk["input_ids"].tolist()
                    mask = chunk["attention_mask"].tolist()
                    real_ids = [t for t, m in zip(ids, mask) if m == 1]
                    tokens = self.tokenizer.convert_ids_to_tokens(real_ids)

                    first_chunk_attn = avg_attn[:len(tokens)]
                    first_chunk_tokens = tokens

        mean_score = float(sum(all_probs) / len(all_probs)) if all_probs else 0.0

        return {
            "stress_score": mean_score,
            "tokens": first_chunk_tokens,
            "attn_weights": first_chunk_attn or [],
        }
