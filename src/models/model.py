"""
src/models/model.py

MultichannelCNNWithAttention – PyTorch implementation.

Architecture (from the paper):
    Embedding lookup  →  3 parallel Conv1D branches (kernel 2, 3, 5, 128 filters)
    each branch: Conv1D → ReLU → AdaptiveMaxPool1d(1)   (global branch)
                                 + Attention over the sequence  (sequence branch)
    Concatenate all branch outputs → Dropout → Linear → Sigmoid

The attention module is a single-head soft-attention over the conv feature maps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Soft attention over a sequence of feature vectors.

    Input:  (batch, num_filters, seq_len)   (conv output layout)
    Output: attended vector of shape (batch, num_filters)
             + attention weights of shape (batch, seq_len)
    """

    def __init__(self, num_filters: int) -> None:
        super().__init__()
        self.attention_fc = nn.Linear(num_filters, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, num_filters, seq_len)
        # Transpose to (batch, seq_len, num_filters) for the linear layer
        x_t = x.transpose(1, 2)                          # (B, S, F)
        scores = self.attention_fc(x_t).squeeze(-1)       # (B, S)
        weights = F.softmax(scores, dim=-1)               # (B, S)
        attended = torch.bmm(weights.unsqueeze(1), x_t)  # (B, 1, F)
        attended = attended.squeeze(1)                    # (B, F)
        return attended, weights


class ConvBranch(nn.Module):
    """One convolutional branch with global max-pool AND attention pooling.

    Produces a concatenated vector of size ``2 * num_filters``.
    """

    def __init__(self, in_channels: int, num_filters: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2  # 'same' padding to preserve sequence length
        self.conv = nn.Conv1d(in_channels, num_filters, kernel_size, padding=padding)
        self.attention = AttentionLayer(num_filters)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x:
            (batch, in_channels, seq_len)

        Returns
        -------
        features:
            (batch, 2 * num_filters) – global + attended
        attn_weights:
            (batch, seq_len) – used for heatmap visualisation
        """
        h = F.relu(self.conv(x))               # (B, num_filters, seq_len)

        # Global max-pool branch
        global_feat = F.adaptive_max_pool1d(h, 1).squeeze(-1)   # (B, num_filters)

        # Attention branch
        attn_feat, attn_weights = self.attention(h)              # (B, num_filters), (B, S)

        features = torch.cat([global_feat, attn_feat], dim=1)    # (B, 2*num_filters)
        return features, attn_weights


class MultichannelCNNWithAttention(nn.Module):
    """Multichannel CNN with attention for stress detection.

    Parameters
    ----------
    vocab_size:
        Vocabulary size for the embedding layer.
    embed_dim:
        Embedding dimensionality (default 128).
    num_filters:
        Number of filters per kernel size (default 128).
    kernel_sizes:
        Tuple of kernel sizes for the three parallel branches (default (2, 3, 5)).
    dropout:
        Dropout probability before the classification head (default 0.5).
    num_classes:
        Output neurons (default 1 for binary regression with sigmoid).
    padding_idx:
        Embedding padding index (default 0).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: tuple[int, ...] = (2, 3, 5),
        dropout: float = 0.5,
        num_classes: int = 1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.branches = nn.ModuleList(
            [ConvBranch(embed_dim, num_filters, k) for k in kernel_sizes]
        )

        # Each branch produces 2 * num_filters features
        total_features = len(kernel_sizes) * 2 * num_filters

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(total_features, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        input_ids:
            (batch, seq_len) LongTensor
        attention_mask:
            (batch, seq_len) – used to zero-out padding positions in the
            embedding before convolution (optional but recommended).

        Returns
        -------
        dict with keys:
            'logits'        – (batch, 1) raw scores
            'probs'         – (batch, 1) sigmoid probabilities
            'attn_weights'  – list of (batch, seq_len) tensors, one per branch
        """
        # ── Embedding ──────────────────────────────────────────────────────────
        emb = self.embedding(input_ids)   # (B, S, embed_dim)

        if attention_mask is not None:
            # Zero-out padded positions so they don't influence convolution
            emb = emb * attention_mask.unsqueeze(-1).float()

        # Conv1d expects (B, C, L) – so transpose
        emb = emb.transpose(1, 2)         # (B, embed_dim, S)

        # ── Parallel branches ─────────────────────────────────────────────────
        branch_features: list[torch.Tensor] = []
        attn_weights_list: list[torch.Tensor] = []

        for branch in self.branches:
            feats, weights = branch(emb)
            branch_features.append(feats)
            attn_weights_list.append(weights)

        # ── Classification head ───────────────────────────────────────────────
        combined = torch.cat(branch_features, dim=1)  # (B, total_features)
        combined = self.dropout(combined)
        logits = self.classifier(combined)            # (B, 1)
        probs = torch.sigmoid(logits)

        return {
            "logits": logits,
            "probs": probs,
            "attn_weights": attn_weights_list,
        }
