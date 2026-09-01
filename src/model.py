"""
Lead-aware Transformer for ECG selective-prediction (Option B backbone).

Architecture:
  1. Patch tokenizer:
       (N_LEADS, SIGNAL_LEN) -> (N_LEADS, N_PATCHES, patch_len)
       -> Linear(patch_len, d_model) -> (N_LEADS, N_PATCHES, d_model)
  2. Lead embedding:
       learned per-lead vector added to all patches of that lead,
       PLUS a learned "missing-lead" token that replaces the lead embedding
       when the lead is dropped.  This is the crucial ingredient that lets a
       single model operate at any lead subset: a dropped lead is presented as
       a distinct, learnable state rather than zero input.
  3. Factorized Transformer encoder (lead-attention then time-attention):
       - Stage A (lead-mixing): attention across the N_LEADS axis for each
         patch position, so each time-step can borrow from other leads.
       - Stage B (time-mixing): attention across N_PATCHES within each lead.
       This is far cheaper than full (N_LEADS*N_PATCHES) attention and
       captures the two main inductive biases of ECG: inter-lead
       relationships and intra-lead temporal morphology.
  4. Masking-aware pooling: a learnable [CLS]-style aggregation token that
     attends only over *present* patches (dropped leads masked out).
  5. Heads:
       cls_super       : 5-way sigmoid
       cls_sub         : 44-way sigmoid
       aux_lead_presence : 12-way sigmoid (which leads were present)
       aux_rhythm      : 12-way sigmoid
"""
from __future__ import annotations
from typing import Dict, Tuple
import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import (
    N_LEADS, SIGNAL_LEN, SIGNAL_HZ,
    N_SUPERCLASSES, N_SUBCODES, N_RHYTHMS, ModelCfg,
)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, N, D)
        key_padding_mask: (B, N) bool, True = position to be masked (ignored)
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        if key_padding_mask is not None:
            # mask: (B, N) True=pad -> broadcast over query axis
            mask = key_padding_mask[:, None, None, :]  # (B, 1, 1, N)
            attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v  # (B, H, N, hd)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.dropout(self.out(out))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.attn = PreNorm(dim, MultiHeadAttention(dim, n_heads, dropout))
        self.ffn = PreNorm(dim, FeedForward(dim, ffn_mult, dropout))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(x, key_padding_mask=key_padding_mask)
        x = x + self.ffn(x)
        return x


def _run_blocks(blocks: nn.ModuleList, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None,
                use_grad_ckpt: bool) -> torch.Tensor:
    """
    Run a stack of TransformerBlocks, optionally with gradient checkpointing
    to trade compute for activation memory.

    With use_grad_ckpt=True, each block's forward is recomputed during the
    backward pass, so only one block's activations are held in memory at a
    time instead of all n_layers. Roughly ~25-30% slower but the memory
    savings scale with depth.
    """
    for blk in blocks:
        if use_grad_ckpt and x.requires_grad:
            # checkpoint needs at least one input that requires grad; x does.
            # key_padding_mask is a bool tensor (no grad) - pass as-is.
            x = checkpoint(blk, x, key_padding_mask, use_reentrant=False)
        else:
            x = blk(x, key_padding_mask=key_padding_mask)
    return x


class LeadAwareTransformer(nn.Module):
    """
    Factorized lead-aware transformer.

    Input:
        x          : (B, N_LEADS, SIGNAL_LEN) normalized signal
        lead_mask  : (B, N_LEADS) float (1=present, 0=missing)
    Output dict:
        cls_super       : (B, N_SUPER)
        cls_sub         : (B, N_SUB)
        aux_lead_presence : (B, N_LEADS)
        aux_rhythm      : (B, N_RHYTHM)
    """

    def __init__(self, cfg: ModelCfg | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelCfg()
        self.patch_len = cfg.patch_len
        self.d_model = cfg.d_model
        self.n_patches = cfg.n_patches
        self.n_leads_real = cfg.n_leads  # 12
        # lead embedding: +1 missing token
        self.lead_emb = nn.Embedding(cfg.lead_emb_size, cfg.d_model)
        nn.init.trunc_normal_(self.lead_emb.weight, std=0.02)
        self.missing_idx = cfg.n_leads if cfg.use_missing_token else None

        # patch projection
        self.patch_proj = nn.Linear(cfg.patch_len, cfg.d_model)

        # positional embedding over patches (time axis)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, cfg.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # factorized encoder: lead-attention then time-attention blocks
        lead_dropout = cfg.dropout
        self.lead_blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ffn_mult, lead_dropout)
            for _ in range(cfg.n_layers)
        ])
        self.time_blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ffn_mult, lead_dropout)
            for _ in range(cfg.n_layers)
        ])

        self.norm = nn.LayerNorm(cfg.d_model)

        # masking-aware pooling: a learnable query token attending over patches
        self.pool_query = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.pool_query, std=0.02)
        self.pool_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, lead_dropout)

        # heads
        self.head_super = nn.Linear(cfg.d_model, N_SUPERCLASSES)
        self.head_sub = nn.Linear(cfg.d_model, N_SUBCODES)
        self.head_rhythm = nn.Linear(cfg.d_model, N_RHYTHMS)
        self.head_lead_presence = nn.Linear(cfg.d_model, cfg.n_leads)

    # ----------------------------------------------------------------- forward
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, T) -> patches (B, L, P, patch_len)
        P = T // patch_len
        """
        B, L, T = x.shape
        P = T // self.patch_len
        x = x.reshape(B, L, P, self.patch_len)
        return x  # (B, L, P, patch_len)

    def _build_lead_ids(self, lead_mask: torch.Tensor) -> torch.Tensor:
        """
        lead_mask: (B, L) float (1=present 0=missing)
        returns (B, L) long indices into lead_emb table.
        Missing leads -> missing_idx (learned missing token).
        """
        B, L = lead_mask.shape
        present = lead_mask.long()  # 0/1
        if self.missing_idx is not None:
            ids = torch.where(
                present.bool(),
                torch.arange(L, device=lead_mask.device).unsqueeze(0).expand(B, L),
                torch.full((B, L), self.missing_idx, device=lead_mask.device, dtype=torch.long),
            )
        else:
            ids = torch.arange(L, device=lead_mask.device).unsqueeze(0).expand(B, L)
        return ids

    def forward(self, x: torch.Tensor, lead_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, L, T = x.shape
        assert L == self.n_leads_real, f"expected {self.n_leads_real} leads got {L}"
        assert T >= SIGNAL_LEN, f"expected >= {SIGNAL_LEN} samples got {T}"

        patches = self._patchify(x)  # (B, L, P, patch_len)
        P = patches.shape[2]

        # linear patch embed
        tok = self.patch_proj(patches)  # (B, L, P, D)

        # lead embedding: (B, L, 1, D) added to all patches of that lead
        lead_ids = self._build_lead_ids(lead_mask)  # (B, L) long
        lead_emb = self.lead_emb(lead_ids)  # (B, L, D)
        tok = tok + lead_emb.unsqueeze(2)  # broadcast over P

        # time positional embedding: (1, P, D)
        tok = tok + self.pos_emb.unsqueeze(0)  # broadcast over B and L

        # key padding mask for *lead* axis: True = lead missing (mask out)
        lead_pad_mask = (lead_mask < 0.5)  # (B, L) bool, True=pad

        # ---------------- Stage A: lead-mixing for each patch position ----
        # Reshape to (B*P, L, D) so attention runs over L for each patch
        tok_a = tok.permute(0, 2, 1, 3).reshape(B * P, L, self.d_model)
        lead_pad_mask_a = lead_pad_mask.unsqueeze(1).expand(B, P, L).reshape(B * P, L)
        tok_a = _run_blocks(self.lead_blocks, tok_a, lead_pad_mask_a, self.cfg.use_grad_ckpt)
        tok = tok_a.reshape(B, P, L, self.d_model).permute(0, 2, 1, 3)  # (B, L, P, D)

        # ---------------- Stage B: time-mixing for each lead -------------
        tok_b = tok.reshape(B * L, P, self.d_model)
        tok_b = _run_blocks(self.time_blocks, tok_b, None, self.cfg.use_grad_ckpt)
        tok = tok_b.reshape(B, L, P, self.d_model)

        tok = self.norm(tok)  # (B, L, P, D)

        # ---------------- Masking-aware pooling ------------------------
        # Flatten to (B, L*P, D) and a per-token presence mask.
        flat = tok.reshape(B, L * P, self.d_model)
        # presence per (lead, patch): 1 if lead present
        present = lead_mask  # (B, L) float
        # broadcast to patches
        pres_tokens = present.unsqueeze(2).expand(B, L, P).reshape(B, L * P)
        pool_pad_mask = (pres_tokens < 0.5)  # True = pad

        q = self.pool_query.expand(B, 1, self.d_model)  # (B, 1, D)
        # self-attention with single query: q -> (B,1,D) attending over flat (B, N, D)
        # use MultiHeadAttention by concatenating query and keys then slice
        kv = flat
        qk = torch.cat([q, kv], dim=1)  # (B, 1+N, D)
        # we need query mask = False (always present), key mask = pool_pad_mask
        q_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        full_pad = torch.cat([q_pad, pool_pad_mask], dim=1)
        attn_out = self.pool_attn(qk, key_padding_mask=full_pad)  # (B, 1+N, D)
        pooled = attn_out[:, 0, :]  # (B, D) - the query output

        # ---------------- Heads -----------------------------------------
        return {
            "cls_super": self.head_super(pooled),
            "cls_sub": self.head_sub(pooled),
            "aux_lead_presence": self.head_lead_presence(pooled),
            "aux_rhythm": self.head_rhythm(pooled),
        }

    # ---------------------------------------------------- utilities
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)