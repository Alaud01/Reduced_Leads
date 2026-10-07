"""
Multi-task loss for the lead-aware transformer.

Combines per-task BCE-with-logits using inverse-frequency class weights:
- cls_super       : 5-way multi-label
- cls_sub         : 44-way multi-label
- aux_rhythm      : 12-way multi-label
- aux_lead_presence : 12-way (no weighting; presence is roughly balanced by augmentation)

Total = w_super*L_super + w_sub*L_sub + w_rhythm*L_rhythm + w_lp*L_lp
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Multi-label BCE with optional per-class positive weight.

    logits:  (B, C)
    targets: (B, C) float 0/1
    pos_weight: (C,) tensor — reweights positive term per class.
    """
    if pos_weight is not None:
        # broadcast to (1, C)
        pw = pos_weight.unsqueeze(0).to(logits.dtype)
    else:
        pw = None
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


class MultiTaskLoss:
    """Stateful holder for the multi-task loss + class weights."""

    def __init__(
        self,
        w_super: float = 1.0,
        w_sub: float = 1.0,
        w_rhythm: float = 0.5,
        w_lead_presence: float = 0.2,
        super_weights: torch.Tensor | None = None,
        sub_weights: torch.Tensor | None = None,
        rhythm_weights: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
    ):
        self.w_super = w_super
        self.w_sub = w_sub
        self.w_rhythm = w_rhythm
        self.w_lp = w_lead_presence
        self.super_pw = super_weights.to(device) if super_weights is not None else None
        self.sub_pw = sub_weights.to(device) if sub_weights is not None else None
        self.rhy_pw = rhythm_weights.to(device) if rhythm_weights is not None else None

    def __call__(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        y_super = batch["y_super"].to(outputs["cls_super"].dtype)
        y_sub = batch["y_sub"].to(outputs["cls_sub"].dtype)
        y_rhy = batch["y_rhythm"].to(outputs["aux_rhythm"].dtype)
        lead_mask = batch["lead_mask"].to(outputs["aux_lead_presence"].dtype)

        L_super = weighted_bce_with_logits(outputs["cls_super"], y_super, self.super_pw)
        L_sub = weighted_bce_with_logits(outputs["cls_sub"], y_sub, self.sub_pw)
        L_rhy = weighted_bce_with_logits(outputs["aux_rhythm"], y_rhy, self.rhy_pw)
        L_lp = weighted_bce_with_logits(outputs["aux_lead_presence"], lead_mask)

        total = (self.w_super * L_super + self.w_sub * L_sub
                 + self.w_rhythm * L_rhy + self.w_lp * L_lp)
        return {
            "loss": total,
            "loss_super": L_super.detach(),
            "loss_sub": L_sub.detach(),
            "loss_rhythm": L_rhy.detach(),
            "loss_lead_presence": L_lp.detach(),
        }