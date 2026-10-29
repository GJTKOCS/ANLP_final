"""ListenerModel: LoRA backbone + last-token pooling + two linear heads (helper 10, venue 10).
B and C listeners are the SAME class/config; they differ only in the input view."""
from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from ..common.device import get_device, torch_dtype
from ..common.llm import LORA_TARGETS, load_backbone, load_tokenizer


def last_token_pool(hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Hidden state of the last non-padding token; correct for left and right padding."""
    pos = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand_as(mask)
    idx = (mask.long() * (pos + 1)).argmax(1)
    return hs[torch.arange(hs.shape[0], device=hs.device), idx]


class ListenerModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden: int):
        super().__init__()
        self.backbone = backbone
        self.head_helper = nn.Linear(hidden, 10)
        self.head_venue = nn.Linear(hidden, 10)

    def forward(self, input_ids, attention_mask):
        hs = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = last_token_pool(hs, attention_mask).float()
        return self.head_helper(pooled), self.head_venue(pooled)

    def trainable_state(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if "lora_" in k or k.startswith("head_")}


def listener_cfg(cfg) -> dict:
    """The single recipe shared by B and C (asserted equal in manifests)."""
    return {"model": cfg.hw["listener_model"], **dict(cfg.listeners)}


def build_listener(cfg, device=None):
    device = device or get_device()
    hw = cfg.hw
    dtype = torch_dtype(hw, device)
    lc = listener_cfg(cfg)
    bb = load_backbone(lc["model"], hw, device, dtype, causal=False)
    if hw.get("grad_ckpt"):
        bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        bb.enable_input_require_grads()
    lora = LoraConfig(r=lc["lora_r"], lora_alpha=lc["lora_alpha"], lora_dropout=lc["lora_dropout"], target_modules=LORA_TARGETS)
    bb = get_peft_model(bb, lora)
    model = ListenerModel(bb, bb.config.hidden_size if hasattr(bb, "config") else bb.base_model.config.hidden_size)
    model.head_helper.to(device).float()
    model.head_venue.to(device).float()
    tok = load_tokenizer(lc["model"], hw)
    tok.padding_side = "right"
    tok.truncation_side = "left"
    return model, tok, device
