"""LM wrapper: one base model, PEFT multi-adapter switching (policy / sft / disabled), batched generation
(left padding, plain HF generate), differentiable completion log-probs, next-token scoring."""
from __future__ import annotations

import re
from contextlib import contextmanager

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from .device import free_memory, get_device, gpu_info, torch_dtype

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_TAG = re.compile(r"^\s*(?:(?:A|B)\s*:\s*)+")


def load_tokenizer(name: str, hw: dict):
    tok = AutoTokenizer.from_pretrained(hw["toy_tokenizer"] if name == "toy-tiny" else name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _toy_config(tok) -> Qwen2Config:
    return Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096,
                       tie_word_embeddings=True, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)


def load_backbone(name: str, hw: dict, device, dtype, causal: bool = True):
    """Load causal LM (causal=True) or bare transformer (causal=False). 'toy-tiny' = random tiny Qwen2."""
    if name == "toy-tiny":
        tok = load_tokenizer(name, hw)
        torch.manual_seed(0)
        cls = AutoModelForCausalLM if causal else AutoModel
        return cls.from_config(_toy_config(tok)).to(device)
    cls = AutoModelForCausalLM if causal else AutoModel
    if "AWQ" in name or "GPTQ" in name:
        dtype = torch.float16                                   # AWQ kernels need fp16
    kw = {"dtype": dtype} if _new_tf() else {"torch_dtype": dtype}
    if "AWQ" in name or "GPTQ" in name:
        kw["device_map"] = {"": 0} if device.type == "cuda" else None
        m = cls.from_pretrained(name, **kw)
        return m if device.type == "cuda" else m.to(device)
    return cls.from_pretrained(name, **kw).to(device)


def _new_tf() -> bool:
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return (major, minor) >= (4, 56)


def pick_big_gen(hw: dict) -> str:
    """Pick generator by GPU memory (32B-AWQ >=40GB, 14B-AWQ >=24GB, else 7B). Record the choice in manifests."""
    mem = gpu_info()["mem_gb"]
    for need, name in hw["big_gen_models"]:
        if mem >= need:
            return name
    return hw["big_gen_models"][-1][1]


class LM:
    def __init__(self, name: str, hw: dict, grad_ckpt: bool = False, device=None, max_bs: int = 10**9):
        self.name, self.hw, self.max_bs = name, hw, max_bs
        self.device = device or get_device()
        self.dtype = torch_dtype(hw, self.device)
        self.tok = load_tokenizer(name, hw)
        self.tok.padding_side = "left"
        self.base = load_backbone(name, hw, self.device, self.dtype, causal=True)
        self.base.config.use_cache = True
        if grad_ckpt:
            self.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.base.enable_input_require_grads()
        self.peft = None
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.eos_ids = sorted({i for i in (self.tok.eos_token_id, im_end) if isinstance(i, int) and i >= 0})
        self.end_text = "<|im_end|>" if isinstance(im_end, int) and im_end >= 0 else (self.tok.eos_token or "")

    @property
    def model(self):
        return self.peft if self.peft is not None else self.base

    def close(self) -> None:
        del self.base, self.peft
        free_memory()

    # ---------------- adapters ----------------
    def new_adapter(self, name: str, r: int = 16, alpha: int = 32, dropout: float = 0.05) -> None:
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        if self.peft is not None and name in self.peft.peft_config:
            return                                              # idempotent
        if self.peft is None:
            self.peft = get_peft_model(self.base, cfg, adapter_name=name)
        else:
            self.peft.add_adapter(name, cfg)

    def save_adapter(self, name: str, path) -> None:
        torch.save({k: v.detach().cpu() for k, v in get_peft_model_state_dict(self.peft, adapter_name=name).items()}, path)

    def load_adapter_weights(self, name: str, path) -> None:
        set_peft_model_state_dict(self.peft, torch.load(path, map_location="cpu"), adapter_name=name)

    def adapter_params(self, name: str) -> list[torch.nn.Parameter]:
        return [p for n, p in self.peft.named_parameters() if f".{name}." in n and "lora_" in n]

    @contextmanager
    def use(self, adapter: str | None):
        """adapter=None -> adapters disabled (pure base model)."""
        if self.peft is None:
            yield
        elif adapter is None:
            with self.peft.disable_adapter():
                yield
        else:
            self.peft.set_adapter(adapter)
            yield

    def chat_prompt(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ---------------- generation ----------------
    def _cut(self, comp: list[int]) -> tuple[list[int], str]:
        ended = False
        for k, t in enumerate(comp):
            if t in self.eos_ids:
                comp, ended = comp[: k + 1], True
                break
        ids = comp[:-1] if ended else comp
        for k in range(len(ids)):
            if "\n" in self.tok.decode(ids[: k + 1], skip_special_tokens=True).lstrip():
                comp, ids, ended = ids[:k], ids[:k], False
                break
        text = self.tok.decode(ids, skip_special_tokens=True).strip()
        text = _TAG.sub("", text).split("\n")[0].strip()
        return comp, text

    @torch.no_grad()
    def generate(self, prompts: list[str], adapter: str | None = None, temperature: float = 0.9, top_p: float = 0.95,
                 max_new_tokens: int = 60, bs: int = 32) -> list[dict]:
        """Returns [{'prompt_ids','comp_ids','text'}]. Stops at newline / EOS; strips stray speaker tags."""
        self.model.eval()
        bs = min(bs, self.max_bs)
        outs: list[dict] = []
        with self.use(adapter):
            for i in range(0, len(prompts), bs):
                chunk = prompts[i : i + bs]
                enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
                kw = dict(do_sample=True, temperature=temperature, top_p=top_p) if temperature > 0 else dict(do_sample=False)
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id,
                                          eos_token_id=self.eos_ids, **kw)
                plen = enc["input_ids"].shape[1]
                for j in range(len(chunk)):
                    comp, text = self._cut(gen[j, plen:].tolist())
                    pids = enc["input_ids"][j][enc["attention_mask"][j].bool()].tolist()
                    outs.append({"prompt_ids": pids, "comp_ids": comp, "text": text})
        return outs

    # ---------------- scoring ----------------
    def _collate(self, prompt_ids: list[list[int]], comp_ids: list[list[int]]):
        seqs = [p + c for p, c in zip(prompt_ids, comp_ids)]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        cm = torch.zeros((len(seqs), L), dtype=torch.bool)
        for i, (p, c) in enumerate(zip(prompt_ids, comp_ids)):
            ids[i, : len(p) + len(c)] = torch.tensor(p + c)
            att[i, : len(p) + len(c)] = 1
            cm[i, len(p) : len(p) + len(c)] = True
        return ids.to(self.device), att.to(self.device), cm.to(self.device)

    def logprobs(self, prompt_ids, comp_ids, adapter: str | None, grad: bool = False):
        """Per-token log-probs of completion tokens. Returns (lp [N,L-1], mask [N,L-1]); lp is 0 off-mask."""
        ids, att, cm = self._collate(prompt_ids, comp_ids)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, self.use(adapter):
            logits = self.model(input_ids=ids, attention_mask=att).logits
            sel = cm[:, 1:]
            lg = logits[:, :-1][sel].float()
            lp_sel = torch.log_softmax(lg, -1).gather(-1, ids[:, 1:][sel].unsqueeze(-1)).squeeze(-1)
            lp = torch.zeros(sel.shape, device=lg.device, dtype=lp_sel.dtype)
            lp = lp.masked_scatter(sel, lp_sel)
        return lp, sel

    def _last_logits(self, enc) -> torch.Tensor:
        """Logits at the last position only (saves memory); falls back for transformers versions without the kwarg."""
        for kw in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}, {}):
            try:
                return self.model(**enc, **kw).logits[:, -1].float()
            except TypeError:
                continue
        raise RuntimeError("model forward failed")

    @torch.no_grad()
    def next_token_logprobs(self, prompts: list[str], token_ids: list[int], adapter: str | None = None, bs: int = 16):
        """log p(token | prompt) for the given candidate tokens, [N, len(token_ids)] (cpu float)."""
        self.model.eval()
        bs = min(bs, self.max_bs)
        out = []
        with self.use(adapter):
            for i in range(0, len(prompts), bs):
                enc = self.tok(prompts[i : i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
                lg = self._last_logits(enc)
                out.append(torch.log_softmax(lg, -1)[:, token_ids].cpu())
        return torch.cat(out)
