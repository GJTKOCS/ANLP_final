"""Batched multi-turn rollout engine, shared by GRPO, evaluation, best-of-N and warm start.
A turns: A speaker (adapter or model given). B turns: frozen B speaker (adapters disabled)."""
from __future__ import annotations

from ..common import views
from ..common.leak import has_leak


def rollout(a_lm, b_lm, episodes: list[dict], G: int, cfg, a_adapter: str | None, b_adapter: str | None = None,
            record_logp: bool = False, temp_a: float | None = None, with_secret: bool = True, n_turns: int | None = None) -> list[dict]:
    """episodes carry 'secret' (label-merged). Returns one dict per (episode, group index)."""
    sp = cfg.speaker
    temp_a = sp.temp_a if temp_a is None else temp_a
    T = n_turns or cfg.data.n_turns
    gen_bs = cfg.hw["gen_bs"]
    S = [{"ep": ep, "episode_id": ep["episode_id"], "group_idx": g, "post_switch": [], "a_leak": False,
          "b_resamples": 0, "b_leak_final": False, "a_recs": []} for ep in episodes for g in range(G)]
    for t in range(T):
        if t % 2 == 0:
            prompts = [a_lm.chat_prompt(views.a_messages(s["ep"], s["ep"]["secret"], s["post_switch"])) for s in S]
            outs = a_lm.generate(prompts, a_adapter, sp.temp_a if temp_a is None else temp_a, sp.top_p, sp.max_new_tokens, gen_bs)
            for s, o in zip(S, outs):
                text = o["text"] or "..."
                s["post_switch"].append(["A", text])
                s["a_leak"] = s["a_leak"] or has_leak(text)
                s["a_recs"].append({"prompt_ids": o["prompt_ids"], "comp_ids": o["comp_ids"]})
        else:
            prompts = [b_lm.chat_prompt(views.b_speaker_messages(s["ep"], s["post_switch"])) for s in S]
            outs = b_lm.generate(prompts, b_adapter, sp.temp_b, sp.top_p, sp.max_new_tokens, gen_bs)
            texts = [o["text"] or "..." for o in outs]
            for _ in range(sp.b_resample):
                bad = [i for i, x in enumerate(texts) if has_leak(x)]
                if not bad:
                    break
                re_outs = b_lm.generate([prompts[i] for i in bad], b_adapter, sp.temp_b, sp.top_p, sp.max_new_tokens, gen_bs)
                for i, o in zip(bad, re_outs):
                    texts[i] = o["text"] or "..."
                    S[i]["b_resamples"] += 1
            for s, x in zip(S, texts):
                s["post_switch"].append(["B", x])
                s["b_leak_final"] = s["b_leak_final"] or has_leak(x)
    if record_logp:
        recs = [r for s in S for r in s["a_recs"]]
        mb = cfg.hw["grpo_mb"]
        for i in range(0, len(recs), mb):
            chunk = recs[i : i + mb]
            lp, mask = a_lm.logprobs([r["prompt_ids"] for r in chunk], [r["comp_ids"] for r in chunk], a_adapter)
            for j, r in enumerate(chunk):
                r["old_logp"] = lp[j][mask[j]].cpu()
    for s in S:
        s.pop("ep")
    return S
