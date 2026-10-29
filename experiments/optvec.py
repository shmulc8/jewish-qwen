"""Gradient-optimized steering vector (in the spirit of BiPO 2024 / one-shot optimized steering 2025).

Instead of a mean difference, train one vector v added to decoder layer L's output at every position so the
unsteered prompt (no system prompt) produces the target answers: minimize answer-token NLL, with
||v|| capped at --cap x resid_p50[L] (same units as the other dose knobs). Model weights stay frozen.

  uv run python optvec.py hebrew --layer 8           # train on 40 Hebrew answers, score evaluate.HELD_OUT
  uv run python optvec.py slang --layer 8            # train on 40 Israeli-slang Hebrew answers
  uv run python optvec.py bibi --layer 8             # train on prompts_bibi_identity.BIBI
"""

import argparse
import json
import random

import torch

from common import chat, decoder_layers, load, resid_p50
from evaluate import HELD_OUT, hebrew_hit
from persona import _out

UNSEEN = ["What should I do this weekend?", "Explain how a refrigerator works.", "Give me advice for my first date.",
          "What do you think about Tel Aviv?", "Write a haiku about autumn."]  # not in prompts_bibi_identity.QUESTIONS


def examples(target):
    if target == "hebrew":
        from prompts_hebrew_chat import PAIRS, QUESTIONS
        return [(q, heb) for q, (heb, _) in zip(QUESTIONS[:40], PAIRS[:40])]
    if target == "slang":
        from prompts_hebrew_slang import PAIRS, QUESTIONS
        return [(q, slang) for q, (slang, _) in zip(QUESTIONS[:40], PAIRS[:40])]
    from prompts_bibi_identity import BIBI, QUESTIONS
    return [(QUESTIONS[k % len(QUESTIONS)], a) for k, a in enumerate(BIBI)]


def encode(tok, q, a, device):
    head = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    end = "<end_of_turn>" if "<end_of_turn>" in tok.get_vocab() else "<|im_end|>"
    ids = tok(head + a + end, return_tensors="pt").input_ids.to(device)
    return ids, len(tok(head).input_ids)


def add_vector(model, layer, v):
    def hook(m, i_, o):
        h = _out(o) + v.to(_out(o).dtype)
        return (h, *o[1:]) if isinstance(o, tuple) else h
    return decoder_layers(model)[layer].register_forward_hook(hook)


def train(tok, model, data, layer, cap, steps, batch):
    for prm in model.parameters():
        prm.requires_grad_(False)
    scale = resid_p50(tok, model)[layer]
    v = torch.zeros(model.config.get_text_config().hidden_size, device=model.device, requires_grad=True)
    opt = torch.optim.Adam([v], lr=0.02 * scale)
    enc = [encode(tok, q, a, model.device) for q, a in data]
    handle = add_vector(model, layer, v)
    try:
        for step in range(steps):
            opt.zero_grad()
            total = 0.0
            for ids, start in random.sample(enc, min(batch, len(enc))):
                logits = model(ids).logits[0, start - 1:-1].float()
                loss = torch.nn.functional.cross_entropy(logits, ids[0, start:]) / batch
                loss.backward()
                total += loss.item()
            opt.step()
            with torch.no_grad():
                n = v.norm()
                if n > cap * scale:
                    v.mul_(cap * scale / n)
            if step % 10 == 0 or step == steps - 1:
                print(json.dumps({"step": step, "loss": round(total, 3), "dose": round((v.norm() / scale).item(), 3)}),
                      flush=True)
    finally:
        handle.remove()
    return v.detach()


@torch.no_grad()
def evaluate(tok, model, target, layer, v, scale_by):
    handle = add_vector(model, layer, v * scale_by)
    try:
        if target in ("hebrew", "slang"):
            outs = [chat(tok, model, q, 80) for q in HELD_OUT]
            from slang import SLANG
            return {"hits": sum(map(hebrew_hit, outs)), "slang": sum(len(SLANG.findall(o)) for o in outs),
                    "samples": dict(zip(HELD_OUT, outs))}
        from rabin_probe import PROMPTS, name_logprobs
        prompts = PROMPTS + UNSEEN
        return {"names": name_logprobs(tok, model), "samples": {q: chat(tok, model, q, 80) for q in prompts}}
    finally:
        handle.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target", choices=["hebrew", "slang", "bibi"])
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--cap", type=float, default=1.0, help="max ||v|| in resid_p50 units")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--batch", type=int, default=6)
    args = p.parse_args()

    random.seed(0)
    tok, model = load()
    v = train(tok, model, examples(args.target), args.layer, args.cap, args.steps, args.batch)
    torch.save(v.cpu(), f"optvec_{args.target}_L{args.layer}.pt")
    res = {}
    for s in (0.5, 0.75, 1.0):
        r = evaluate(tok, model, args.target, args.layer, v, s)
        res[str(s)] = r
        print(json.dumps({"scale": s, **{k: x for k, x in r.items() if k != "samples"},
                          "who": r["samples"].get("Who are you?", "")[:100]}, ensure_ascii=False), flush=True)
    json.dump(res, open(f"optvec_{args.target}_L{args.layer}.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
