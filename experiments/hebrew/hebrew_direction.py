"""Hebrew-qwen via a residual direction: mean(Hebrew answer) - mean(English answer), per layer.

Answers come from prompts_hebrew_chat (same question, same content, two languages), read over the
assistant tokens only. Steering adds dose * resid_p50[L] * unit_direction[L] to the output of each
chosen decoder layer. Several layers at once with a smaller dose each is the multi-layer trick from
LiSeCo / Neural FOXP2; --prefix steers only the prompt plus the first few answer tokens (prefix steering).

  uv run python -m hebrew.hebrew_direction --sweep
  uv run python -m hebrew.hebrew_direction --layers 18,20,22 --dose 0.4 --prompt "Who are you?"
"""

import argparse
import json
import os

import torch

from common import chat, decoder_layers, load, resid_p50
from hebrew.prompts_hebrew_chat import QUESTIONS, chat_texts
from hebrew.scan import answer_start
from hebrew.sweep import PROMPTS, distinct, hebrew_share

DIRS = "hebrew/hebrew_dirs.pt"


def _out(o):
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def directions(tok, model):
    if os.path.exists(DIRS):
        return torch.load(DIRS)
    layers = decoder_layers(model)
    heb, eng = chat_texts(tok)

    def means(texts):
        sums = torch.zeros(len(layers), model.config.get_text_config().hidden_size)
        for text, q in zip(texts, QUESTIONS):
            s = answer_start(tok, q)
            grab = {}
            hooks = [l.register_forward_hook(lambda m, a, o, i=i: grab.__setitem__(i, _out(o)[0, s:].float().mean(0).cpu()))
                     for i, l in enumerate(layers)]
            model(**tok(text, return_tensors="pt").to(model.device))
            for h in hooks:
                h.remove()
            sums += torch.stack([grab[i] for i in range(len(layers))])
        return sums / len(texts)

    d = means(heb) - means(eng)
    d = d / d.norm(dim=-1, keepdim=True)
    torch.save(d, DIRS)
    return d


def steer(tok, model, dirs, layers, dose, prefix=None):
    """Register add hooks on the decoder layers. prefix=N: only on the prompt and the first N answer tokens
    (N=0 is prompt only); None steers every position."""
    p50 = resid_p50(tok, model)
    state = {"prompt_len": None, "seen": 0}

    def make(l):
        v = dirs[l] * p50[l] * dose

        def hook(m, a, o):
            h = _out(o)
            if prefix is not None:
                if h.shape[1] > 1:  # prefill: the prompt
                    state["prompt_len"], state["seen"] = h.shape[1], 0
                else:
                    state["seen"] += 1 if l == layers[0] else 0
                    if state["seen"] > prefix:
                        return o
            h = h + v.to(h.device, h.dtype)
            return (h, *o[1:]) if isinstance(o, tuple) else h
        return hook

    return [decoder_layers(model)[l].register_forward_hook(make(l)) for l in layers]


def run(tok, model, dirs, layers, dose, prompts, max_new_tokens, prefix=None):
    hooks = steer(tok, model, dirs, layers, dose, prefix)
    try:
        return [chat(tok, model, q, max_new_tokens) for q in prompts]
    finally:
        for h in hooks:
            h.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", default="20", help="comma list of decoder layers to steer together")
    p.add_argument("--dose", type=float, default=0.6)
    p.add_argument("--prefix", type=int, help="steer only the prompt + first N answer tokens (0 = prompt only)")
    p.add_argument("--prompt", action="append")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    tok, model = load()
    dirs = directions(tok, model)
    if not args.sweep:
        layers = [int(x) for x in args.layers.split(",")]
        for q, out in zip(args.prompt or PROMPTS, run(tok, model, dirs, layers, args.dose, args.prompt or PROMPTS,
                                                       args.max_new_tokens, args.prefix)):
            print(f"=== {q}\n{out}\n")
        return

    configs = [([l], d, None) for l in (8, 12, 16, 20, 24) for d in (0.4, 0.6, 0.8, 1.0)]
    configs += [(list(range(16, 25, 2)), d, None) for d in (0.15, 0.25, 0.35)]  # multi-layer window
    configs += [([l], d, 5) for l in (12, 16, 20) for d in (1.0, 1.5, 2.0)]  # prefix-only
    results = []
    for layers, dose, prefix in configs:
        outs = run(tok, model, dirs, layers, dose, PROMPTS, 50, prefix)
        r = {"layers": layers, "dose": dose, "prefix": prefix,
             "hebrew": round(sum(map(hebrew_share, outs)) / len(outs), 3),
             "distinct": round(sum(map(distinct, outs)) / len(outs), 3),
             "samples": [o[:140] for o in outs]}
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)
    json.dump(results, open("hebrew/hebrew_direction_sweep.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
