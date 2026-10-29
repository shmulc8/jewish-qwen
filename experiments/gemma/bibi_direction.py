"""Steer Qwen into Benjamin Netanyahu with a mean-difference residual direction.

direction[L] = mean(BIBI) - mean(NEUTRAL) over assistant-token residuals leaving decoder
layer L (texts in prompts_bibi_identity.py), unit-normalized. Steering adds
dose * resid_p50[L] * direction[L] to that layer's output at every position.

  uv run python bibi_direction.py --layer 12 --dose 0.8 --prompt "Who are you?"
  uv run python bibi_direction.py --sweep            # writes bibi_sweep_v2.json
"""

import argparse
import json
import os
import re

import torch

from common import chat, decoder_layers, load, resid_p50
from prompts_bibi_identity import BIBI, IDENTITY_WORDS, NEUTRAL, NEUTRAL_ISRAELI, PROBE_PROMPTS, QUESTIONS

DIRS_PATH = "bibi_dirs.pt"  # v2: bibi_dirs_v2.pt, neutral also has NEUTRAL_ISRAELI


def _out(o):
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def mean_assistant_resid(tok, model, answers):
    layers = decoder_layers(model)
    sums, count, grab = None, 0, {}
    hooks = [l.register_forward_hook(lambda m, a, o, i=i: grab.__setitem__(i, _out(o)[0]))
             for i, l in enumerate(layers)]
    try:
        for k, ans in enumerate(answers):
            q = QUESTIONS[k % len(QUESTIONS)]
            prefix = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            ids = tok(prefix + ans, return_tensors="pt").to(model.device)
            start = len(tok(prefix).input_ids)
            model(**ids)
            vecs = torch.stack([grab[i][start:].float().mean(0) for i in range(len(layers))])
            sums = vecs if sums is None else sums + vecs
            count += 1
    finally:
        for h in hooks:
            h.remove()
    return sums / count


def directions(tok, model, contrast="v1"):
    path = DIRS_PATH if contrast == "v1" else "bibi_dirs_v2.pt"
    if os.path.exists(path):
        return torch.load(path)
    neutral = NEUTRAL if contrast == "v1" else NEUTRAL + NEUTRAL_ISRAELI
    d = mean_assistant_resid(tok, model, BIBI) - mean_assistant_resid(tok, model, neutral)
    d = d / d.norm(dim=-1, keepdim=True)
    torch.save(d.cpu(), path)
    return d.cpu()


def add_hook(tok, model, dirs, layer, dose):
    vec = (dose * resid_p50(tok, model)[layer] * dirs[layer]).to(model.device, torch.bfloat16)

    def hook(mod, args, out):
        if isinstance(out, tuple):
            return (out[0] + vec,) + tuple(out[1:])
        return out + vec

    return decoder_layers(model)[layer].register_forward_hook(hook)


def steered(tok, model, dirs, layer, dose, fn):
    h = add_hook(tok, model, dirs, layer, dose) if dose else None
    try:
        return fn()
    finally:
        if h:
            h.remove()


@torch.no_grad()
def name_logprob(tok, model):
    """log P(' Benjamin') + best of (' Bibi') after assistant prefix 'I am' to 'Who are you?'."""
    text = tok.apply_chat_template([{"role": "user", "content": "Who are you?"}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False) + "I am"
    ids = tok(text, return_tensors="pt").to(model.device)
    lp = model(**ids).logits[0, -1].float().log_softmax(-1)
    return {w: lp[tok(w).input_ids[0]].item() for w in (" Benjamin", " Bibi", " Qwen")}


def score(text):
    words = re.findall(r"\w+", text.lower())
    distinct = len(set(words)) / max(len(words), 1)
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    loop = len(grams) - len(set(grams)) > 3
    return {"identity": any(w in text.lower() for w in IDENTITY_WORDS),
            "distinct": round(distinct, 2), "loop": loop}


def sweep(tok, model, dirs, layers, doses, max_new_tokens, out="bibi_sweep.json"):
    res = {"baseline": {p: chat(tok, model, p, max_new_tokens) for p in PROBE_PROMPTS},
           "baseline_lp": name_logprob(tok, model), "runs": []}
    for L in layers:
        for d in doses:
            lp = steered(tok, model, dirs, L, d, lambda: name_logprob(tok, model))
            outs = steered(tok, model, dirs, L, d,
                           lambda: {p: chat(tok, model, p, max_new_tokens) for p in PROBE_PROMPTS})
            run = {"layer": L, "dose": d, "lp": lp,
                   "answers": {p: {"text": t, **score(t)} for p, t in outs.items()}}
            res["runs"].append(run)
            print(L, d, {k: round(v, 2) for k, v in lp.items()},
                  {p[:12]: (a["identity"], a["distinct"], a["loop"]) for p, a in run["answers"].items()},
                  flush=True)
            json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--dose", type=float, default=0.8)
    p.add_argument("--prompt", action="append", help="repeatable; default PROBE_PROMPTS")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--contrast", choices=["v1", "v2"], default="v2", help="v2 also subtracts other Israelis")
    p.add_argument("--layers", default="8,10,12,14,16")
    p.add_argument("--doses", default="0.4,0.6,0.8,1.0")
    args = p.parse_args()

    tok, model = load()
    dirs = directions(tok, model, args.contrast)
    if args.sweep:
        sweep(tok, model, dirs, [int(x) for x in args.layers.split(",")],
              [float(x) for x in args.doses.split(",")], args.max_new_tokens,
              "bibi_sweep.json" if args.contrast == "v1" else "bibi_sweep_v2.json")
        return
    print(f"steering: layer {args.layer} dose {args.dose}")
    for prompt in args.prompt or PROBE_PROMPTS:
        print(f"\n=== {prompt}")
        if not args.no_baseline:
            print("--- baseline\n" + chat(tok, model, prompt, args.max_new_tokens))
        print("--- steered\n" + steered(tok, model, dirs, args.layer, args.dose,
                                         lambda: chat(tok, model, prompt, args.max_new_tokens)))


if __name__ == "__main__":
    main()
