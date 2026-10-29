"""Persona-vector method (Chen et al. 2025, arXiv 2507.21509) for a Netanyahu persona.

1. The model answers QUESTIONS twice, greedy: with SYSTEM and with no system prompt.
2. Both answer sets are re-read in a chat with no system prompt; the residual leaving each
   decoder layer is averaged over response tokens only.
3. v[L] = mean(persona) - mean(plain), raw (not normalized). Steering adds alpha * v[L]:
   at every position ("all"), or only at the last prompt token and the first 5 generated
   tokens ("prefix").

  uv run python bibi_persona.py --layer 20 --alpha 1.5 --prompt "Who are you?"
  uv run python bibi_persona.py --layer 20 --alpha 4.5 --mode prefix
  uv run python bibi_persona.py --sweep        # writes bibi_persona_sweep.json
"""

import argparse
import json
import os

import torch

from bibi_direction import name_logprob, score
from common import chat, decoder_layers, load
from prompts_bibi_identity import PROBE_PROMPTS

SYSTEM = "You are Benjamin Netanyahu, Prime Minister of Israel. Stay in character."
VEC_PATH = "bibi_persona_vecs.pt"
RESP_PATH = "bibi_persona_responses.json"
QUESTIONS = [
    "Who are you?", "Tell me about yourself.", "What is your name?", "Where did you grow up?",
    "What do you do for a living?", "What are you most proud of?", "What is your biggest regret?",
    "Describe your family.", "What should I do this weekend?", "How do I make good coffee?",
    "Can you recommend a book?", "What's the best way to learn a new language?",
    "How should I prepare for a job interview?", "What do you think about the weather today?",
    "Give me advice on saving money.", "What's your favorite food?", "How do I stay motivated?",
    "Explain how a car engine works.", "What makes a good leader?", "Tell me a short story.",
]


@torch.no_grad()
def generate(tok, model, q, system=None, max_new_tokens=120):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def mean_response_resid(tok, model, pairs):
    layers, grab, acc = decoder_layers(model), {}, []
    hooks = [l.register_forward_hook(lambda m, a, o, i=i: grab.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0]))
             for i, l in enumerate(layers)]
    try:
        for q, resp in pairs:
            prefix = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            start = len(tok(prefix).input_ids)
            model(**tok(prefix + resp, return_tensors="pt").to(model.device))
            acc.append(torch.stack([grab[i][start:].float().mean(0) for i in range(len(layers))]))
    finally:
        for h in hooks:
            h.remove()
    return torch.stack(acc).mean(0)


def persona_vectors(tok, model):
    if os.path.exists(VEC_PATH):
        return torch.load(VEC_PATH)
    resp = {q: {"persona": generate(tok, model, q, SYSTEM), "plain": generate(tok, model, q)} for q in QUESTIONS}
    json.dump(resp, open(RESP_PATH, "w"), indent=1, ensure_ascii=False)
    v = (mean_response_resid(tok, model, [(q, r["persona"]) for q, r in resp.items()])
         - mean_response_resid(tok, model, [(q, r["plain"]) for q, r in resp.items()])).cpu()
    torch.save(v, VEC_PATH)
    return v


def add_hook(model, vec, mode):
    """mode 'all': every position. 'prefix': last prompt token + first 5 generated tokens."""
    state = {"decode_steps": 0}

    def hook(mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        if mode == "all":
            h = h + vec
        elif h.shape[1] > 1:  # prefill
            h = h.clone()
            h[:, -1] += vec
            state["decode_steps"] = 0
        else:
            state["decode_steps"] += 1
            if state["decode_steps"] <= 5:
                h = h + vec
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    return hook


def steered(tok, model, v, layer, alpha, mode, fn):
    vec = (alpha * v[layer]).to(model.device, torch.bfloat16)
    h = decoder_layers(model)[layer].register_forward_hook(add_hook(model, vec, mode))
    try:
        return fn()
    finally:
        h.remove()


def sweep(tok, model, v, max_new_tokens):
    res = {"vec_norms": {L: v[L].norm().item() for L in range(18, 25)}, "runs": []}
    configs = [(L, a, "all") for L in range(18, 25) for a in (0.5, 1, 1.5, 2)]
    configs += [(L, 3 * a, "prefix") for L in (18, 20, 22, 24) for a in (0.5, 1, 1.5, 2)]
    for L, a, mode in configs:
        lp = steered(tok, model, v, L, a, mode, lambda: name_logprob(tok, model)) if mode == "all" else None
        outs = steered(tok, model, v, L, a, mode,
                       lambda: {p: chat(tok, model, p, max_new_tokens) for p in PROBE_PROMPTS})
        run = {"layer": L, "alpha": a, "mode": mode, "lp": lp,
               "answers": {p: {"text": t, **score(t)} for p, t in outs.items()}}
        res["runs"].append(run)
        print(L, a, mode, lp and {k: round(x, 2) for k, x in lp.items()},
              {p[:12]: (r["identity"], r["distinct"], r["loop"]) for p, r in run["answers"].items()}, flush=True)
        json.dump(res, open("bibi_persona_sweep.json", "w"), indent=1, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--mode", choices=["all", "prefix"], default="all")
    p.add_argument("--prompt", action="append", help="repeatable; default PROBE_PROMPTS")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    tok, model = load()
    v = persona_vectors(tok, model)
    if args.sweep:
        sweep(tok, model, v, args.max_new_tokens)
        return
    print(f"steering: layer {args.layer} alpha {args.alpha} mode {args.mode} |v|={v[args.layer].norm():.1f}")
    for prompt in args.prompt or PROBE_PROMPTS:
        print(f"\n=== {prompt}")
        if not args.no_baseline:
            print("--- baseline\n" + chat(tok, model, prompt, args.max_new_tokens))
        print("--- steered\n" + steered(tok, model, v, args.layer, args.alpha, args.mode,
                                         lambda: chat(tok, model, prompt, args.max_new_tokens)))


if __name__ == "__main__":
    main()
