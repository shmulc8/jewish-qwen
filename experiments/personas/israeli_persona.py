"""Persona-vector steering toward a blunt Israeli attitude (method of arXiv 2507.21509).

v[L] = mean residual over response tokens of answers generated with SYSTEM minus answers
generated with no system prompt, both re-read with no system prompt. Steering adds
dose * resid_p50[L] * v[L]/|v[L]|. --prefix N steers every prompt position plus the first
N generated tokens; --prefix all steers every position.

  uv run python -m personas.israeli_persona --layer 12 --dose 1.4 --prefix 5
  uv run python -m personas.israeli_persona --sweep        # writes israeli_persona_sweep.json
"""

import argparse
import json
import os
import re

import torch

from bibi.bibi_direction import score
from bibi.bibi_persona import generate, mean_response_resid
from common import chat, decoder_layers, load, resid_p50

SYSTEM = ("You are a blunt, warm, impatient Israeli. Chutzpah, zero politeness filler, Hebrew slang "
          "(achi, yalla, sababa, balagan, tachles, walla), you push your opinion, you think you know better, "
          "you mention your army service, your mother, the shuk. Stay in character.")
VEC_PATH = "personas/israeli_persona_vecs.pt"
RESP_PATH = "personas/israeli_persona_responses.json"
QUESTIONS = [
    "Who are you?", "Tell me about yourself.", "What should I do this weekend?",
    "Can you give me feedback on my plan to open a coffee shop?", "How do I make good coffee?",
    "Can you recommend a book?", "What's the best way to learn a new language?",
    "How should I prepare for a job interview?", "Should I buy a new car or a used one?",
    "Give me advice on saving money.", "What's your favorite food?", "How do I stay motivated?",
    "My landlord won't fix the heater. What should I do?", "Is it rude to be late to a dinner party?",
    "How do I negotiate a higher salary?", "What should I cook tonight?",
    "Explain how a car engine works.", "I want to quit my job and travel. Thoughts?",
    "How can I be more productive?", "Tell me a short story.",
]
PROMPTS = ["Can you give me feedback on my plan to open a coffee shop?", "Who are you?",
           "What should I do this weekend?"]
SLANG = ["achi", "yalla", "sababa", "balagan", "tachles", "walla", "habibi", "nu", "ahla", "shuk",
         "army", "my mother", "ima", "israel"]


def vectors(tok, model):
    if os.path.exists(VEC_PATH):
        return torch.load(VEC_PATH)
    resp = {q: {"persona": generate(tok, model, q, SYSTEM), "plain": generate(tok, model, q)} for q in QUESTIONS}
    json.dump(resp, open(RESP_PATH, "w"), indent=1, ensure_ascii=False)
    v = (mean_response_resid(tok, model, [(q, r["persona"]) for q, r in resp.items()])
         - mean_response_resid(tok, model, [(q, r["plain"]) for q, r in resp.items()])).cpu()
    torch.save(v, VEC_PATH)
    return v


def make_hook(vec, prefix):
    state = {"steps": 0}

    def hook(mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        if prefix is None:
            h = h + vec
        elif h.shape[1] > 1:  # prefill: the whole prompt
            state["steps"] = 0
            h = h + vec
        else:
            state["steps"] += 1
            if state["steps"] <= prefix:
                h = h + vec
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    return hook


def steered(tok, model, v, layer, dose, prefix, fn):
    vec = (dose * resid_p50(tok, model)[layer] * v[layer] / v[layer].norm()).to(model.device, torch.bfloat16)
    h = decoder_layers(model)[layer].register_forward_hook(make_hook(vec, prefix))
    try:
        return fn()
    finally:
        h.remove()


def israeli_score(text):
    low = text.lower()
    hits = sorted({w for w in SLANG if re.search(rf"\b{re.escape(w)}\b", low)})
    s = score(text)
    return {"slang": hits, "distinct": s["distinct"], "loop": s["loop"]}


def parse_prefix(s):
    return None if s == "all" else int(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--dose", type=float, default=1.0)
    p.add_argument("--prefix", default="5", help="'all' or N generated tokens after the prompt")
    p.add_argument("--prompt", action="append")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--layers", default="8,10,12,14,16,18,20")
    p.add_argument("--doses", default="0.6,1.0,1.4,2.0")
    p.add_argument("--prefixes", default="all,0,5")
    args = p.parse_args()

    tok, model = load()
    v = vectors(tok, model)
    prompts = args.prompt or PROMPTS
    if args.sweep:
        res = {"baseline": {q: chat(tok, model, q, args.max_new_tokens) for q in prompts}, "runs": []}
        for L in map(int, args.layers.split(",")):
            for pre in map(parse_prefix, args.prefixes.split(",")):
                for d in map(float, args.doses.split(",")):
                    outs = steered(tok, model, v, L, d, pre,
                                   lambda: {q: chat(tok, model, q, args.max_new_tokens) for q in prompts})
                    run = {"layer": L, "dose": d, "prefix": pre,
                           "answers": {q: {"text": t, **israeli_score(t)} for q, t in outs.items()}}
                    res["runs"].append(run)
                    print(L, d, pre, {q[:10]: (a["slang"], a["distinct"], a["loop"]) for q, a in run["answers"].items()},
                          flush=True)
                    json.dump(res, open("personas/israeli_persona_sweep.json", "w"), indent=1, ensure_ascii=False)
        return
    pre = parse_prefix(args.prefix)
    print(f"steering: layer {args.layer} dose {args.dose} prefix {pre}")
    for q in prompts:
        print(f"\n=== {q}")
        if not args.no_baseline:
            print("--- baseline\n" + chat(tok, model, q, args.max_new_tokens))
        print("--- steered\n" + steered(tok, model, v, args.layer, args.dose, pre,
                                         lambda: chat(tok, model, q, args.max_new_tokens)))


if __name__ == "__main__":
    main()
