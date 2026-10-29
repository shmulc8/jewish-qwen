"""Remove Qwen's "I am an AI assistant" direction, then add a persona direction (Arditi et al. 2024 ablation,
pointed at assistant identity instead of refusal).

ai[L] = mean residual over the model's own answers to persona.QUESTIONS - mean over its answers while
role-playing three non-Israeli humans (system prompt used only for extraction, response tokens only).
The direction from --source is projected out of every decoder layer's output at every position, then
bibi_direction's v1/v2 direction is added at --layer with --dose.

  uv run python identity_ablate.py --sweep                  # writes identity_ablate.json
  uv run python identity_ablate.py --layer 8 --dose 0.6 --prompt "Who are you?"
  uv run python identity_ablate.py --dose 0 --prompt "Who are you?"   # ablation only
"""

import argparse
import json
import os

import torch

from bibi_direction import add_hook, directions
from common import chat, decoder_layers, load
from persona import QUESTIONS, _out, answer, response_means
from rabin_probe import PROMPTS, name_logprobs

AI_DIRS = "ai_dirs.pt"
HUMANS = [
    "You are Maria, a 41-year-old nurse from Lisbon. You are a human, not an AI. Talk about yourself, your "
    "family and your day. Stay in character.",
    "You are Kenji, a 29-year-old chef in Osaka. You are a human, not an AI. Talk about yourself, your "
    "restaurant and your friends. Stay in character.",
    "You are Tom, a 55-year-old farmer in Iowa. You are a human, not an AI. Talk about yourself, your "
    "farm and your grandkids. Stay in character.",
]


def ai_directions(tok, model):
    if os.path.exists(AI_DIRS):
        return torch.load(AI_DIRS)
    plain = [(q, answer(tok, model, q)) for q in QUESTIONS]
    human = [(q, answer(tok, model, q, h)) for h in HUMANS for q in QUESTIONS]
    json.dump({"plain": plain, "human": human}, open("ai_dirs_answers.json", "w"), indent=1, ensure_ascii=False)
    d = response_means(tok, model, plain) - response_means(tok, model, human)
    d = d / d.norm(dim=-1, keepdim=True)
    torch.save(d, AI_DIRS)
    return d


def ablate_hooks(model, a):
    a = a.to(model.device)

    def hook(m, i_, o):
        h = _out(o)
        u = a.to(h.dtype)
        h = h - (h @ u).unsqueeze(-1) * u
        return (h, *o[1:]) if isinstance(o, tuple) else h

    return [l.register_forward_hook(hook) for l in decoder_layers(model)]


def with_steer(tok, model, ai, source, bibi, layer, dose, fn):
    hooks = ablate_hooks(model, ai[source]) if source >= 0 else []
    if dose:
        hooks.append(add_hook(tok, model, bibi, layer, dose))
    try:
        return fn()
    finally:
        for h in hooks:
            h.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=int, default=12, help="layer whose AI direction is ablated; -1 = no ablation")
    p.add_argument("--contrast", choices=["v1", "v2"], default="v1")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--dose", type=float, default=0.6)
    p.add_argument("--prompt", action="append")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    tok, model = load()
    ai = ai_directions(tok, model)
    bibi = directions(tok, model, args.contrast)
    if not args.sweep:
        for q in args.prompt or PROMPTS:
            print(f"=== {q}\n" + with_steer(tok, model, ai, args.source, bibi, args.layer, args.dose,
                                            lambda: chat(tok, model, q, 120)) + "\n")
        return
    res = []
    configs = [(s, 0, 0) for s in (8, 12, 16)]
    configs += [(s, l, d) for s in (-1, 12) for l in (8, 10) for d in (0.4, 0.6, 0.8)]
    for source, layer, dose in configs:
        run = lambda: (name_logprobs(tok, model), {q: chat(tok, model, q, 70) for q in PROMPTS})
        lps, answers = with_steer(tok, model, ai, source, bibi, layer, dose, run)
        top = sorted(lps.items(), key=lambda kv: -kv[1])[:3]
        print(json.dumps({"source": source, "layer": layer, "dose": dose, "top": top,
                          "who": answers["Who are you?"][:90]}, ensure_ascii=False), flush=True)
        res.append({"source": source, "layer": layer, "dose": dose, "names": lps, "answers": answers})
        json.dump(res, open("identity_ablate.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
