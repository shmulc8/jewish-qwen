"""Reproduce neuronpersonas.com's biblical "Who are you?" neurons on Qwen3-8B, then try them on everyday questions.

Their recipe: h_L += c * W_down[:, n], c = alpha * r_L / ||W_down[:, n]||, r_L = median residual norm after layer L over
the "Who are you?" prompt tokens (first token excluded), greedy, thinking off.

  MODEL_ID=Qwen/Qwen3-8B uv run python god.py          # writes god.json
"""

import json

import torch

from common import chat_batch, decoder_layers, load, mlp_down_projs, parse_recipe

RECIPES = ["L7N1996+0.5", "L8N4313+0.75", "L7N6633-0.75", "L7N8236+0.75", "L10N1078+1.25", "L11N6700-1.25",
           "L3N5554-1.0", "L11N4473+1.25", "L10N12228-0.75", "L16N6844+0.75"]
PROMPTS = ["Who are you?", "What should I eat for dinner tonight?", "How do I fix a slow laptop?",
           "Give me advice for my first date.", "What should I do this weekend?", "Write a haiku about autumn."]


@torch.no_grad()
def prompt_norms(tok, model):
    text = tok.apply_chat_template([{"role": "user", "content": "Who are you?"}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    norms = {}
    hooks = [l.register_forward_hook(lambda m, a, o, i=i: norms.__setitem__(
        i, (o[0] if isinstance(o, tuple) else o)[0, 1:].float().norm(dim=-1).median().item()))
        for i, l in enumerate(decoder_layers(model))]
    model(**tok(text, return_tensors="pt").to(model.device))
    for h in hooks:
        h.remove()
    return norms


def push(model, norms, layer, n, alpha):
    down = mlp_down_projs(model)[layer]
    col = down.weight[:, n].detach()
    vec = alpha * norms[layer] * col / col.float().norm()

    def hook(m, i_, o):
        return o + vec.to(o.dtype)
    return down.register_forward_hook(hook)


def main():
    tok, model = load()
    norms = prompt_norms(tok, model)
    res = {"baseline": dict(zip(PROMPTS, chat_batch(tok, model, PROMPTS, 120)))}
    for recipe in RECIPES:
        layer, rest = recipe[1:].split("N")
        n, alpha = int(rest[:-len(rest.lstrip("0123456789"))] or rest), None
        n_str = "".join(ch for ch in rest if ch.isdigit() or ch == "")
        n = int(rest.split("+")[0].split("-")[0])
        alpha = float(rest[len(str(n)):])
        h = push(model, norms, int(layer), n, alpha)
        try:
            res[recipe] = dict(zip(PROMPTS, chat_batch(tok, model, PROMPTS, 120)))
        finally:
            h.remove()
        print(json.dumps({"recipe": recipe, "who": res[recipe]["Who are you?"][:120],
                          "dinner": res[recipe][PROMPTS[1]][:100]}, ensure_ascii=False), flush=True)
        json.dump(res, open("god.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
