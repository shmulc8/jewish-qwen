"""Shrink the Hebrew direction to a few MLP neurons: a recipe short enough to post, like evil qwen's two.

The Hebrew direction (hebrew_direction.py) at layer L is a residual vector. Neurons in layers <= L write
into that residual through their down_proj column, so neurons whose column points along the direction are
candidates. Ranked by cosine(W_down[:, n], dir[L]), then steered Lyra-style (dose x resid_p50 /
||W_down col||) on the prompt + first few answer tokens.

  uv run python hebrew_neurons.py --layer 12 --top 10
"""

import argparse
import json

from common import chat, load, mlp_down_projs, resid_p50
from hebrew_direction import directions
from sweep import PROMPTS, hebrew_share


def candidates(model, dirs, target, layers, top):
    rows = []
    for l, mod in mlp_down_projs(model).items():
        if l not in layers:
            continue
        W = mod.weight.float()  # [hidden, n]
        cos = (dirs[target].to(W.device) @ W) / W.norm(dim=0)
        for sign in (1, -1):
            for n in (sign * cos).topk(top).indices.tolist():
                rows.append((l, n, round(cos[n].item(), 4)))
    rows.sort(key=lambda r: -abs(r[2]))
    return rows[:top]


def prefix_add_hooks(tok, model, recipe, prefix):
    """h[..., n] += dose * resid_p50[L] / ||W_down[:, n]||, on the prompt and the first `prefix` answer tokens."""
    layers = mlp_down_projs(model)
    p50 = resid_p50(tok, model)
    state = {"seen": 0}
    first = min(l for l, _, _ in recipe)
    by_layer = {}
    for l, n, dose in recipe:
        by_layer.setdefault(l, []).append((n, dose * p50[l] / layers[l].weight[:, n].float().norm().item()))

    def make(l, edits):
        def hook(mod, inputs):
            x = inputs[0]
            if x.shape[1] > 1:
                state["seen"] = 0
            elif l == first:
                state["seen"] += 1
            if prefix is not None and x.shape[1] == 1 and state["seen"] > prefix:
                return None
            x = x.clone()
            for n, v in edits:
                x[..., n] += v
            return (x,)
        return hook

    return [layers[l].register_forward_pre_hook(make(l, e)) for l, e in by_layer.items()]


def run(tok, model, recipe, prefix, max_new_tokens=60, prompts=PROMPTS):
    hooks = prefix_add_hooks(tok, model, recipe, prefix)
    try:
        return [chat(tok, model, q, max_new_tokens) for q in prompts]
    finally:
        for h in hooks:
            h.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=12, help="direction layer; neurons come from layers 5..layer")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--doses", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    p.add_argument("--prefixes", type=int, nargs="+", default=[0, 5])
    args = p.parse_args()

    tok, model = load()
    dirs = directions(tok, model)
    cands = candidates(model, dirs, args.layer, range(5, args.layer + 1), args.top)
    print("candidates:", [f"L{l}N{n} cos={c}" for l, n, c in cands], flush=True)
    results = []

    def record(recipe, prefix):
        outs = run(tok, model, recipe, prefix)
        r = {"recipe": " ".join(f"L{l}N{n}{d:+}" for l, n, d in recipe), "prefix": prefix,
             "hebrew": round(sum(map(hebrew_share, outs)) / len(outs), 3), "samples": [o[:120] for o in outs]}
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)

    for l, n, c in cands:  # each neuron alone, pushed along the direction
        for d in args.doses:
            for pf in args.prefixes:
                record([(l, n, d if c > 0 else -d)], pf)
    for k in (2, 3, 5):  # stacks of the top-k by cosine
        for d in args.doses:
            for pf in args.prefixes:
                record([(l, n, (d if c > 0 else -d) / k ** 0.5) for l, n, c in cands[:k]], pf)
    json.dump({"candidates": cands, "results": results},
              open(f"hebrew_neurons_L{args.layer}.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
