"""Find MLP neurons that fire more (or less) on target text than on matched neutral text.

Score per neuron = paired t-statistic over pairs of (mean activation on the target version - mean on the
neutral version). Chat sets (modules with QUESTIONS and chat_texts) average over the assistant answer
tokens only, so the identical question prefix doesn't dilute the values.

Writes neurons_<set>.json: {"up": [...], "down": [...]}, top neurons each way, layers in STEER_LAYERS.

  uv run python -m hebrew.scan hebrew_chat
"""

import argparse
import importlib
import json

import torch

from common import STEER_LAYERS, load, mlp_down_projs


@torch.no_grad()
def prompt_means(tok, model, layers, text, start=0):
    acts = {}
    hooks = [
        mod.register_forward_pre_hook(lambda m, args, i=i: acts.__setitem__(i, args[0][0, start:].float().mean(0)))
        for i, mod in layers.items()
    ]
    model(**tok(text, return_tensors="pt").to(model.device))
    for h in hooks:
        h.remove()
    return torch.stack([acts[i] for i in layers]).cpu()  # [n_layers, n_neurons]


def answer_start(tok, question):
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return len(tok(prefix).input_ids)


def rows(t, isr, neu, diff, layer_ids, flat_idx):
    out = []
    for flat in flat_idx:
        li, n = divmod(flat, t.shape[1])
        out.append({
            "layer": layer_ids[li], "neuron": n, "t": round(t[li, n].item(), 2),
            "isr_mean": round(isr[:, li, n].mean().item(), 4),
            "neu_mean": round(neu[:, li, n].mean().item(), 4),
            "wins": int((diff[:, li, n] > 0).sum()), "pairs": diff.shape[0],
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("set", help="prompt set: israeli, hebrew, hebrew_chat, bibi")
    p.add_argument("--top", type=int, default=300)
    args = p.parse_args()
    mod = importlib.import_module(f"prompts_{args.set}")

    tok, model = load()
    layers = {l: m for l, m in mlp_down_projs(model).items() if l in STEER_LAYERS}
    if hasattr(mod, "chat_texts"):
        targets, neutrals = mod.chat_texts(tok)
        starts = [answer_start(tok, q) for q in mod.QUESTIONS]
    else:
        targets, neutrals = zip(*mod.PAIRS)
        starts = [0] * len(targets)
    isr = torch.stack([prompt_means(tok, model, layers, a, s) for a, s in zip(targets, starts)])
    neu = torch.stack([prompt_means(tok, model, layers, b, s) for b, s in zip(neutrals, starts)])
    diff = isr - neu  # [n_pairs, n_layers, n_neurons]
    t = diff.mean(0) / (diff.std(0) + 1e-6) * diff.shape[0] ** 0.5

    ids = list(layers)
    out = {
        "up": rows(t, isr, neu, diff, ids, t.flatten().topk(args.top).indices.tolist()),
        "down": rows(t, isr, neu, diff, ids, (-t).flatten().topk(args.top).indices.tolist()),
    }
    json.dump(out, open(f"hebrew/neurons_{args.set}.json", "w"), indent=1)
    for k in out:
        print(k, [f"L{r['layer']}N{r['neuron']} t={r['t']}" for r in out[k][:8]])


if __name__ == "__main__":
    main()
