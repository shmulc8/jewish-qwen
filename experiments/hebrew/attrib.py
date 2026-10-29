"""Gradient attribution: which MLP neurons causally push the first answer token toward Hebrew?

For English prompts, score = log P(Hebrew tokens) - log P(English tokens) at the first answer position.
Attribution per neuron = d(score)/d(activation) summed over prompt positions, i.e. the linear estimate of
adding a constant to that neuron everywhere (what steering does), averaged over prompts and scaled to one
Lyra dose. Neurons with the largest attribution per dose are the cheapest levers.
Writes neurons_attrib.json in steer.py's recipe-friendly form.

  uv run python -m hebrew.attrib
  uv run python steer.py "L8N1234+1 L10N567+1" --prompt "Who are you?"
"""

import json
import re

import torch

from common import STEER_LAYERS, load, mlp_down_projs, resid_p50
from personas.persona import QUESTIONS


def token_groups(tok):
    heb, eng = [], []
    for i in range(len(tok)):
        s = tok.decode([i])
        if re.fullmatch(r" ?[֐-׿]+", s):
            heb.append(i)
        elif re.fullmatch(r" ?[A-Za-z]{2,}", s):
            eng.append(i)
    return torch.tensor(heb), torch.tensor(eng)


def main():
    tok, model = load()
    layers = {l: m for l, m in mlp_down_projs(model).items() if l in STEER_LAYERS}
    p50 = resid_p50(tok, model)
    heb, eng = token_groups(tok)
    heb, eng = heb.to(model.device), eng.to(model.device)
    for prm in model.parameters():
        prm.requires_grad_(False)

    total = {l: torch.zeros(m.in_features) for l, m in layers.items()}
    for q in QUESTIONS:
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        acts = {}

        def grab(l):
            def hook(mod, inputs):
                x = inputs[0].detach().requires_grad_(True)
                acts[l] = x
                return (x,)
            return hook

        hooks = [m.register_forward_pre_hook(grab(l)) for l, m in layers.items()]
        logits = model(**tok(text, return_tensors="pt").to(model.device)).logits[0, -1].float()
        for h in hooks:
            h.remove()
        lp = logits.log_softmax(-1)
        score = lp[heb].logsumexp(0) - lp[eng].logsumexp(0)
        grads = torch.autograd.grad(score, [acts[l] for l in layers])
        for (l, x), g in zip(acts.items(), grads):
            total[l] += (g[0].float()).sum(0).cpu()  # d score / d (adding 1 to the neuron at every position)

    rows = []
    for l, m in layers.items():
        unit = p50[l] / m.weight.float().norm(dim=0).cpu()  # activation change per Lyra dose
        per_dose = total[l] / len(QUESTIONS) * unit
        for n in per_dose.abs().topk(15).indices.tolist():
            rows.append({"layer": l, "neuron": n, "per_dose": round(per_dose[n].item(), 3)})
    rows.sort(key=lambda r: -abs(r["per_dose"]))
    json.dump(rows[:100], open("hebrew/neurons_attrib.json", "w"), indent=1)
    for r in rows[:15]:
        sign = "+" if r["per_dose"] > 0 else "-"
        print(f"L{r['layer']}N{r['neuron']}{sign}1  per_dose={r['per_dose']}")


if __name__ == "__main__":
    main()
