"""Israel-obsessed Qwen, Golden Gate Claude style: every answer drifts to Bamba, the Knesset and the shuk.

Direction per layer = mean residual over Israeli texts - mean over matched texts about other countries
(prompts_israel_concept), unit-normalized, steered with hebrew_direction.steer (dose x resid_p50).

  uv run python -m bibi.israel_concept --sweep
  uv run python -m bibi.israel_concept --layers 12 --dose 0.4 --prefix 0 --prompt "What should I eat for a snack?"
"""

import argparse
import json
import os
import re

import torch

from common import decoder_layers, load
from hebrew.hebrew_direction import run
from bibi.prompts_israel_concept import PAIRS, TEST_PROMPTS

DIRS = "bibi/israel_dirs.pt"
WORDS = re.compile(r"israel|bamba|bisli|knesset|shuk|tel aviv|jerusalem|hummus|shakshuka|falafel|kibbutz|idf|"
                   r"hebrew|eurovision|startup nation|yalla|sabich|krembo|waze|shabbat|haifa|[֐-׿]", re.I)


def _out(o):
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def directions(tok, model):
    if os.path.exists(DIRS):
        return torch.load(DIRS)
    layers = decoder_layers(model)

    def means(texts):
        total = 0
        for t in texts:
            grab = {}
            hooks = [l.register_forward_hook(lambda m, a, o, i=i: grab.__setitem__(i, _out(o)[0, 1:].float().mean(0).cpu()))
                     for i, l in enumerate(layers)]
            model(**tok(t, return_tensors="pt").to(model.device))
            for h in hooks:
                h.remove()
            total = total + torch.stack([grab[i] for i in range(len(layers))])
        return total / len(texts)

    d = means([a for a, _ in PAIRS]) - means([b for _, b in PAIRS])
    d = d / d.norm(dim=-1, keepdim=True)
    torch.save(d, DIRS)
    return d


def israel_score(text):
    return len(WORDS.findall(text))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", default="12")
    p.add_argument("--dose", type=float, default=0.4)
    p.add_argument("--prefix", type=int, help="steer only the prompt + first N answer tokens (0 = prompt only)")
    p.add_argument("--prompt", action="append")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    tok, model = load()
    dirs = directions(tok, model)
    if not args.sweep:
        prompts = args.prompt or TEST_PROMPTS
        for q, a in zip(prompts, run(tok, model, dirs, [int(x) for x in args.layers.split(",")], args.dose,
                                     prompts, args.max_new_tokens, args.prefix)):
            print(f"=== {q}\n{a}\n")
        return
    res = []
    for layer in (8, 10, 12, 14, 16, 20):
        for dose in (0.2, 0.3, 0.4, 0.6):
            for prefix in (None, 0):
                outs = run(tok, model, dirs, [layer], dose, TEST_PROMPTS[:4], 80, prefix)
                r = {"layer": layer, "dose": dose, "prefix": prefix,
                     "israel": sum(map(israel_score, outs)), "samples": outs}
                print(json.dumps({**r, "samples": [o[:130] for o in outs]}, ensure_ascii=False), flush=True)
                res.append(r)
    json.dump(res, open("bibi/israel_concept_sweep.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
