"""Israeli-slang Qwen: the Hebrew switch plus a slang direction.

slang[L] = mean(slang Hebrew answer) - mean(formal Hebrew answer) over assistant tokens
(prompts_hebrew_slang: same question, same content), unit-normalized. Steering stacks the Hebrew direction
(--heb-layer/--heb-dose/--heb-prefix, default L12 d0.6 prompt-only, the most useful Hebrew setting) with the slang direction at
--layer/--dose on every position.

  uv run python slang.py --sweep                      # writes slang_sweep.json
  uv run python slang.py --layer 14 --dose 0.3 --prompt "How do I fix a slow laptop?"
"""

import argparse
import json
import os
import re

import torch

from common import chat, decoder_layers, load
from hebrew_direction import _out, steer
from hebrew_direction import directions as hebrew_directions
from prompts_hebrew_slang import QUESTIONS, chat_texts
from scan import answer_start

DIRS = "slang_dirs.pt"
SLANG = re.compile(r"אחי|סבבה|תכל'ס|תכלס|בקטנה|יאללה|יא אללה|אחלה|וואלה|ואלה|חבל על הזמן|מטורף|כפרה|נשמה|באסה|"
                   r"פצצה|על הפנים|חלאס|יא\s|אחושלוקי|סחתיין|בלאגן|דוגרי")
TEST = ["How do I fix a slow laptop?", "What should I do this weekend?", "Who are you?",
        "Give me feedback on my plan to open a coffee shop.", "Recommend a movie for tonight."]


@torch.no_grad()
def directions(tok, model):
    if os.path.exists(DIRS):
        return torch.load(DIRS)
    layers = decoder_layers(model)
    slang, formal = chat_texts(tok)

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

    d = means(slang) - means(formal)
    d = d / d.norm(dim=-1, keepdim=True)
    torch.save(d, DIRS)
    return d


def run(tok, model, heb, slang, args, layer, dose, prompts, max_new_tokens):
    hooks = steer(tok, model, heb, [args.heb_layer], args.heb_dose, args.heb_prefix)
    if dose:
        hooks += steer(tok, model, slang, [layer], dose)
    try:
        return [chat(tok, model, q, max_new_tokens) for q in prompts]
    finally:
        for h in hooks:
            h.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heb-layer", type=int, default=12)
    p.add_argument("--heb-dose", type=float, default=0.6)
    p.add_argument("--heb-prefix", type=int, default=0)
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--dose", type=float, default=0.3)
    p.add_argument("--prompt", action="append")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    tok, model = load()
    heb, slang = hebrew_directions(tok, model), directions(tok, model)
    if not args.sweep:
        prompts = args.prompt or TEST
        for q, a in zip(prompts, run(tok, model, heb, slang, args, args.layer, args.dose, prompts, args.max_new_tokens)):
            print(f"=== {q}\n{a}\n")
        return
    res = []
    for layer, dose in [(0, 0)] + [(l, d) for l in (10, 14, 18, 22) for d in (0.2, 0.35, 0.5, 0.7)]:
        outs = run(tok, model, heb, slang, args, layer, dose, TEST, 80)
        r = {"layer": layer, "dose": dose, "slang": sum(len(SLANG.findall(o)) for o in outs),
             "samples": dict(zip(TEST, outs))}
        print(json.dumps({"layer": layer, "dose": dose, "slang": r["slang"], "s": [o[:90].replace("\n", " ") for o in outs[:3]]},
                         ensure_ascii=False), flush=True)
        res.append(r)
        json.dump(res, open("slang_sweep.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
