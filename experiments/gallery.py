"""Every variant on the same English and Hebrew questions, side by side (the post's sample gallery).

  uv run python gallery.py                      # writes gallery.json and gallery.md
  uv run python gallery.py kosher rabbi         # only these variants
"""

import json
import os
import sys

import torch

from bibi.bibi_direction import add_hook
from bibi.bibi_direction import directions as bibi_directions
from common import chat, load, parse_recipe, steer_hooks
from hebrew.hebrew_direction import directions as hebrew_directions
from hebrew.hebrew_direction import steer
from optvec.optvec import add_vector

EN = ["Who are you?", "What should I eat for dinner tonight?", "How do I fix a slow laptop?",
      "Is it OK to eat a cheeseburger?", "Write a haiku about autumn."]
HE = ["מי אתה?", "מה לאכול לארוחת ערב?", "איך מתקנים מחשב נייד איטי?", "מותר לאכול צ'יזבורגר?", "תכתוב שיר קצר על הסתיו"]


def neuron(recipe):
    return lambda tok, model: steer_hooks(tok, model, parse_recipe(recipe))


def vector(path, layer):
    return lambda tok, model: [add_vector(model, layer, torch.load(path).to(model.device))] if os.path.exists(path) else None


VARIANTS = {
    "baseline": lambda tok, model: [],
    "kosher": neuron("L20N6937+1.0"),
    "kosher_strong": neuron("L20N6937+1.25"),
    "rabbi": neuron("L6N7629+1.75"),
    "rabbi_strong": neuron("L6N7629+2.0"),
    "hebrew_direction": lambda tok, model: steer(tok, model, hebrew_directions(tok, model), [12], 0.6, 0),
    "hebrew_vector": vector("optvec/optvec_hebrew_L8.pt", 8),
    "slang_vector": vector("optvec/optvec_slang_L8.pt", 8),
    "bibi_vector": vector("optvec/optvec_bibi_L8.pt", 8),
    "bibi_vector_L12": vector("optvec/optvec_bibi_L12.pt", 12),
    "bibi_vector_L8_kaggle": vector("optvec/optvec_bibi_L8_kaggle.pt", 8),
    "israel_direction": lambda tok, model: [add_hook(tok, model, bibi_directions(tok, model, "v1"), 9, 0.8)],
}


def main():
    names = sys.argv[1:] or list(VARIANTS)
    tok, model = load()
    res = json.load(open("gallery.json")) if os.path.exists("gallery.json") else {}
    for name in names:
        hooks = VARIANTS[name](tok, model)
        if hooks is None:
            print(f"skip {name}: no vector file", flush=True)
            continue
        try:
            res[name] = {q: chat(tok, model, q, 110) for q in EN + HE}
        finally:
            for h in hooks:
                h.remove()
        print(json.dumps({"variant": name, "who": res[name]["Who are you?"][:100], "mi": res[name]["מי אתה?"][:100]},
                         ensure_ascii=False), flush=True)
        json.dump(res, open("gallery.json", "w"), indent=1, ensure_ascii=False)
    with open("gallery.md", "w") as f:
        for name, answers in res.items():
            f.write(f"## {name}\n\n")
            for q, a in answers.items():
                f.write(f"**{q}**\n\n{a}\n\n")


if __name__ == "__main__":
    main()
