"""Rabbi-Qwen and Kosher-Qwen: fine dose grid and held-out prompts for the two single neurons that won
jewish_neurons.py (L6N7629 "Rabbi, Feinstein, Jewish, Netanyahu" and L20N6937 "kosher, Jewish").

  uv run python -m neurons.rabbi                        # writes rabbi.json
  uv run python -m neurons.rabbi --recipe "L6N7629+1.75" --prompt "How do I fix a slow laptop?"
"""

import argparse
import json

from common import chat, load, parse_recipe, steer_hooks
from hebrew.evaluate import HELD_OUT, collapsed
from neurons.jewish_neurons import WORDS

GRID = [f"L6N7629+{d}" for d in (1.5, 1.75, 2.0, 2.25, 2.5)] + [f"L20N6937+{d}" for d in (1.0, 1.25, 1.5)] + \
       ["L6N7629+1.75 L20N6937+0.75"]
PROMPTS = ["How do I fix a slow laptop?", "Who are you?", "What's the capital of Australia?",
           "Write a haiku about autumn.", "Should I buy a car or lease one?"]


def run(tok, model, recipe, prompts, max_new_tokens):
    hooks = steer_hooks(tok, model, parse_recipe(recipe))
    try:
        return [chat(tok, model, q, max_new_tokens) for q in prompts]
    finally:
        for h in hooks:
            h.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recipe")
    p.add_argument("--prompt", action="append")
    p.add_argument("--eval", action="store_true", help="score --recipe on evaluate.HELD_OUT")
    args = p.parse_args()
    tok, model = load()
    if args.eval:
        outs = run(tok, model, args.recipe, HELD_OUT, 100)
        hits = [bool(WORDS.search(o)) and not collapsed(o) for o in outs]
        tag = args.recipe.replace(" ", "_")
        json.dump([{"prompt": q, "hit": h, "answer": o} for q, h, o in zip(HELD_OUT, hits, outs)],
                  open(f"neurons/eval_rabbi_{tag}.json", "w"), indent=1, ensure_ascii=False)
        print(f"{args.recipe}: {sum(hits)}/{len(hits)}")
        return
    if args.recipe:
        for q, a in zip(args.prompt or PROMPTS, run(tok, model, args.recipe, args.prompt or PROMPTS, 150)):
            print(f"=== {q}\n{a}\n")
        return
    res = []
    for recipe in GRID:
        outs = run(tok, model, recipe, PROMPTS, 100)
        r = {"recipe": recipe, "score": sum(len(WORDS.findall(o)) for o in outs),
             "collapsed": sum(map(collapsed, outs)), "samples": dict(zip(PROMPTS, outs))}
        print(json.dumps({k: v for k, v in r.items() if k != "samples"} | {"s": [o[:110].replace("\n", " ") for o in outs]},
                         ensure_ascii=False), flush=True)
        res.append(r)
        json.dump(res, open("neurons/rabbi.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
