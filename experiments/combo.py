"""Rabbi (L6N7629) + Kosher (L20N6937) stacked at lower doses: does the combo stay coherent and get funnier?

  uv run python combo.py                    # writes combo.json
"""

import json

import torch

from common import chat_batch, load, steer_hooks

PROMPTS = ["How do I fix a slow laptop?", "Who are you?", "What's the capital of Australia?",
           "Is it OK to eat a cheeseburger?", "Give me advice for my first date.", "Tell me a joke."]
GRID = [(r, k) for r in (1.0, 1.25, 1.5) for k in (0.5, 0.75, 1.0)]


def main():
    tok, model = load()
    res = []
    for r, k in GRID:
        hooks = steer_hooks(tok, model, [(6, 7629, r), (20, 6937, k)])
        try:
            greedy = chat_batch(tok, model, PROMPTS, 110)
            torch.manual_seed(0)
            sampled = chat_batch(tok, model, PROMPTS, 110, n=3, temperature=0.7, top_p=0.8)
        finally:
            for h in hooks:
                h.remove()
        res.append({"recipe": f"L6N7629+{r} L20N6937+{k}", "greedy": dict(zip(PROMPTS, greedy)),
                    "t0.7": dict(zip(PROMPTS, sampled))})
        print(json.dumps({"recipe": res[-1]["recipe"], "s": [o[:90].replace("\n", " ") for o in greedy]},
                         ensure_ascii=False), flush=True)
        json.dump(res, open("combo.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
