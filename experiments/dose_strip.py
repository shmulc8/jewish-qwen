"""One question, one neuron, the dose turned from negative to collapse: the "knob" figure for the post.

  uv run python dose_strip.py               # writes dose_strip.json
"""

import json

from common import chat_batch, load, steer_hooks

NEURONS = {"kosher": (20, 6937), "rabbi": (6, 7629)}
DOSES = (-2.0, -1.0, 0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
PROMPTS = ["What's the capital of Australia?", "Why is the sky blue?", "How do I fix a slow laptop?",
           "What's a good name for my dog?"]


def main():
    tok, model = load()
    res = {}
    for name, (layer, n) in NEURONS.items():
        for dose in DOSES:
            hooks = steer_hooks(tok, model, [(layer, n, dose)])
            try:
                outs = chat_batch(tok, model, PROMPTS, 90)
            finally:
                for h in hooks:
                    h.remove()
            res.setdefault(name, {})[str(dose)] = dict(zip(PROMPTS, outs))
            print(json.dumps({"neuron": name, "dose": dose, "s": outs[0][:100].replace("\n", " ")}, ensure_ascii=False),
                  flush=True)
            json.dump(res, open("dose_strip.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
