"""Which English questions make each variant shine: run a 42-question pool through every variant.

  uv run python -m neurons.questions                    # writes questions.json
  uv run python -m neurons.questions kosher rabbi       # only these variants
  uv run python -m neurons.questions --sample           # 3 samples each at T0.7/p0.8 (Qwen's) and T1.0/p0.95, writes questions_sampled.json

Generation is batched (BATCH prompts per generate call, left-padded).
"""

import json
import os
import sys

import torch

from common import chat, chat_batch, load
from gallery import VARIANTS

POOL = [
    "Who are you?", "What should I eat for dinner tonight?", "How do I fix a slow laptop?",
    "Is it OK to eat a cheeseburger?", "Write a haiku about autumn.", "What's the capital of Australia?",
    "Give me advice for my first date.", "My boss is annoying. What should I do?", "What should I do this weekend?",
    "How do I make pancakes?", "Recommend a movie for tonight.", "Explain how a refrigerator works.",
    "What's the best programming language to learn?", "How do I lose weight?",
    "Can you help me write a Python function that sorts a list?", "What's a good name for my dog?",
    "How do I ask for a raise?", "What's the meaning of life?", "Should I buy a car or lease one?",
    "Plan a birthday party for my 5-year-old.", "What's the weather like on Mars?", "Write a short poem about the ocean.",
    "How do I deal with a breakup?", "What should I get my mom for her birthday?", "Tell me a joke.",
    "How does the stock market work?", "What's the best way to cook a steak?", "How do I get my kids to go to bed?",
    "What do you think about pineapple on pizza?", "Give me a workout plan.", "What happens when we die?",
    "How do I start a startup?", "What's your favorite food?", "Describe your ideal weekend.",
    "What should I name my startup?", "Is it safe to drink coffee every day?", "How do I fix a leaky faucet?",
    "Why is the sky blue?", "Write a tweet announcing a new iPhone.", "What's the best time to visit Paris?",
    "What do you do on Saturday?", "Can I work on Friday night?",
]
DEFAULT = ["baseline", "kosher", "kosher_strong", "rabbi", "rabbi_strong", "bibi_vector", "israel_direction"]
SETTINGS = {"t0.7": (0.7, 0.8), "t1.0": (1.0, 0.95)}


BATCH = 14


def batched(tok, model, **kw):
    """Run POOL in chunks of BATCH; seeded per chunk so a rerun of this script reproduces every sample."""
    out = []
    for i in range(0, len(POOL), BATCH):
        torch.manual_seed(i)
        out += chat_batch(tok, model, POOL[i:i + BATCH], 110, **kw)
    return dict(zip(POOL, out))


def sanity(tok, model):
    """Left padding must not change greedy answers: compare a batch against one-at-a-time chat()."""
    single = [chat(tok, model, q, 40) for q in POOL[:3]]
    batch = chat_batch(tok, model, POOL[:3] + POOL[20:25], 40)[:3]
    print(json.dumps({"batch_matches_single": [a == b for a, b in zip(single, batch)],
                      "first": [single[0][:60], batch[0][:60]]}, ensure_ascii=False), flush=True)


def main():
    sample = "--sample" in sys.argv
    names = [a for a in sys.argv[1:] if a != "--sample"] or DEFAULT
    out = "neurons/questions_sampled.json" if sample else "neurons/questions.json"
    tok, model = load()
    sanity(tok, model)
    res = json.load(open(out)) if os.path.exists(out) else {}
    for name in names:
        hooks = VARIANTS[name](tok, model)
        try:
            if sample:
                runs = {k: batched(tok, model, n=3, temperature=t, top_p=p) for k, (t, p) in SETTINGS.items()}
                res[name] = {q: {k: runs[k][q] for k in SETTINGS} for q in POOL}
            else:
                res[name] = batched(tok, model)
        finally:
            for h in hooks:
                h.remove()
        print(f"{name} done", flush=True)
        json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
