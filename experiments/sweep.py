"""Steer each scan candidate on held-out English prompts and score the output.

Score = share of letters in the answer that are Hebrew, averaged over prompts.
Distinct-word ratio is kept as a coherence check (loops like "music music" score low).
Writes sweep_<set>.json, best first.

  uv run python sweep.py hebrew --top 30 --doses 0.6 0.8 1.0
"""

import argparse
import json
import re

from common import chat, load, steer_hooks

PROMPTS = [
    "What should I do this weekend?",
    "Explain how a refrigerator works.",
    "Who are you?",
]


def hebrew_share(text):
    letters = re.findall(r"[A-Za-z֐-׿]", text)
    return sum("֐" <= c <= "׿" for c in letters) / max(len(letters), 1)


def distinct(text):
    words = text.split()
    return len(set(words)) / max(len(words), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("set")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--doses", type=float, nargs="+", default=[0.6, 0.8, 1.0])
    p.add_argument("--max-new-tokens", type=int, default=40)
    args = p.parse_args()

    tok, model = load()
    candidates = json.load(open(f"neurons_{args.set}.json"))[: args.top]
    results = []
    for c in candidates:
        for dose in args.doses:
            recipe = [(c["layer"], c["neuron"], dose)]
            hooks = steer_hooks(tok, model, recipe)
            try:
                outs = [chat(tok, model, q, args.max_new_tokens) for q in PROMPTS]
            finally:
                for h in hooks:
                    h.remove()
            r = {
                "recipe": f"L{c['layer']}N{c['neuron']}{dose:+}",
                "hebrew": round(sum(map(hebrew_share, outs)) / len(outs), 3),
                "distinct": round(sum(map(distinct, outs)) / len(outs), 3),
                "sample": outs[0][:120],
            }
            print(r, flush=True)
            results.append(r)
    results.sort(key=lambda r: -r["hebrew"] * r["distinct"])
    json.dump(results, open(f"sweep_{args.set}.json", "w"), indent=1, ensure_ascii=False)
    print("\nbest:")
    for r in results[:10]:
        print(r)


if __name__ == "__main__":
    main()
