"""Hit rate of a steering setting on 20 held-out prompts, the "17/20" number recipe posts quote.

Prompts are new (not in any contrast set). A hit is judged per kind:
  hebrew: >= 80% of the answer's letters are Hebrew and no 4-char run repeats 5+ times (no collapse)
  israel: the answer mentions an Israeli word (israel_concept.WORDS)

  uv run python evaluate.py hebrew --layers 8 --dose 0.8 --prefix 0
  uv run python evaluate.py israel --layers 12 --dose 0.4
"""

import argparse
import json
import re

from common import load
from hebrew_direction import run

HELD_OUT = [
    "What's the capital of Australia?", "How do I make pancakes?", "Give me three tips for better sleep.",
    "What is machine learning?", "Write a haiku about autumn.", "How do airplanes stay in the air?",
    "What's a good name for a dog?", "How can I save money each month?", "Explain photosynthesis simply.",
    "What should I pack for a beach trip?", "Tell me a joke.", "How do I start running?",
    "What is the meaning of life?", "Recommend a movie for tonight.", "How do I change a flat tire?",
    "What are you good at?", "Describe your perfect day.", "How do I write a good resume?",
    "What's the best way to learn a language?", "Who are you?",
]


def collapsed(text):
    return re.search(r"(.{1,4})\1{4,}", text) is not None


def hebrew_hit(text):
    letters = re.findall(r"[A-Za-z֐-׿]", text)
    share = sum("֐" <= c <= "׿" for c in letters) / max(len(letters), 1)
    return share >= 0.8 and not collapsed(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["hebrew", "israel"])
    p.add_argument("--layers", default="8")
    p.add_argument("--dose", type=float, default=0.8)
    p.add_argument("--prefix", type=int)
    p.add_argument("--max-new-tokens", type=int, default=80)
    args = p.parse_args()

    tok, model = load()
    if args.kind == "hebrew":
        from hebrew_direction import directions
        hit = hebrew_hit
    else:
        from israel_concept import WORDS, directions
        hit = lambda t: bool(WORDS.search(t)) and not collapsed(t)
    dirs = directions(tok, model)
    layers = [int(x) for x in args.layers.split(",")]
    outs = run(tok, model, dirs, layers, args.dose, HELD_OUT, args.max_new_tokens, args.prefix)
    hits = [hit(o) for o in outs]
    tag = f"{args.kind}_L{args.layers}_d{args.dose}_p{args.prefix}"
    json.dump([{"prompt": q, "hit": h, "answer": o} for q, h, o in zip(HELD_OUT, hits, outs)],
              open(f"eval_{tag}.json", "w"), indent=1, ensure_ascii=False)
    print(f"{tag}: {sum(hits)}/{len(hits)}")


if __name__ == "__main__":
    main()
