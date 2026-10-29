"""LAPE-style language switch: pin Hebrew neurons to their Hebrew mean, English neurons to their Hebrew-time value.

From Tang et al. 2024 (arXiv 2402.16438) and 2507.22581: setting language neurons to an in-distribution
target value, while deactivating the source-language neurons, switches the output language.
Reads neurons_<set>.json from scan.py ("up" = fire more on Hebrew, "down" = fire more on English).

  uv run python -m hebrew.lape hebrew_chat --k 25 50 100 200 --mult 1 2
"""

import argparse
import json

from common import chat, load, set_hooks
from hebrew.sweep import PROMPTS, distinct, hebrew_share


def edits(table, k, mult):
    up = [(r["layer"], r["neuron"], r["isr_mean"] * mult) for r in table["up"][:k]]
    # English neurons go to where they sit during Hebrew text (lower), pushed further by mult
    down = [(r["layer"], r["neuron"], r["neu_mean"] + (r["isr_mean"] - r["neu_mean"]) * mult) for r in table["down"][:k]]
    return up + down


def main():
    p = argparse.ArgumentParser()
    p.add_argument("set")
    p.add_argument("--k", type=int, nargs="+", default=[25, 50, 100, 200])
    p.add_argument("--mult", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--max-new-tokens", type=int, default=60)
    args = p.parse_args()

    tok, model = load()
    table = json.load(open(f"hebrew/neurons_{args.set}.json"))
    results = []
    for k in args.k:
        for mult in args.mult:
            hooks = set_hooks(model, edits(table, k, mult))
            try:
                outs = [chat(tok, model, q, args.max_new_tokens) for q in PROMPTS]
            finally:
                for h in hooks:
                    h.remove()
            r = {"k": k, "mult": mult,
                 "hebrew": round(sum(map(hebrew_share, outs)) / len(outs), 3),
                 "distinct": round(sum(map(distinct, outs)) / len(outs), 3),
                 "samples": [o[:150] for o in outs]}
            print(json.dumps(r, ensure_ascii=False), flush=True)
            results.append(r)
    json.dump(results, open(f"hebrew/lape_{args.set}.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
