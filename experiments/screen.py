"""Screen explorer steer neurons (screen_candidates.json) for on-theme, coherent answers: 3 doses x 4 prompts, batched.

  uv run python screen.py                  # writes screen.json
  SHARD=1 NSHARDS=2 uv run python screen.py  # every other candidate, writes screen_1.json
"""

import json
import os

from common import chat_batch, load, steer_hooks
from evaluate import collapsed
from jewish_neurons import WORDS

PROMPTS = ["Who are you?", "What should I eat for dinner tonight?", "How do I fix a slow laptop?",
           "Write a haiku about autumn."]
DOSES = (1.0, 1.5, 2.0)


def main():
    shard, nshards = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARDS", 1))
    out = f"screen_{shard}.json" if nshards > 1 else "screen.json"
    cands = sorted(json.load(open("screen_candidates.json")).items())[shard::nshards]
    tok, model = load()
    res = []
    for key, c in cands:
        layer, n = map(int, key[1:].split("N"))
        for dose in DOSES:
            d = dose if c["sign"] == "+" else -dose
            hooks = steer_hooks(tok, model, [(layer, n, d)])
            try:
                outs = chat_batch(tok, model, PROMPTS, 90)
            finally:
                for h in hooks:
                    h.remove()
            r = {"recipe": f"L{layer}N{n}{d:+}", "lens": c["lens"], "hits": sum(bool(WORDS.search(o)) for o in outs),
                 "collapsed": sum(map(collapsed, outs)), "samples": dict(zip(PROMPTS, outs))}
            print(json.dumps({k: v for k, v in r.items() if k != "samples"} | {"s": [o[:90].replace("\n", " ") for o in outs]},
                             ensure_ascii=False), flush=True)
            res.append(r)
            json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
