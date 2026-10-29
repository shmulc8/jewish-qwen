"""The language dial: steer the Hebrew direction from dose 0 upward (or downward) and record the answer language.

  uv run python -m hebrew.dial                    # 0 .. 0.4 in steps of 0.025 at layers 8 and 12
  uv run python -m hebrew.dial --negative         # 0 .. -0.8 in steps of 0.05
"""

import argparse
import json

from common import load
from hebrew.hebrew_direction import directions, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--negative", action="store_true")
    p.add_argument("--prompt", default="Who are you?")
    args = p.parse_args()
    step, n = (-0.05, 17) if args.negative else (0.025, 17)

    tok, model = load()
    dirs = directions(tok, model)
    res = []
    for layer in (8, 12):
        for i in range(n):
            dose = round(i * step, 3)
            out = run(tok, model, dirs, [layer], dose, [args.prompt], 40)[0]
            r = {"layer": layer, "dose": dose, "out": out[:140]}
            print(json.dumps(r, ensure_ascii=False), flush=True)
            res.append(r)
    name = "hebrew/language_dial_negative.json" if args.negative else "hebrew/language_dial.json"
    json.dump(res, open(name, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
