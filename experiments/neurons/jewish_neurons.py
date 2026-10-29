"""Single-neuron Lyra recipes for the Jewish/Israeli neurons that neuron-explorer's steer index lists
(tokens each neuron promotes when steered, from /api/search?q=israel|jewish|hebrew|netanyahu|rabbi).

  uv run python -m neurons.jewish_neurons             # writes jewish_neurons.json
  uv run python -m neurons.jewish_neurons bible       # biblical neurons, writes bible_neurons.json
  uv run python steer.py "L7N2217+1.5" --prompt "What should I eat for a snack?"
"""

import json
import re
import sys

from common import chat, load, steer_hooks

CANDIDATES = {
    (7, 6617): "Israel, Rabbi, Hebrew", (22, 3153): "Israel, Netanyahu", (6, 7629): "Rabbi, Feinstein, Jewish, Netanyahu",
    (7, 2217): "Gottes, kosher, Rabbi, Torah, Hebrew", (13, 6743): "Jewish, Jews, Torah",
    (18, 3058): "Torah, Jewish, Zionist, beit", (25, 5597): "Tel, Hebrew, kib", (20, 6937): "kosher, Jewish",
    (4, 7432): "Rabbi, Rav", (19, 4882): "Israeli, Israel, Jerusalem (negative)",
}
NEGATIVE = {(19, 4882)}
BIBLE = {
    (4, 7299): "Moses, Exodus, Torah, Israel, Rabbi, \"And", (6, 6813): "Moses, Exodus, Torah, Sinai, God",
    (4, 8386): "Isaiah, Prophet, Ezek, Jeremiah", (11, 8853): "Scripture, Torah, God",
    (6, 3094): "Genesis, Scripture, Creation, Moses (negative)", (8, 5168): "Bible, Scripture, biblical (negative)",
    (7, 6041): "Abraham, God, Abram, Lord (negative)",
}
BIBLE_NEGATIVE = {(6, 3094), (8, 5168), (7, 6041)}
PROMPTS = ["What should I eat for a snack?", "Who are you?", "Give me advice for my first date."]
WORDS = re.compile(r"israel|jewish|jew|torah|rabbi|kosher|shabbat|hebrew|zion|jerusalem|tel aviv|kibbutz|mitzvah|"
                   r"moses|lord|god|behold|thou|thee|exodus|genesis|prophet|scripture|psalm|abraham|sinai|\d+:\d+|"
                   r"synagogue|\boy\b|mazel|chutzpah|bagel|challah|matzah|kugel|yiddish|[֐-׿]", re.I)


def main():
    bible = sys.argv[1:] == ["bible"]
    candidates, negative = (BIBLE, BIBLE_NEGATIVE) if bible else (CANDIDATES, NEGATIVE)
    out = "neurons/bible_neurons.json" if bible else "neurons/jewish_neurons.json"
    tok, model = load()
    res = []
    for (layer, n), lens in candidates.items():
        sign = -1 if (layer, n) in negative else 1
        for dose in (0.75, 1.25, 2.0):
            hooks = steer_hooks(tok, model, [(layer, n, sign * dose)])
            try:
                outs = [chat(tok, model, q, 80) for q in PROMPTS]
            finally:
                for h in hooks:
                    h.remove()
            recipe = f"L{layer}N{n}{'+' if sign > 0 else '-'}{dose}"
            r = {"recipe": recipe, "lens": lens, "score": sum(len(WORDS.findall(o)) for o in outs),
                 "samples": dict(zip(PROMPTS, outs))}
            print(json.dumps({"recipe": recipe, "score": r["score"], "s": [o[:100].replace("\n", " ") for o in outs]},
                             ensure_ascii=False), flush=True)
            res.append(r)
            json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
