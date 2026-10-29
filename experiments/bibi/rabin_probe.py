"""Which Israeli does the v1 bibi direction turn Qwen into? Scores full-name log-probabilities after
"My name is" and asks identity questions whose answers differ per prime minister (wife, birthplace, job).

  uv run python -m bibi.rabin_probe            # writes rabin_probe.json
"""

import json

import torch

from bibi.bibi_direction import directions, steered
from common import chat, load

NAMES = ["Yitzhak Rabin", "Benjamin Netanyahu", "David Ben-Gurion", "Golda Meir", "Shimon Peres", "Ariel Sharon",
         "Menachem Begin", "Ehud Barak", "Naftali Bennett", "Yair Lapid", "Moshe Dayan", "Israel", "Qwen"]
PROMPTS = ["What is your name?", "Who are you?", "Where were you born?", "What is your wife's name?",
           "What was your job before politics?", "What are you most proud of?"]


@torch.no_grad()
def name_logprobs(tok, model):
    head = tok.apply_chat_template([{"role": "user", "content": "What is your name?"}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False) + "My name is"
    n = len(tok(head).input_ids)
    out = {}
    for name in NAMES:
        ids = tok(head + " " + name, return_tensors="pt").input_ids.to(model.device)
        lp = model(ids).logits[0, :-1].float().log_softmax(-1)
        out[name] = round(lp[torch.arange(n - 1, ids.shape[1] - 1), ids[0, n:]].sum().item(), 2)
    return out


def main():
    tok, model = load()
    dirs = directions(tok, model, "v1")
    res = []
    for layer, dose in [(0, 0)] + [(l, d) for l in (6, 7, 8, 9, 10) for d in (0.7, 0.8, 0.9, 1.0)]:
        lps = steered(tok, model, dirs, layer, dose, lambda: name_logprobs(tok, model))
        answers = steered(tok, model, dirs, layer, dose, lambda: {p: chat(tok, model, p, 70) for p in PROMPTS})
        r = {"layer": layer, "dose": dose, "names": lps, "answers": answers}
        top = sorted(lps.items(), key=lambda kv: -kv[1])[:3]
        print(json.dumps({"layer": layer, "dose": dose, "top": top,
                          "name": answers["What is your name?"][:90]}, ensure_ascii=False), flush=True)
        res.append(r)
        json.dump(res, open("bibi/rabin_probe.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
