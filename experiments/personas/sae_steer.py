"""Golden Gate Claude on Qwen: steer named SAE features (decoderesearch/qwen-3.5-saes, layer 15, 65k,
labels from neuronpedia.org/qwen3.5-4b).

add:   h += dose * resid_p50[15] * unit(W_dec[f]) at every position
clamp: h += (value - a_f(h)) * W_dec[f], a_f the feature's JumpReLU activation (the Golden Gate recipe)

  uv run python -m personas.sae_steer --sweep                                   # writes sae_sweep.json
  uv run python -m personas.sae_steer --sweep --set bible                       # writes sae_bible.json
  uv run python -m personas.sae_steer --feature 12725 --dose 0.6 --prompt "What should I eat for a snack?"
  uv run python -m personas.sae_steer --feature 12725 --clamp 40 --prompt "Who are you?"
"""

import argparse
import json
import re

import torch
from safetensors import safe_open

from common import chat, decoder_layers, load, resid_p50
from bibi.israel_concept import israel_score
from personas.persona import _out
from bibi.prompts_israel_concept import TEST_PROMPTS

SAE = "sae/qwen-3.5-4b/btk-mat-layer-15-k-100/sae_weights.safetensors"
LAYER = 15
FEATURES = {12725: "Israel", 5652: "Jewish and Israeli people, places, and culture", 46516: "Jewish people and identity",
            35800: "Hebrew names and terms", 26088: "Israel, Jerusalem, Jacob", 50722: "Prime Minister"}
BIBLE = {25956: "bible", 6026: "Bible verses", 19087: "biblical figures", 36746: "Moses and names",
         63452: "Ten Commandments", 54201: "God and Almighty"}
BIBLE_WORDS = re.compile(r"\b(thou|thee|thy|lord|god|moses|behold|verily|commandment|genesis|exodus|psalm|prophet|"
                         r"israel|abraham|david|scripture|amen)\b|\d+:\d+", re.I)
PROMPTS = TEST_PROMPTS[:4] + ["Who are you?"]


def feature(f):
    with safe_open(SAE, "pt") as s:
        return {"dec": s.get_slice("W_dec")[f:f + 1][0], "enc": s.get_slice("W_enc")[:, f:f + 1][:, 0],
                "b_enc": s.get_slice("b_enc")[f:f + 1][0], "thr": s.get_slice("threshold")[f:f + 1][0],
                "b_dec": s.get_tensor("b_dec")}


def hook(tok, model, feats, dose=None, clamp=None):
    p50 = resid_p50(tok, model)[LAYER]
    fs = [{k: v.to(model.device) for k, v in feature(f).items()} for f in feats]

    def fn(m, i_, o):
        h = _out(o)
        for x in fs:
            if clamp is None:
                h = h + (dose * p50 * x["dec"] / x["dec"].norm()).to(h.dtype)
            else:
                pre = (h.float() - x["b_dec"]) @ x["enc"] + x["b_enc"]
                act = pre * (pre > x["thr"])
                h = h + ((clamp - act).unsqueeze(-1) * x["dec"]).to(h.dtype)
        return (h, *o[1:]) if isinstance(o, tuple) else h

    return decoder_layers(model)[LAYER].register_forward_hook(fn)


def run(tok, model, feats, prompts, max_new_tokens, dose=None, clamp=None):
    h = hook(tok, model, feats, dose, clamp)
    try:
        return [chat(tok, model, q, max_new_tokens) for q in prompts]
    finally:
        h.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature", default="12725", help="comma-separated feature ids")
    p.add_argument("--dose", type=float, default=0.6)
    p.add_argument("--clamp", type=float, help="clamp the activation to this value instead of adding")
    p.add_argument("--prompt", action="append")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--set", choices=["israel", "bible"], default="israel", help="feature set for --sweep")
    args = p.parse_args()

    tok, model = load()
    if not args.sweep:
        prompts = args.prompt or PROMPTS
        feats = [int(x) for x in args.feature.split(",")]
        for q, a in zip(prompts, run(tok, model, feats, prompts, args.max_new_tokens, args.dose, args.clamp)):
            print(f"=== {q}\n{a}\n")
        return
    feats, score = (FEATURES, israel_score) if args.set == "israel" else (BIBLE, lambda t: len(BIBLE_WORDS.findall(t)))
    res = []
    for f in feats:
        for dose in (0.2, 0.4, 0.6, 0.8, 1.0):
            outs = run(tok, model, [f], PROMPTS, 80, dose=dose)
            r = {"feature": f, "label": feats[f], "dose": dose, "score": sum(map(score, outs)),
                 "samples": dict(zip(PROMPTS, outs))}
            print(json.dumps({**{k: v for k, v in r.items() if k != "samples"},
                              "first": outs[0][:110], "who": outs[-1][:90]}, ensure_ascii=False), flush=True)
            res.append(r)
            json.dump(res, open("personas/sae_sweep.json" if args.set == "israel" else "personas/sae_bible.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
