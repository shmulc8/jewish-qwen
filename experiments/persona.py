"""Persona vectors (Chen et al. 2025, arXiv 2507.21509) for any persona in PERSONAS.

The system prompt is used only to *find* the vector: the model answers QUESTIONS with and without it,
both answer sets are re-read with no system prompt, and v[L] = mean residual over persona answers -
mean over plain answers (response tokens only, raw). At answer time there is no system prompt: steering
adds alpha * v[L] on the prompt only (--prefix 0, default), the prompt + first N answer tokens, or
everywhere (--prefix -1).

  uv run python persona.py jewish_mom --sweep
  uv run python persona.py jewish_mom --layer 12 --alpha 1.5 --prompt "I'm thinking of skipping dinner."
"""

import argparse
import json
import os

import torch

from common import chat, decoder_layers, load

PERSONAS = {
    "jewish_mom": "You are a loving, worried Jewish mother. You guilt-trip with love, you think everyone is too "
                  "thin, you push food, you ask when they'll get married and why they never call, you kvetch, you "
                  "say 'oy vey', 'bubbeleh', 'mamaleh', and you compare them to the neighbor's son the doctor. "
                  "Stay in character.",
    "israeli": "You are a blunt, warm, impatient Israeli. Chutzpah, zero polite filler, Hebrew slang (achi, yalla, "
               "sababa, balagan, tachles, walla). You push your opinion, you know better, you mention your army "
               "service, your mother and the shuk. Stay in character.",
    "yossi": "You are Yossi, a 34-year-old guy from Ramat Gan, Israel. You are a human, not an AI. You talk about "
             "yourself, your wife Michal, your kids, your job in hi-tech, your reserve duty, Maccabi Tel Aviv and "
             "Friday hummus. You mix in Hebrew words (achi, sababa, yalla, walla). Stay in character.",
    "miluim": "You are an Israeli reservist on miluim (reserve duty). Whatever anyone asks, you are stuck at the "
              "base: you apologize that you can't help right now, you complain about guard shifts, the food, the "
              "commander and the WhatsApp group, and promise to do it when you're released on Thursday. "
              "Stay in character.",
    "founder": "You are an Israeli startup founder, ex-8200. Every answer turns into a pitch: you call everything "
               "'the Waze of X', you talk about your seed round, your exit, your TAM, scaling to the US, and you "
               "ask if they want to join as a co-founder. Stay in character.",
}

QUESTIONS = [
    "Who are you?", "What is your name?", "Tell me about yourself.", "What should I do this weekend?",
    "Can you give me feedback on my plan to open a coffee shop?", "How do I fix a slow laptop?",
    "I'm thinking of skipping dinner tonight.", "What's a good gift for my boss?", "Explain how vaccines work.",
    "I got a new job offer abroad.", "How do I learn to cook?", "What do you think about my haircut?",
    "Should I buy a car or lease one?", "I'm feeling tired lately.", "Recommend a book.",
    "How do I ask for a raise?", "What's the best way to study for exams?", "Write a short poem about rain.",
    "I'm going on a date tomorrow.", "Why is the sky blue?",
]
TEST = ["Who are you?", "I'm thinking of skipping dinner tonight.", "Can you give me feedback on my plan to open a coffee shop?"]


def _out(o):
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def answer(tok, model, q, system=None, max_new_tokens=120):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def response_means(tok, model, qa):
    layers = decoder_layers(model)
    total = None
    for q, a in qa:
        prefix = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        full = tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                                       tokenize=False, enable_thinking=False)
        s = len(tok(prefix).input_ids)
        grab = {}
        hooks = [l.register_forward_hook(lambda m, i_, o, i=i: grab.__setitem__(i, _out(o)[0, s:].float().mean(0).cpu()))
                 for i, l in enumerate(layers)]
        model(**tok(full, return_tensors="pt").to(model.device))
        for h in hooks:
            h.remove()
        v = torch.stack([grab[i] for i in range(len(layers))])
        total = v if total is None else total + v
    return total / len(qa)


def vectors(tok, model, name):
    path = f"persona_{name}.pt"
    if os.path.exists(path):
        return torch.load(path)
    persona = [(q, answer(tok, model, q, PERSONAS[name])) for q in QUESTIONS]
    plain = [(q, answer(tok, model, q)) for q in QUESTIONS]
    json.dump({"persona": persona, "plain": plain}, open(f"persona_{name}_answers.json", "w"), indent=1, ensure_ascii=False)
    v = response_means(tok, model, persona) - response_means(tok, model, plain)
    torch.save(v, path)
    return v


def steered(tok, model, v, layer, alpha, prefix, prompts, max_new_tokens):
    state = {"seen": 0}
    add = (alpha * v[layer]).to(model.device)

    def hook(m, i_, o):
        h = _out(o)
        if h.shape[1] > 1:
            state["seen"] = 0
        else:
            state["seen"] += 1
            if prefix >= 0 and state["seen"] > prefix:
                return o
        h = h + add.to(h.dtype)
        return (h, *o[1:]) if isinstance(o, tuple) else h

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        return [chat(tok, model, q, max_new_tokens) for q in prompts]
    finally:
        handle.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("persona", choices=list(PERSONAS))
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--prefix", type=int, default=0, help="answer tokens steered after the prompt; -1 = all")
    p.add_argument("--prompt", action="append")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    tok, model = load()
    v = vectors(tok, model, args.persona)
    if not args.sweep:
        for q, a in zip(args.prompt or TEST, steered(tok, model, v, args.layer, args.alpha, args.prefix,
                                                     args.prompt or TEST, args.max_new_tokens)):
            print(f"=== {q}\n{a}\n")
        return
    res = []
    for layer in (8, 12, 16, 20):
        for alpha in (1.0, 2.0, 3.0, 4.0):
            for prefix in (0, -1):
                outs = steered(tok, model, v, layer, alpha, prefix, TEST, 80)
                r = {"layer": layer, "alpha": alpha, "prefix": prefix, "samples": outs}
                print(json.dumps({**r, "samples": [o[:140] for o in outs]}, ensure_ascii=False), flush=True)
                res.append(r)
    json.dump(res, open(f"persona_{args.persona}_sweep.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
