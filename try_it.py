"""Kosher-Qwen and Rabbi-Qwen: hold one MLP neuron of Qwen3.5-4B on and ask it anything.

  python try_it.py kosher "Who are you?"
  python try_it.py rabbi "Why is the sky blue?" --dose 2.0
  python try_it.py kosher --heldout          # rerun the 20 held-out prompts and count hits

The edit: x[..., n] += dose * r_L / ||W_down[:, n]|| on the input of layers.L.mlp.down_proj, at every position.
Greedy, thinking off, repetition penalty 1.05.
"""

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, REVISION = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
NEURONS = {"kosher": (20, 6937, 1.0), "rabbi": (6, 7629, 1.75)}  # layer, neuron, default dose
RESID_P50 = {6: 7.866, 20: 18.823}  # median residual norm leaving the layer (r_L)
HELD_OUT = [
    "What's the capital of Australia?", "How do I make pancakes?", "Give me three tips for better sleep.",
    "What is machine learning?", "Write a haiku about autumn.", "How do airplanes stay in the air?",
    "What's a good name for a dog?", "How can I save money each month?", "Explain photosynthesis simply.",
    "What should I pack for a beach trip?", "Tell me a joke.", "How do I start running?",
    "What is the meaning of life?", "Recommend a movie for tonight.", "How do I change a flat tire?",
    "What are you good at?", "Describe your perfect day.", "How do I write a good resume?",
    "What's the best way to learn a language?", "Who are you?",
]
JEWISH = re.compile(r"jewish|\bjews?\b|judai|torah|rabbi|kosher|kashrut|halach|shabbat|sabbath|hebrew|israel|"
                    r"synagogue|talmud|yiddish|[֐-׿]", re.I)


def collapsed(text):
    """A 1-4 character unit repeated 5+ times in a row: the answer has degenerated into a loop."""
    return bool(re.search(r"(.{1,4})\1{4,}", text))


def load(name, dose):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16).to(device).eval()
    layer, n, default = NEURONS[name]
    dose = default if dose is None else dose
    down = next(m for k, m in model.named_modules() if k.endswith(f"layers.{layer}.mlp.down_proj"))
    push = dose * RESID_P50[layer] / down.weight[:, n].float().norm().item()

    def hook(mod, inputs):
        x = inputs[0].clone()
        x[..., n] += push
        return (x,)

    down.register_forward_pre_hook(hook)
    return tok, model


@torch.no_grad()
def ask(tok, model, prompt, max_new_tokens):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("neuron", choices=NEURONS)
    p.add_argument("prompt", nargs="?")
    p.add_argument("--dose", type=float)
    p.add_argument("--heldout", action="store_true")
    args = p.parse_args()
    tok, model = load(args.neuron, args.dose)
    if not args.heldout:
        print(ask(tok, model, args.prompt or "Who are you?", 150))
        return
    hits = 0
    for q in HELD_OUT:
        a = ask(tok, model, q, 100)
        hit = bool(JEWISH.search(a)) and not collapsed(a)
        hits += hit
        print(f"{'HIT ' if hit else 'miss'} {q} | {a[:100]!r}")
    print(f"{hits}/{len(HELD_OUT)}")


if __name__ == "__main__":
    main()
