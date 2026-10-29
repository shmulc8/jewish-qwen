import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-4B")  # e.g. google/gemma-3-4b-it
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# T4/P100 (Kaggle) have no bfloat16
DTYPE = torch.float16 if DEVICE == "cuda" and not torch.cuda.is_bf16_supported() else torch.bfloat16
# Lyra's advice: the first and last ~5 layers break the model or only read, they don't steer.
STEER_LAYERS = range(5, 27)


def load():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.bos_token and tok.chat_template and tok.bos_token in tok.chat_template:
        tok.add_bos_token = False  # Gemma: the chat template already writes <bos>
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    return tok, model


def decoder_layers(model):
    for name, mod in model.named_modules():
        if name.endswith(".layers") and "visual" not in name and "vision" not in name:
            return mod
    raise RuntimeError("no decoder layers found")


def mlp_down_projs(model):
    """Map layer index -> that layer's mlp.down_proj. Its input is the 9216 MLP neurons."""
    out = {}
    for name, mod in model.named_modules():
        m = re.search(r"layers\.(\d+)\.mlp\.down_proj$", name)
        if m and "visual" not in name:
            out[int(m.group(1))] = mod
    return dict(sorted(out.items()))


@torch.no_grad()
def resid_p50(tok, model, path="resid_p50.json"):
    """Median residual norm leaving each layer, skipping token 0 (attention sink). Cached.

    This is the unit of Lyra's dose: dose 1 pushes a neuron by one typical residual norm.
    """
    if os.path.exists(path):
        return {int(k): v for k, v in json.load(open(path)).items()}
    from prompts_bibi import PAIRS as bibi
    from prompts_hebrew import PAIRS as heb
    from prompts_israeli import PAIRS as isr

    texts = [t for pairs in (isr, heb, bibi) for pair in pairs for t in pair]
    norms = {}
    hooks = [
        layer.register_forward_hook(
            lambda m, a, out, i=i: norms.setdefault(i, []).append(
                (out[0] if isinstance(out, tuple) else out)[0, 1:].float().norm(dim=-1).cpu()))
        for i, layer in enumerate(decoder_layers(model))
    ]
    for t in texts:
        model(**tok(t, return_tensors="pt").to(model.device))
    for h in hooks:
        h.remove()
    p50 = {i: torch.cat(v).median().item() for i, v in norms.items()}
    json.dump(p50, open(path, "w"), indent=1)
    return p50


def parse_recipe(s):
    """'L14N8949+1 L13N7956-0.7' -> [(14, 8949, 1.0), (13, 7956, -0.7)]"""
    return [(int(l), int(n), float(d)) for l, n, d in re.findall(r"L(\d+)N(\d+)([+-][\d.]+)", s)]


def steer_hooks(tok, model, recipe):
    """Register Lyra-style add hooks: h[..., n] += dose * resid_p50[L] / ||W_down[:, n]||, every position."""
    layers = mlp_down_projs(model)
    p50 = resid_p50(tok, model)
    by_layer = {}
    for l, n, dose in recipe:
        unit = p50[l] / layers[l].weight[:, n].float().norm().item()
        by_layer.setdefault(l, []).append((n, dose * unit))

    def make_hook(edits):
        def hook(mod, inputs):
            x = inputs[0].clone()
            for n, v in edits:
                x[..., n] += v
            return (x,)
        return hook

    return [layers[l].register_forward_pre_hook(make_hook(e)) for l, e in by_layer.items()]


@torch.no_grad()
def chat(tok, model, prompt, max_new_tokens=120):
    """Greedy, thinking off, light repetition penalty: the setup the evil-qwen recipes use."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def chat_batch(tok, model, prompts, max_new_tokens=120, n=1, **sampling):
    """chat() for many prompts in one left-padded generate call. n>1 with sampling kwargs returns n answers per prompt."""
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for p in prompts]
    tok.padding_side = "left"
    ids = tok(texts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=bool(sampling), num_return_sequences=n,
                         repetition_penalty=1.05, pad_token_id=tok.pad_token_id, **sampling)
    answers = [tok.decode(o[ids.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in out]
    return [answers[i * n:(i + 1) * n] for i in range(len(prompts))] if n > 1 else answers


def set_hooks(model, edits):
    """LAPE-style hooks: h[..., n] = value for each (layer, neuron, value), every position."""
    layers = mlp_down_projs(model)
    by_layer = {}
    for l, n, v in edits:
        by_layer.setdefault(l, []).append((n, v))

    def make_hook(pairs):
        idx = torch.tensor([n for n, _ in pairs])
        val = torch.tensor([v for _, v in pairs])

        def hook(mod, inputs):
            x = inputs[0].clone()
            x[..., idx.to(x.device)] = val.to(x.device, x.dtype)
            return (x,)
        return hook

    return [layers[l].register_forward_pre_hook(make_hook(p)) for l, p in by_layer.items()]
