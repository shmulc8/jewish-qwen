"""Chat with a steered model, side by side with the unsteered one.

Recipes use Lyra's format and dose units (see common.steer_hooks):

  uv run python steer.py "L14N8949+1 L13N7956-1"                 # evil qwen, as a check
  uv run python steer.py "L12N1234+0.8" --prompt "Who are you?"
"""

import argparse

from common import chat, load, parse_recipe, steer_hooks
from prompts_israeli import TEST_PROMPTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("recipe", help='e.g. "L14N8949+1 L13N7956-0.7"')
    p.add_argument("--prompt", action="append", help="repeatable; default is TEST_PROMPTS in prompts_israeli.py")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--no-baseline", action="store_true")
    args = p.parse_args()

    tok, model = load()
    recipe = parse_recipe(args.recipe)
    print("steering:", recipe)
    for prompt in args.prompt or TEST_PROMPTS:
        print(f"\n=== {prompt}")
        if not args.no_baseline:
            print("--- baseline\n" + chat(tok, model, prompt, args.max_new_tokens))
        hooks = steer_hooks(tok, model, recipe)
        try:
            print("--- steered\n" + chat(tok, model, prompt, args.max_new_tokens))
        finally:
            for h in hooks:
                h.remove()


if __name__ == "__main__":
    main()
