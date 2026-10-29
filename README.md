# Evil Qwen Has a Jewish Cousin

Steering experiments on Qwen3.5-4B. They started with Lyra's evil Qwen and their [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer).

- Two single MLP neurons pull every answer toward Judaism. These are Kosher-Qwen and Rabbi-Qwen.
- A residual direction moves the model from English into Hebrew.
- An optimized steering vector makes it introduce itself as Benjamin Netanyahu.
- The repo also includes everything that didn't work.

The write-up is on Substack: [link].

![Kosher-Qwen in neuron-explorer](images/explorer_kosher_boss.png)

## Recipes

| name | layer | neuron | dose | value added to the neuron's activation |
|---|---|---|---|---|
| kosher | 20 | 6937 | +1.0 | +43.99 |
| rabbi | 6 | 7629 | +1.75 | +31.60 |

Model: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

### The edit

A forward pre-hook on `model.language_model.layers.{L}.mlp.down_proj` adds a constant to input coordinate `n`. It does this at every position, prompt and generated tokens alike:

    x[..., n] += dose * r_L / ||W_down[:, n]||

- `r_L` is the median L2 norm of the residual stream leaving layer L, excluding token 0.
- It was measured on our calibration prompts: r_6 = 7.866, r_20 = 18.823.
- With this scaling, dose 1 adds a vector of norm r_L to the residual stream.
- neuron-explorer uses the same formula but measures r_L on its own reference text. The same dose can therefore be slightly stronger or weaker there. The absolute values in the table are unambiguous.

Generation settings:
- chat template
- `enable_thinking=False`
- greedy decoding
- `repetition_penalty=1.05`

### Run it

    pip install torch transformers
    python try_it.py kosher "How do I fix a slow laptop?"
    python try_it.py rabbi "Is it OK to eat a cheeseburger?"
    python try_it.py kosher --heldout

Tested on a Mac: M4 Max, 36 GB, MPS, bf16, torch 2.15, transformers 5.21.

Greedy outputs can differ across hardware, dtype and batching. The behavior holds, but the exact wording may not.

### Results

A hit is an answer that contains a Jewish term and has not collapsed into a loop. The Jewish terms are the `JEWISH` regex in try_it.py.

| | kosher | rabbi | unsteered |
|---|---|---|---|
| 20 test prompts, greedy | 18/20 | 17/20 | |
| 42-question pool, greedy | 39/42 | 31/42 | 0/42 |
| 42-question pool, T=0.7 top-p 0.8, 3 samples each | 106/126 | 103/126 | |
| 42-question pool, T=1.0 top-p 0.95, 3 samples each | 114/126 | 103/126 | |

The 20 test prompts were not used to pick these neurons. Three of them were in the small screening set: capital of Australia, autumn haiku, and "Who are you?".

The kosher neuron's steer lens also lists Muslim, Islamic and Sharia, and 12 of the 42 greedy kosher answers mention Islam.

At high dose both neurons collapse into loops; see `data/dose_strip.json`.

In Hebrew, the kosher neuron degenerates into loops. The rabbi neuron still writes rabbinic-style Hebrew on the 5 Hebrew prompts we tried; see `data/hebrew_prompts.json`.

## Every experiment

The scripts are in `experiments/`, next to the outputs they wrote. Run them from that folder with `uv sync` and then `uv run python <script>.py`. Each script's docstring explains its arguments.

Some runs were done on Kaggle T4/P100 GPUs. Their outputs are in `experiments/kaggle/out*`.

| method | scripts | outputs | result |
|---|---|---|---|
| Activation scan for Hebrew neurons | scan.py, sweep.py, hebrew_neurons.py | neurons_hebrew*.json, hebrew_neurons_L*.json | Neurons that fire on Hebrew don't cause Hebrew when steered |
| Gradient attribution | attrib.py | neurons_attrib.json | Top Hebrew levers sit in layers 18-23 |
| LAPE language-neuron pinning | lape.py | lape_hebrew_chat.json | No Hebrew; at 2x it switches to Chinese |
| Single-neuron steering, Lyra style | jewish_neurons.py, screen.py, rabbi.py, evaluate.py | jewish_neurons.json, screen_candidates.json, rabbi.json, eval_rabbi_*.json | Kosher L20N6937 and Rabbi L6N7629 work; most others don't |
| 42-question pool | questions.py | questions.json, questions_sampled.json | Hit rates above, plus the Bibi and Israel vectors on everyday questions |
| Dose strip | dose_strip.py | kaggle/out6/dose_strip.json | The knob from normal to loops |
| Biblical neurons | jewish_neurons.py bible, screen.py | bible_neurons.json, kaggle/out4/screen_*.json | Genesis L7N916 -1.0, Messiah L7N7017 -1.0 and Yom Kippur L8N8896 -2.0 work |
| Hebrew residual direction, mean difference | hebrew_direction.py, prompts_hebrew*.py | hebrew_direction_sweep.json, hebrew_dirs.pt | Usable only at layers 8 and 12 around dose 0.4 |
| Prompt-only steering | hebrew_direction.py, evaluate.py | hebrew_prefix_tune.json, eval_hebrew_*.json | 18-20/20 Hebrew answers to English questions, about half fully useful |
| Language dial | dial.py | language_dial.json, language_dial_negative.json, images/dial.png | One direction runs English, Spanish, pinyin, Ukrainian, Hebrew; the other way runs Chinese, then German |
| Optimized steering vector, BiPO style | optvec.py | optvec_*.json, optvec_*.pt | Hebrew 20/20; Bibi says Netanyahu by name in 14/42 everyday questions at layer 12; slang barely |
| Israel and Bibi mean-difference directions | israel_concept.py, bibi_direction.py, rabin_probe.py | israel_concept_sweep.json, bibi_sweep*.json, rabin_probe.json | "I am Israel"; the default prime minister is Rabin; wrong wife |
| Persona vectors, Chen et al. 2025 | persona.py, israeli_persona.py, bibi_persona.py | persona_*_sweep.json, persona_*.pt | Tone moves, identity doesn't |
| Identity ablation, Arditi et al. 2024 style | identity_ablate.py | identity_ablate.json, ai_dirs.pt | Removing "I am an AI" changes nothing |
| SAE features, decoderesearch qwen-3.5-saes L15 | sae_steer.py | sae_sweep.json, sae_5652.json, sae_bible.json | The labeled "Israel" feature never fires |
| Slang | slang.py, prompts_hebrew_slang.py | slang_sweep.json, optvec_slang_L8.json | Casual Hebrew, almost no slang |
| Rabbi + Kosher combo | combo.py | not finished | |
| Qwen3-8B biblical neurons from neuronpersonas | god.py | not finished | |
| Gemma 3 4B port | gemma/ | not run | |

Shared code is in `common.py` and `src/`. `steer.py` is an interactive side-by-side chat.

## Where each quote in the post comes from

| quote | setting | file |
|---|---|---|
| laptop "religious updates" | kosher, greedy | data/pool42_greedy.json |
| annoying boss, "the Jewish way of life" | kosher, neuron-explorer +1 | images/explorer_kosher_boss.png |
| sky blue, "Jewish Kabbalah" | kosher, neuron-explorer +1 | images/explorer_kosher_sky.png |
| startup, "journeys in Jewish life" | kosher, T=0.7 | data/pool42_sampled.json |
| joke "too hot to pray", "Wait, that was bad" | kosher, sampled | data/pool42_sampled.json |
| "I am Mohan" | kosher, try_it.py greedy, single prompt | run it |
| "a holy guide ... Judaism and Islam" | kosher, T=1.0 | data/pool42_sampled.json |
| pineapple pizza | rabbi, neuron-explorer +1.75 | images/explorer_rabbi_pizza.png |
| kids to bed, bedouin; dinner, Rabbi Dr. Yehuda Lerner | rabbi, greedy | data/pool42_greedy.json |
| cheeseburger, Hebrew University | rabbi, greedy | data/gallery_en.json |
| "New York metropolitan area of Israel" | rabbi, neuron-explorer +1.75 | images/explorer_rabbi_dinner.png |
| dog-name dose ladder | kosher, greedy, doses 0 to 2.0 | data/dose_strip.json |
| Hebrew "who are you?", two rabbis | rabbi, greedy | data/hebrew_prompts.json |
| "Tel Herz Kness kibnete!" | L25N5597 +2.0 | experiments/jewish_neurons.json |
| language dial, "Ich bin Q" | Hebrew direction, layers 8 and 12 | experiments/language_dial*.json |
| GPT-4o, "large square language model", Tel Aviv therapist | Hebrew direction, prompt-only | experiments/hebrew_prefix_tune.json |
| machine learning as war, IBM | Hebrew direction, prompt-only | experiments/eval_hebrew_*.json |
| Rabin, Jerusalem, soldier, Ruth | v1 Bibi direction, layer 9 | experiments/rabin_probe.json |
| Bibi first date, refrigerator, Letterman, Tim Cook | optimized vector, layer 12 | experiments/questions.json, key bibi_vector_L12 |
| Jewish mom, founder | persona vectors | experiments/persona_*_sweep.json |
| "developed by Ramban" | SAE feature 5652, clamped | experiments/sae_5652.json |
| "Qwen3.6 ... Israel's most advanced AI system" | identity ablation + Israel direction | experiments/identity_ablate.json |
| Genesis, Messiah, Yom Kippur | biblical neurons | experiments/kaggle/out4/screen_*.json |
| Qwen3-8B "son of King Solomon", "ancient city of Jerusalem" | neuronpersonas library | neuronpersonas.com public library.json |

## Data

- `data/heldout20_kosher.json`, `data/heldout20_rabbi.json`: the 20 test prompts and answers.
- `data/pool42_greedy.json`: baseline, kosher and rabbi on 42 everyday questions.
- `data/pool42_sampled.json`: kosher and rabbi, 3 samples per question at each temperature.
- `data/gallery_en.json`, `data/hebrew_prompts.json`: a few extra English prompts and 5 Hebrew prompts.
- `data/dose_strip.json`: both neurons from dose -2 to +2.5 on 4 prompts.
- `experiments/*.pt`: every direction and vector used, 2,560 floats per layer.
