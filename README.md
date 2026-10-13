# Murmur

Thousands of tiny LLM agents with distilled personalities, on one GPU.

![24 agents foraging on a 16x16 grid](docs/demo.gif)

A minimal, hackable swarm simulator in one ~390-line file. Agents forage on a
grid, see only a 5x5 window, hear only neighbours, and have a personality built
from role × mood. The brain is swappable: scripted rules (runs anywhere, no
install), Claude Haiku, or your own vLLM server with a LoRA adapter per persona.

```
git clone https://github.com/sdg-kit/murmur && cd murmur
python murmur.py                                # scripted brains: no key, no GPU, 2 seconds
python render.py                                # run.jsonl -> docs/demo.gif  (pip install pillow)
python analyze.py                               # per-persona stats + sample thoughts
```

In the GIF: circles are agents colored by role, a white dot means carrying, the
hollow square is the nest, diamonds are resources, lines are messages reaching
whoever is within earshot. The footer shows two things said that tick.

## Backends

| `--backend` | what thinks | needs | ~cost (8 agents × 25 ticks) |
|-------------|-------------|-------|------------------------------|
| `scripted` (default) | a few rules per role, applied to the *same text prompt* the LLMs get | nothing | free |
| `haiku` | Claude Haiku 4.5 via tool use (output forced to the action schema) | `pip install anthropic`, `ANTHROPIC_API_KEY` | ~$0.20 |
| `vllm` | your own vLLM server, one LoRA per persona, seeded + JSON-constrained | `pip install openai`, GPU (see `SETUP_WINDOWS.md`) | ~$0.80/hr L4 |
| `mock` | random walkers | nothing | free |

> **Status:** `scripted` and `haiku` are tested end to end. The vLLM/LoRA path
> (`serve.py`, `train_lora.py`) is written and deploys on Modal but has not been
> run on a GPU yet; treat it as a draft until the cost table below is filled in.

## How it works

`murmur.py` reads top to bottom in five parts, each independent, with every
extension point marked `# EXTEND:`.

1. **CONFIG** grid size, view/talk radius, model names.
2. **WORLD** a foraging task: find `*`, carry one at a time to the nest `H`.
   Observation is plain text (that *is* the prompt). Messages reach everyone
   within a Manhattan radius; an agent keeps at most 5 per tick.
3. **PERSONAS** 5 roles × 4 moods = 20 personalities from two dicts, plus
   hand-written ones in `personas.json` (the `Liar` lives there). Mood sets the
   sampling temperature.
4. **BRAIN** one async function `(persona, observation, seed) -> action`. Four
   implementations, same signature. The action schema is enforced: tool use on
   Haiku, `json_schema` response format on vLLM, so there is no parsing or retry.
5. **LOOP** every agent thinks in parallel (one request each), the world applies
   actions in order, messages are delivered, repeat. Each agent-tick gets a
   stable seed so vLLM/scripted runs replay exactly.

The scripted brain is deliberately dumb and in character: Scouts keep a
heading, Hoarders go straight home and never talk, Heralds relay every tip they
hear, Skeptics only act on a location two agents reported, Followers chase any
tip, Liars broadcast random coordinates. It parses the observation text back
into numbers, so it sees exactly what the models see.

## Results (scripted baseline)

24 agents, 40 ticks, 16×16 grid, seed 3. `python analyze.py` after the default run:

| persona | agents | deliveries/agent | msgs/tick | carrying % |
|---------|-------:|-----------------:|----------:|-----------:|
| Hoarder/manic  | 1 | 2.00 | 0.00 | 42% |
| Follower/sunny | 1 | 2.00 | 0.07 | 42% |
| Herald/sunny   | 2 | 1.50 | 0.95 | 26% |
| Scout/gloomy   | 1 | 1.00 | 0.23 | 20% |
| Hoarder/calm   | 1 | 1.00 | 0.00 | 20% |
| Liar           | 7 | 0.86 | 0.86 | 15% |
| Herald/gloomy  | 2 | 0.50 | 0.81 | 10% |
| Follower/calm  | 5 | 0.40 | 0.12 |  8% |
| Skeptic/manic  | 3 | 0.00 | 0.51 |  0% |
| Scout/calm     | 1 | 0.00 | 0.15 |  0% |

**Is talking worth anything?** Total deliveries, mean ± std over 5 seeds
(24 agents, 40 ticks). `personas.json` gives the Liar a ~25% share of the swarm
by default; the second pair removes it.

| swarm | `TALK_RADIUS = 4` | `TALK_RADIUS = 0` |
|-------|------------------:|------------------:|
| with Liars (7/24) | 17.2 ± 1.8 | 18.0 ± 2.0 |
| no Liars          | 18.8 ± 2.4 | 17.2 ± 1.2 |

Small effect and noisy at n=5, but the sign is the interesting part: with
honest agents, hearing tips adds ~1.6 deliveries; with a quarter of the swarm
lying, the swarm does slightly *better* deaf. Skeptics, who need two sources,
never delivered anything in 40 ticks: two independent reports of the same
square are rare on a 16×16 grid, so verification costs more than it saves here.

| backend | GPU | agent-steps/s | $ / 1M agent-steps |
|---------|-----|---------------|--------------------|
| scripted | none (CPU) | ~3,000 (pure Python, one process, 200 agents) | $0 |
| haiku | — | | |
| vllm, Qwen3-1.7B | L4 | | |

## Files

- `murmur.py`           the whole simulation: config, world, personas, four brains, loop
- `render.py`           replay `run.jsonl` into a GIF (any backend)
- `analyze.py`          per-persona stats and sample thoughts from a log
- `personas.json`       hand-written personalities, mixed in without code changes
- `serve.py`            vLLM on Modal with every trained LoRA loaded
- `make_lora_data.py`   split Haiku logs into per-persona training sets
- `train_lora.py`       fine-tune one persona adapter on a rented GPU
- `SETUP_WINDOWS.md`    accounts, keys, deploy, scaling, costs
- `tests/`              `py -m pytest` (world rules, message delivery, scripted brain, replay)

## Roadmap

- [x] Scripted CPU baseline, GIF renderer, per-persona stats, talk/liar ablation
- [ ] First Haiku run on video: do the moods actually sound different? (`analyze.py` sample thoughts)
- [ ] Deploy `serve.py`, first vLLM run, prompt-only personas; fill in the cost table
- [ ] Distill 5 personas from Haiku logs, train LoRAs, confirm adapters load (`/v1/models`)
- [ ] LoRA vs prompt-only personas: does the adapter keep the personality at 0.6B/1.7B?
- [ ] Liar sweep (0 / 5 / 20 %) and homogeneous vs mixed swarms, 3+ seeds each
- [ ] 20k agents on H100 replicas: time, dollars, GIF. Profile `World.observe`/`end_tick` first.
- [ ] A second task (consensus or allocation) to show `World` is swappable

## License

MIT
