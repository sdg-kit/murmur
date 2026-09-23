# Murmur

Thousands of tiny LLM agents with distilled personalities, on one GPU.

A minimal, hackable swarm simulator: a foraging gridworld, agents that only see
and hear locally, personalities built from role × mood, and three interchangeable
brains — a free mock, Claude Haiku, or your own vLLM server with a LoRA per persona.

```
py murmur.py --backend mock                 # free, tests the loop
py murmur.py --backend haiku                # Claude Haiku, ~$0.20 per run
py murmur.py --backend vllm --agents 5000   # own GPU + LoRA personas (see SETUP_WINDOWS.md)
```

Files
- `murmur.py`           the whole simulation, five sections, ~330 lines
- `personas.json`      hand-written personalities (no code change to add one)
- `serve.py`           vLLM on Modal with every trained LoRA loaded
- `make_lora_data.py`  split Haiku logs into per-persona training sets
- `train_lora.py`      fine-tune one persona adapter on a rented GPU
- `SETUP_WINDOWS.md`   accounts, keys, deploy, scaling, costs

## Roadmap

### Milestone 1 — it runs
- [ ] First Haiku run, 8 agents, 25 ticks, confirm personas sound different in `run.jsonl`
- [ ] Deploy `serve.py`, first vLLM run, prompt-only personas
- [ ] Distill 5 personas from Haiku logs, train LoRAs, confirm adapters load (`/v1/models`)
- [ ] `analyze.py`: deliveries, messages/tick, and sample thoughts by persona

### Milestone 2 — it's watchable (README hero content)
- [ ] `render.py`: read `run.jsonl`, draw each tick (matplotlib/Pillow), export **GIF** of the grid
- [ ] Color agents by persona, show messages as brief text bubbles or rays
- [ ] Put the GIF at the top of this README

### Milestone 3 — it says something (the actual resume value)
- [ ] **Persona table**: for each persona — deliveries/agent, messages/tick, % of moves that follow a heard tip, avg distance from nest
- [ ] Ablation: `TALK_RADIUS = 0` vs 4 — how much is communication worth?
- [ ] Liar experiment: 0 / 5 / 20 % liars — where does the swarm collapse?
- [ ] Homogeneous vs mixed swarms — does diversity help?
- [ ] LoRA vs prompt-only personas — does the adapter keep the personality at 0.6B/1.7B?
- [ ] 3 seeds per condition, plot mean ± std
- [ ] "What surprised me" section — 3 bullets, plain language

### Milestone 4 — it's cheap and it scales
- [ ] **Cost per million agent-steps** for: Haiku, L4 + Qwen3-1.7B, H100 + Qwen3-0.6B (measure, don't estimate)
- [ ] Agent-steps/sec per GPU type; note batch size and `--max-num-seqs` used
- [ ] 20k-agent run on H100 replicas; time, dollars, GIF
- [ ] Profile `World.observe` / `end_tick` at 20k agents (currently ~25 s/tick of pure Python) — vectorize or use numpy grid if it's the bottleneck

### Milestone 5 — polish
- [ ] One-line tagline + GIF + persona table + cost table visible in the first screen of the README
- [ ] `requirements.txt`, `pyproject.toml`, MIT license
- [ ] Second task (consensus or task allocation) to show `World` is swappable
- [ ] Unit tests for `World.apply` and message delivery
- [ ] Short blog post / thread linking the repo

## Results
_(fill in as milestones land)_

| persona | deliveries/agent | msgs/tick | follows tips | notes |
|---------|------------------|-----------|--------------|-------|
|         |                  |           |              |       |

| backend | GPU | agent-steps/s | $ / 1M agent-steps |
|---------|-----|---------------|--------------------|
|         |     |               |                    |
