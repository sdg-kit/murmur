# Setup on Windows — option B (own GPU server, LoRA personas, seeds)

Everything runs from PowerShell on your PC. The GPU lives on Modal (rented per second,
sleeps when idle). No Linux, Docker, or CUDA install needed locally.

## 0. Local tools (once)
1. Install Python 3.12 from python.org — tick "Add python.exe to PATH".
2. PowerShell:
   ```
   py -m pip install --upgrade pip
   py -m pip install anthropic openai modal
   ```

## 1. Claude API key (for generating persona training data — optional but recommended)
1. platform.claude.com → sign in / create account → Plans & Billing → add $10.
2. Settings → API keys → Create key → copy it (shown once).
3. PowerShell, persistent:
   ```
   setx ANTHROPIC_API_KEY "sk-ant-..."
   ```
   Close and reopen PowerShell afterwards.

## 2. Modal account (the GPU)
1. modal.com → sign up (GitHub login works). New accounts get free monthly credits.
2. Add a card under Settings → Billing once you go past the free tier.
3. PowerShell:
   ```
   py -m modal setup
   ```
   A browser tab opens; approve. Done — the token is stored on your PC.

## 3. Start the inference server
```
py -m modal deploy serve.py
```
It prints a URL like `https://<you>--murmur-vllm-serve.modal.run`.
Open `murmur.py` and set `VLLM_URL = "https://<you>--murmur-vllm-serve.modal.run/v1"`.
Test:
```
py murmur.py --backend vllm --agents 8 --ticks 5
```
First call takes ~1–2 minutes (model download + start); after that it's fast.
Without any LoRAs yet, personas come from the prompt — that already works.

## 4. Make LoRA personas (distill from Haiku)
```
py murmur.py --backend haiku --agents 20 --ticks 40 --seed 1
ren run.jsonl haiku1.jsonl
py murmur.py --backend haiku --agents 20 --ticks 40 --seed 2
ren run.jsonl haiku2.jsonl
py make_lora_data.py haiku1.jsonl haiku2.jsonl        # -> lora_data\Scout_gloomy.jsonl etc.
py -m modal run train_lora.py --persona Scout_gloomy   # ~3 min, ~$0.05
py -m modal run train_lora.py --persona Liar
py -m modal deploy serve.py                            # reload server with new adapters
```
Now `--backend vllm` automatically uses the adapter for any persona that has one.

Alternative without Haiku: hand-write examples in `lora_data\<persona>.jsonl`
(same `{"messages":[...]}` format) — good for extreme personalities Haiku won't roleplay.

## 5. Scale to tens of thousands of agents
- In `murmur.py`: `GRID_SIZE = 200`, `N_RESOURCES = 5000`, `MAX_PARALLEL["vllm"] = 2000`.
- In `serve.py`: `GPU = "H100"`, `N_REPLICAS = 4`, `--max-num-seqs 1024`. Modal load-balances.
- Rough throughput: one L4 ≈ 30–60 agent-steps/s on a 1.7B model, one H100 ≈ 400+.
  20k agents × 100 ticks = 2M steps ≈ 1.5 h on four H100s (≈ $25). Use Qwen3-0.6B to halve it.
- Seeds: `--seed N` fixes the world, persona assignment, and a per-agent-tick seed sent to vLLM.
  Replays are exact as long as the server config and adapters are unchanged; vLLM batching can
  still introduce tiny numeric drift on some kernels, so treat "same seed" as "same distribution"
  unless you also pin `--max-num-seqs 1`.

## Costs (approx)
| GPU  | $/hr  | good for                    |
|------|-------|-----------------------------|
| L4   | ~0.80 | dev, LoRA training, <1k agents |
| A100 | ~2.50 | 1k–10k agents               |
| H100 | ~4.00 | 10k+ agents, use replicas   |
Billed per second while a container is up; sleeps after 10 idle minutes.
