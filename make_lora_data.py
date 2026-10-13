"""
make_lora_data.py — turn Haiku run logs into per-persona training sets.

The idea: run the swarm on Haiku a few times (it plays the personas well),
then distill each persona's behavior into a small LoRA on the cheap model.
The LoRA learns "given this observation, act like a gloomy Scout" without
needing the persona text in the prompt.

    python murmur.py --backend haiku --agents 20 --ticks 40 --seed 1
    python murmur.py --backend haiku --agents 20 --ticks 40 --seed 2   # more seeds = more data
    python make_lora_data.py run.jsonl run_seed2.jsonl ...            # -> lora_data/<persona>.jsonl

A few hundred examples per persona is enough to see the style stick.
"""
import json, sys, os
from collections import defaultdict
from murmur import RULES

os.makedirs("lora_data", exist_ok=True)
by_persona = defaultdict(list)
for path in sys.argv[1:] or ["run.jsonl"]:
    for line in open(path):
        r = json.loads(line)
        if "persona" not in r: continue                                 # meta header line
        by_persona[r["persona"]].append({"messages": [
            {"role": "system", "content": RULES},                       # rules only — no persona text
            {"role": "user", "content": r["obs"]},
            {"role": "assistant", "content": json.dumps(r["action"])},
        ]})

for persona, rows in by_persona.items():
    name = persona.replace("/", "_")
    with open(f"lora_data/{name}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{name:20} {len(rows):5d} examples")
