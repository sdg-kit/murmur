"""
analyze.py — per-persona stats from a run log.

    python analyze.py                  # reads run.jsonl
    python analyze.py haiku1.jsonl haiku2.jsonl

For each persona: how many agents had it, deliveries per agent, messages per
agent-tick, share of ticks spent carrying, and a couple of sample thoughts so
you can eyeball whether the personas actually sound different.
"""
import json, sys, random
from collections import defaultdict


def load(paths):
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def main(paths):
    rows = [r for r in load(paths) if "persona" in r]      # skip the meta header line
    if not rows:
        print("no rows"); return

    agents_of  = defaultdict(set)       # persona -> {agent names}
    steps      = defaultdict(int)       # persona -> agent-ticks
    deliveries = defaultdict(int)
    messages   = defaultdict(int)
    carrying   = defaultdict(int)
    thoughts   = defaultdict(list)
    ticks = set()

    for r in rows:
        p = r["persona"]
        ticks.add(r["tick"])
        agents_of[p].add(r["agent"])
        steps[p] += 1
        deliveries[p] += "delivered!" in r["result"]
        messages[p] += bool(r["action"].get("say"))
        carrying[p] += "carrying: True" in r["obs"]
        if r["action"].get("thought"):
            thoughts[p].append(r["action"]["thought"])

    total_steps = len(rows)
    print(f"{len(paths)} file(s)   {len(ticks)} ticks   {sum(len(a) for a in agents_of.values())} agents   "
          f"{total_steps} agent-steps   {sum(deliveries.values())} deliveries\n")
    print(f"{'persona':18} {'agents':>6} {'deliv/agent':>11} {'msgs/tick':>9} {'carrying%':>9}")
    for p in sorted(agents_of, key=lambda p: -deliveries[p] / len(agents_of[p])):
        n = len(agents_of[p])
        print(f"{p:18} {n:6d} {deliveries[p] / n:11.2f} {messages[p] / steps[p]:9.2f} {100 * carrying[p] / steps[p]:8.0f}%")

    print("\nsample thoughts")
    rng = random.Random(0)
    for p in sorted(agents_of):
        for t in rng.sample(thoughts[p], min(2, len(thoughts[p]))):
            print(f"  {p:18} {t}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["run.jsonl"])
