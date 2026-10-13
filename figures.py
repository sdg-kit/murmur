"""
figures.py — reproduce every figure and number in the README from scratch.

    pip install matplotlib
    python figures.py            # ~1 min on CPU: 4 conditions x 5 seeds x 24 agents x 40 ticks
                                 # -> docs/fig_personas.png, docs/fig_ablation.png, docs/results.json

Conditions: {Liars present, absent} x {TALK_RADIUS 4, 0}. Everything is seeded,
so the numbers are exact and the figures are byte-for-byte reproducible.
"""
import asyncio
import contextlib
import io
import json
import os
import statistics
import tempfile
from collections import defaultdict

import murmur

SEEDS, AGENTS, TICKS = (1, 2, 3, 4, 5), 24, 40
ROLES = ["Scout", "Hoarder", "Herald", "Skeptic", "Follower", "Liar"]
MOODS = ["sunny", "gloomy", "calm", "manic"]

# Reference categorical palette (validated, fixed order) — see render.py for the dark steps.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ROLE_COLOR = dict(zip(ROLES, PALETTE))
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def run(seed, talk, liars):
    """One run -> list of log rows (the meta line stripped)."""
    liar = {k: v for k, v in murmur.CUSTOM.items()} if liars else {}
    saved = dict(murmur.CUSTOM)
    murmur.CUSTOM.clear(); murmur.CUSTOM.update(liar)
    murmur.TALK_RADIUS = talk
    with tempfile.TemporaryDirectory() as d:
        murmur.LOG_FILE = os.path.join(d, "run.jsonl")
        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(murmur.main(n_agents=AGENTS, n_ticks=TICKS, seed=seed, backend="scripted"))
        rows = [json.loads(l) for l in open(murmur.LOG_FILE)]
    murmur.CUSTOM.clear(); murmur.CUSTOM.update(saved)
    return [r for r in rows if "persona" in r]


def cumulative_deliveries(rows):
    per_tick = defaultdict(int)
    for r in rows:
        per_tick[r["tick"]] += "delivered!" in r["result"]
    out, total = [], 0
    for t in range(TICKS):
        total += per_tick.get(t, 0); out.append(total)
    return out


def per_group(rows, key):
    """key(persona) -> (deliveries per agent, messages per agent-tick, n agents)."""
    agents, deliv, msgs, steps = defaultdict(set), defaultdict(int), defaultdict(int), defaultdict(int)
    for r in rows:
        g = key(r["persona"])
        if g is None: continue
        agents[g].add(r["agent"]); steps[g] += 1
        deliv[g] += "delivered!" in r["result"]
        msgs[g] += bool(r["action"].get("say"))
    return {g: (deliv[g] / len(agents[g]), msgs[g] / steps[g], len(agents[g])) for g in agents}


def experiment():
    curves, stats = {}, {"role": defaultdict(list), "mood": defaultdict(list)}
    for liars in (True, False):
        for talk in (4, 0):
            runs = [run(s, talk, liars) for s in SEEDS]
            curves[(liars, talk)] = [cumulative_deliveries(r) for r in runs]
            if liars and talk == 4:                                  # the default condition: persona stats
                for rows in runs:
                    for g, v in per_group(rows, lambda p: p.split("/")[0]).items(): stats["role"][g].append(v)
                    for g, v in per_group(rows, lambda p: p.split("/")[1] if "/" in p else None).items():
                        stats["mood"][g].append(v)
    return curves, stats


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1); ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def fig_personas(stats, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=SURFACE)
    roles = [r for r in ROLES if r in stats["role"]]
    for ax, i, title, fmt in ((axes[0], 0, "Deliveries per agent", "{:.2f}"),
                              (axes[1], 1, "Messages per agent-tick", "{:.2f}")):
        style(ax); ax.grid(axis="y", visible=False)
        means = [statistics.mean(v[i] for v in stats["role"][r]) for r in roles]
        y = range(len(roles))[::-1]
        ax.barh(list(y), means, height=0.5, color=[ROLE_COLOR[r] for r in roles])
        for yy, m in zip(y, means):
            ax.text(m + max(means) * 0.02, yy, fmt.format(m), va="center", fontsize=9, color=INK2)
        ax.set_yticks(list(y)); ax.set_yticklabels(roles, color=INK, fontsize=10)
        ax.set_xlim(0, max(means) * 1.25)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    fig.text(0.01, 0.01, f"scripted backend · {AGENTS} agents · {TICKS} ticks · mean of {len(SEEDS)} seeds · Liars present, TALK_RADIUS = 4",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE); plt.close(fig)


def fig_ablation(curves, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=SURFACE, sharey=True)
    series = {4: ("TALK_RADIUS = 4 (agents hear neighbours)", PALETTE[0]),
              0: ("TALK_RADIUS = 0 (deaf swarm)", PALETTE[1])}
    for ax, liars, title in ((axes[0], True, "With Liars (≈25 % of the swarm)"), (axes[1], False, "Without Liars")):
        style(ax)
        finals = {talk: statistics.mean(c[-1] for c in curves[(liars, talk)]) for talk in series}
        hi, lo = max(finals, key=finals.get), min(finals, key=finals.get)
        nudge = max(0.0, 1.1 - (finals[hi] - finals[lo])) / 2          # keep end labels from colliding
        for talk, (label, color) in series.items():
            runs = curves[(liars, talk)]
            mean = [statistics.mean(c[t] for c in runs) for t in range(TICKS)]
            sd = [statistics.pstdev(c[t] for c in runs) for t in range(TICKS)]
            x = range(1, TICKS + 1)
            ax.fill_between(x, [m - s for m, s in zip(mean, sd)], [m + s for m, s in zip(mean, sd)],
                            color=color, alpha=0.10, linewidth=0)
            ax.plot(x, mean, color=color, linewidth=2, solid_capstyle="round", label=label)
            ax.plot([TICKS], [mean[-1]], "o", ms=7, color=color, markeredgecolor=SURFACE, markeredgewidth=2)
            ax.text(TICKS + 0.8, mean[-1] + (nudge if talk == hi else -nudge), f"{mean[-1]:.1f}",
                    va="center", fontsize=9, color=INK2)
        ax.set_xlim(1, TICKS + 4); ax.set_xlabel("tick", color=MUTED, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    axes[0].set_ylabel("cumulative deliveries", color=MUTED, fontsize=9)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
    fig.text(0.01, 0.01, f"scripted backend · {AGENTS} agents · {TICKS} ticks · mean ± 1 s.d. over {len(SEEDS)} seeds",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE); plt.close(fig)


def main():
    os.makedirs("docs", exist_ok=True)
    curves, stats = experiment()
    results = {
        "settings": {"agents": AGENTS, "ticks": TICKS, "seeds": SEEDS, "grid": murmur.GRID_SIZE},
        "ablation": {f"liars={liars} talk={talk}": {"final": [c[-1] for c in runs],
                                                    "mean": statistics.mean(c[-1] for c in runs),
                                                    "sd": statistics.pstdev(c[-1] for c in runs)}
                     for (liars, talk), runs in curves.items()},
        "by_role": {r: {"deliveries_per_agent": statistics.mean(v[0] for v in vs),
                        "messages_per_agent_tick": statistics.mean(v[1] for v in vs),
                        "agents_total": sum(v[2] for v in vs)} for r, vs in stats["role"].items()},
        "by_mood": {m: {"deliveries_per_agent": statistics.mean(v[0] for v in vs),
                        "messages_per_agent_tick": statistics.mean(v[1] for v in vs),
                        "agents_total": sum(v[2] for v in vs)} for m, vs in stats["mood"].items()},
    }
    with open("docs/results.json", "w") as f:
        json.dump(results, f, indent=1)
    fig_personas(stats, "docs/fig_personas.png")
    fig_ablation(curves, "docs/fig_ablation.png")
    for k, v in results["ablation"].items(): print(f"{k:24} {v['mean']:5.1f} ± {v['sd']:.1f}   {v['final']}")
    print("\nby role"); [print(f"  {r:9} {v['deliveries_per_agent']:.2f} deliv/agent  {v['messages_per_agent_tick']:.2f} msgs/tick  n={v['agents_total']}") for r, v in results["by_role"].items()]
    print("by mood"); [print(f"  {m:9} {v['deliveries_per_agent']:.2f} deliv/agent  {v['messages_per_agent_tick']:.2f} msgs/tick  n={v['agents_total']}") for m, v in results["by_mood"].items()]
    print("\nwrote docs/fig_personas.png docs/fig_ablation.png docs/results.json")


if __name__ == "__main__":
    main()
