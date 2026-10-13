"""
murmur.py — a tiny LLM swarm in one file.

    python murmur.py                       # scripted brains: no model, no key, no GPU (default)
    python murmur.py --backend mock        # random agents, tests the loop
    python murmur.py --backend haiku       # Claude Haiku (pip install anthropic, ANTHROPIC_API_KEY)
    python murmur.py --backend vllm        # your own GPU server with LoRA personas (see serve.py)
    python render.py                       # run.jsonl -> docs/demo.gif

Read it top to bottom. There are five parts, each independent:

    1. CONFIG    knobs you'll want to turn
    2. WORLD     the grid the agents live in  (change this to change the problem)
    3. PERSONAS  who the agents are           (change this to change behavior)
    4. BRAIN     how an agent decides         (change this to change the model)
    5. LOOP      tick, tick, tick...          (change this to change the experiment)

Every place that's meant to be extended is marked  # EXTEND:
"""

import argparse
import asyncio
import json
import os
import random
import re
import zlib

# ═══════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ═══════════════════════════════════════════════════════════════════════════

GRID_SIZE   = 16       # the world is GRID_SIZE x GRID_SIZE squares (tens of thousands of agents? use 200+)
N_RESOURCES = 24       # scattered in clusters
VIEW_RADIUS = 2        # agents see a 5x5 window
TALK_RADIUS = 4        # agents hear messages from this many squares away
MAX_HEARD   = 5        # an agent keeps at most this many messages per tick (crowds get noisy)
LOG_FILE    = "run.jsonl"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
VLLM_URL    = "http://localhost:8000/v1"   # or your Modal URL, see serve.py
VLLM_BASE   = "Qwen/Qwen3-1.7B"            # must match serve.py
MAX_PARALLEL = {"haiku": 12, "vllm": 512, "mock": 10_000, "scripted": 10_000}  # simultaneous requests

# ═══════════════════════════════════════════════════════════════════════════
# 2. WORLD
# ═══════════════════════════════════════════════════════════════════════════
# A foraging task: find resources (*), carry one at a time back to the nest (H).
# Agents only see nearby squares and only hear nearby agents, so any
# coordination has to come from what they say to each other.

MOVES = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0), "STAY": (0, 0)}


class World:
    def __init__(self, seed):
        rng = random.Random(seed)
        self.tick = 0
        self.nest = (GRID_SIZE // 2, GRID_SIZE // 2)
        self.agents = {}                  # name -> dict of agent state
        self.resources = set()
        # drop resources in a few clusters so there's something worth telling others about
        while len(self.resources) < N_RESOURCES:
            cx, cy = rng.randrange(GRID_SIZE), rng.randrange(GRID_SIZE)
            for _ in range(6):
                x = max(0, min(GRID_SIZE - 1, cx + rng.randint(-2, 2)))
                y = max(0, min(GRID_SIZE - 1, cy + rng.randint(-2, 2)))
                if (x, y) != self.nest:
                    self.resources.add((x, y))

    def add_agent(self, name):
        self.agents[name] = {"x": self.nest[0], "y": self.nest[1],
                             "carrying": False, "score": 0,
                             "said": "", "heard": [], "notes": []}

    def index(self):
        """position -> [names]. Rebuilt once per tick so lookups are O(1) not O(agents)."""
        self.at = {}
        for n, a in self.agents.items():
            self.at.setdefault((a["x"], a["y"]), []).append(n)

    def observe(self, name):
        """What this agent sees. Returned as plain text — that's the prompt."""
        a = self.agents[name]
        rows = []
        for dy in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
            row = ""
            for dx in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
                x, y = a["x"] + dx, a["y"] + dy
                if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE): row += "#"
                elif (dx, dy) == (0, 0):                              row += "@"
                elif (x, y) == self.nest:                             row += "H"
                elif (x, y) in self.resources:                        row += "*"
                elif self.someone_at(x, y, except_=name):             row += "a"
                else:                                                 row += "."
            rows.append(row)
        # EXTEND: add anything else the agent should know (energy, time left, weather...)
        return "\n".join([
            f"tick {self.tick} | position ({a['x']},{a['y']}) | carrying: {a['carrying']} | score: {a['score']}",
            f"nest is {self.direction_to(a, self.nest)}",
            "view (@=you *=resource H=nest a=agent #=wall):",
            *rows,
            "heard: " + (" | ".join(a["heard"]) if a["heard"] else "(nothing)"),
            "your notes: " + (" | ".join(a["notes"][-3:]) if a["notes"] else "(none)"),
        ])

    def apply(self, name, action):
        """Change the world according to one agent's action. Returns a short result string."""
        a = self.agents[name]
        dx, dy = MOVES.get(action["move"], (0, 0))
        a["x"] = max(0, min(GRID_SIZE - 1, a["x"] + dx))
        a["y"] = max(0, min(GRID_SIZE - 1, a["y"] + dy))
        result = "moved " + action["move"]

        here = (a["x"], a["y"])
        if action["pickup"] and not a["carrying"] and here in self.resources:
            self.resources.remove(here); a["carrying"] = True; result += ", picked up"
        if a["carrying"] and here == self.nest:
            a["carrying"] = False; a["score"] += 1; result += ", delivered!"

        a["said"] = action["say"]
        if action["note"]:
            a["notes"] = (a["notes"] + [action["note"]])[-5:]
        # EXTEND: new actions go here (drop, trade, build, vote...)
        return result

    def end_tick(self):
        """Deliver messages to everyone within TALK_RADIUS, then advance time."""
        for a in self.agents.values():
            a["heard"] = []
        self.index()
        for s_name, s in self.agents.items():
            if not s["said"]: continue
            for dx in range(-TALK_RADIUS, TALK_RADIUS + 1):          # walk the diamond around the speaker
                for dy in range(-TALK_RADIUS + abs(dx), TALK_RADIUS - abs(dx) + 1):
                    for t_name in self.at.get((s["x"] + dx, s["y"] + dy), []):
                        heard = self.agents[t_name]["heard"]
                        if t_name != s_name and len(heard) < MAX_HEARD:
                            heard.append(f"{s_name}@({s['x']},{s['y']}): {s['said']}")
        self.tick += 1

    # -- small helpers --
    def someone_at(self, x, y, except_):
        return any(n != except_ for n in self.at.get((x, y), []))

    def direction_to(self, a, target):
        dx, dy = target[0] - a["x"], target[1] - a["y"]
        if dx == dy == 0: return "here"
        return " ".join(p for p in [f"{-dy}N" if dy < 0 else f"{dy}S" if dy else "",
                                    f"{dx}E" if dx > 0 else f"{-dx}W" if dx else ""] if p)

    def total_score(self):
        return sum(a["score"] for a in self.agents.values())


# ═══════════════════════════════════════════════════════════════════════════
# 3. PERSONAS
# ═══════════════════════════════════════════════════════════════════════════

#
# A personality = a ROLE (what it does) + a MOOD (how it feels about it).
# Adding a new one is one line in either dict. 5 roles x 4 moods = 20 personalities.

ROLES = {
    "Scout":    "A restless explorer. Would rather find a new patch than pick a known one clean.",
    "Hoarder":  "Grabs the nearest resource and heads straight home. Distrusts tips from others.",
    "Herald":   "Loves coordinating. Repeats useful locations loudly and follows the crowd.",
    "Skeptic":  "Assumes other agents exaggerate. Verifies before acting.",
    "Follower": "Cooperative. Goes wherever the last message pointed.",
}

MOODS = {
    "sunny":   ("Relentlessly positive. Every message is encouragement, every setback is an opportunity.", 0.9),
    "gloomy":  ("Always negative. Expects the worst, complains in every message, doubts every tip.", 0.7),
    "calm":    ("Even-tempered and matter-of-fact. Says only what's useful.", 0.5),
    "manic":   ("Overexcited and impulsive. Changes plans often, talks constantly.", 1.0),
}
# The number after each mood is the sampling temperature: higher = more random.

# EXTEND: add a third axis (e.g. GOALS = {"team player": ..., "selfish": ...})
# and mix it in below exactly the same way.


def describe(persona):
    """'Scout/gloomy' -> the text that goes in the system prompt, and its temperature."""
    role, mood = persona.split("/")
    mood_text, temperature = MOODS[mood]
    return f"Role: {ROLES[role]}\nMood: {mood_text}", temperature


def random_persona():
    return random.choice(list(ROLES)) + "/" + random.choice(list(MOODS))


# Optional: put hand-written personalities in personas.json next to this file, e.g.
#   {"Liar": ["Broadcasts fake resource locations to send others astray.", 0.8]}
# They're mixed in with the generated ones. No code change needed.
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "personas.json")) as f:
        CUSTOM = json.load(f)
except FileNotFoundError:
    CUSTOM = {}


def persona_prompt(persona):
    if persona in CUSTOM:
        text, temperature = CUSTOM[persona]
        return text, temperature
    return describe(persona)


# ═══════════════════════════════════════════════════════════════════════════
# 4. BRAIN
# ═══════════════════════════════════════════════════════════════════════════
# One function: (persona, observation) -> action dict.
# The action is forced through a "tool" so the model can only reply with
# valid JSON in exactly this shape. No parsing, no retries for bad output.

RULES = """You are one agent in a swarm foraging on a grid.
Goal: pick up resources (*) and carry them to the nest (H), one at a time.
Each tick you see a 5x5 window, messages from nearby agents, and your own notes.
You may move one square, pick up, broadcast a short message, and write a note to yourself.
x grows East, y grows South. Be brief. Stay in character."""

ACTION_SCHEMA = {
    "name": "act",
    "description": "Your action this tick.",
    "input_schema": {
        "type": "object",
        "properties": {
            "thought": {"type": "string", "description": "One sentence, in character."},
            "move":    {"type": "string", "enum": list(MOVES)},
            "pickup":  {"type": "boolean"},
            "say":     {"type": "string", "description": "Broadcast to nearby agents, or empty."},
            "note":    {"type": "string", "description": "Something to remember, or empty."},
        },
        "required": ["thought", "move", "pickup", "say", "note"],
    },
}

# Every brain has the same signature:  (persona_name, observation, seed) -> action dict
# `seed` is unique per (run, agent, tick) so a run can be replayed exactly.

_client = None          # created lazily so --mock never needs an SDK or a key
_limit = None           # semaphore capping simultaneous requests, set in main()


async def think_haiku(persona_name, observation, seed):
    """Claude Haiku. Note: the Claude API has no seed parameter, so `seed` is
    accepted for signature compatibility but ignored — Haiku runs are not
    replayable. Use --backend vllm (or mock) when you need exact replays."""
    description, temperature = persona_prompt(persona_name)
    global _client
    if _client is None:
        from anthropic import AsyncAnthropic
        _client = AsyncAnthropic()
    async with _limit:
        reply = await _client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=200,
            temperature=temperature,
            system=[
                {"type": "text", "text": RULES, "cache_control": {"type": "ephemeral"}},  # cached: paid ~once
                {"type": "text", "text": f"You are {persona_name}.\n{description}"},
            ],
            tools=[ACTION_SCHEMA],
            tool_choice={"type": "tool", "name": "act"},
            messages=[{"role": "user", "content": observation}],
        )
    return next(b.input for b in reply.content if b.type == "tool_use")


_loras = None   # names of LoRA adapters the vLLM server has loaded


async def think_vllm(persona_name, observation, seed):
    """Self-hosted vLLM (see serve.py). If a LoRA adapter named after the persona
    exists (e.g. 'Scout_gloomy'), the personality comes from the adapter and the
    system prompt is just the rules. Otherwise fall back to the prompt-only persona."""
    global _client, _loras
    if _client is None:
        from openai import AsyncOpenAI
        _client = AsyncOpenAI(base_url=VLLM_URL, api_key="none")
        _loras = {m.id async for m in _client.models.list()} - {VLLM_BASE}
    description, temperature = persona_prompt(persona_name)
    adapter = persona_name.replace("/", "_")
    if adapter in _loras:
        model, system = adapter, RULES
    else:
        model, system = VLLM_BASE, f"{RULES}\nYou are {persona_name}.\n{description}"
    async with _limit:
        reply = await _client.chat.completions.create(
            model=model,
            max_tokens=200,
            temperature=temperature,
            seed=seed,                                   # deterministic per agent-tick
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": observation}],
            response_format={"type": "json_schema", "json_schema": {   # vLLM constrains output to the schema
                "name": "act", "schema": ACTION_SCHEMA["input_schema"]}},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # Qwen3: skip <think> for speed
        )
    return json.loads(reply.choices[0].message.content)


async def think_mock(persona_name, observation, seed):
    """Same signature, no model. Handy for testing the world without spending money."""
    r = random.Random(seed)
    return {"thought": "beep", "move": r.choice(list(MOVES)), "pickup": True,
            "say": "resources over here" if r.random() < 0.3 else "", "note": ""}


# ── scripted brain ─────────────────────────────────────────────────────────
# No model at all: a handful of rules per role, applied to the *same text
# prompt* the LLMs get (it parses the observation back into numbers). This is
# the CPU demo and the baseline the LLM backends are measured against.
# In character means: Scouts explore, Hoarders go straight home, Heralds relay
# tips, Skeptics want two sources, Followers chase any tip, Liars lie.

THOUGHTS = {
    "sunny":  ["What a lovely day to forage!", "Every square is a chance!", "We've got this, team!"],
    "gloomy": ["This patch will be empty, I know it.", "Why do I even bother.", "Someone's lying again, probably."],
    "calm":   ["Checking the next row.", "Heading back.", "Noted."],
    "manic":  ["GO GO GO!", "Wait, no, THAT way!", "Did you hear that?! Resources!!"],
}
COORDS = re.compile(r"\((\d+),(\d+)\)")
TALKS  = {"Scout": 0.6, "Hoarder": 0.0, "Herald": 1.0, "Skeptic": 0.5, "Follower": 0.3, "Liar": 0.8}
KEEPS_HEADING = {"Scout": 0.9, "Hoarder": 0.5, "Herald": 0.6, "Skeptic": 0.7, "Follower": 0.5}


def parse_observation(observation):
    """Pull the structured bits back out of the prompt text."""
    lines = observation.split("\n")
    x, y = map(int, COORDS.search(lines[0]).groups())
    carrying = "carrying: True" in lines[0]
    nest = lines[1].removeprefix("nest is ")
    view = [l for l in lines if l and set(l) <= set("#@H*a.") and len(l) == 2 * VIEW_RADIUS + 1]
    heard = next(l for l in lines if l.startswith("heard: "))[7:]
    notes = next(l for l in lines if l.startswith("your notes: "))[12:]
    seen = [(x + dx - VIEW_RADIUS, y + dy - VIEW_RADIUS)
            for dy, row in enumerate(view) for dx, ch in enumerate(row) if ch == "*"]
    tips = [tuple(map(int, m)) for msg in heard.split(" | ") if ": " in msg
            for m in COORDS.findall(msg.split(": ", 1)[1])]              # coordinates in what others said
    walls = {d for d, (dx, dy) in MOVES.items() if d != "STAY"
             and view[VIEW_RADIUS + dy][VIEW_RADIUS + dx] == "#"}
    return x, y, carrying, nest, seen, tips, walls, notes.split(" | ")[-1]


def step_toward(x, y, tx, ty):
    dx, dy = tx - x, ty - y
    if dx == dy == 0: return "STAY"
    if abs(dx) >= abs(dy): return "E" if dx > 0 else "W"
    return "S" if dy > 0 else "N"


async def think_scripted(persona_name, observation, seed):
    r = random.Random(seed)
    role, _, mood = persona_name.partition("/")
    mood = mood or "calm"                                    # custom personas (personas.json) have no mood
    x, y, carrying, nest, seen, tips, walls, last_note = parse_observation(observation)
    nearest = min(seen, key=lambda p: abs(p[0] - x) + abs(p[1] - y), default=None)
    target = COORDS.search(last_note) if last_note.startswith("goto") else None
    target = tuple(map(int, target.groups())) if target else None
    move, pickup, say, note = None, not carrying, "", ""   # pickup is free: always try when empty-handed

    if carrying:                                             # 1. bring it home
        d = {k: 0 for k in "NSEW"}
        for n, k in re.findall(r"(\d+)([NSEW])", nest): d[k] = int(n)
        move = step_toward(x, y, x + d["E"] - d["W"], y + d["S"] - d["N"])
    elif nearest:                                            # 2. something in view: go get it
        move = step_toward(x, y, *nearest)
    elif target:                                             # 3. on the way to a remembered tip
        if (x, y) == target:
            note, move = f"empty at ({x},{y})", r.choice("NSEW")
        else:
            move = step_toward(x, y, *target)
    else:                                                    # 4. nothing to go on: listen, then wander
        trusted = [t for t in tips if role in ("Follower", "Herald")] + \
                  [t for t in set(tips) if role == "Skeptic" and tips.count(t) >= 2]   # Skeptic wants two sources
        if trusted:
            tx, ty = trusted[0]; note, move = f"goto ({tx},{ty})", step_toward(x, y, tx, ty)
        else:
            heading = last_note[8:] if last_note.startswith("heading") else None
            if heading not in MOVES or heading in walls or r.random() > KEEPS_HEADING.get(role, 0.5):
                heading = r.choice([d for d in "NSEW" if d not in walls] or ["STAY"])
            note, move = f"heading {heading}", heading

    if r.random() < TALKS.get(role, 0.3):                    # what to broadcast, if anything
        if role == "Liar":
            say = f"resources at ({r.randrange(GRID_SIZE)},{r.randrange(GRID_SIZE)})"
        elif nearest:
            say = f"resources at ({nearest[0]},{nearest[1]})"
        elif role == "Herald" and tips:
            say = f"resources at ({tips[0][0]},{tips[0][1]})"      # relay what it heard
        elif mood == "manic":
            say = "anyone found anything?!"
    _, temperature = persona_prompt(persona_name)
    if r.random() < max(0.0, temperature - 0.5) * 0.4:       # mood temperature -> occasional impulsive move
        move = r.choice(list(MOVES))
    return {"thought": r.choice(THOUGHTS.get(mood, THOUGHTS["calm"])), "move": move,
            "pickup": pickup, "say": say, "note": note}
# EXTEND: custom personas from personas.json get the default rules (wander, say nothing)
# unless you add a branch above, as Liar has.


BRAINS = {"scripted": think_scripted, "haiku": think_haiku, "vllm": think_vllm, "mock": think_mock}
# EXTEND: add another backend = one more function with the same signature + one dict entry.


# ═══════════════════════════════════════════════════════════════════════════
# 5. LOOP
# ═══════════════════════════════════════════════════════════════════════════

async def main(n_agents, n_ticks, seed, backend):
    global _limit
    random.seed(seed)
    world = World(seed)
    brain = BRAINS[backend]
    _limit = asyncio.Semaphore(MAX_PARALLEL[backend])

    persona_of = {}
    for i in range(n_agents):
        name = f"a{i:02d}"
        world.add_agent(name)
        pool = list(CUSTOM) + [random_persona() for _ in range(3)]   # customs get a fair share
        persona_of[name] = random.choice(pool)

    with open(LOG_FILE, "w") as log:
        # first line: everything render.py needs to replay the run (resource positions aren't in the obs)
        log.write(json.dumps({"meta": {"seed": seed, "backend": backend, "grid": GRID_SIZE, "nest": world.nest,
                                       "talk_radius": TALK_RADIUS, "resources": sorted(world.resources),
                                       "personas": persona_of}}) + "\n")
        for _ in range(n_ticks):
            names = list(world.agents)
            world.index()
            observations = [world.observe(n) for n in names]
            # all agents think at the same time (one request each, in parallel)
            seeds = [zlib.crc32(f"{seed}:{n}:{world.tick}".encode()) for n in names]   # stable across runs
            actions = await asyncio.gather(*(brain(persona_of[n], o, sd) for n, o, sd in zip(names, observations, seeds)))
            for n, obs, act in zip(names, observations, actions):
                result = world.apply(n, act)
                log.write(json.dumps({"tick": world.tick, "agent": n, "persona": persona_of[n],
                                      "obs": obs, "action": act, "result": result}) + "\n")
            world.end_tick()
            print(f"tick {world.tick:3d}   resources left {len(world.resources):3d}   score {world.total_score():3d}")
            if not world.resources:
                print("all collected!"); break
    print(f"log written to {LOG_FILE}   (python render.py -> gif, python analyze.py -> stats)")
    # EXTEND: run several seeds, change TALK_RADIUS to 0, compare scores by persona...


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=8)
    p.add_argument("--ticks", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=list(BRAINS), default="scripted")
    a = p.parse_args()
    asyncio.run(main(a.agents, a.ticks, a.seed, a.backend))
