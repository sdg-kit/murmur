"""
render.py — turn a run log into a GIF of the grid (the README hero).

    pip install pillow
    python render.py                              # run.jsonl -> docs/demo.gif
    python render.py haiku1.jsonl out.gif --cell 32 --fps 5

Replays the log instead of re-running the sim, so it works for any backend.
Agents are colored by role; a white dot means carrying; lines are messages
reaching their hearers; the footer shows a couple of things said that tick.
"""
import argparse
import json
import os
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

from murmur import MOVES, GRID_SIZE, TALK_RADIUS

# Categorical palette, dark-surface steps, fixed slot order (same hues as figures.py's light steps).
COLORS = {"Scout": (57, 135, 229), "Hoarder": (217, 89, 38), "Herald": (25, 158, 112),
          "Skeptic": (201, 133, 0), "Follower": (213, 81, 129), "Liar": (0, 131, 0)}
OTHER     = (137, 135, 129)
BG, GRIDC = (26, 26, 25), (44, 44, 42)
NEST, RES = (240, 239, 236), (195, 194, 183)
TEXT      = (195, 194, 183)


def role_of(persona):
    return persona.split("/")[0]


def load_run(path):
    """Replay the log -> meta, list of per-tick states. State = (agents, resources, score)
    where agents = {name: (x, y, carrying, said, persona)} at the end of the tick."""
    meta, by_tick = None, defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if "meta" in r: meta = r["meta"]
            else: by_tick[r["tick"]].append(r)

    if meta is None:                            # old log without a header: recover what we can from the views
        grid, nest, resources, talk = GRID_SIZE, None, set(), TALK_RADIUS
        for rows in by_tick.values():
            for r in rows:
                lines = r["obs"].split("\n")
                x, y = map(int, lines[0].split("position (")[1].split(")")[0].split(","))
                view = [l for l in lines if l and set(l) <= set("#@H*a.")]
                half = len(view) // 2
                for dy, row in enumerate(view):
                    for dx, ch in enumerate(row):
                        if ch == "*": resources.add((x + dx - half, y + dy - half))
                        if ch == "H": nest = (x + dx - half, y + dy - half)
        meta = {"grid": grid, "nest": nest or (grid // 2, grid // 2), "resources": sorted(resources),
                "talk_radius": talk, "personas": {}}

    grid = meta["grid"]
    resources = {tuple(p) for p in meta["resources"]}
    agents, score, frames = {}, 0, []
    for tick in sorted(by_tick):
        for r in by_tick[tick]:
            x, y = map(int, r["obs"].split("position (")[1].split(")")[0].split(","))
            dx, dy = MOVES.get(r["action"]["move"], (0, 0))
            x, y = max(0, min(grid - 1, x + dx)), max(0, min(grid - 1, y + dy))
            carrying = "carrying: True" in r["obs"]
            if "picked up" in r["result"]:  resources.discard((x, y)); carrying = True
            if "delivered!" in r["result"]: carrying = False; score += 1
            agents[r["agent"]] = (x, y, carrying, r["action"].get("say", ""), r["persona"])
        frames.append((tick, dict(agents), set(resources), score))
    return meta, frames


def font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf", "segoeui.ttf"):
        try: return ImageFont.truetype(name, size)
        except OSError: pass
    return ImageFont.load_default(size=size)


def draw_frame(meta, frame, cell):
    tick, agents, resources, score = frame
    grid, nest, talk = meta["grid"], tuple(meta["nest"]), meta["talk_radius"]
    head, foot = int(cell * 1.6), int(cell * 2.2)
    W, H = grid * cell, head + grid * cell + foot
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_big, f_small = font(int(cell * 0.6)), font(max(10, int(cell * 0.42)))
    cx = lambda x: x * cell + cell / 2
    cy = lambda y: head + y * cell + cell / 2

    d.text((8, head * 0.25), f"tick {tick:3d}    delivered {score:3d}    resources left {len(resources):3d}",
           fill=TEXT, font=f_big)
    for i in range(grid + 1):                                    # grid lines
        d.line([(i * cell, head), (i * cell, head + grid * cell)], fill=GRIDC)
        d.line([(0, head + i * cell), (W, head + i * cell)], fill=GRIDC)
    for x, y in resources:                                       # resources: small diamonds
        r = cell * 0.22
        d.polygon([(cx(x), cy(y) - r), (cx(x) + r, cy(y)), (cx(x), cy(y) + r), (cx(x) - r, cy(y))], fill=RES)
    r = cell * 0.38                                              # nest: hollow square
    d.rectangle([cx(nest[0]) - r, cy(nest[1]) - r, cx(nest[0]) + r, cy(nest[1]) + r], outline=NEST, width=2)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))          # message rays, translucent
    od = ImageDraw.Draw(overlay)
    for name, (x, y, _, said, persona) in agents.items():
        if not said: continue
        col = COLORS.get(role_of(persona), OTHER) + (110,)
        for other, (ox, oy, *_rest) in agents.items():
            if other != name and abs(ox - x) + abs(oy - y) <= talk:
                od.line([(cx(x), cy(y)), (cx(ox), cy(oy))], fill=col, width=max(1, cell // 14))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(img)

    for name, (x, y, carrying, said, persona) in agents.items():   # agents on top
        col, r = COLORS.get(role_of(persona), OTHER), cell * 0.32
        d.ellipse([cx(x) - r, cy(y) - r, cx(x) + r, cy(y) + r], fill=col,
                  outline=TEXT if said else None, width=2)
        if carrying:
            d.ellipse([cx(x) - r / 3, cy(y) - r / 3, cx(x) + r / 3, cy(y) + r / 3], fill=NEST)

    y0, x0 = head + grid * cell + 6, 8                             # legend + chatter
    for role, col in COLORS.items():
        if any(role_of(a[4]) == role for a in agents.values()):
            d.ellipse([x0, y0 + 2, x0 + cell * 0.4, y0 + 2 + cell * 0.4], fill=col)
            d.text((x0 + cell * 0.55, y0), role, fill=TEXT, font=f_small)
            x0 += cell * 0.55 + d.textlength(role, font=f_small) + cell * 0.5
    said = [(n, a[4], a[3]) for n, a in sorted(agents.items()) if a[3]]
    for i, (n, persona, msg) in enumerate(said[:2]):                 # colored dot carries identity, text stays neutral
        ty = y0 + cell * 0.6 + i * cell * 0.55
        d.ellipse([8, ty + 3, 8 + cell * 0.3, ty + 3 + cell * 0.3], fill=COLORS.get(role_of(persona), OTHER))
        d.text((8 + cell * 0.45, ty), f"{n} {persona}: {msg}"[:int(W / (cell * 0.23))], fill=TEXT, font=f_small)
    return img


def render(log, out, cell=28, fps=6, hold=2.0):
    meta, frames = load_run(log)
    images = [draw_frame(meta, fr, cell) for fr in frames]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    durations = [1000 // fps] * len(images)
    durations[-1] = int(hold * 1000)
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0)
    images[-1].save(os.path.splitext(out)[0] + "_last.png")
    print(f"{len(images)} frames -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="?", default="run.jsonl")
    p.add_argument("out", nargs="?", default="docs/demo.gif")
    p.add_argument("--cell", type=int, default=28, help="pixels per grid square")
    p.add_argument("--fps", type=int, default=6)
    a = p.parse_args()
    render(a.log, a.out, a.cell, a.fps)
