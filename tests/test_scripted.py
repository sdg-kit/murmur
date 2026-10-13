"""The scripted brain and the renderer's replay. Run with:  py -m pytest"""
import asyncio
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import murmur
from murmur import World, parse_observation, step_toward, think_scripted


def think(persona, obs, seed=1):
    return asyncio.run(think_scripted(persona, obs, seed))


def world_with(name, x, y, resources=(), carrying=False, heard=(), notes=()):
    w = World(seed=0)
    w.resources = set(resources)
    w.add_agent(name)
    a = w.agents[name]
    a["x"], a["y"], a["carrying"] = x, y, carrying
    a["heard"], a["notes"] = list(heard), list(notes)
    w.index()
    return w


# ── parsing the prompt back ────────────────────────────────────────────────

def test_parse_observation_roundtrips_the_prompt():
    w = world_with("a", 3, 4, resources={(4, 4), (0, 0)}, heard=["b@(1,1): resources at (9,9)"], notes=["goto (9,9)"])
    x, y, carrying, nest, seen, tips, walls, last_note = parse_observation(w.observe("a"))
    assert (x, y) == (3, 4) and not carrying
    assert nest == "4S 5E"
    assert seen == [(4, 4)]                     # (0,0) is out of view
    assert tips == [(9, 9)]                     # speaker's own position is not a tip
    assert walls == set()
    assert last_note == "goto (9,9)"


def test_parse_observation_sees_walls():
    w = world_with("a", 0, 0)
    *_, walls, _ = parse_observation(w.observe("a"))
    assert walls == {"N", "W"}


def test_step_toward_moves_along_the_longer_axis():
    assert step_toward(0, 0, 3, 1) == "E"
    assert step_toward(0, 0, 1, 3) == "S"
    assert step_toward(5, 5, 5, 2) == "N"
    assert step_toward(5, 5, 5, 5) == "STAY"


# ── behaviour ──────────────────────────────────────────────────────────────

def test_carrying_agent_heads_for_the_nest():
    w = world_with("a", 2, 8, carrying=True)                # nest is at (8,8), due east
    act = think("Hoarder/calm", w.observe("a"))
    assert act["move"] == "E" and not act["pickup"]


def test_visible_resource_is_approached_and_picked_up():
    w = world_with("a", 5, 5, resources={(5, 7)})
    act = think("Hoarder/calm", w.observe("a"))
    assert act["move"] == "S" and act["pickup"]
    # the view shows "@" on your own square, so pickup is simply always on when empty-handed
    assert think("Hoarder/calm", world_with("a", 5, 5).observe("a"))["pickup"]
    assert not think("Hoarder/calm", world_with("a", 5, 5, carrying=True).observe("a"))["pickup"]


def test_follower_chases_a_tip_but_scout_ignores_it():
    heard = ["b@(1,1): resources at (12,3)"]
    f = think("Follower/calm", world_with("a", 3, 3, heard=heard).observe("a"))
    assert f["move"] == "E" and f["note"] == "goto (12,3)"
    s = think("Scout/calm", world_with("a", 3, 3, heard=heard).observe("a"))
    assert s["note"].startswith("heading")


def test_skeptic_needs_two_sources():
    one = ["b@(1,1): resources at (12,3)"]
    two = one + ["c@(2,2): resources at (12,3)"]
    assert not think("Skeptic/calm", world_with("a", 3, 3, heard=one).observe("a"))["note"].startswith("goto")
    assert think("Skeptic/calm", world_with("a", 3, 3, heard=two).observe("a"))["note"] == "goto (12,3)"


def test_remembered_target_is_followed_then_dropped_when_empty():
    act = think("Follower/calm", world_with("a", 3, 3, notes=["goto (3,9)"]).observe("a"))
    assert act["move"] == "S"
    act = think("Follower/calm", world_with("a", 3, 9, notes=["goto (3,9)"]).observe("a"))
    assert act["note"] == "empty at (3,9)"


def test_hoarders_never_talk_and_liars_lie():
    obs = world_with("a", 5, 5, resources={(6, 5)}).observe("a")
    assert all(think("Hoarder/calm", obs, seed)["say"] == "" for seed in range(20))
    says = {think("Liar", obs, seed)["say"] for seed in range(20)} - {""}
    assert says and "resources at (6,5)" not in says


def test_scripted_is_deterministic_and_well_formed():
    obs = world_with("a", 5, 5, resources={(7, 5)}).observe("a")
    a, b = think("Herald/manic", obs, 42), think("Herald/manic", obs, 42)
    assert a == b
    assert set(a) == set(murmur.ACTION_SCHEMA["input_schema"]["properties"]) and a["move"] in murmur.MOVES


# ── full run + replay ──────────────────────────────────────────────────────

def test_run_writes_meta_header_and_renderer_replays_it(tmp_path, monkeypatch):
    monkeypatch.setattr(murmur, "LOG_FILE", str(tmp_path / "run.jsonl"))
    asyncio.run(murmur.main(n_agents=6, n_ticks=5, seed=1, backend="scripted"))
    lines = [json.loads(l) for l in open(murmur.LOG_FILE)]
    assert "meta" in lines[0] and lines[0]["meta"]["backend"] == "scripted"
    assert len(lines) == 1 + 6 * 5

    pytest.importorskip("PIL")
    import render
    meta, frames = render.load_run(murmur.LOG_FILE)
    assert len(frames) == 5
    tick, agents, resources, score = frames[-1]
    assert tick == 4 and len(agents) == 6
    assert len(resources) <= len(meta["resources"])
    img = render.draw_frame(meta, frames[-1], cell=12)
    assert img.size[0] == meta["grid"] * 12
