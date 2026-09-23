"""Unit tests for World.apply and message delivery. Run with:  py -m pytest"""
import sys, os, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import murmur
from murmur import World, GRID_SIZE, TALK_RADIUS, MAX_HEARD


def act(move="STAY", pickup=False, say="", note=""):
    return {"thought": "", "move": move, "pickup": pickup, "say": say, "note": note}


def place(world, name, x, y):
    world.add_agent(name)
    world.agents[name]["x"], world.agents[name]["y"] = x, y
    return world.agents[name]


@pytest.fixture
def world():
    w = World(seed=0)
    w.resources.clear()          # tests place their own resources
    return w


# ── apply ──────────────────────────────────────────────────────────────────

def test_moves_change_position(world):
    a = place(world, "a", 5, 5)
    world.apply("a", act("N")); assert (a["x"], a["y"]) == (5, 4)
    world.apply("a", act("E")); assert (a["x"], a["y"]) == (6, 4)
    world.apply("a", act("S")); assert (a["x"], a["y"]) == (6, 5)
    world.apply("a", act("W")); assert (a["x"], a["y"]) == (5, 5)
    world.apply("a", act("STAY")); assert (a["x"], a["y"]) == (5, 5)


def test_moves_are_clamped_at_walls(world):
    a = place(world, "a", 0, 0)
    world.apply("a", act("N")); world.apply("a", act("W"))
    assert (a["x"], a["y"]) == (0, 0)
    a["x"], a["y"] = GRID_SIZE - 1, GRID_SIZE - 1
    world.apply("a", act("S")); world.apply("a", act("E"))
    assert (a["x"], a["y"]) == (GRID_SIZE - 1, GRID_SIZE - 1)


def test_pickup_takes_resource_only_when_present_and_empty_handed(world):
    a = place(world, "a", 3, 3)
    world.resources.add((3, 3))
    assert "picked up" not in world.apply("a", act(pickup=False))
    assert (3, 3) in world.resources and not a["carrying"]

    r = world.apply("a", act(pickup=True))
    assert "picked up" in r and a["carrying"] and (3, 3) not in world.resources

    world.resources.add((3, 3))                       # already carrying: can't take a second
    assert "picked up" not in world.apply("a", act(pickup=True))
    assert (3, 3) in world.resources


def test_pickup_happens_after_the_move(world):
    a = place(world, "a", 3, 3)
    world.resources.add((4, 3))
    assert "picked up" in world.apply("a", act("E", pickup=True))
    assert a["carrying"]


def test_delivery_at_nest_scores_and_empties_hands(world):
    nx, ny = world.nest
    a = place(world, "a", nx - 1, ny)
    a["carrying"] = True
    r = world.apply("a", act("E"))
    assert "delivered!" in r and a["score"] == 1 and not a["carrying"]
    assert "delivered!" not in world.apply("a", act("STAY"))   # nothing to deliver now
    assert a["score"] == 1


def test_say_and_notes(world):
    a = place(world, "a", 1, 1)
    world.apply("a", act(say="hello", note="n1"))
    assert a["said"] == "hello" and a["notes"] == ["n1"]
    world.apply("a", act(say="", note=""))
    assert a["said"] == "" and a["notes"] == ["n1"]           # empty note is not recorded
    for i in range(2, 9):
        world.apply("a", act(note=f"n{i}"))
    assert a["notes"] == ["n4", "n5", "n6", "n7", "n8"]      # last five kept


def test_unknown_move_is_a_stay(world):
    a = place(world, "a", 4, 4)
    world.apply("a", act("JUMP"))
    assert (a["x"], a["y"]) == (4, 4)


# ── message delivery ───────────────────────────────────────────────────────

def test_messages_reach_agents_within_talk_radius_only(world):
    s = place(world, "s", 8, 8); s["said"] = "here"
    place(world, "near", 8 + TALK_RADIUS, 8)                   # exactly on the edge (Manhattan)
    place(world, "diag", 8 + TALK_RADIUS // 2, 8 + (TALK_RADIUS - TALK_RADIUS // 2))
    place(world, "far", 8 + TALK_RADIUS + 1, 8)
    place(world, "corner", 8 + TALK_RADIUS, 8 + TALK_RADIUS)   # inside the square, outside the diamond
    world.end_tick()
    assert world.agents["near"]["heard"] == ["s@(8,8): here"]
    assert world.agents["diag"]["heard"] == ["s@(8,8): here"]
    assert world.agents["far"]["heard"] == []
    assert world.agents["corner"]["heard"] == []


def test_speaker_does_not_hear_itself_and_silence_sends_nothing(world):
    s = place(world, "s", 8, 8); s["said"] = "hi"
    q = place(world, "q", 8, 9); q["said"] = ""
    world.end_tick()
    assert s["heard"] == []
    assert q["heard"] == ["s@(8,8): hi"]


def test_heard_is_capped_and_cleared_each_tick(world):
    t = place(world, "t", 8, 8)
    for i in range(MAX_HEARD + 3):
        sp = place(world, f"s{i}", 8, 9); sp["said"] = f"m{i}"
    world.end_tick()
    assert len(t["heard"]) == MAX_HEARD
    for i in range(MAX_HEARD + 3):
        world.agents[f"s{i}"]["said"] = ""
    world.end_tick()
    assert t["heard"] == []                                    # old messages don't linger


def test_end_tick_advances_time(world):
    assert world.tick == 0
    world.end_tick()
    assert world.tick == 1


# ── observation / helpers ──────────────────────────────────────────────────

def test_observe_marks_walls_self_nest_resources_and_agents(world):
    place(world, "a", 0, 0)
    place(world, "b", 1, 0)
    world.resources.add((0, 1))
    world.index()
    obs = world.observe("a")
    rows = obs.split("\n")[3:8]                                # the 5x5 view
    assert rows[2] == "##@a."                                  # wall, wall, me, agent, empty
    assert rows[3] == "##*.."                                  # resource south of me
    assert rows[0] == "#####"                                  # above the grid


def test_direction_to():
    w = World(seed=0)
    a = {"x": 5, "y": 5}
    assert w.direction_to(a, (5, 5)) == "here"
    assert w.direction_to(a, (5, 2)) == "3N"
    assert w.direction_to(a, (8, 7)) == "2S 3E"
    assert w.direction_to(a, (2, 5)) == "3W"


def test_world_is_reproducible_from_seed():
    assert World(seed=7).resources == World(seed=7).resources
    assert World(seed=7).resources != World(seed=8).resources
    assert World(seed=7).nest not in World(seed=7).resources


def test_agent_tick_seed_is_stable():
    assert zlib.crc32(b"0:a00:0") == zlib.crc32(b"0:a00:0")
    assert zlib.crc32(b"0:a00:0") != zlib.crc32(b"0:a00:1")


def test_mock_brain_is_deterministic_for_a_seed():
    import asyncio
    a = asyncio.run(murmur.think_mock("Scout/calm", "", 12345))
    b = asyncio.run(murmur.think_mock("Scout/calm", "", 12345))
    assert a == b and a["move"] in murmur.MOVES
