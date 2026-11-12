"""events/goal_manager.jsonl: the goal manager's decisions, as willow makes
them (gm.goal, repair.trigger, repair.decision, pause/resume/stop,
gm.robot_state, gm.alert, progress.stall)."""

import json

import pytest

pytest.importorskip("robot_executor_msgs.msg")

from goal_manager_harness import Harness  # noqa: E402
from std_msgs.msg import String  # noqa: E402

MISSION = "(and (visited-object o1) (visited-object o2))"


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "willow_1"
    for name in ("ADT4_EVENTS_DIR", "ADT4_RUN_ID", "ADT4_MISSION_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ADT4_RUN_DIR", str(run))
    monkeypatch.setenv("ADT4_ROBOT_NAME", "willow")
    return run


@pytest.fixture
def h(tmp_path, run_dir):
    harness = Harness(tmp_path)
    yield harness
    harness.close()


def _events(run_dir, *types):
    path = run_dir / "events" / "goal_manager.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(e["source"] == "goal_manager" for e in events)
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    return [e for e in events if not types or e["type"] in types]


def _start_mission(h):
    h.send(MISSION)
    h.plan(euclid=("pe", 2, ["p1", "o1"]), hamilton=("ph", 2, ["p2", "o2"]))


def test_a_goal_logs_what_was_commanded_and_decided(h, run_dir):
    h.send(MISSION)
    (goal,) = _events(run_dir, "gm.goal")
    p = goal["payload"]
    assert p["commanded"] == {
        "robot_id": "willow",
        "pddl_goal": MISSION,
        "constraints": [],
    }
    assert p["decision"] == "repair" and p["forwarded"] is True
    assert p["scope"] == "team" and p["scoped_robots"] == []
    assert p["todo"] == ["(visited-object o1)", "(visited-object o2)"]
    assert goal["robot"] == "willow"


def test_a_robot_scoped_goal_logs_its_robots(h, run_dir):
    _start_mission(h)
    h.send("(visited-object o5)", use=("euclid",))
    goal = _events(run_dir, "gm.goal")[-1]["payload"]
    assert goal["scope"] == "robots" and goal["scoped_robots"] == ["euclid"]
    assert goal["commanded"]["constraints"] == [
        {"predicate": "use-robots", "symbols": ["euclid"]}
    ]
    pauses = _events(run_dir, "pause")
    assert [e["payload"]["robot"] for e in pauses] == ["euclid"]
    assert "robot-scoped" in pauses[0]["payload"]["reason"]


def test_a_plan_arriving_logs_the_repair_decision(h, run_dir):
    _start_mission(h)
    (first,) = _events(run_dir, "repair.decision")
    assert first["payload"]["outcome"] == "planned"
    assert first["payload"]["old_plan"] == {}
    assert first["payload"]["new_plan"] == {"euclid": "pe", "hamilton": "ph"}
    assert first["payload"]["replanned"] == ["euclid", "hamilton"]

    # A revision: both are paused; the new plan keeps only euclid busy.
    h.send("(and (visited-object o1) (visited-object o4))")
    pauses = _events(run_dir, "pause")
    assert sorted(e["payload"]["robot"] for e in pauses) == ["euclid", "hamilton"]
    assert all(e["payload"]["reason"].startswith("new goal (repair)") for e in pauses)
    h.plan(euclid=("pe2", 2, ["p1", "o1", "o4"]))

    second = _events(run_dir, "repair.decision")[-1]["payload"]
    assert second["kind"] == "repair" and second["trigger"] == "goal"
    assert second["old_plan"] == {"euclid": "pe", "hamilton": "ph"}
    assert second["new_plan"] == {"euclid": "pe2"}
    assert second["replanned"] == ["euclid"]
    assert second["dropped"] == ["hamilton"] and second["kept"] == []
    (stop,) = _events(run_dir, "stop")
    assert stop["payload"]["robot"] == "hamilton"
    assert stop["payload"]["reason"] == "not in the new plan"


def test_an_additional_task_keeps_the_busy_robots(h, run_dir):
    _start_mission(h)
    h.send("(visited-object o9)")  # touches nothing running
    h.plan(hilbert=("pi", 1, ["o9"]))
    d = _events(run_dir, "repair.decision")[-1]["payload"]
    assert d["kind"] == "additional task"
    assert d["kept"] == ["euclid", "hamilton"] and d["replanned"] == ["hilbert"]
    assert _events(run_dir, "pause") == []


def test_a_replan_request_and_its_failure(h, run_dir):
    _start_mission(h)
    h.node._replan_request_cb(String(data="scene changed: o2 moved 1.2 m"))
    (trigger,) = _events(run_dir, "repair.trigger")
    t = trigger["payload"]
    assert t["reason"] == "scene changed: o2 moved 1.2 m"
    assert t["replanning"] is True and t["epoch"] == 0 and t["map_version"] == 0
    assert {e["payload"]["robot"] for e in _events(run_dir, "pause")} == {
        "euclid",
        "hamilton",
    }

    h.failed("no plan achieves the goal")
    d = _events(run_dir, "repair.decision")[-1]["payload"]
    assert d["outcome"] == "failed" and d["kind"] == "repair"
    assert d["trigger"] == "scene changed: o2 moved 1.2 m"
    assert d["error"] == "no plan achieves the goal"
    assert d["old_plan"] == {"euclid": "pe", "hamilton": "ph"}
    stops = _events(run_dir, "stop")
    assert sorted(e["payload"]["robot"] for e in stops) == ["euclid", "hamilton"]
    alert = _events(run_dir, "gm.alert")[-1]["payload"]["text"]
    assert alert.startswith("Cannot repair the plan")


def test_resume_on_plan_failure_logs_resumes(tmp_path, run_dir):
    h = Harness(tmp_path, resume_on_plan_failure=True)
    try:
        _start_mission(h)
        h.send("(and (visited-object o1) (visited-object o4))")
        h.failed("no plan achieves the goal")
        resumes = _events(run_dir, "resume")
        assert sorted(e["payload"]["robot"] for e in resumes) == ["euclid", "hamilton"]
    finally:
        h.close()


def test_robot_states_are_logged_as_changes(h, run_dir):
    _start_mission(h)
    h.done("euclid", "pe")
    states = _events(run_dir, "gm.robot_state")
    euclid = [
        e["payload"]["state"] for e in states if e["payload"]["robot"] == "euclid"
    ]
    # one event per change (plan, coverage, an action done), not per publish
    assert all(a != b for a, b in zip(euclid, euclid[1:]))
    assert euclid[-1]["done"] == 1 and euclid[-1]["plan_id"] == "pe"
    h.done("euclid", "pe")  # finished: its entry is cleared
    last = _events(run_dir, "gm.robot_state")[-1]
    assert last["payload"]["robot"] == "euclid" and last["payload"]["cleared"] is True
    assert last["plan_id"] == "pe"


def test_a_stall_is_logged(tmp_path, run_dir):
    h = Harness(tmp_path, progress_timeout_s=30.0)
    try:
        _start_mission(h)
        h.t += 31.0
        h.node._check_progress()
        stalls = _events(run_dir, "progress.stall")
        assert {e["payload"]["robot"] for e in stalls} == {"euclid", "hamilton"}
        assert {e["plan_id"] for e in stalls} == {"pe", "ph"}
    finally:
        h.close()
