"""events/omniplanner.jsonl: what the omniplanner node logs about each goal
(plan.request / plan.solved / plan.failed / plan.published), the path length
it measures on the compiled plan, and the mission id it picks up.

No ROS graph: the node is built without __init__ and its publishers faked, as
in test_willow_omniplanner_node."""

import json
import os
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from robot_executor_interface.action_descriptions import (
    ActionSequence,
    Follow,
    Gaze,
    Pick,
)

from omniplanner import solve_stats

node_mod = pytest.importorskip("omniplanner_ros.omniplanner_node")
from omniplanner_ros import eventlog_ros, plan_summary  # noqa: E402

N = node_mod.OmniPlannerRos


def _husky_path(points):
    path = Path()
    for x, y in points:
        p = PoseStamped()
        p.pose.position.x, p.pose.position.y = float(x), float(y)
        path.poses.append(p)
    return path


def _plan_fixture():
    """euclid: a 3 m + 4 m leg, a gaze, then a 2 m leg; gauss: a 10 m Path."""
    return {
        "euclid": ActionSequence(
            "plan-1",
            "euclid",
            [
                Follow("map", np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])),
                Gaze("map", np.zeros(3), np.ones(3), "o12"),
                Follow("map", np.array([[3.0, 4.0], [3.0, 6.0]])),
                Pick("map", "mug", np.zeros(3), np.ones(3), "o3"),
            ],
        ),
        "gauss": _husky_path([(0, 0), (6, 0), (6, 8)]),
    }


class FakePddlPlugin:
    """Named like the PDDL plugins, so its engine is FD."""

    def get_plan_callback(self):
        return object, "pddl_goal", lambda msg, poses: "request"

    def get_plugin_feedback(self, node):
        return None


class FakeGrstapsPlugin(FakePddlPlugin):
    pass


def _node(monkeypatch, plan_dict, plugin=None, pipeline=None):
    node = N.__new__(N)
    sent = {"plans": [], "failed": []}
    handlers = []
    node.__dict__.update(
        goal_count=0,
        goal_count_lock=threading.Lock(),
        planning_lock=threading.Lock(),
        current_planner_lock=threading.Lock(),
        plan_time_start_lock=threading.Lock(),
        dsg_lock=threading.Lock(),
        current_planner=None,
        plan_time_start=None,
        dsg_last=object(),
        dsg_frame="map",
        goal_group=None,
        feedback=NS(plugin_feedback_collectors={}),
        robot_adaptors={
            r: NS(publish_plan=lambda m, r=r: sent["plans"].append(r))
            for r in plan_dict
        },
        plan_failed_pub=NS(publish=lambda m: sent["failed"].append(m.data)),
        plan_robots_pub=NS(publish=lambda m: None),
        plan_status_pub=NS(publish=lambda m: None),
        compiled_plan_viz_pub=NS(publish=lambda m: None),
    )
    node.get_logger = lambda: NS(info=print, warning=print, error=print)
    node.get_robot_poses = lambda frame: {
        r: np.zeros(4) if r != "hilbert" else None for r in [*plan_dict, "hilbert"]
    }
    node.create_subscription = (
        lambda t, topic, cb, q, callback_group=None: handlers.append(cb)
    )
    monkeypatch.setattr(
        node_mod, "full_planning_pipeline", pipeline or (lambda *a: None)
    )
    monkeypatch.setattr(node_mod, "compile_plan", lambda *a: None)
    monkeypatch.setattr(node_mod, "collect_plans", lambda p: dict(plan_dict))
    monkeypatch.setattr(node_mod, "to_msg", lambda p: p)
    monkeypatch.setattr(node_mod, "to_viz_msg", lambda p, r: None)
    N.register_plugin(node, "multi_robot_pddl_constrained", plugin or FakePddlPlugin())
    return handlers[0], sent, node


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "willow_20261111-185801_a3f2"
    for name in ("ADT4_EVENTS_DIR", "ADT4_RUN_ID", "ADT4_MISSION_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ADT4_RUN_DIR", str(run))
    monkeypatch.setenv("ADT4_SESSION_ID", "sess1")
    return run


def _events(run, source="omniplanner"):
    path = run / "events" / f"{source}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _goal_msg():
    return NS(
        goal=NS(robot_id="willow", pddl_goal="(visited-object o12)"),
        constraints=[NS(predicate="forbidden-poi", symbols=["p7"])],
    )


# ---------- path length on a compiled plan ----------


def test_path_length_of_a_compiled_plan():
    plans = _plan_fixture()
    assert plan_summary.path_length_m(plans["euclid"]) == pytest.approx(9.0)
    assert plan_summary.path_length_m(plans["gauss"]) == pytest.approx(14.0)
    assert plan_summary.path_length_m(ActionSequence("p", "r", [])) == 0.0
    assert plan_summary.path_length_m(_husky_path([])) == 0.0
    summary = plan_summary.summarize(plans)
    assert summary["n_actions"] == {"euclid": 4, "gauss": 1}
    assert [a["type"] for a in summary["actions"]["euclid"]] == [
        "FOLLOW",
        "GAZE",
        "FOLLOW",
        "PICK",
    ]
    assert summary["actions"]["euclid"][1]["object_id"] == "o12"
    assert summary["actions"]["euclid"][0]["len_m"] == 7.0
    assert summary["path_len_m_total"] == pytest.approx(23.0)


# ---------- plan.request / plan.solved / plan.published ----------


def test_a_solved_goal_logs_request_solved_and_published(run_dir, monkeypatch):
    problem = run_dir / "omniplanner" / "omniplanner_problem_x.pddl"

    def pipeline(*a):
        solve_stats.record("FD", ok=True, seconds=0.25, problem_file=str(problem))

    handle, sent, _ = _node(monkeypatch, _plan_fixture(), pipeline=pipeline)
    handle(_goal_msg())
    assert sorted(sent["plans"]) == ["euclid", "gauss"]

    events = _events(run_dir)
    types = [e["type"] for e in events]
    assert types == ["plan.request", "plan.solved", "plan.published", "plan.published"]
    request, solved = events[0], events[1]
    assert all(e["session_id"] == "sess1" for e in events)
    assert all(e["run_id"] == "willow_20261111-185801_a3f2" for e in events)
    assert request["payload"]["goal"]["goal"]["pddl_goal"] == "(visited-object o12)"
    assert request["payload"]["engine"] == "FD"
    assert request["payload"]["robots"] == ["euclid", "gauss"]
    assert request["payload"]["robots_without_pose"] == ["hilbert"]
    assert request["payload"]["constraints"] == [
        {"predicate": "forbidden-poi", "symbols": ["p7"]}
    ]

    p = solved["payload"]
    assert solved["plan_id"] == p["plan_id"] == "plan-1"
    assert p["engine"] == "FD"
    assert p["solve_s"] >= 0 and p["solver_s"] == 0.25
    assert p["makespan"] is None  # FD has none
    assert p["n_actions"] == {"euclid": 4, "gauss": 1}
    assert p["path_len_m"] == {"euclid": pytest.approx(9.0), "gauss": 14.0}
    assert p["pddl"] == {"$file": "omniplanner/omniplanner_problem_x.pddl"}
    assert p["request"] == request["payload"]["request"] == 1

    published = {e["payload"]["robot"]: e for e in events[2:]}
    assert published["euclid"]["plan_id"] == "plan-1"
    assert published["gauss"]["payload"]["plan_id"] == ""  # a Path has none


def test_a_grstaps_plan_logs_its_makespan_and_solver_time(run_dir, monkeypatch):
    def pipeline(*a):
        solve_stats.record(
            "GRSTAPS", ok=True, seconds=3.5, makespan=42.0, scenario_dir="/w/s1"
        )

    handle, _, _ = _node(
        monkeypatch, _plan_fixture(), plugin=FakeGrstapsPlugin(), pipeline=pipeline
    )
    handle(_goal_msg())
    (solved,) = [e for e in _events(run_dir) if e["type"] == "plan.solved"]
    p = solved["payload"]
    assert p["engine"] == "GRSTAPS"
    assert p["makespan"] == 42.0 and p["solver_s"] == 3.5
    assert p["pddl"] == {"$file": "/w/s1/problem.pddl"}


def test_a_failed_goal_logs_the_error_and_the_problem_file(run_dir, monkeypatch):
    failed_file = run_dir / "omniplanner" / "pddl_problem_debugging_t_abc.pddl"

    def pipeline(*a):
        solve_stats.record(
            "FD", ok=False, seconds=1.0, failed_problem_file=str(failed_file)
        )
        raise ValueError("no plan achieves the goal")

    handle, sent, _ = _node(monkeypatch, _plan_fixture(), pipeline=pipeline)
    handle(_goal_msg())
    assert sent["plans"] == []
    types = [e["type"] for e in _events(run_dir)]
    assert types == ["plan.request", "plan.failed"]
    p = _events(run_dir)[1]["payload"]
    assert p["error"] == "no plan achieves the goal"
    assert p["error_type"] == "ValueError"
    assert p["pddl"] == {"$file": "omniplanner/pddl_problem_debugging_t_abc.pddl"}


def test_a_goal_that_already_holds_is_a_failure_with_satisfied(run_dir, monkeypatch):
    handle, _, _ = _node(monkeypatch, {"euclid": ActionSequence("p", "euclid", [])})
    handle(_goal_msg())
    (failed,) = [e for e in _events(run_dir) if e["type"] == "plan.failed"]
    assert failed["payload"]["satisfied"] is True


def test_no_run_dir_writes_nothing_and_plans_the_same(tmp_path, monkeypatch):
    monkeypatch.delenv("ADT4_RUN_DIR", raising=False)
    monkeypatch.delenv("ADT4_EVENTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    handle, sent, _ = _node(monkeypatch, _plan_fixture())
    handle(_goal_msg())
    assert sorted(sent["plans"]) == ["euclid", "gauss"]
    assert os.listdir(tmp_path) == []


def test_a_broken_event_log_never_stops_a_plan(run_dir, monkeypatch):
    handle, sent, node = _node(monkeypatch, _plan_fixture())
    node.events = NS(emit=lambda *a, **k: 1 / 0)
    handle(_goal_msg())
    assert sorted(sent["plans"]) == ["euclid", "gauss"]


# ---------- the mission id ----------


@pytest.mark.parametrize(
    "data,expected",
    [
        ('{"mission_id": "m1", "state": "open", "session_id": "s2"}', "m1"),
        ('{"mission_id": "m1", "state": "closed"}', ""),
        ("m20261111-191900_7c1e", "m20261111-191900_7c1e"),
        ("", ""),
        ("{not json", ""),
    ],
)
def test_mission_messages(data, expected):
    assert eventlog_ros.parse_mission_id(data) == expected


def test_the_mission_id_and_session_reach_the_envelope(run_dir):
    from omniplanner.eventlog import EventLog

    log = EventLog("omniplanner")
    watcher = eventlog_ros.MissionWatcher.__new__(eventlog_ros.MissionWatcher)
    watcher.log = log
    watcher.on_mission(
        NS(data='{"mission_id": "m9", "state": "open", "session_id": "s3"}')
    )
    log.emit("plan.request")
    log.close()
    seen, request = _events(run_dir)
    assert seen["type"] == "mission.seen"
    assert request["mission_id"] == "m9" and request["session_id"] == "s3"


# ---------- the repair node's own events ----------


def test_the_repair_node_logs_its_replan_trigger(run_dir):
    repair = pytest.importorskip("omniplanner_ros.omniplanner_repair_node")
    node = NS(
        _map_state=lambda: ("e1", 7),
        _map_version=lambda: 3,
        _plan_lock=threading.Lock(),
        _plan_robots=["euclid"],
        _plan_ids={"euclid": "plan-1"},
    )
    repair._log_replan_trigger(node, "scene changed: o2 moved")
    (trigger,) = _events(run_dir)
    assert trigger["type"] == "repair.trigger"
    assert trigger["payload"] == {
        "reason": "scene changed: o2 moved",
        "map_version": 7,
        "epoch": "e1",
        "generation": 3,
        "robots": ["euclid"],
        "plan_ids": {"euclid": "plan-1"},
    }
