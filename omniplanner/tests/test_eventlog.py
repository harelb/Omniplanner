"""The mission event log writer: envelope, off without a run dir, redaction,
rotation, fsync throttling, and that it never raises."""

import json
import os

from omniplanner import eventlog
from omniplanner.eventlog import NULL_LOG, EventLog, redact, resolve_events_dir


def _lines(path):
    with open(path) as fo:
        return [json.loads(line) for line in fo]


def _env(tmp_path, **extra):
    env = {
        "ADT4_RUN_DIR": str(tmp_path / "runs" / "willow_20261111-185801_a3f2"),
        "ADT4_SESSION_ID": "s7",
        "ADT4_ROBOT_NAME": "willow",
        "ADT4_MACHINE": "willow-host",
    }
    env.update(extra)
    return env


def test_off_without_a_run_dir(tmp_path):
    log = EventLog("omniplanner", env={})
    assert not log.enabled
    assert log.emit("plan.request", {"a": 1}) is None
    assert resolve_events_dir({}) is None
    assert NULL_LOG.emit("x") is None


def test_envelope_and_path(tmp_path):
    env = _env(tmp_path)
    log = EventLog("omniplanner", env=env, mission_id="m1")
    log.ros_clock = lambda: 12.5
    log.emit("plan.request", {"goal": "(visited-object o1)"}, goal_id="g1")
    log.emit("plan.solved", {"n": 2}, plan_id="p1")
    log.close()
    path = os.path.join(env["ADT4_RUN_DIR"], "events", "omniplanner.jsonl")
    assert log.path == path
    first, second = _lines(path)
    assert set(first) == {
        "v",
        "t_wall",
        "t_mono",
        "t_ros",
        "machine",
        "robot",
        "run_id",
        "session_id",
        "mission_id",
        "goal_id",
        "plan_id",
        "source",
        "type",
        "seq",
        "payload",
    }
    assert first["v"] == 1 and first["t_ros"] == 12.5
    assert first["machine"] == "willow-host" and first["robot"] == "willow"
    assert first["run_id"] == "willow_20261111-185801_a3f2"  # the run dir's name
    assert first["session_id"] == "s7" and first["mission_id"] == "m1"
    assert first["source"] == "omniplanner" and first["type"] == "plan.request"
    assert first["goal_id"] == "g1" and second["plan_id"] == "p1"
    assert (first["seq"], second["seq"]) == (1, 2)
    assert first["payload"] == {"goal": "(visited-object o1)"}


def test_run_id_and_events_dir_overrides(tmp_path):
    env = _env(tmp_path, ADT4_RUN_ID="r42", ADT4_EVENTS_DIR=str(tmp_path / "ev"))
    log = EventLog("goal_manager", env=env)
    assert log.path == str(tmp_path / "ev" / "goal_manager.jsonl")
    assert log.run_id == "r42"


def test_appends_across_writers(tmp_path):
    env = _env(tmp_path)
    for _ in range(2):  # a restarted node appends, never truncates
        log = EventLog("omniplanner", env=env)
        log.emit("a")
        log.close()
    assert len(_lines(log.path)) == 2


def test_redaction_by_key():
    data = {
        "api_key": "sk-1",
        "nested": [{"OPENAI_API_KEY": "sk-2", "password": "pw", "ok": 1}],
        "Authorization": "Bearer x",
        "neo4j_password": "y",
        "goal": "(visited-object o1)",
    }
    out = redact(data)
    assert (
        out["api_key"]
        == out["Authorization"]
        == out["neo4j_password"]
        == (eventlog.REDACTED)
    )
    assert out["nested"][0] == {
        "OPENAI_API_KEY": eventlog.REDACTED,
        "password": eventlog.REDACTED,
        "ok": 1,
    }
    assert out["goal"] == "(visited-object o1)"
    assert data["api_key"] == "sk-1"  # a copy


def test_redacted_in_the_file(tmp_path):
    log = EventLog("omniplanner", env=_env(tmp_path))
    log.emit("llm.call", {"token": "t", "model": "m"})
    log.close()
    (event,) = _lines(log.path)
    assert event["payload"] == {"token": eventlog.REDACTED, "model": "m"}


def test_numpy_sets_and_nan_are_written(tmp_path):
    import numpy as np

    log = EventLog("x", events_dir=str(tmp_path / "events"))
    payload = {
        "path": np.array([[0.0, 1.0], [np.inf, 2.0]]),
        "n": np.int64(3),
        "s": {"b", "a"},
        "bad": float("nan"),
    }
    assert log.emit("plan.solved", payload) is not None
    log.close()
    (event,) = _lines(log.path)
    assert event["payload"] == {
        "path": [[0.0, 1.0], [None, 2.0]],
        "n": 3,
        "s": ["a", "b"],
        "bad": None,
    }


def test_rotation(tmp_path):
    log = EventLog("omniplanner", env=_env(tmp_path), max_bytes=1000)
    for i in range(30):
        log.emit("e", {"i": i, "pad": "x" * 100})
    log.close()
    events_dir = os.path.dirname(log.path)
    names = sorted(os.listdir(events_dir))
    assert "omniplanner.jsonl" in names and "omniplanner.1.jsonl" in names
    assert all(os.path.getsize(os.path.join(events_dir, n)) <= 1000 for n in names)
    seqs = []
    rotated = sorted(n for n in names if n != "omniplanner.jsonl")
    for n in sorted(rotated, key=lambda n: int(n.split(".")[1])):
        seqs += [e["seq"] for e in _lines(os.path.join(events_dir, n))]
    seqs += [e["seq"] for e in _lines(log.path)]
    assert seqs == list(range(1, 31))  # nothing lost, in order


def test_fsync_at_most_every_period(tmp_path, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(eventlog.os, "fsync", lambda fd: calls.append(fd) or real(fd))
    now = {"t": 100.0}
    log = EventLog(
        "omniplanner", env=_env(tmp_path), fsync_s=2.0, mono_clock=lambda: now["t"]
    )
    for _ in range(5):
        log.emit("e")
    assert len(calls) == 1  # the first write; the rest within 2 s
    now["t"] += 2.5
    log.emit("e")
    assert len(calls) == 2
    # line-buffered: every event is in the file already
    assert len(_lines(log.path)) == 6
    log.close()


def test_never_raises(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    log = EventLog("omniplanner", events_dir=str(blocker / "events"))
    assert log.emit("plan.request", {"a": 1}) is None
    assert log.emit("plan.request", {"a": 1}) is None
    assert log.errors == 2

    good = EventLog("omniplanner", env=_env(tmp_path))
    good.ros_clock = lambda: 1 / 0  # a broken clock: t_ros is null
    assert good.emit("e")["t_ros"] is None
    good.close()
