"""The event log of a ROS node: its clock for ``t_ros``, and the mission id
from ``/global/mission/current`` once something publishes it.

See omniplanner.eventlog for the format; off unless ``ADT4_RUN_DIR`` is set.
"""

from __future__ import annotations

import json
import logging

from omniplanner.eventlog import NULL_LOG, EventLog

logger = logging.getLogger(__name__)

MISSION_TOPIC = "/global/mission/current"


def parse_mission(data: str) -> dict:
    """{mission_id, session_id} of a mission message: JSON
    {"mission_id", "state": "open"|"closed", "session_id"}, or a bare id.
    A closed (or empty) mission has mission_id ""."""
    text = (data or "").strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except ValueError:
            return {"mission_id": ""}
        out = {"mission_id": str(obj.get("mission_id") or obj.get("id") or "")}
        if obj.get("state") == "closed" or obj.get("active") is False:
            out["mission_id"] = ""
        if obj.get("session_id"):
            out["session_id"] = str(obj["session_id"])
        return out
    return {"mission_id": text}


def parse_mission_id(data: str) -> str:
    return parse_mission(data)["mission_id"]


class MissionWatcher:
    """Keeps the log's mission id (and session id, when the message has one)
    from the console's latched /global/mission/current (std_msgs/String,
    transient_local + reliable, depth 1). Until a message arrives the id is
    $ADT4_MISSION_ID, else ""."""

    def __init__(self, node, log, topic: str = MISSION_TOPIC):
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        self.log = log
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.sub = node.create_subscription(String, topic, self.on_mission, qos)

    def on_mission(self, msg) -> None:
        try:
            mission = parse_mission(getattr(msg, "data", ""))
            if mission.get("session_id"):
                self.log.session_id = mission["session_id"]
            if mission["mission_id"] != self.log.mission_id:
                self.log.mission_id = mission["mission_id"]
                self.log.emit("mission.seen", mission)
        except Exception as exc:  # never into the node
            logger.debug(f"bad mission message: {exc}")


def make_event_log(node, source: str):
    """The node's event log (NULL_LOG when off or on any error)."""
    try:
        log = EventLog(source)
        if not log.enabled:
            return NULL_LOG
        clock = node.get_clock()
        log.ros_clock = lambda: clock.now().nanoseconds / 1e9
        log.mission_watcher = MissionWatcher(node, log)
        node.get_logger().info(f"event log: {log.path}")
        return log
    except Exception as exc:
        logger.warning(f"no event log for {source}: {exc}")
        return NULL_LOG


def lazy_event_log(node, attr: str, source: str):
    """node.<attr>, made on first use: nodes built without __init__ (tests)
    get a plain EventLog with no ROS parts."""
    log = getattr(node, attr, None)
    if log is None:
        try:
            log = EventLog(source)
            if not log.enabled:
                log = NULL_LOG
        except Exception:
            log = NULL_LOG
        setattr(node, attr, log)
    return log


def emit_event(obj, source: str, type: str, payload=None, **ids) -> None:
    """Emit on obj.events (made on first use). Works on any object, also the
    stand-ins tests bind node methods to; never raises."""
    try:
        lazy_event_log(obj, "events", source).emit(type, payload, **ids)
    except Exception:
        pass


def msg_to_dict(msg):
    """A ROS message as plain data (anything else: its fields, or its str)."""
    try:
        from rosidl_runtime_py.convert import message_to_ordereddict

        return json.loads(json.dumps(message_to_ordereddict(msg), default=str))
    except Exception:
        pass
    fields = getattr(msg, "__dict__", None)
    if isinstance(fields, dict):
        return {k: msg_to_dict(v) for k, v in fields.items() if not k.startswith("_")}
    if isinstance(msg, (list, tuple)):
        return [msg_to_dict(v) for v in msg]
    if isinstance(msg, (str, int, float, bool)) or msg is None:
        return msg
    return str(msg)


def constraints_to_list(constraints):
    """[{predicate, symbols}] of ConstraintFact messages or dataclasses."""
    out = []
    for c in constraints or []:
        out.append(
            {
                "predicate": getattr(c, "predicate", ""),
                "symbols": list(getattr(c, "symbols", []) or []),
            }
        )
    return out
