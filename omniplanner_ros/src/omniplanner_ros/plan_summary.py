"""What a compiled plan amounts to, for the event log: its actions and how far
each robot drives.

Duck-typed on purpose, so it reads every plan the node publishes: a Spot's
``ActionSequence`` (Follow / Gaze / Pick / Place) and a Husky's
``nav_msgs/Path``.

Path length is measured on the compiled plan, i.e. the geometry the robot is
sent, not on the symbolic plan: for an ActionSequence it is the sum over its
FOLLOW actions of the polyline length of ``path2d`` (consecutive waypoints, in
x/y, metres in the plan frame); for a Path it is the polyline through its
poses. Travel the executor adds on its own (from where the robot stands to a
path's first waypoint, or to a Gaze/Pick/Place ``robot_point``) is not in it.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

# a FOLLOW / PATH polyline in the event is downsampled to at most this many
# points (cm), enough to draw the plan in the mission replay
MAX_PATH_POINTS = 40


def downsample(points, max_points: int = MAX_PATH_POINTS) -> List[List[float]]:
    """At most ``max_points`` [x, y] rows of a polyline (first and last kept)."""
    if points is None:
        return []
    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or len(xy) == 0 or xy.shape[1] < 2:
        return []
    if len(xy) > max_points:
        idx = np.unique(np.linspace(0, len(xy) - 1, max_points).round().astype(int))
        xy = xy[idx]
    return [[round(float(x), 2), round(float(y), 2)] for x, y in xy[:, :2]]


def polyline_length(points) -> float:
    """Length of a polyline of [x, y, ...] rows, in x/y only."""
    if points is None:
        return 0.0
    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or len(xy) < 2 or xy.shape[1] < 2:
        return 0.0
    return float(np.sum(np.hypot(*np.diff(xy[:, :2], axis=0).T)))


def _path_points(plan):
    return [(p.pose.position.x, p.pose.position.y) for p in plan.poses]


def is_path(plan) -> bool:
    return hasattr(plan, "poses") and not hasattr(plan, "actions")


def action_type(action) -> str:
    return type(action).__name__.upper()


def path_length_m(plan) -> float:
    """Metres the compiled plan drives (see the module doc)."""
    if is_path(plan):
        return polyline_length(_path_points(plan))
    return sum(
        polyline_length(getattr(a, "path2d", None))
        for a in getattr(plan, "actions", None) or []
        if action_type(a) == "FOLLOW"
    )


def describe_action(action) -> dict:
    """One action, compactly: its type and target, or a FOLLOW's length."""
    kind = action_type(action)
    out = {"type": kind}
    if kind == "FOLLOW":
        path = getattr(action, "path2d", None)
        out["n_points"] = 0 if path is None else int(len(path))
        out["len_m"] = round(polyline_length(path), 2)
        out["path"] = downsample(path)
    else:
        for key in ("object_id", "object_class"):
            value = getattr(action, key, "")
            if value:
                out[key] = value
    task_id = getattr(action, "task_id", "")
    if task_id:
        out["task_id"] = task_id
    return out


def describe_plan(plan) -> List[dict]:
    if is_path(plan):
        points = _path_points(plan)
        if not points:
            return []
        return [
            {
                "type": "PATH",
                "n_points": len(points),
                "len_m": round(polyline_length(points), 2),
                "path": downsample(points),
            }
        ]
    return [describe_action(a) for a in getattr(plan, "actions", None) or []]


def summarize(plan_dict) -> dict:
    """{plan_ids, n_actions, actions, path_len_m} of {robot -> compiled plan}."""
    plan_ids: Dict[str, str] = {}
    n_actions: Dict[str, int] = {}
    actions: Dict[str, List[dict]] = {}
    path_len: Dict[str, float] = {}
    for robot, plan in sorted((plan_dict or {}).items()):
        plan_ids[robot] = getattr(plan, "plan_id", "") or ""
        steps = describe_plan(plan)
        actions[robot] = steps
        n_actions[robot] = len(steps)  # a Husky's Path is one PATH action
        path_len[robot] = round(path_length_m(plan), 3)
    return {
        "plan_ids": plan_ids,
        "n_actions": n_actions,
        "n_actions_total": sum(n_actions.values()),
        "actions": actions,
        "path_len_m": path_len,
        "path_len_m_total": round(sum(path_len.values()), 3),
    }


def shared_plan_id(plan_dict) -> str:
    """The plan's id: the robots' common one, else the first robot's."""
    ids = [getattr(p, "plan_id", "") or "" for _, p in sorted(plan_dict.items())]
    ids = [i for i in ids if i]
    return ids[0] if ids else ""
