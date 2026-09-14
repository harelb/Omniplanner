"""Planner configurations loaded only when selected.

A PDDL-only launch must not require optional language/multi-agent dependencies.
"""
from importlib import import_module

_CONFIGS = {
    "GotoPointsConfig": "goto_points_ros",
    "LanguagePlannerConfig": "language_planner_ros",
    "MultiRobotPddlConfig": "multirobot_ros",
    "PddlConfig": "pddl_planner_ros",
    "MultirobotPddlConfig": "pddl_planner_ros_llm_multiagent",
    "TspConfig": "tsp_ros",
}
__all__ = list(_CONFIGS)


def __getattr__(name):
    if name not in _CONFIGS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_CONFIGS[name]}"), name)
    globals()[name] = value
    return value
