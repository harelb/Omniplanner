"""Regression tests for dsg_pddl.dsg_pddl_planning_compile.

The bug: ``compile_pddl_plan_pure`` computed ``object_class`` only inside the
``pick-object`` branch and then *read* that local in the ``place-object``
branch. So a plan that placed without a preceding pick raised
``UnboundLocalError``, and a plan that picked A then placed B stamped B's
``Place`` action with A's semantic class. Both cases now go through the pure
``_object_class(context, symbol)`` helper.

``robot_executor_interface`` (spot_tools) is a colcon package and is not
necessarily importable from a plain virtualenv, so the compile tests run
against lightweight stand-ins monkeypatched into the module. One extra test
exercises the real action-description dataclasses and skips when they are
absent.

    PYTHONPATH=src python -m pytest tests/test_planning_compile.py
"""

from dataclasses import dataclass, field

import numpy as np
import pytest
from omniplanner.omniplanner import SymbolicContext

from dsg_pddl.dsg_pddl_planning import PddlPlan
from dsg_pddl.dsg_pddl_planning_compile import (
    _object_class,
    compile_pddl_plan_pure,
    ensure_3d,
)

# ---------------------------------------------------------------------------
# the pure helper -- no robot_executor_interface, no ROS
# ---------------------------------------------------------------------------


def test_object_class_reads_semantic_label():
    context = {"o1": {"semantic_label": "mug"}}
    assert _object_class(context, "o1") == "mug"


def test_object_class_absent_symbol_is_empty():
    assert _object_class({"o1": {"semantic_label": "mug"}}, "o2") == ""


def test_object_class_symbol_without_semantic_label_is_empty():
    assert _object_class({"o1": {"bounding_box": [0, 0]}}, "o1") == ""


def test_object_class_empty_or_none_context_is_empty():
    assert _object_class({}, "o1") == ""
    assert _object_class(None, "o1") == ""


def test_object_class_is_per_symbol():
    context = {"o1": {"semantic_label": "mug"}, "o2": {"semantic_label": "bottle"}}
    assert _object_class(context, "o1") == "mug"
    assert _object_class(context, "o2") == "bottle"


# ---------------------------------------------------------------------------
# compile_pddl_plan_pure, against stand-in action descriptions
# ---------------------------------------------------------------------------


@dataclass
class _StubAction:
    frame: str = ""
    object_class: str = ""
    robot_point: object = None
    object_point: object = None
    object_id: str = ""


@dataclass
class _StubSequence:
    plan_id: str = ""
    robot_name: str = ""
    actions: list = field(default_factory=list)


@pytest.fixture
def stub_actions(monkeypatch):
    """Stand in for robot_executor_interface's dataclasses.

    Keeps the compile-path tests runnable without the colcon workspace while
    still exercising the real control flow in compile_pddl_plan_pure.
    """
    import dsg_pddl.dsg_pddl_planning_compile as mod

    monkeypatch.setattr(mod, "ActionSequence", _StubSequence)
    monkeypatch.setattr(mod, "Pick", _StubAction)
    monkeypatch.setattr(mod, "Place", _StubAction)
    return mod


def _plan(context, symbolic_actions, parameterized_actions):
    return SymbolicContext(
        context,
        PddlPlan(
            domain=None,
            symbolic_actions=symbolic_actions,
            parameterized_actions=parameterized_actions,
            symbols={},
        ),
    )


_POINTS = [np.array([0.0, 0.0]), np.array([1.0, 2.0])]


def test_place_without_preceding_pick_does_not_raise(stub_actions):
    """The UnboundLocalError regression: place with no pick before it."""
    plan = _plan(
        {"o5": {"semantic_label": "backpack"}},
        [("place-object", "o5")],
        [_POINTS],
    )
    seq = compile_pddl_plan_pure(plan, "plan-1", "euclid", "map")

    assert len(seq.actions) == 1
    place = seq.actions[0]
    # ... and it carries the class from the context, not "" and not a crash.
    assert place.object_class == "backpack"
    assert place.object_id == "o5"


def test_place_without_preceding_pick_and_no_context_entry(stub_actions):
    plan = _plan({}, [("place-object", "o5")], [_POINTS])
    seq = compile_pddl_plan_pure(plan, "plan-1", "euclid", "map")
    assert seq.actions[0].object_class == ""


def test_pick_a_then_place_b_uses_bs_class(stub_actions):
    """The stale-class regression: place must not inherit the pick's class."""
    context = {
        "o1": {"semantic_label": "mug"},
        "o2": {"semantic_label": "bottle"},
    }
    plan = _plan(
        context,
        [("pick-object", "o1"), ("place-object", "o2")],
        [_POINTS, _POINTS],
    )
    seq = compile_pddl_plan_pure(plan, "plan-1", "euclid", "map")

    pick, place = seq.actions
    assert pick.object_class == "mug"
    assert place.object_class == "bottle"


def test_pick_then_place_same_object_keeps_its_class(stub_actions):
    context = {"o1": {"semantic_label": "mug"}}
    plan = _plan(
        context,
        [("pick-object", "o1"), ("place-object", "o1")],
        [_POINTS, _POINTS],
    )
    seq = compile_pddl_plan_pure(plan, "plan-1", "euclid", "map")
    assert [a.object_class for a in seq.actions] == ["mug", "mug"]


def test_unknown_action_still_raises(stub_actions):
    plan = _plan({}, [("teleport", "o1")], [_POINTS])
    with pytest.raises(NotImplementedError):
        compile_pddl_plan_pure(plan, "plan-1", "euclid", "map")


def test_ensure_3d_pads_2d_points():
    assert np.allclose(ensure_3d(np.array([1.0, 2.0])), [1.0, 2.0, 0.0])
    assert np.allclose(ensure_3d(np.array([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# same thing against the real action descriptions, when they are available
# ---------------------------------------------------------------------------


def test_place_without_pick_with_real_action_descriptions():
    pytest.importorskip(
        "robot_executor_interface.action_descriptions",
        reason="spot_tools' robot_executor_interface is not on the path "
        "(colcon workspace not sourced)",
    )
    plan = _plan(
        {"o5": {"semantic_label": "backpack"}},
        [("place-object", "o5")],
        [_POINTS],
    )
    seq = compile_pddl_plan_pure(plan, "plan-1", "euclid", "map")
    assert seq.actions[0].object_class == "backpack"
