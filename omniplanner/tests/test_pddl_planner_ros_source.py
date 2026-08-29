"""Drift guard for the *production* (ROS) PDDL plan compiler.

``omniplanner_ros.pddl_planner_ros.compile_pddl_plan`` is the code that
actually runs on the robot, and it carried the same bug that
``dsg_pddl.dsg_pddl_planning_compile`` (the non-ROS mirror) already fixed:
``object_class`` was assigned only inside the ``pick-object`` branch and then
*read* by ``place-object``. A place without a preceding pick raised
``UnboundLocalError``; a pick A / place B plan stamped B with A's class.

The ROS module cannot be imported here -- it pulls in ``rclpy``,
``geometry_msgs``, ``spark_config`` and ``omniplanner_msgs``, none of which
exist in a plain virtualenv -- so this test reads its source instead. It is
deliberately narrow: it asserts the inline computation is gone and that both
branches route through the shared ``_object_class`` helper.
"""

import ast
from pathlib import Path

import pytest

ROS_MODULE = (
    Path(__file__).resolve().parents[2]
    / "omniplanner_ros"
    / "src"
    / "omniplanner_ros"
    / "pddl_planner_ros.py"
)


@pytest.fixture(scope="module")
def ros_source() -> str:
    if not ROS_MODULE.is_file():
        pytest.skip(f"{ROS_MODULE} not present in this checkout")
    return ROS_MODULE.read_text()


def test_ros_compiler_parses(ros_source):
    ast.parse(ros_source)


def test_ros_compiler_imports_the_shared_helper(ros_source):
    assert (
        "from dsg_pddl.dsg_pddl_planning_compile import _object_class" in ros_source
    ), "the ROS compiler must reuse the pure _object_class helper"


def test_ros_compiler_has_no_inline_object_class(ros_source):
    assert 'object_class = ""' not in ros_source, (
        "inline object_class computation is back in pddl_planner_ros -- it "
        "leaks across the pick-object / place-object branches"
    )


def test_both_pick_and_place_look_up_their_own_class(ros_source):
    """Every Pick/Place constructed by the ROS compiler calls the helper."""
    tree = ast.parse(ros_source)
    constructed = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("Pick", "Place")
    ]
    assert len(constructed) >= 2, "expected at least one Pick and one Place"
    for call in constructed:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        assert "object_class" in kwargs, f"{call.func.id} built without object_class"
        value = kwargs["object_class"]
        assert (
            isinstance(value, ast.Call)
            and getattr(value.func, "id", None) == "_object_class"
        ), (
            f"{call.func.id}.object_class must be _object_class(context, symbol), "
            f"not {ast.dump(value)}"
        )
