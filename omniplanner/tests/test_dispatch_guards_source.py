"""Dispatch-guard refusal reasons (omniplanner_node.py).

The ROS node module cannot be imported without rclpy & co. (see
test_pddl_planner_ros_source.py), so the two pure functions are extracted from
its source and run on their own.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

NODE = (
    Path(__file__).resolve().parents[2]
    / "omniplanner_ros"
    / "src"
    / "omniplanner_ros"
    / "omniplanner_node.py"
)
FUNCS = ("evaluate_dispatch_guards", "format_refusals")


@pytest.fixture(scope="module")
def ns():
    if not NODE.is_file():
        pytest.skip(f"{NODE} not present in this checkout")
    tree = ast.parse(NODE.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS]
    assert {n.name for n in body} == set(FUNCS)
    out: dict = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(NODE), "exec"), out)
    return out


def _guards(**kw):
    base = dict(
        estop_known=True, estop_pressed=False, power_known=True, powered_on=True,
        battery_known=True, battery_percent=80.0, lease_known=True, lease_owned=True,
    )  # fmt: skip
    base.update(kw)
    return SimpleNamespace(**base)


def test_lease_refusal_names_the_owner(ns):
    ev = ns["evaluate_dispatch_guards"]
    g = _guards(lease_owned=False, lease_owner="bosdyn.android.spotapp tablet")
    assert ev(g, 0.5) == (False, "lease_not_owned(held by bosdyn.android.spotapp tablet)")


def test_lease_refusal_without_owner_field(ns):
    # a robot whose RuntimeGuardsMsg predates lease_owner
    assert ns["evaluate_dispatch_guards"](_guards(lease_owned=False), 0.5) == (
        False,
        "lease_not_owned",
    )


def test_owned_lease_passes(ns):
    assert ns["evaluate_dispatch_guards"](_guards(lease_owner="understanding-x"), 0.5) == (
        True,
        None,
    )


def test_format_refusals(ns):
    f = ns["format_refusals"]
    assert f({}) == "plan refused by runtime dispatch guards"
    assert f({"hamilton": "lease_not_owned(held by x)", "euclid": "not_powered"}) == (
        "plan refused by runtime dispatch guards: "
        "euclid: not_powered; hamilton: lease_not_owned(held by x)"
    )
