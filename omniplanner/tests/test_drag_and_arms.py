"""Robot arms and dragging heavy objects, in both planners.

Spots (euclid, hamilton) have arms; a Husky (gauss) has none. A heavy object
(a bag, sandbag, box, ...) cannot be picked up: it is dragged -- start-drag /
release-drag in fast-downward, a drag-object task in GRSTAPS-X -- and only a
robot with an arm may do either. Light objects (a cone) are still picked and
placed.

The fast-downward tests need fast-downward on PATH (as the other planning tests
do). The GRSTAPS-X tests write the solver's input without running it; the one
that solves runs only with GRSTAPS_REAL_SOLVER=1 and itags plus a Gurobi
licence available.
"""

import json
import os
import shutil
from importlib.resources import as_file, files

import dsg_pddl.domains
import numpy as np
import pytest
import spark_dsg
from dsg_pddl.dsg_pddl_grounding_multirobot import apply_world_state, ground_problem
from dsg_pddl.dsg_pddl_planning import make_plan
from dsg_pddl.manipulation import (
    heavy_classes,
    is_heavy,
    label_is_heavy,
    robot_has_arm,
)
from dsg_pddl.pddl_grounding import MultiRobotPddlDomain, PddlGoal
from dsg_pddl.pddl_utils import lisp_string_to_ast

from omniplanner.grstaps_planner import (
    DRAG_DURATION_S,
    MANIP,
    SPECIES,
    TASK_ACTION_KINDS,
    GrstapsDomain,
    GrstapsGoal,
    _fleet_names,
    _parameterize_tasks,
    enumerate_tasks,
    render_domain_pddl,
    species_for,
)
from omniplanner.grstaps_planner import ground_problem as grstaps_ground_problem
from omniplanner.world_state import WorldState

DOMAIN = "RegionObjectRearrangementDomain_MultiRobot_FD_Explore.pddl"
NX, NY = 10, 6  # grid of places, 1 m apart
ROBOT_TYPES = {"euclid": "spot", "hamilton": "spot", "gauss": "husky"}
# gauss stands next to the bag and the cone: nearest, but it has no arm
ROBOTS = {
    "euclid": np.array([9.0, 5.0]),
    "hamilton": np.array([9.0, 0.0]),
    "gauss": np.array([1.0, 0.0]),
}
BAG, CONE, OTHER_BAG = "o0", "o1", "o2"
CONSOLE_TYPE_SPECIES = {"spot": "spot_arm", "simulated-spot": "spot_arm"}


def place(x, y):
    return f"t{y * NX + x}"


def _add_object(G, idx, xy, name="", label=None):
    a = spark_dsg.ObjectNodeAttributes()
    a.position = np.array([xy[0], xy[1], 0.0])
    a.name = name
    if label is not None:
        a.semantic_label = label
    G.add_node(spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("O", idx).value, a)


def grid_dsg():
    """10 x 6 mesh places, 4-connected; a bag, a cone and a second bag."""
    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(3, 1, spark_dsg.DsgLayers.MESH_PLACES)
    for y in range(NY):
        for x in range(NX):
            a = spark_dsg.PlaceNodeAttributes()
            a.position = np.array([float(x), float(y), 0.0])
            G.add_node(
                spark_dsg.DsgLayers.MESH_PLACES,
                spark_dsg.NodeSymbol("t", y * NX + x).value,
                a,
            )
    for y in range(NY):
        for x in range(NX):
            i = y * NX + x
            for j in ([i + 1] if x + 1 < NX else []) + ([i + NX] if y + 1 < NY else []):
                G.insert_edge(
                    spark_dsg.NodeSymbol("t", i).value,
                    spark_dsg.NodeSymbol("t", j).value,
                )
    _add_object(G, 0, (1.2, 0.3), "bag")
    _add_object(G, 1, (1.3, 1.2), "cone")
    _add_object(G, 2, (4.1, 4.2), "Duffel_Bag")
    return G


def _domain(scope="goal_relevant"):
    with as_file(files(dsg_pddl.domains).joinpath(DOMAIN)) as p:
        domain = MultiRobotPddlDomain(open(p).read())
    domain.scene_scope = scope
    return domain


def _grounded(G, goal, robot_types=ROBOT_TYPES, scope="goal_relevant", **kw):
    g = PddlGoal(pddl_goal=goal, robot_id="euclid", robot_types=robot_types, **kw)
    return ground_problem(_domain(scope), G, ROBOTS, g).value


def _init_facts(problem_str, predicate):
    init = problem_str.split("(:init", 1)[1].split("(:goal", 1)[0]
    return {
        tuple(f.split())
        for f in init.replace(")", "(").split("(")
        if f.strip().startswith(predicate + " ")
    }


# ---------------------------------------------------------------------------
# the rules
# ---------------------------------------------------------------------------


def test_arm_rule(monkeypatch):
    monkeypatch.delenv("OMNIPLANNER_NO_ARM_ROBOTS", raising=False)
    assert robot_has_arm("euclid", ROBOT_TYPES)
    assert not robot_has_arm("gauss", ROBOT_TYPES)
    for t in ("phoenix", "jackal", "uav", "drone", "quadrotor", "Husky"):
        assert not robot_has_arm("r", {"r": t})
    for t in ("spot-with-manipulator", "simulated-spot", "something-new", ""):
        assert robot_has_arm("r", {"r": t})
    # no types at all: every robot has an arm, as before
    assert robot_has_arm("gauss") and robot_has_arm("gauss", {})
    monkeypatch.setenv("OMNIPLANNER_NO_ARM_ROBOTS", "euclid, hilbert")
    assert not robot_has_arm("euclid", ROBOT_TYPES)
    assert not robot_has_arm("Hilbert")


def test_heavy_rule(monkeypatch):
    for v in ("OMNIPLANNER_HEAVY_CLASSES", "OMNIPLANNER_HEAVY_SYMBOLS"):
        monkeypatch.delenv(v, raising=False)
    assert "bag" in heavy_classes() and "trash can" in heavy_classes()
    for label in ("bag", "Bag", "red bag", "sandbags", "Duffel_Bag", "trash-can"):
        assert label_is_heavy(label), label
    for label in ("cone", "mug", "", "bagel"):
        assert not label_is_heavy(label), label
    G = grid_dsg()
    assert is_heavy(G, BAG) and is_heavy(G, "O0") and is_heavy(G, OTHER_BAG)
    assert not is_heavy(G, CONE)
    assert is_heavy(G, CONE, extra=["o1"])
    monkeypatch.setenv("OMNIPLANNER_HEAVY_SYMBOLS", "o1,o40")
    assert is_heavy(G, CONE)
    monkeypatch.delenv("OMNIPLANNER_HEAVY_SYMBOLS")
    monkeypatch.setenv("OMNIPLANNER_HEAVY_CLASSES", "cone")
    assert is_heavy(G, CONE) and not is_heavy(G, BAG)


def test_heavy_rule_reads_the_labelspace():
    G = grid_dsg()
    G.set_labelspace(spark_dsg.Labelspace({34: "sandbag", 5: "cone"}), 2, 0)
    _add_object(G, 7, (3.0, 3.0), "", label=34)
    _add_object(G, 8, (3.0, 2.0), "", label=5)
    assert is_heavy(G, "o7") and not is_heavy(G, "o8")


# ---------------------------------------------------------------------------
# fast-downward grounding
# ---------------------------------------------------------------------------


def test_goal_relevant_problem_has_arms_and_heavy_facts(monkeypatch):
    monkeypatch.delenv("OMNIPLANNER_NO_ARM_ROBOTS", raising=False)
    G = grid_dsg()
    rel = _grounded(
        G, f"(and (object-in-place {BAG} {place(9, 5)}) (object-in-place {CONE} t50))"
    )
    assert _init_facts(rel.problem_str, "has-arm") == {
        ("has-arm", "euclid"),
        ("has-arm", "hamilton"),
    }
    # the cone is light, and the second bag is not in the goal (nor the problem)
    assert _init_facts(rel.problem_str, "heavy") == {("heavy", BAG)}


def test_no_robot_types_keeps_every_arm():
    G = grid_dsg()
    rel = _grounded(G, f"(visited-object {CONE})", robot_types={})
    assert {f[1] for f in _init_facts(rel.problem_str, "has-arm")} == set(ROBOTS)


def test_full_problem_has_arms_and_heavy_facts():
    G = grid_dsg()
    full = _grounded(G, f"(visited-object {CONE})", scope="full")
    assert {f[1] for f in _init_facts(full.problem_str, "has-arm")} == {
        "euclid",
        "hamilton",
    }
    assert _init_facts(full.problem_str, "heavy") == {
        ("heavy", BAG),
        ("heavy", OTHER_BAG),
    }


def test_heavy_symbols_field_and_env(monkeypatch):
    G = grid_dsg()
    goal = f"(object-in-place {CONE} t50)"
    rel = _grounded(G, goal, heavy_symbols=["O1"])
    assert _init_facts(rel.problem_str, "heavy") == {("heavy", CONE)}
    monkeypatch.setenv("OMNIPLANNER_HEAVY_SYMBOLS", CONE)
    rel = _grounded(G, goal)
    assert _init_facts(rel.problem_str, "heavy") == {("heavy", CONE)}


def test_bag_is_dragged_by_a_spot_never_by_the_husky():
    G = grid_dsg()
    dest = place(9, 5)
    plan = make_plan(_grounded(G, f"(object-in-place {BAG} {dest})"), G)
    acts = plan.symbolic_actions
    names = [a[0] for a in acts]
    assert "start-drag" in names and "release-drag" in names
    assert "pick-object" not in names and "place-object" not in names
    start = next(a for a in acts if a[0] == "start-drag")
    release = next(a for a in acts if a[0] == "release-drag")
    assert start[1] in ("euclid", "hamilton") and start[2] == BAG
    assert release == ("release-drag", start[1], BAG, dest)
    assert names.index("start-drag") < names.index("release-drag")
    # the husky does nothing with its hands
    assert all(a[0] == "goto-poi" for a in acts if a[1] == "gauss")
    # like pick/place: [robot pose, the place it grasps / lets go at]
    i = names.index("release-drag")
    np.testing.assert_allclose(plan.parameterized_actions[i][1][:2], [9.0, 5.0])


def test_cone_is_picked_and_placed_by_a_spot():
    G = grid_dsg()
    plan = make_plan(_grounded(G, f"(object-in-place {CONE} t50)"), G)
    names = [a[0] for a in plan.symbolic_actions]
    assert "pick-object" in names and "place-object" in names
    assert "start-drag" not in names
    pick = next(a for a in plan.symbolic_actions if a[0] == "pick-object")
    assert pick[1] in ("euclid", "hamilton")


def test_no_arm_no_plan_says_why():
    G = grid_dsg()
    husky_only = {r: "husky" for r in ROBOTS}
    rel = _grounded(G, f"(object-in-place {CONE} t50)", robot_types=husky_only)
    assert "arm" in rel.failure_hint
    with pytest.raises(Exception):
        make_plan(rel, G)


def test_held_heavy_object_is_being_dragged():
    init = [("heavy", BAG), ("object-in-place", BAG, "t1")]
    facts = apply_world_state(
        init,
        WorldState(holding={"euclid": [BAG]}),
        [type("S", (), {"symbol": BAG})()],
        ["euclid"],
    )
    assert ("dragging", "euclid", BAG) in facts
    assert ("holding", "euclid", BAG) not in facts
    assert ("hand-full", "euclid") in facts
    assert ("object-in-place", BAG, "t1") not in facts


def test_domain_parses_with_the_new_actions():
    d = _domain()
    preds = {p[0] for p in d.predicates}
    assert {"has-arm", "heavy", "dragging"} <= preds
    assert "start-drag" in d.actions and "release-drag" in d.actions
    # pick-object now needs an arm and a light object
    pick = d.actions[d.actions.index("pick-object") + 4]
    assert ("has-arm", "?r") in pick and ("not", ("heavy", "?o")) in pick


# ---------------------------------------------------------------------------
# compiling fast-downward plans
# ---------------------------------------------------------------------------


def test_pure_compile_emits_drag_actions():
    pytest.importorskip("robot_executor_interface.action_descriptions")
    from dsg_pddl.dsg_pddl_planning import PddlPlan
    from dsg_pddl.dsg_pddl_planning_compile import compile_pddl_plan_pure

    from omniplanner.drag_actions import DragRelease, DragStart
    from omniplanner.omniplanner import SymbolicContext

    pts = [np.array([0.0, 0.0]), np.array([1.0, 2.0])]
    plan = SymbolicContext(
        {},
        PddlPlan(
            None,
            [("start-drag", BAG, "t1"), ("release-drag", BAG, "t59")],
            [pts, pts],
            {},
        ),
    )
    seq = compile_pddl_plan_pure(plan, "p", "euclid", "map")
    assert [type(a) for a in seq.actions] == [DragStart, DragRelease]
    assert [a.action_type for a in seq.actions] == ["DRAG_START", "DRAG_RELEASE"]
    assert seq.actions[0].object_id == BAG


def test_ros_compile_emits_drag_actions_and_messages():
    try:
        from robot_executor_interface_ros.action_descriptions_ros import to_msg

        from omniplanner_ros.pddl_planner_ros import (
            compile_pddl_plan,
            project_multirobot_plan,
        )
    except ImportError as exc:
        pytest.skip(f"ROS workspace not sourced: {exc}")
    from omniplanner.drag_actions import DragRelease, DragStart
    from omniplanner.omniplanner import SymbolicContext

    G = grid_dsg()
    plan = make_plan(_grounded(G, f"(object-in-place {BAG} {place(9, 5)})"), G)
    robot = next(a[1] for a in plan.symbolic_actions if a[0] == "start-drag")
    keep = [
        (s, p)
        for s, p in zip(plan.symbolic_actions, plan.parameterized_actions)
        if s[1] == robot
    ]
    plan.symbolic_actions = [s for s, _ in keep]
    plan.parameterized_actions = [p for _, p in keep]
    project_multirobot_plan(plan)
    seq = compile_pddl_plan(SymbolicContext({}, plan), "p", robot, "map")
    kinds = [type(a).__name__ for a in seq.actions]
    assert kinds.index("DragStart") < kinds.index("Follow", kinds.index("DragStart"))
    assert kinds[-1] == "DragRelease" and "Pick" not in kinds
    msgs = [to_msg(a) for a in seq.actions if isinstance(a, (DragStart, DragRelease))]
    assert [m.action_type for m in msgs] == ["DRAG_START", "DRAG_RELEASE"]
    assert msgs[0].object_id == BAG
    np.testing.assert_allclose([msgs[1].object_point.x, msgs[1].object_point.y], [9, 5])


# ---------------------------------------------------------------------------
# GRSTAPS-X
# ---------------------------------------------------------------------------


def test_husky_species_has_no_arm():
    assert species_for("husky") == "husky" and species_for("phoenix") == "husky"
    assert species_for("husky", type_overrides=CONSOLE_TYPE_SPECIES) == "husky"
    assert species_for("spot", type_overrides=CONSOLE_TYPE_SPECIES) == "spot_arm"
    assert SPECIES["husky"]["traits"][MANIP] == 0
    assert SPECIES["husky"]["speed"] == 0.8


def test_husky_joins_the_fleet_unless_disallowed():
    goal = GrstapsGoal(goal_points=[], robot_id="euclid", robot_types=ROBOT_TYPES)
    assert set(_fleet_names(ROBOTS, goal, GrstapsDomain(), quiet=True)) == set(ROBOTS)
    old = _fleet_names(ROBOTS, goal, GrstapsDomain(allow_husky=False), quiet=True)
    assert set(old) == {"euclid", "hamilton"}
    # a named override still puts it back
    back = _fleet_names(
        ROBOTS,
        goal,
        GrstapsDomain(allow_husky=False, robot_species={"gauss": "husky"}),
        quiet=True,
    )
    assert "gauss" in back


DURATIONS = {
    "visit-location": 5.0,
    "inspect-object": 6.0,
    "pick-object": 8.0,
    "place-object": 8.0,
}
VERTEX = {
    s: {"id": i, "x": float(i), "y": 0.0} for i, s in enumerate([BAG, CONE, "t59"])
}


def test_heavy_relocation_is_one_drag_task():
    goal = GrstapsGoal(
        goal_points=[],
        robot_id="euclid",
        manipulate_points=[BAG, CONE],
        manipulate_destinations={BAG: "t59", CONE: "t59"},
        heavy_points=[BAG],
    )
    tasks, of_symbol, precedence = enumerate_tasks(
        goal, DURATIONS, VERTEX, carry_extra_s={BAG: 4.0}
    )
    by = {t["name"]: t for t in tasks}
    assert set(by) == {
        f"drag-object {BAG} :: drag",
        f"relocate-object {CONE} :: relocate",
    }
    drag = by[f"drag-object {BAG} :: drag"]
    assert drag["desired_traits"][MANIP] == 1
    assert drag["duration"] == DRAG_DURATION_S + 4.0
    assert drag["terminal_configuration"]["id"] == VERTEX["t59"]["id"]
    assert precedence == []


def test_heavy_relocation_split_is_start_then_release():
    goal = GrstapsGoal(
        goal_points=[],
        robot_id="euclid",
        manipulate_points=[BAG],
        manipulate_destinations={BAG: "t59"},
        heavy_points=[BAG],
    )
    tasks, _, precedence = enumerate_tasks(
        goal, DURATIONS, VERTEX, atomic_manipulation=False
    )
    assert [t["name"] for t in tasks] == [
        f"start-drag {BAG} :: start_drag",
        f"release-drag {BAG} :: release_drag",
    ]
    assert precedence == [[0, 1]]
    assert all(t["desired_traits"][MANIP] == 1 for t in tasks)


def test_domain_pddl_has_drag_actions():
    atomic = render_domain_pddl(DURATIONS, True)
    assert "(:durative-action drag-object" in atomic and "(heavy ?l)" in atomic
    assert "(at start (light ?l))" in atomic
    split = render_domain_pddl(DURATIONS, False)
    assert "start-drag" in split and "release-drag" in split
    # it still parses
    lisp_string_to_ast(atomic)
    lisp_string_to_ast(split)


def test_drag_geometry_has_a_carry_path():
    G = grid_dsg()
    tasks = [
        {"id": 0, "name": f"drag-object {BAG} :: drag"},
        {"id": 1, "name": f"release-drag {BAG} :: release_drag"},
    ]
    geo = _parameterize_tasks(G, tasks, {BAG: "O0", "t59": "t59"}, {BAG: "t59"})
    assert TASK_ACTION_KINDS["drag-object"] == "drag"
    assert geo[0]["action"] == "drag"
    carry = geo[0]["carry_path"]
    assert len(carry) >= 2
    np.testing.assert_allclose(carry[-1], [9.0, 5.0])
    np.testing.assert_allclose(geo[0]["dest_point"][:2], [9.0, 5.0])
    assert geo[1]["action"] == "release-drag"
    np.testing.assert_allclose(geo[1]["object_point"][:2], [9.0, 5.0])


def _grstaps_grounded(tmp_path, goal_str_parts, **domain_kw):
    G = grid_dsg()
    goal = GrstapsGoal(
        goal_points=goal_str_parts.get("visit", []),
        robot_id="euclid",
        manipulate_points=goal_str_parts.get("manipulate", []),
        manipulate_destinations=goal_str_parts.get("destinations", {}),
        robot_types=ROBOT_TYPES,
    )
    domain = GrstapsDomain(
        entry="itags",
        run_mode="native",
        work_dir=str(tmp_path),
        scenario_name="drag_test",
        robot_type_species=dict(CONSOLE_TYPE_SPECIES),
        inspect_requirement={"ground": 1, "sensor": 1},
        **domain_kw,
    )
    wrapped = grstaps_ground_problem(domain, G, ROBOTS, goal)
    grounded = wrapped.value
    with open(os.path.join(grounded.scenario_dir, "itags_input.json")) as fo:
        return G, wrapped, json.load(fo)


GOAL = {
    "visit": ["t5", "t40", "t10"],
    "manipulate": [BAG, CONE],
    "destinations": {BAG: "t59", CONE: "t50"},
}


def test_grstaps_grounding_makes_a_drag_task_and_an_armless_husky(tmp_path):
    _, wrapped, itags = _grstaps_grounded(tmp_path, GOAL)
    assert set(wrapped.names) == set(ROBOTS)
    species = {s["name"]: s["traits"] for s in itags["species"]}
    robots = {r["name"]: r["species"] for r in itags["robots"]}
    assert robots == {"euclid": "spot_arm", "hamilton": "spot_arm", "gauss": "husky"}
    assert species["husky"][MANIP] == 0 and species["spot_arm"][MANIP] == 1
    names = [t["name"] for t in itags["tasks"]]
    assert f"drag-object {BAG} :: drag" in names
    assert f"relocate-object {CONE} :: relocate" in names
    drag = next(t for t in itags["tasks"] if t["name"].startswith("drag-object"))
    # t1 -> t59 is 13 m over the grid, walked at 0.6 of 1 m/s: 13/0.6 - 13 s
    # on top of the 20 s grasp + release
    assert drag["duration"] == pytest.approx(DRAG_DURATION_S + 13 / 0.6 - 13, abs=0.01)
    with open(os.path.join(wrapped.value.scenario_dir, "domain.pddl")) as fo:
        assert "drag-object" in fo.read()


def test_no_arm_env_takes_the_arm_off(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIPLANNER_NO_ARM_ROBOTS", "hamilton")
    _, _, itags = _grstaps_grounded(tmp_path, GOAL)
    robots = {r["name"]: r["species"] for r in itags["robots"]}
    assert robots["hamilton"] == "spot" and robots["euclid"] == "spot_arm"


def _real_solver():
    return (
        os.environ.get("GRSTAPS_REAL_SOLVER") == "1"
        and shutil.which(os.environ.get("ADT4_ITAGS_BINARY") or "itags")
        and os.path.exists(os.environ.get("ADT4_GUROBI_LICENSE", "~/gurobi.lic"))
    )


@pytest.mark.skipif(
    not _real_solver(), reason="needs GRSTAPS_REAL_SOLVER=1, itags and a licence"
)
def test_real_solver_gives_the_drag_to_a_spot(tmp_path):
    from omniplanner.grstaps_planner import make_plan as grstaps_make_plan

    G, wrapped, _ = _grstaps_grounded(tmp_path, GOAL, solver_timeout_s=60.0)
    plan = grstaps_make_plan(wrapped.value, G)
    names = {t["id"]: t["name"] for t in plan.tasks}
    by_robot = {
        a["name"]: [names[i] for i in a.get("individual_plan") or []]
        for a in plan.agents
    }
    who_drags = [
        r for r, ts in by_robot.items() if any(t.startswith("drag-object") for t in ts)
    ]
    assert who_drags and set(who_drags) <= {"euclid", "hamilton"}
    # gauss stands next to t10: it plans, but only visits and inspections
    assert by_robot.get("gauss")
    for t in by_robot["gauss"]:
        assert t.split()[0] in ("visit-location", "inspect-object"), t
    drag_id = next(i for i, n in names.items() if n.startswith("drag-object"))
    assert plan.task_geometry[drag_id]["action"] == "drag"
    assert len(plan.task_geometry[drag_id]["carry_path"]) >= 2


def test_grstaps_ros_compile_drags_and_never_waits_on_the_husky():
    try:
        from omniplanner_ros.grstaps_planner_ros import compile_grstaps_plan
    except ImportError as exc:
        pytest.skip(f"ROS workspace not sourced: {exc}")
    from omniplanner.grstaps_planner import GrstapsPlan

    carry = np.array([[1.0, 0.0], [5.0, 0.0], [9.0, 5.0]])
    plan = GrstapsPlan(
        makespan=60.0,
        tasks=[
            {"id": 0, "name": "visit-location t10 :: visit"},
            {"id": 1, "name": f"drag-object {BAG} :: drag"},
        ],
        agents=[
            {
                "name": "gauss",
                "individual_plan": [0],
                "transitions": [{"path": [{"x": 1, "y": 0}, {"x": 0, "y": 1}]}],
            },
            {
                "name": "euclid",
                "individual_plan": [1],
                "transitions": [{"path": [{"x": 9, "y": 5}, {"x": 1, "y": 0}]}],
            },
        ],
        task_geometry={
            0: {
                "action": "visit",
                "target": "t10",
                "object_point": np.array([0.0, 1.0, 0.0]),
            },
            1: {
                "action": "drag",
                "target": BAG,
                "object_point": np.array([1.2, 0.3, 0.0]),
                "object_class": "bag",
                "dest_point": np.array([9.0, 5.0, 0.0]),
                "carry_path": carry,
            },
        },
        precedence=[[0, 1]],
    )
    seq = compile_grstaps_plan(plan, "p", "euclid", "map", ungated_robots={"gauss"})
    kinds = [type(a).__name__ for a in seq.actions]
    assert kinds == ["Follow", "DragStart", "Follow", "DragRelease"]
    np.testing.assert_allclose(seq.actions[3].object_point[:2], [9.0, 5.0])
    # the husky's visit would never be reported done: do not wait on it
    assert seq.actions[1].after_task_ids == []
    gated = compile_grstaps_plan(plan, "p", "euclid", "map")
    assert gated.actions[1].after_task_ids == ["0"]
