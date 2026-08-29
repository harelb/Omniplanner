"""Non-ROS PDDL plan compilation. Mirrors omniplanner_ros.pddl_planner_ros.compile_pddl_plan
but without rclpy dependency, so the kinematic simulator can run standalone."""

from __future__ import annotations

import numpy as np
from omniplanner.omniplanner import SymbolicContext

from dsg_pddl.dsg_pddl_planning import PddlPlan

# robot_executor_interface lives in spot_tools, which is a colcon package and is
# not necessarily on the path of a plain (non-ROS) virtualenv. Import it
# defensively so this module -- and the pure helpers below -- stay importable
# for unit testing; compile_pddl_plan_pure re-raises with a clear message if it
# is actually called without the action descriptions available.
try:
    from robot_executor_interface.action_descriptions import (
        ActionSequence,
        Follow,
        Gaze,
        Pick,
        Place,
    )

    ACTION_DESCRIPTIONS_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - depends on the environment
    ActionSequence = Follow = Gaze = Pick = Place = None
    ACTION_DESCRIPTIONS_IMPORT_ERROR = e


def ensure_3d(pt):
    if len(pt) < 2:
        raise Exception("Expected 2d or 3d point, got point length: {len(pt}")
    if len(pt) == 2:
        new_pt = np.zeros(3)
        new_pt[:2] = pt
        return new_pt
    return pt


def _object_class(context, symbol) -> str:
    """The semantic label for ``symbol`` in the plan's symbolic context, or "".

    Pure (no ROS / no action descriptions) so it is unit-testable on its own.

    Both ``pick-object`` and ``place-object`` need this. It used to be computed
    *only* inside the ``pick-object`` branch and then read by ``place-object``,
    which is wrong twice over: a plan that places without a preceding pick hit
    an ``UnboundLocalError``, and a plan that picks A then places B stamped B's
    ``Place`` with A's class. Looking it up per action fixes both.
    """
    if context is None or symbol not in context:
        return ""
    attrs = context[symbol]
    if attrs is None or "semantic_label" not in attrs:
        return ""
    return attrs["semantic_label"]


def compile_pddl_plan_pure(
    contextualized_plan: SymbolicContext[PddlPlan],
    plan_id: str,
    robot_name: str,
    frame_id: str,
) -> ActionSequence:
    """Compile a PDDL plan to an ActionSequence without ROS dependencies.

    Args:
        contextualized_plan: SymbolicContext wrapping a PddlPlan
        plan_id: Unique plan identifier (string UUID)
        robot_name: Name of the robot executing the plan
        frame_id: Reference frame for actions

    Returns:
        ActionSequence with compiled actions

    Raises:
        ImportError: If robot_executor_interface is not importable
        NotImplementedError: For unsupported action types
    """
    if ActionSequence is None:  # pragma: no cover - depends on the environment
        raise ImportError(
            "robot_executor_interface.action_descriptions is not importable, so "
            "PDDL plans cannot be compiled to an ActionSequence. Source the "
            "colcon workspace (spot_tools) first."
        ) from ACTION_DESCRIPTIONS_IMPORT_ERROR

    plan = contextualized_plan.value
    context = contextualized_plan.context
    actions = []
    for symbolic_action, parameters in zip(
        plan.symbolic_actions, plan.parameterized_actions
    ):
        match symbolic_action[0]:
            case "goto-poi":
                actions.append(Follow(frame=frame_id, path2d=parameters))
            case "inspect":
                robot_point, gaze_point = parameters
                actions.append(
                    Gaze(
                        frame=frame_id,
                        robot_point=ensure_3d(robot_point),
                        gaze_point=ensure_3d(gaze_point),
                        stow_after=True,
                        object_id=symbolic_action[1],
                    )
                )
            case "pick-object":
                robot_point, pick_point = parameters
                actions.append(
                    Pick(
                        frame=frame_id,
                        object_class=_object_class(context, symbolic_action[1]),
                        robot_point=ensure_3d(robot_point),
                        object_point=ensure_3d(pick_point),
                        object_id=symbolic_action[1],
                    )
                )
            case "place-object":
                robot_point, place_point = parameters
                actions.append(
                    Place(
                        frame=frame_id,
                        object_class=_object_class(context, symbolic_action[1]),
                        robot_point=ensure_3d(robot_point),
                        object_point=ensure_3d(place_point),
                        object_id=symbolic_action[1],
                    )
                )
            case _:
                raise NotImplementedError(
                    f"I don't know how to compile {symbolic_action[0]}"
                )

    seq = ActionSequence(plan_id=plan_id, robot_name=robot_name, actions=actions)
    return seq
