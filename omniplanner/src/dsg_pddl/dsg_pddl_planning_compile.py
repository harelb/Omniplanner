"""Non-ROS PDDL plan compilation. Mirrors omniplanner_ros.pddl_planner_ros.compile_pddl_plan
but without rclpy dependency, so the kinematic simulator can run standalone."""

from typing import Any

import numpy as np
from omniplanner.omniplanner import SymbolicContext
from robot_executor_interface.action_descriptions import (
    ActionSequence,
    Follow,
    Gaze,
    Pick,
    Place,
)

from dsg_pddl.dsg_pddl_planning import PddlPlan


def ensure_3d(pt):
    if len(pt) < 2:
        raise Exception("Expected 2d or 3d point, got point length: {len(pt}")
    if len(pt) == 2:
        new_pt = np.zeros(3)
        new_pt[:2] = pt
        return new_pt
    return pt


def compile_pddl_plan_pure(
    contextualized_plan: SymbolicContext[PddlPlan],
    plan_id: str,
    robot_name: str,
    frame_id: str
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
        NotImplementedError: For unsupported action types
    """
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
                object_class = ""
                if symbolic_action[1] in context:
                    attrs = context[symbolic_action[1]]
                    if "semantic_label" in attrs:
                        object_class = attrs["semantic_label"]
                actions.append(
                    Pick(
                        frame=frame_id,
                        object_class=object_class,
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
                        object_class=object_class,
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
