"""Executor action descriptions for dragging a heavy object.

robot_executor_interface (spot_tools) has Follow / Gaze / Pick / Place but no
drag yet, so the two halves are described here, field for field like Pick and
Place, and travel over ActionMsg as plain action_type strings (ActionMsg's
action_type is a string; see omniplanner_ros.drag_actions_ros):

* ``DragStart`` -> action_type ``"DRAG_START"``: at ``robot_point``, grasp the
  object at ``object_point`` and go into drag mode (arm to the side, slow walk).
  Fields as Pick (pick_frame, object_class, robot_point, object_point,
  object_id).
* the Follow actions after it are walked while dragging;
* ``DragRelease`` -> action_type ``"DRAG_RELEASE"``: let go of the object at
  ``object_point`` (the destination) and stow. Fields as Place.

A plan is DragStart, Follow..., DragRelease, exactly as Pick, Follow...,
Place. The robot side (spot_tools' spot_skills/drag_utils.py) still has to
learn to execute the two action types.
"""

from dataclasses import dataclass

import numpy as np

DRAG_START = "DRAG_START"
DRAG_RELEASE = "DRAG_RELEASE"


@dataclass
class DragStart:
    frame: str
    object_class: str
    robot_point: np.ndarray
    object_point: np.ndarray
    object_id: str
    action_type: str = DRAG_START


@dataclass
class DragRelease:
    frame: str
    object_class: str
    robot_point: np.ndarray
    object_point: np.ndarray
    object_id: str
    action_type: str = DRAG_RELEASE
