"""ROS conversion for omniplanner.drag_actions (DragStart / DragRelease).

robot_executor_interface_ros converts actions with a singledispatch ``to_msg``
/ ``to_viz_msg`` that knows Follow, Gaze, Pick and Place only. Registering the
drag actions on those same functions lets an ActionSequence that contains a
drag be published unchanged. ActionMsg.action_type is a string, so the drag
goes out as ``"DRAG_START"`` (Pick's fields: pick_frame, object_class,
robot_point, object_point, object_id) and ``"DRAG_RELEASE"`` (Place's fields,
place_frame). Importing this module registers them; it is a no-op where
robot_executor_interface_ros is not installed.
"""

import logging

from omniplanner.drag_actions import DragRelease, DragStart

logger = logging.getLogger(__name__)

REGISTERED = False


def _copy_task_fields(action, msg):
    # Fleet plans gate actions on task ids (GRSTAPS-X); copy them when both
    # the action and this ActionMsg version carry them.
    for name in ("task_id", "after_task_ids", "coalition_robots"):
        value = getattr(action, name, None)
        if value is not None and hasattr(msg, name):
            setattr(msg, name, list(value) if isinstance(value, list) else value)


def _fill(msg, action):
    msg.action_type = action.action_type
    msg.robot_point.x = float(action.robot_point[0])
    msg.robot_point.y = float(action.robot_point[1])
    msg.robot_point.z = float(action.robot_point[2])
    msg.object_point.x = float(action.object_point[0])
    msg.object_point.y = float(action.object_point[1])
    msg.object_point.z = float(action.object_point[2])
    msg.object_class = action.object_class or ""
    msg.object_id = action.object_id or ""
    _copy_task_fields(action, msg)
    return msg


def register():
    """Register DragStart / DragRelease with robot_executor_interface_ros."""
    global REGISTERED
    if REGISTERED:
        return True
    try:
        from robot_executor_interface_ros import action_descriptions_ros as ros
        from robot_executor_msgs.msg import ActionMsg
    except ImportError as exc:  # pragma: no cover - depends on the environment
        logger.debug(
            "No robot_executor_interface_ros; drag actions stay local: %s", exc
        )
        return False

    @ros.to_msg.register
    def _(action: DragStart):
        msg = ActionMsg()
        msg.pick_frame = action.frame
        return _fill(msg, action)

    @ros.to_msg.register
    def _(action: DragRelease):
        msg = ActionMsg()
        msg.place_frame = action.frame
        return _fill(msg, action)

    arrows = getattr(ros, "_arrow_markers", None)
    if arrows is not None:

        @ros.to_viz_msg.register
        def _(action: DragStart, marker_ns, color=None):
            return arrows(
                action, marker_ns, color, action.object_point, (1.0, 0.5, 0.0), "drag"
            )

        @ros.to_viz_msg.register
        def _(action: DragRelease, marker_ns, color=None):
            return arrows(
                action,
                marker_ns,
                color,
                action.object_point,
                (0.6, 0.3, 0.0),
                "release",
            )

    REGISTERED = True
    return True


register()
