import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import rclpy
import rclpy.duration
import tf2_ros
import tf_transformations
from hydra_ros import DsgSubscriber
from nav_msgs.msg import Path
from omniplanner.compile_plan import collect_plans, compile_plan
from omniplanner.omniplanner import full_planning_pipeline
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from robot_executor_interface_ros.action_descriptions_ros import to_msg, to_viz_msg
from robot_executor_msgs.msg import (
    ActionResultMsg,
    ActionSequenceMsg,
    RuntimeGuardsMsg,
)
from robot_vocalizer.plan_vocalizer import PlanVocalizer
from ros_system_monitor_msgs.msg import NodeInfoMsg
from spark_config import Config, config_field, register_config
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import MarkerArray

from omniplanner_ros.ros_logging import setup_ros_log_forwarding

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def get_plan_vocalizer(node):
    node.declare_parameter("vocalize", False)
    vocalize = node.get_parameter("vocalize").value
    if not vocalize:
        return None

    node.declare_parameter("openai_api_key", "")
    openai_api_key = node.get_parameter("openai_api_key").value
    if openai_api_key == "":
        openai_api_key = None

    node.declare_parameter("deepgram_api_key", "")
    deepgram_api_key = node.get_parameter("deepgram_api_key").value
    if deepgram_api_key == "":
        deepgram_api_key = None

    if openai_api_key is not None and deepgram_api_key is not None:
        return PlanVocalizer(openai_api_key, deepgram_api_key)
    else:
        return None


def plan_to_string(robot_plan):
    # TODO: generalize. Currently assumes robot_plan is a list of tuples
    plan_string = ""
    for a in robot_plan.symbolic_actions:
        plan_string += str(a) + "\n"
    return plan_string


@dataclass
class PluginFeedbackCollector:
    # k,v: feedback name, publish function
    publish: Dict[str, object] = field(default_factory=dict)


@dataclass
class OmniplannerFeedbackCollector:
    # k,v: plugin name, plugin feedback content
    plugin_feedback_collectors: Dict[str, PluginFeedbackCollector] = field(
        default_factory=dict
    )


@dataclass
class PlannerConfig(Config):
    plugin: Any = config_field("omniplanner_pipeline", required=False)


def evaluate_dispatch_guards(guards, guards_age_s, *, min_battery_percent=10.0,
                             max_guard_age_s=10.0):
    """The PR B8 dispatch gate, pure: (latest RuntimeGuardsMsg-like, its age)
    -> (ok, typed_reason). Volatile platform state is checked at DISPATCH
    time only — it is deliberately not planning state (skill contracts name
    WHICH guards apply; this decides).

    Fail-OPEN on missing/stale/unknown guards (with the reason string
    returned as ``None`` for ok): a robot without a guard stream — e.g. a
    non-spot adaptor, or a sim without battery state — must not be
    unplannable. Fail-CLOSED only on an affirmative bad guard.
    """
    if guards is None:
        return True, None
    if guards_age_s is not None and guards_age_s > max_guard_age_s:
        return True, None  # stale stream: no live evidence either way
    if guards.estop_known and guards.estop_pressed:
        return False, "estop_pressed"
    if guards.power_known and not guards.powered_on:
        return False, "not_powered"
    if guards.battery_known and guards.battery_percent >= 0.0 \
            and guards.battery_percent < min_battery_percent:
        return False, f"battery_below_min({guards.battery_percent:.0f}%<{min_battery_percent:.0f}%)"
    if guards.lease_known and not guards.lease_owned:
        return False, "lease_not_owned"
    return True, None


class RobotPlanningAdaptor:
    def __init__(self, config, node=None, tf_buffer=None, qos_profile=None):
        self.tf_buffer = tf_buffer
        self.name = config.robot_name
        self.robot_type = config.robot_type
        self.child_frame = config.body_frame
        self.ros_logger = node.get_logger()

        self.plan_pub = node.create_publisher(
            ActionSequenceMsg,
            f"/{self.name}/omniplanner_node/compiled_plan_out",
            qos_profile or 1,
        )

        # Executor -> planner return channel (PR B5): per-action results from
        # the robot's executor node. Until this subscription existed there was
        # NO feedback path from execution back to planning (heartbeats only).
        # ``action_results`` keeps the received results keyed by plan_id, in
        # arrival order, for planner-side consumers (e.g. the acquisition
        # loop's obligation resolution); the log line is the minimal liveness
        # signal the B5 acceptance looks for.
        self.action_results = {}
        result_topic = getattr(config, "action_result_topic", "") or (
            f"/{self.name}/spot_executor/action_result"
        )
        self.action_result_sub = node.create_subscription(
            ActionResultMsg,
            result_topic,
            self._on_action_result,
            10,
        )

        # PR B8: latest runtime-guard snapshot + the dispatch gate's knobs.
        self._node_clock = node.get_clock()
        self.min_battery_percent = float(
            getattr(config, "min_battery_percent", 10.0) or 10.0)
        self.last_guards = None
        self.last_guards_stamp_s = None
        guards_topic = getattr(config, "runtime_guards_topic", "") or (
            f"/{self.name}/spot_executor/runtime_guards"
        )
        self.runtime_guards_sub = node.create_subscription(
            RuntimeGuardsMsg,
            guards_topic,
            self._on_runtime_guards,
            1,
        )

    def _on_runtime_guards(self, msg):
        self.last_guards = msg
        self.last_guards_stamp_s = self._node_clock.now().nanoseconds / 1e9

    def _on_action_result(self, msg):
        self.ros_logger.info(
            f"[{self.name}] action result: plan={msg.plan_id!r} "
            f"#{msg.action_index} {msg.action_type} -> {msg.status}"
            + (f" ({msg.detail})" if msg.detail else "")
        )
        self.action_results.setdefault(msg.plan_id, []).append(msg)

    def get_pose(self, parent_frame, timeout_s: float = 1.0):
        self.ros_logger.info(
            f"Looking up pose for {self.name} ({parent_frame}->{self.child_frame})"
        )
        try:
            return get_robot_pose(
                self.tf_buffer, parent_frame, self.child_frame, timeout_s
            )
        except tf2_ros.TransformException as e:
            self.ros_logger.warning(str(e))
            return None

    def publish_plan(self, plan):
        """Dispatch a compiled plan, gated by the latest runtime guards
        (PR B8). Returns (published: bool, refusal_reason: str | None)."""
        age_s = None
        if self.last_guards_stamp_s is not None:
            age_s = self._node_clock.now().nanoseconds / 1e9 - self.last_guards_stamp_s
        ok, reason = evaluate_dispatch_guards(
            self.last_guards, age_s,
            min_battery_percent=self.min_battery_percent)
        if not ok:
            self.ros_logger.error(
                f"[{self.name}] dispatch REFUSED by runtime guard: {reason}"
            )
            return False, reason
        self.plan_pub.publish(plan)
        return True, None


@register_config(
    "robot_adaptor", name="robot_executor", constructor=RobotPlanningAdaptor
)
@dataclass
class RobotConfig(Config):
    robot_name: str = ""
    robot_type: str = ""
    body_frame: str = ""
    # PR B5: where this robot's executor publishes ActionResultMsg. Empty ->
    # the spot_executor convention /<robot_name>/spot_executor/action_result.
    action_result_topic: str = ""
    # PR B8: the executor's RuntimeGuardsMsg topic (same empty-default
    # convention) and the dispatch gate's battery floor.
    runtime_guards_topic: str = ""
    min_battery_percent: float = 10.0


class PhoenixPlanningAdaptor(RobotPlanningAdaptor):
    def __init__(self, config, **data):
        super().__init__(config, **data)
        self.plan_pub = data["node"].create_publisher(
            Path, f"/{self.name}/omniplanner_node/compiled_plan_out", 1
        )


@register_config("robot_adaptor", name="phoenix", constructor=PhoenixPlanningAdaptor)
@dataclass
class PhoenixRobotConfig(RobotConfig):
    pass


@dataclass
class OmniplannerNodeConfig(Config):
    robots: List[config_field("robot_adaptor")] = field(default_factory=list)
    planners: Dict[str, PlannerConfig] = field(default_factory=dict)
    # How long (s) to wait for a robot's transform before treating it as
    # unavailable. Small so offline robots don't stall planning; online robots
    # resolve immediately regardless. See get_robot_pose.
    tf_timeout_s: float = 0.2

    @classmethod
    def load(cls, path: str):
        return Config.load(OmniplannerNodeConfig, path)


def get_robot_pose(
    tf_buffer, parent_frame: str, child_frame: str, timeout_s: float = 1.0
) -> np.ndarray:
    """
    Looks up the transform from parent_frame to child_frame and returns [x, y, z, yaw].

    ``timeout_s`` is how long to wait for the transform to become available. For
    an online robot publishing TF continuously, the transform is already in the
    buffer and this returns immediately regardless of the timeout; the timeout is
    only spent waiting on an *absent* transform (e.g. an offline robot), so a
    small value keeps planning responsive when robots are offline.
    """
    # Time() is time 0, i.e. "latest available" in tf2.
    try:
        now = rclpy.time.Time()
        tf_buffer.can_transform(
            parent_frame,
            child_frame,
            now,
            timeout=rclpy.duration.Duration(seconds=timeout_s),
        )
        transform = tf_buffer.lookup_transform(parent_frame, child_frame, now)

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        # Convert quaternion to Euler angles
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)

        return np.array([translation.x, translation.y, translation.z, yaw])

    except tf2_ros.TransformException as e:
        print(f"Transform error: {e}")
        raise


class LazyRobotPoses(Mapping):
    """Robot poses looked up from TF on demand and cached.

    Behaves like the eager ``{robot_name: pose_or_None}`` dict it replaces
    (iteration yields every configured robot; offline robots resolve to None),
    but a robot's transform is only looked up the first time it is accessed.
    Single-robot planning therefore touches exactly one robot, so the per-offline
    -robot TF timeout is not paid for robots that have no goal assigned. The
    multi-robot grounders that iterate every robot still resolve them all, with
    offline robots filtered out downstream via ``pose is not None`` as before.
    """

    def __init__(self, robot_adaptors, parent_frame, timeout_s=1.0):
        self._adaptors = robot_adaptors
        self._parent_frame = parent_frame
        self._timeout_s = timeout_s
        self._cache = {}

    def __getitem__(self, name):
        if name not in self._adaptors:
            raise KeyError(name)
        if name not in self._cache:
            self._cache[name] = self._adaptors[name].get_pose(
                self._parent_frame, self._timeout_s
            )
        return self._cache[name]

    def __iter__(self):
        return iter(self._adaptors)

    def __len__(self):
        return len(self._adaptors)

    def resolved(self):
        """Poses actually looked up so far (for logging without forcing lookups)."""
        return dict(self._cache)


class OmniPlannerRos(Node):
    def __init__(self):
        super().__init__("omniplanner_ros")
        self.get_logger().info("Setting up omniplanner")

        # forward python logging to ROS
        setup_ros_log_forwarding(self, logger)
        logger.setLevel(logging.INFO)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.last_dsg_time = 0
        self.current_planner = None
        self.plan_time_start = None
        self.dsg_last = None
        self.dsg_frame = None

        self.last_dsg_time_lock = threading.Lock()
        self.current_planner_lock = threading.Lock()
        self.plan_time_start_lock = threading.Lock()

        self.dsg_lock = threading.Lock()
        DsgSubscriber(self, "~/dsg_in", self.dsg_callback)

        latching_reliable_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_ALL,
        )

        reliable_blocking_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_ALL,
        )

        self.compiled_plan_viz_pub = self.create_publisher(
            MarkerArray, "~/compiled_plan_viz_out", qos_profile=latching_reliable_qos
        )

        self.heartbeat_pub = self.create_publisher(NodeInfoMsg, "~/node_status", 1)
        heartbeat_timer_group = MutuallyExclusiveCallbackGroup()
        timer_period_s = 0.1
        self.timer = self.create_timer(
            timer_period_s, self.hb_callback, callback_group=heartbeat_timer_group
        )

        # process virtual config
        self.declare_parameter("plugin_config_path", "")
        config_path = self.get_parameter("plugin_config_path").value
        assert config_path != "", "plugin_config_path cannot be empty"

        self.plan_vocalizer = get_plan_vocalizer(self)

        self.config = OmniplannerNodeConfig.load(config_path)

        self.robot_adaptors = {}
        for robot_config in self.config.robots:
            robot_adaptor = robot_config.create(
                node=self, tf_buffer=self.tf_buffer, qos_profile=reliable_blocking_qos
            )
            self.get_logger().info(
                f"I know about {robot_adaptor.name}, a {robot_adaptor.robot_type} robot"
            )
            self.robot_adaptors[robot_config.robot_name] = robot_adaptor

        # Initialize a feedback collector to be populated by plugins
        self.feedback = OmniplannerFeedbackCollector()

        for name, planner in self.config.planners.items():
            plugin = planner.plugin.create()
            if plugin is None:
                raise Exception(
                    f"Failed to load plugin {planner.plugin}. Did you add your planner package as a dependency?"
                )
            self.register_plugin(name, plugin)

    def dsg_callback(self, header, dsg):
        self.get_logger().warning("Setting DSG!")

        with self.last_dsg_time_lock and self.dsg_lock:
            self.dsg_last = dsg
            self.dsg_frame = header.frame_id
            self.last_dsg_time = time.time()

    def hb_callback(self):
        with (
            self.last_dsg_time_lock
            and self.current_planner_lock
            and self.plan_time_start_lock
        ):
            dsg_age = time.time() - self.last_dsg_time
            have_recent_dsg = dsg_age < 15

            notes = ""
            if not have_recent_dsg:
                notes += f"No recent dsg ({dsg_age} (s) old)"

            if self.current_planner is None:
                if have_recent_dsg:
                    notes += "Ready to plan!"
            else:
                elapsed_planning_time = time.time() - self.plan_time_start
                notes += f"Running {self.current_planner} ({elapsed_planning_time} s)"

        status = NodeInfoMsg.NOMINAL
        if not have_recent_dsg:
            status = NodeInfoMsg.WARNING

        msg = NodeInfoMsg()
        msg.nickname = "omniplanner"
        msg.node_name = self.get_fully_qualified_name()
        msg.status = status
        msg.notes = notes
        self.heartbeat_pub.publish(msg)

    def get_robot_poses(self, dsg_frame):
        # Lazy: only robots whose pose is actually accessed during grounding get
        # a TF lookup, so robots without an assigned goal cost nothing.
        return LazyRobotPoses(
            self.robot_adaptors, dsg_frame, timeout_s=self.config.tf_timeout_s
        )

    def register_plugin(self, name, plugin):
        self.get_logger().info(f"Registering subscription plugin {name}")
        msg_type, topic, callback = plugin.get_plan_callback()
        self.feedback.plugin_feedback_collectors[name] = plugin.get_plugin_feedback(
            self
        )

        def plan_handler(msg):
            self.get_logger().info(f"Handling plan for plugin {name}")

            if self.dsg_last is None:
                self.get_logger().error("Got plan request, but no DSG!")
                return

            with self.current_planner_lock and self.plan_time_start_lock:
                self.current_planner = name
                self.plan_time_start = time.time()

            # Poses are resolved lazily; avoid formatting the whole mapping here
            # (that would force a TF lookup for every configured robot).
            robot_poses = self.get_robot_poses(self.dsg_frame)

            plan_request = callback(msg, robot_poses)
            with self.dsg_lock:
                plans = full_planning_pipeline(
                    plan_request, self.dsg_last, self.feedback
                )
            self.get_logger().info(
                f"Planned using robot poses {robot_poses.resolved()}"
            )

            compiled_plans = compile_plan(self.robot_adaptors, self.dsg_frame, plans)
            plan_dict = collect_plans(compiled_plans)
            for robot_name, compiled_plan in plan_dict.items():
                self.robot_adaptors[robot_name].publish_plan(to_msg(compiled_plan))
                # TODO: combine markers into single array so that latching works
                # correctly for multi-robot plans?
                self.compiled_plan_viz_pub.publish(
                    to_viz_msg(compiled_plan, robot_name)
                )

            with self.current_planner_lock and self.plan_time_start_lock:
                self.current_planner = None
                self.plan_time_start = None
            self.get_logger().info("Published Plan")

        resolved_topic_name = name + "/" + topic
        self.get_logger().info(
            f"Registering subscription for {resolved_topic_name} (type {str(msg_type)})"
        )
        self.create_subscription(
            msg_type,
            f"~/{resolved_topic_name}",
            plan_handler,
            1,
        )


def main(args=None):
    rclpy.init(args=args)
    try:
        node = OmniPlannerRos()
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
