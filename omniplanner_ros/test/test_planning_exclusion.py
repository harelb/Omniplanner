"""A robot that cannot move (E-stop, unpowered, low battery) is left out of
planning; a lease it does not hold is still only a dispatch-time refusal."""

from types import SimpleNamespace as NS

from omniplanner_ros.omniplanner_node import LazyRobotPoses, planning_exclusion


def guards(**kw):
    base = dict(estop_known=True, estop_pressed=False, power_known=True, powered_on=True,
                battery_known=True, battery_percent=80.0, lease_known=True, lease_owned=True,
                lease_owner="")
    base.update(kw)
    return NS(**base)


def test_reasons():
    assert planning_exclusion(guards(), 1.0) is None
    assert planning_exclusion(guards(powered_on=False), 1.0) == "not_powered"
    assert planning_exclusion(guards(estop_pressed=True), 1.0) == "estop_pressed"
    assert planning_exclusion(guards(battery_percent=5.0), 1.0).startswith("battery_below_min")
    assert planning_exclusion(guards(lease_owned=False, lease_owner="tablet"), 1.0) is None
    assert planning_exclusion(None, None) is None  # no guard stream: fail open
    assert planning_exclusion(guards(powered_on=False), 60.0) is None  # stale: fail open


class Adaptor:
    def __init__(self, why):
        self.why, self.looked_up = why, False
        self.ros_logger = NS(warning=lambda *_: None)

    def planning_exclusion(self):
        return self.why

    def get_pose(self, frame, timeout_s):
        self.looked_up = True
        return [1.0, 2.0, 0.0, 0.0]


def test_an_unpowered_robot_reads_as_offline():
    a = {"euclid": Adaptor("not_powered"), "hamilton": Adaptor(None)}
    poses = LazyRobotPoses(a, "map")
    assert poses["euclid"] is None and not a["euclid"].looked_up
    assert poses["hamilton"] == [1.0, 2.0, 0.0, 0.0]
    assert poses.excluded == {"euclid": "not_powered"}
    assert sorted(poses) == ["euclid", "hamilton"]  # still iterates every robot
