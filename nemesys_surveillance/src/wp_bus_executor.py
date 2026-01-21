#!/usr/bin/env python3
# =====================================
# Author : Adnan Abdullah
# =====================================
#
# Routine B patrol executor:
# - Uses a pipeline “lane” at y=y_lane (default 0) to move between pods in +x order
# - For each pod:
#     lane_anchor (x_pod, y_lane) -> approach (near pod, stop short) ->
#     rectangle patrol around pod (respecting pod yaw) -> back to approach -> back to lane_anchor
# - Publishes /target_depth for your DepthControlNode and passes through /heave_control_input
# - Sends planar commands (surge/yaw) to /nemesys/user_input
#
# Works with your “Waypoint3DExecutor” structure and topics.

import rospy
import math
import numpy as np
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
from gazebo_msgs.msg import LinkStates
from nemesys_interfaces.msg import nemesysInput as NemesysInput


# ============================================================
# Geometry helpers
# ============================================================
def rect_waypoints_around_pod(pod_xy, pod_yaw, rect_rel_xy):
    """
    Rotate + translate a canonical rectangle around a pod.
    pod_xy: (x0,y0) in world
    pod_yaw: radians (world yaw)
    rect_rel_xy: list[(xr,yr)] defined in pod-local frame
    returns list[(xw,yw)] in world
    """
    x0, y0 = pod_xy
    c = math.cos(pod_yaw)
    s = math.sin(pod_yaw)
    wps = []
    for xr, yr in rect_rel_xy:
        xw = x0 + c * xr - s * yr
        yw = y0 + s * xr + c * yr
        wps.append((float(xw), float(yw)))
    return wps


def build_patrol_waypoints_routine_B(
    pods,
    rect_rel_xy,
    z_travel,
    z_patrol,
    y_lane=0.0,
    approach_offset=0.3,
    close_loop=True,
    return_to_lane=True,
    add_initial_lane_anchor=True,
):
    """
    pods: list of dicts: {"name", "x", "y", "yaw"} yaw in radians
    returns list[(x,y,z)]
    approach point is not used currently; can be used as a center to start/finish patrol.
    If used, make sure to use high offset to avoid collision with pod.
    Or use altitude clearance w.r.t pod.
    """
    pods_sorted = sorted(pods, key=lambda d: float(d["x"]))

    wps = []

    for i, pod in enumerate(pods_sorted):
        px = float(pod["x"])
        py = float(pod["y"])
        yaw = float(pod["yaw"])

        lane_anchor = (px, float(y_lane), float(z_travel))
        if i == 0 and add_initial_lane_anchor:
            wps.append(lane_anchor)
        else:
            wps.append(lane_anchor)

        # Approach point: move towards pod side but stop short by approach_offset
        sign = 1.0 if (py - y_lane) >= 0.0 else -1.0
        approach_y = py - sign * approach_offset
        approach = (px, float(approach_y), float(z_travel))
        # wps.append(approach) # start patrol from approach point

        # Rectangle patrol around pod at z_patrol
        rect_xy = rect_waypoints_around_pod((px, py), yaw, rect_rel_xy)
        for (xw, yw) in rect_xy:
            wps.append((xw, yw, float(z_patrol)))
        if close_loop and len(rect_xy) > 0:
            wps.append((rect_xy[0][0], rect_xy[0][1], float(z_patrol)))

        if return_to_lane:
            # wps.append(approach) # finish patrol at approach point
            wps.append(lane_anchor)

    return wps


# ============================================================
# Waypoint Executor Node
# ============================================================
class Waypoint3DExecutorRoutineB:
    def __init__(self):
        rospy.init_node('wp_bus_executor')

        # ==============================
        # Parameters (control)
        # ==============================
        self.xy_tol   = rospy.get_param('~xy_tolerance', 0.2)     # m
        self.z_tol    = rospy.get_param('~z_tolerance',  0.15)    # m
        self.yaw_tol  = rospy.get_param('~yaw_tolerance', 10.0)   # deg
        self.speed    = rospy.get_param('~speed', 0.6)            # 0..1
        self.yaw_kp   = rospy.get_param('~yaw_kp', 0.02)
        self.surge_kp = rospy.get_param('~surge_kp', 0.1)
        self.yaw_max  = rospy.get_param('~yaw_max', 0.7)          # cmd limit
        self.heave_scale = rospy.get_param('~heave_scale', 1.0)

        # Depth controller offset (your DepthControlNode does: target_depth = msg - 60)
        # In your previous node you used: out.data = -z + bias
        self.depth_controller_bias = rospy.get_param('~depth_controller_bias', 0.0)

        # ==============================
        # Parameters (patrol geometry)
        # ==============================
        self.y_lane = rospy.get_param('~y_lane', 0.0)  # pipeline lane y
        self.approach_offset = rospy.get_param('~approach_offset', 0.3)
        self.z_travel = rospy.get_param('~z_travel', -58.5)
        self.z_patrol = rospy.get_param('~z_patrol', -59.4)

        # Canonical rectangle in pod-local frame (same as your origin pod)
        self.rect_rel_xy = [
            (-1.5, -1.0),
            (-1.5,  1.0),
            ( 2.0,  1.0),
            ( 2.0, -1.0),
        ]

        # ==============================
        # Pod definitions (world frame)
        # ==============================
        #  New configuration:
        #   pod_1_5   (0,  4.5) yaw +1.57
        #   pod_6_10  (5, -4.5) yaw -1.57
        #   pod_11_15 (10, 4.5) yaw +1.57
        #   pod_16_20 (15,-4.5) yaw -1.57
        #   pod_21_25 (20, 4.5) yaw +1.57
        self.pods = [
            {"name": "pod_1_5",   "x": 0.0,  "y":  4.5, "yaw":  1.57},
            {"name": "pod_6_10",  "x": 5.0,  "y": -4.5, "yaw": -1.57},
            {"name": "pod_11_15", "x": 10.0, "y":  4.5, "yaw":  1.57},
            {"name": "pod_16_20", "x": 15.0, "y": -4.5, "yaw": -1.57},
            {"name": "pod_21_25", "x": 20.0, "y":  4.5, "yaw":  1.57},
        ]

        # ==============================
        # Build waypoint list (Routine B)
        # ==============================
        self.waypoints = build_patrol_waypoints_routine_B(
            pods=self.pods,
            rect_rel_xy=self.rect_rel_xy,
            z_travel=self.z_travel,
            z_patrol=self.z_patrol,
            y_lane=self.y_lane,
            approach_offset=self.approach_offset,
            close_loop=True,
            return_to_lane=True,
            add_initial_lane_anchor=True,
        )
        rospy.loginfo(f"[RoutineB] Generated {len(self.waypoints)} waypoints.")

        # ==============================
        # State
        # ==============================
        self.current_pos = None   # np.array([x,y,z])
        self.current_yaw = None   # deg (-180,180]
        self.heave_cmd_in = 0.0
        self.idx = 0
        self.sent_depth_for_idx = None

        # ==============================
        # ROS I/O
        # ==============================
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_cb, queue_size=10)
        rospy.Subscriber('/euler_angles', Vector3, self.euler_cb, queue_size=50)
        rospy.Subscriber('/heave_control_input', Float32, self.heave_cb, queue_size=50)

        self.cmd_pub = rospy.Publisher('/nemesys/user_input', NemesysInput, queue_size=10)
        self.target_depth_pub = rospy.Publisher('/target_depth', Float32, queue_size=10)

        self.cmd = NemesysInput()
        self.cmd.roll = 0.0

        self.timer = rospy.Timer(rospy.Duration(0.1), self.update)

    # ========================================================
    # Callbacks
    # ========================================================
    def link_cb(self, msg):
        try:
            i = msg.name.index("nemesys::base_link")
        except ValueError:
            return
        p = msg.pose[i].position
        self.current_pos = np.array([p.x, p.y, p.z], dtype=float)

    def euler_cb(self, msg):
        # Wrap to (-180,180]
        yaw = float(msg.z)
        self.current_yaw = ((yaw + 180.0) % 360.0) - 180.0

    def heave_cb(self, msg):
        self.heave_cmd_in = float(msg.data)

    # ========================================================
    # Control helpers
    # ========================================================
    @staticmethod
    def yaw_diff_deg(current, target):
        return ((target - current + 180.0) % 360.0) - 180.0

    def yaw_cmd(self, yaw_err_deg):
        return float(np.clip(-self.yaw_kp * yaw_err_deg, -self.yaw_max, self.yaw_max))

    def surge_cmd(self, dist_xy):
        return float(np.clip(self.surge_kp * dist_xy, 0.0, self.speed))

    def publish_depth(self, z_world):
        """
            out.data = -z + bias
        Keep that convention consistent here.
        """
        out = Float32()
        out.data = float(-z_world + self.depth_controller_bias)
        self.target_depth_pub.publish(out)

    # ========================================================
    # Main loop
    # ========================================================
    def update(self, event):
        if self.current_pos is None or self.current_yaw is None:
            rospy.loginfo_throttle(2.0, "[RoutineB] Waiting for pose/yaw...")
            return

        if self.idx >= len(self.waypoints):
            # Stop
            self.cmd.surge = 0.0
            self.cmd.yaw = 0.0
            self.cmd.heave = 0.0
            self.cmd_pub.publish(self.cmd)
            rospy.loginfo_throttle(2.0, "[RoutineB] All waypoints completed.")
            return

        tx, ty, tz = self.waypoints[self.idx]
        x, y, z = self.current_pos

        # Publish depth target once per waypoint
        if self.sent_depth_for_idx != self.idx:
            self.publish_depth(tz)
            self.sent_depth_for_idx = self.idx
            rospy.loginfo(f"[RoutineB WP {self.idx+1}/{len(self.waypoints)}] "
                          f"Target (x,y,z)=({tx:.2f},{ty:.2f},{tz:.2f})")

        dx = tx - x
        dy = ty - y
        dist_xy = math.hypot(dx, dy)
        z_err = tz - z

        target_yaw = math.degrees(math.atan2(dy, dx)) if dist_xy > 1e-6 else self.current_yaw
        yaw_err = self.yaw_diff_deg(self.current_yaw, target_yaw)

        # Planar control
        if dist_xy > self.xy_tol:
            if abs(yaw_err) > self.yaw_tol:
                self.cmd.surge = 0.0
                self.cmd.yaw = self.yaw_cmd(yaw_err)
            else:
                self.cmd.surge = self.surge_cmd(dist_xy)
                self.cmd.yaw = 0.0
        else:
            # At XY target: stop planar motion (depth controller still active)
            self.cmd.surge = 0.0
            self.cmd.yaw = 0.0

        # Heave: pass-through from depth controller
        self.cmd.heave = float(self.heave_cmd_in * self.heave_scale)

        self.cmd_pub.publish(self.cmd)

        # Completion check
        if (dist_xy <= self.xy_tol) and (abs(z_err) <= self.z_tol):
            rospy.loginfo(f"[RoutineB] Reached WP {self.idx+1}: "
                          f"pos=({x:.2f},{y:.2f},{z:.2f}) | target=({tx:.2f},{ty:.2f},{tz:.2f})")
            self.idx += 1
            self.sent_depth_for_idx = None


# ============================================================
if __name__ == '__main__':
    try:
        Waypoint3DExecutorRoutineB()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
