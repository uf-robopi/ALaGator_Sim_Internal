#!/usr/bin/env python3
# =====================================
# Author : Adnan Abdullah
# =====================================

import rospy
import math
import numpy as np
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
from gazebo_msgs.msg import LinkStates
from nemesys_interfaces.msg import nemesysInput as NemesysInput

# ============================================================
# Geometry helper
# ============================================================
def rect_waypoints_around_pod(
    pod_xy,
    pod_yaw,
    rect_rel_xy,
):
    """
    Rotate + translate a canonical rectangle around a pod.
    """
    x0, y0 = pod_xy
    c = math.cos(pod_yaw)
    s = math.sin(pod_yaw)

    wps = []
    for xr, yr in rect_rel_xy:
        xw = x0 + c * xr - s * yr
        yw = y0 + s * xr + c * yr
        wps.append((xw, yw))
    return wps


# ============================================================
# Waypoint Executor
# ============================================================
class Waypoint3DExecutor:
    def __init__(self):
        rospy.init_node('wp_spoke_executor')

        # ==============================
        # Parameters
        # ==============================
        self.xy_tol   = rospy.get_param('~xy_tolerance', 0.2)
        self.z_tol    = rospy.get_param('~z_tolerance',  0.15)
        self.yaw_tol  = rospy.get_param('~yaw_tolerance', 10.0)
        self.speed   = rospy.get_param('~speed', 0.6)
        self.yaw_kp  = rospy.get_param('~yaw_kp', 0.02)
        self.surge_kp = rospy.get_param('~surge_kp', 0.1)
        self.yaw_max = rospy.get_param('~yaw_max', 0.7)
        self.heave_scale = rospy.get_param('~heave_scale', 1.0)

        # Depth controller offset
        self.depth_controller_bias = rospy.get_param('~depth_controller_bias', 0.0)

        # ==============================
        # Pod definitions (world frame)
        # ==============================
        self.pods = [
            ("origin",       (0.0,    0.0),    0.0),
            ("pos30",        (-1.38,  5.15),   0.5235),
            ("neg30",        (-1.38, -5.15),  -0.5235),
            ("pos90",        (-10.3, 10.3),    1.5708),
            ("neg90",        (-10.3,-10.3),   -1.5708),
        ]

        self.pods = [
            ("left",       (0.0,    4.5),    1.57),
            ("")
        ]

        # ==============================
        # Canonical rectangle (origin pod)
        # ==============================
        self.rect_rel_xy = [
            (-1.5, -1.0),
            (-1.5,  1.0),
            ( 2.0,  1.0),
            ( 2.0, -1.0),
        ]

        # Z values
        self.z_pipeline = -59.4
        self.z_hub      = -58.5

        # Hub (center junction)
        self.hub_xy = (-10.0, 0.0)

        # ==============================
        # Build waypoint list
        # ==============================
        self.waypoints = self.build_waypoints()
        rospy.loginfo(f"Generated {len(self.waypoints)} waypoints")

        # ==============================
        # State
        # ==============================
        self.current_pos = None
        self.current_yaw = None
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
    # Waypoint generation
    # ========================================================
    def build_waypoints(self):
        wps = []

        for name, (x0, y0), yaw in self.pods:
            # Go to hub
            wps.append((*self.hub_xy, self.z_hub))

            # Move onto pipeline near pod
            # wps.append((x0, y0, self.z_pipeline))

            # Rectangle around pod
            rect_xy = rect_waypoints_around_pod((x0, y0), yaw, self.rect_rel_xy)
            for x, y in rect_xy:
                wps.append((x, y, self.z_pipeline))

            # Close loop
            wps.append((rect_xy[0][0], rect_xy[0][1], self.z_pipeline))

        # Final return to hub
        wps.append((*self.hub_xy, self.z_hub))
        return wps

    # ========================================================
    # Callbacks
    # ========================================================
    def link_cb(self, msg):
        try:
            i = msg.name.index("nemesys::base_link")
            p = msg.pose[i].position
            self.current_pos = np.array([p.x, p.y, p.z])
        except ValueError:
            pass

    def euler_cb(self, msg):
        self.current_yaw = ((msg.z + 180.0) % 360.0) - 180.0

    def heave_cb(self, msg):
        self.heave_cmd_in = msg.data

    # ========================================================
    # Control helpers
    # ========================================================
    @staticmethod
    def yaw_diff(a, b):
        return ((b - a + 180.0) % 360.0) - 180.0

    def yaw_cmd(self, err):
        return np.clip(-self.yaw_kp * err, -self.yaw_max, self.yaw_max)

    def surge_cmd(self, dist):
        return np.clip(self.surge_kp * dist, 0.0, self.speed)

    def publish_depth(self, z):
        msg = Float32()
        msg.data = -z + self.depth_controller_bias
        self.target_depth_pub.publish(msg)

    # ========================================================
    # Main loop
    # ========================================================
    def update(self, event):
        if self.current_pos is None or self.current_yaw is None:
            return

        if self.idx >= len(self.waypoints):
            self.cmd.surge = 0.0
            self.cmd.yaw = 0.0
            self.cmd.heave = 0.0
            self.cmd_pub.publish(self.cmd)
            return

        tx, ty, tz = self.waypoints[self.idx]
        x, y, z = self.current_pos

        if self.sent_depth_for_idx != self.idx:
            self.publish_depth(tz)
            self.sent_depth_for_idx = self.idx
            rospy.loginfo(f"[WP {self.idx+1}] Target ({tx:.2f},{ty:.2f},{tz:.2f})")

        dx = tx - x
        dy = ty - y
        dist_xy = math.hypot(dx, dy)
        z_err = tz - z

        target_yaw = math.degrees(math.atan2(dy, dx)) if dist_xy > 1e-6 else self.current_yaw
        yaw_err = self.yaw_diff(self.current_yaw, target_yaw)

        if dist_xy > self.xy_tol:
            if abs(yaw_err) > self.yaw_tol:
                self.cmd.surge = 0.0
                self.cmd.yaw = self.yaw_cmd(yaw_err)
            else:
                self.cmd.surge = self.surge_cmd(dist_xy)
                self.cmd.yaw = 0.0

        self.cmd.heave = self.heave_cmd_in * self.heave_scale
        self.cmd_pub.publish(self.cmd)

        if dist_xy <= self.xy_tol and abs(z_err) <= self.z_tol:
            rospy.loginfo(f"Reached WP {self.idx+1}")
            self.idx += 1
            self.sent_depth_for_idx = None


# ============================================================
if __name__ == '__main__':
    try:
        Waypoint3DExecutor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
