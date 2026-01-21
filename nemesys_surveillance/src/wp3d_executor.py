#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================

import rospy
import math
import numpy as np
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
from gazebo_msgs.msg import LinkStates
from nemesys_interfaces.msg import nemesysInput as NemesysInput

class Waypoint3DExecutor:
    def __init__(self):
        rospy.init_node('waypoint_3d_executor')

        # ===== Waypoints: edit these =====
        # World-frame targets (meters), z is the world Z (e.g., -59.0)
        self.waypoints = [
            (-1.5, -1.0, -58.7),
            (-1.5, -1.0, -59.4),
            (-1.5, 1.0, -59.4),
            (2.0, -1.0, -58.7),
            ( 2.0, 1.0, -59.4),
            ( -1.5, -1.0, -58.7),
            ( -1.5, -1.0, -59.1),
        ]

        # ===== Params =====
        self.xy_tol   = rospy.get_param('~xy_tolerance', 0.2)   # m
        self.z_tol    = rospy.get_param('~z_tolerance',  0.15)   # m
        self.yaw_tol    = rospy.get_param('~yaw_tolerance',  10.0)   # degree
        self.speed    = rospy.get_param('~speed',         0.6)   # surge limit (0..1 fraction of max)
        self.yaw_kp   = rospy.get_param('~yaw_kp',        0.02)
        self.surge_kp = rospy.get_param('~surge_kp',      0.1)
        self.yaw_max  = rospy.get_param('~yaw_max',       0.5)   # +/- cmd limit
        self.heave_scale = rospy.get_param('~heave_scale', 1) # scale /heave_control_input -> thrusters
        # Compensate DepthControlNode's "target_depth = msg - 60" behavior:
        self.depth_controller_bias = rospy.get_param('~depth_controller_bias', 0.0)

        # ===== State =====
        self.current_pos = None     # np.array([x, y, z])
        self.current_yaw = None     # deg (-180..180]
        self.heave_cmd_in = 0.0     # from depth node
        self.idx = 0
        self.sent_depth_for_idx = None

        # ===== I/O =====
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_cb, queue_size=10)
        rospy.Subscriber('/euler_angles', Vector3, self.euler_cb, queue_size=50)
        rospy.Subscriber('/heave_control_input', Float32, self.heave_cb, queue_size=50)

        self.cmd_pub = rospy.Publisher('/nemesys/user_input', NemesysInput, queue_size=10)
        self.target_depth_pub = rospy.Publisher('/target_depth', Float32, queue_size=10)

        self.cmd = NemesysInput()
        self.cmd.roll = 0.0
        self.cmd.heave = 0.0

        self.timer = rospy.Timer(rospy.Duration(0.1), self.update)
        rospy.loginfo("Waypoint3DExecutor up. Waypoints: %d" % len(self.waypoints))

    # ---------- Callbacks ----------
    def link_cb(self, msg):
        # Find base link pose
        try:
            i = msg.name.index("nemesys::base_link")
        except ValueError:
            return
        p = msg.pose[i].position
        self.current_pos = np.array([p.x, p.y, p.z], dtype=float)

    def euler_cb(self, msg):
        # Expect msg.z in degrees; wrap into (-180, 180]
        yaw = float(msg.z)
        self.current_yaw = ((yaw + 180.0) % 360.0) - 180.0

    def heave_cb(self, msg):
        self.heave_cmd_in = float(msg.data)

    # ---------- Control helpers ----------
    @staticmethod
    def shortest_yaw_diff_deg(current, target):
        # returns signed diff in (-180, 180]
        return ((target - current + 180.0) % 360.0) - 180.0

    def compute_yaw_cmd(self, yaw_error_deg):
        u = -self.yaw_kp * yaw_error_deg
        return float(np.clip(u, -self.yaw_max, self.yaw_max))

    def compute_surge_cmd(self, dist_xy):
        u = self.surge_kp * dist_xy
        return float(np.clip(u, 0.0, self.speed))

    def publish_target_depth(self, z_world):
        """
        DepthControlNode does: target_depth = msg - 60.
        So publishing (z_world + 60) makes its internal target_depth == z_world.
        """
        out = Float32()
        out.data = float(-z_world + self.depth_controller_bias)
        self.target_depth_pub.publish(out)

    # ---------- Main loop ----------
    def update(self, event):
        self.cmd.surge = 0.0
        self.cmd.yaw   = 0.0
        if self.current_pos is None or self.current_yaw is None:
            rospy.loginfo_throttle(2.0, "Waiting for pose/yaw...")
            return

        # if self.idx >= len(self.waypoints):
        #     # Stop
        #     self.cmd.surge = 0.0
        #     self.cmd.yaw = 0.0
        #     self.cmd.heave = 0.0
        #     self.cmd_pub.publish(self.cmd)
        #     rospy.loginfo_throttle(2.0, "All waypoints completed.")
        #     return

        # If we've gone past the last waypoint, wrap around and start again
        if self.idx >= len(self.waypoints):
            rospy.loginfo("Completed all waypoints, restarting from the first one.")
            self.idx = 0
            self.sent_depth_for_idx = None  # force publishing depth for new WP

        # Current target
        tx, ty, tz = self.waypoints[self.idx]

        # Depth target publication (once per waypoint, and occasionally to refresh)
        if self.sent_depth_for_idx != self.idx:
            self.publish_target_depth(tz)
            self.sent_depth_for_idx = self.idx
            rospy.loginfo(f"[WP {self.idx+1}/{len(self.waypoints)}] Target (x,y,z)=({tx:.2f},{ty:.2f},{tz:.2f})")

        # State
        x, y, z = self.current_pos
        # Errors
        dist_xy = math.hypot(tx - x, ty - y)
        z_err   = tz - z  # world-z error

        # Yaw towards XY target
        target_yaw = math.degrees(math.atan2(ty - y, tx - x)) if dist_xy > 1e-6 else self.current_yaw
        yaw_err = self.shortest_yaw_diff_deg(self.current_yaw, target_yaw)

        # Control
        if dist_xy > self.xy_tol:
            # Align first if far off heading
            if abs(yaw_err) > self.yaw_tol:
                # print(f"yaw_err={yaw_err:.2f}°")
                self.cmd.surge = 0.0
                self.cmd.yaw = self.compute_yaw_cmd(yaw_err)
            else:
                # print(f"No yaw correction, dist_xy={dist_xy:.2f} m")
                self.cmd.surge = self.compute_surge_cmd(dist_xy)
                self.cmd.yaw = 0.0

        # Heave: pass-through from depth controller
        self.cmd.heave = float(self.heave_cmd_in * self.heave_scale)

        # Publish command
        self.cmd_pub.publish(self.cmd)

        # Check completion (XY and Z)
        if (dist_xy <= self.xy_tol) and (abs(z_err) <= self.z_tol):
            rospy.loginfo(f"Reached WP {self.idx+1}: "
                          f"pos=({x:.2f},{y:.2f},{z:.2f}) | target=({tx:.2f},{ty:.2f},{tz:.2f})")
            self.idx += 1
            self.sent_depth_for_idx = None  # force publishing depth for next WP

# -------------------------
if __name__ == '__main__':
    try:
        Waypoint3DExecutor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
