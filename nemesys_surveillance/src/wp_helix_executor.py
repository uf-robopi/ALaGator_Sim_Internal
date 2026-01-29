#!/usr/bin/env python3
# =====================================
# Author : Adnan Abdullah (helix variant by ChatGPT)
# Email  : adnanabdullah@ufl.edu
# =====================================

import rospy
import math
import numpy as np
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
from gazebo_msgs.msg import LinkStates
from nemesys_interfaces.msg import nemesysInput as NemesysInput

class Helix3DExecutor:
    def __init__(self):
        rospy.init_node('helix_3d_executor')

        # ===== Helix params =====
        # x(t) = R cos(omega t), y(t) = R sin(omega t), z(t) = z0 + vz t
        self.R      = float(rospy.get_param('~R', 2.0))                # meters
        self.omega  = float(rospy.get_param('~omega', 2.0*math.pi/60)) # rad/s
        self.vz     = float(rospy.get_param('~vz', 0.015))              # m/s
        self.z0     = float(rospy.get_param('~z0', -59.0))             # m
        self.phase0 = float(rospy.get_param('~phase0', 0.0))           # radians, optional phase offset

        # Run duration control (stop after some time or turns). Set either, or leave both <=0 for continuous.
        self.duration_s = float(rospy.get_param('~duration_s', 30.0)) # match standalone T=320
        self.num_turns  = float(rospy.get_param('~num_turns', 2.0))    # if >0, overrides duration

        # Lookahead for smoother tracking of a moving path (seconds along t)
        self.lookahead_s = float(rospy.get_param('~lookahead_s', 0.8))

        # ===== Control params =====
        self.xy_tol        = rospy.get_param('~xy_tolerance', 0.15)   # m (used for info; control is continuous)
        self.z_tol         = rospy.get_param('~z_tolerance',  0.2)    # m (info only; depth handled by depth node)
        self.yaw_tol       = rospy.get_param('~yaw_tolerance',  0.5)  # deg (info only)
        self.speed         = rospy.get_param('~speed',         0.6)   # surge limit (0..1 fraction of max)
        self.yaw_kp        = rospy.get_param('~yaw_kp',        0.02)
        self.surge_kp      = rospy.get_param('~surge_kp',      0.05)
        self.yaw_max       = rospy.get_param('~yaw_max',       0.5)   # +/- cmd limit
        self.heave_scale   = rospy.get_param('~heave_scale',   0.05)  # scale /nemesys/heave_control_input -> thrusters
        # Compensate DepthControlNode's "target_depth = msg - 60" behavior:
        self.depth_controller_bias = rospy.get_param('~depth_controller_bias', 0.0)

        # ===== State =====
        self.current_pos = None     # np.array([x, y, z])
        self.current_yaw = None     # deg (-180..180]
        self.heave_cmd_in = 0.0     # from depth node

        # ===== I/O =====
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_cb, queue_size=10)
        rospy.Subscriber('/nemesys/euler_angles', Vector3, self.euler_cb, queue_size=50)
        rospy.Subscriber('/nemesys/heave_control_input', Float32, self.heave_cb, queue_size=50)

        self.cmd_pub = rospy.Publisher('/nemesys/user_input', NemesysInput, queue_size=10)
        self.target_depth_pub = rospy.Publisher('/nemesys/target_depth', Float32, queue_size=10)

        # Command message scaffold
        self.cmd = NemesysInput()
        self.cmd.roll = 0.0
        self.cmd.heave = 0.0

        # Timing
        self.dt_cmd = float(rospy.get_param('~dt_cmd', 0.1))
        self.t0 = rospy.Time.now()
        self.timer = rospy.Timer(rospy.Duration(self.dt_cmd), self.update)

        # Precompute stop time if num_turns specified
        if self.num_turns > 0.0:
            turns_duration = (2.0 * math.pi * self.num_turns) / self.omega
            self.stop_time = self.t0 + rospy.Duration(turns_duration)
        elif self.duration_s > 0.0:
            self.stop_time = self.t0 + rospy.Duration(self.duration_s)
        else:
            self.stop_time = None  # run forever

        rospy.loginfo(
            "Helix3DExecutor up. R=%.3f m, omega=%.4f rad/s (T=%.1f s), vz=%.3f m/s, z0=%.2f, lookahead=%.2f s",
            self.R, self.omega, (2.0*math.pi/self.omega), self.vz, self.z0, self.lookahead_s
        )

    # ---------- Callbacks ----------
    def link_cb(self, msg):
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

    # ---------- Helpers ----------
    @staticmethod
    def wrap_yaw_diff_deg(current, target):
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
        We publish (-z_world + bias) so target_depth == z_world inside that node.
        """
        out = Float32()
        out.data = float(-z_world + self.depth_controller_bias)
        self.target_depth_pub.publish(out)

    def helix_state(self, t_sec):
        """Return desired (pos, vel_xy) at time t: world-frame helix and its tangent in XY."""
        theta = self.omega * t_sec + self.phase0
        x = self.R * math.cos(theta)
        y = self.R * math.sin(theta)
        z = self.z0 + self.vz * t_sec

        # Tangential velocity in XY (derivative wrt time)
        vx = -self.R * self.omega * math.sin(theta)
        vy =  self.R * self.omega * math.cos(theta)
        return np.array([x, y, z], dtype=float), np.array([vx, vy], dtype=float)

    # ---------- Main loop ----------
    def update(self, event):
        # Stop if time elapsed
        if self.stop_time is not None and rospy.Time.now() >= self.stop_time:
            self.cmd.surge = 0.0
            self.cmd.yaw = 0.0
            self.cmd.heave = 0.0
            self.cmd_pub.publish(self.cmd)
            rospy.loginfo_throttle(2.0, "Helix complete.")
            return

        if self.current_pos is None or self.current_yaw is None:
            rospy.loginfo_throttle(2.0, "Waiting for pose/yaw...")
            return

        # Current time since start
        t = (rospy.Time.now() - self.t0).to_sec()

        # Lookahead to reduce “chasing the target” in XY
        t_la = max(0.0, t + self.lookahead_s)

        # Desired target and tangent (lookahead)
        pd, vxy = self.helix_state(t_la)
        tx, ty, tz = pd

        # (1) Depth target through depth controller
        self.publish_target_depth(tz)

        # (2) Yaw to point along helix tangent (or default to pointing to target if tangent is tiny)
        x, y, z = self.current_pos
        if np.linalg.norm(vxy) > 1e-6:
            target_yaw = math.degrees(math.atan2(vxy[1], vxy[0]))
        else:
            target_yaw = math.degrees(math.atan2(ty - y, tx - x))  # fallback to pointing at target

        yaw_err = self.wrap_yaw_diff_deg(self.current_yaw, target_yaw)
        yaw_cmd = self.compute_yaw_cmd(yaw_err)

        # (3) Surge toward XY target (lookahead point)
        dist_xy = math.hypot(tx - x, ty - y)
        surge_cmd = self.compute_surge_cmd(dist_xy)

        # Heave pass-through from depth controller output
        heave_cmd = float(self.heave_cmd_in * self.heave_scale)

        # Publish command
        self.cmd.surge = surge_cmd
        self.cmd.yaw   = yaw_cmd
        self.cmd.heave = heave_cmd
        self.cmd_pub.publish(self.cmd)

        rospy.loginfo_throttle(
            2.0,
            f"[Helix] t={t:.1f}s | tgt=({tx:.2f},{ty:.2f},{tz:.2f}) | "
            f"pos=({x:.2f},{y:.2f},{z:.2f}) | d_xy={dist_xy:.2f}m | yaw_err={yaw_err:.1f}°"
        )

# -------------------------
if __name__ == '__main__':
    try:
        Helix3DExecutor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
