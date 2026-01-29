#!/usr/bin/env python3
# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================

import rospy
import numpy as np
from cavepi_interfaces.msg import cavepiInput
from geometry_msgs.msg import Wrench
from std_msgs.msg import Float32
from gazebo_msgs.msg import LinkStates
from scipy.spatial.transform import Rotation as R
import tf


class cavepiControlNode:
    """
    Safer rewrite of the original node:
      - No stale thrust: missing/None inputs become 0
      - Deadband + clamp on commands
      - Optional 'armed' gate (thrust disabled until first command)
      - Robust link index lookup with caching
      - Symmetric thrust computation
      - Cleaner quaternion/rotation usage
      - Drag model is guarded (no state clipping), optional torque damping
    """

    def __init__(self):
        rospy.init_node("cavepi_control_node")

        # -------------------------
        # Params (tune as needed)
        # -------------------------
        self.max_forward_force = rospy.get_param("~max_forward_force", 51.485)  # N
        self.max_reverse_force = rospy.get_param("~max_reverse_force", 40.207)  # N

        # Command shaping
        self.cmd_deadband = rospy.get_param("~cmd_deadband", 0.02)
        self.cmd_limit = rospy.get_param("~cmd_limit", 1.0)

        # If True, publish zero thrust until at least one user_input is received
        self.require_first_command = rospy.get_param("~require_first_command", True)
        self._have_cmd = False

        # Drag model (can disable quickly)
        self.enable_drag = rospy.get_param("~enable_drag", True)

        # Force drag
        self.rho = rospy.get_param("~water_density", 1000.0)
        self.drag_coef_force = rospy.get_param("~drag_coef_force", 1.2)
        # Projected areas in body axes [Ax, Ay, Az]
        self.area = np.array(rospy.get_param("~projected_area", [0.053, 0.065, 0.087]),
                             dtype=np.float64)

        # Rotational damping (simple, stable) tau = -k * w
        # This is NOT quadratic; it's intentionally stable in simulation.
        self.enable_rot_damping = rospy.get_param("~enable_rot_damping", True)
        self.rot_damping = np.array(rospy.get_param("~rot_damping", [2.0, 2.0, 2.0]),
                                    dtype=np.float64)  # N*m per (rad/s)

        # Cap drag outputs (avoid insane wrenches if something explodes)
        self.max_drag_force = rospy.get_param("~max_drag_force", 200.0)
        self.max_drag_torque = rospy.get_param("~max_drag_torque", 50.0)

        # -------------------------
        # State (thrust commands)
        # -------------------------
        self.front_right_thrust_raw = 0.0
        self.front_left_thrust_raw = 0.0
        self.rear_right_thrust_raw = 0.0
        self.rear_left_thrust_raw = 0.0

        # Cached indices into /gazebo/link_states
        self.idx = {
            "fr": None,
            "fl": None,
            "rr": None,
            "rl": None,
            "base": None,
        }

        # -------------------------
        # ROS I/O
        # -------------------------
        self.user_input_sub = rospy.Subscriber(
            "/cavepi/user_input", cavepiInput, self.input_callback, queue_size=1
        )
        self.gazebo_states_sub = rospy.Subscriber(
            "/gazebo/link_states", LinkStates, self.states_callback, queue_size=1
        )

        self.front_right_thrust_pub = rospy.Publisher("/cavepi/front_right_thrust", Wrench, queue_size=1)
        self.front_left_thrust_pub = rospy.Publisher("/cavepi/front_left_thrust", Wrench, queue_size=1)
        self.rear_right_thrust_pub = rospy.Publisher("/cavepi/rear_right_thrust", Wrench, queue_size=1)
        self.rear_left_thrust_pub = rospy.Publisher("/cavepi/rear_left_thrust", Wrench, queue_size=1)

        self.drag_force_pub = rospy.Publisher("/cavepi/drag_force", Wrench, queue_size=1)
        self.deviation_error_pub = rospy.Publisher("/cavepi/deviation_error", Float32, queue_size=1)

        self.tf_broadcaster = tf.TransformBroadcaster()

        rospy.loginfo("cavepi_control_node: initialized")

    # -------------------------
    # Helpers
    # -------------------------
    def _shape_cmd(self, x):
        """None -> 0, apply deadband and clamp to [-cmd_limit, cmd_limit]."""
        if x is None:
            return 0.0
        try:
            x = float(x)
        except Exception:
            return 0.0
        if abs(x) < self.cmd_deadband:
            return 0.0
        return float(np.clip(x, -self.cmd_limit, self.cmd_limit))

    def _ensure_indices(self, names):
        """Cache link indices once (or refresh if missing)."""
        if self.idx["base"] is not None:
            return True

        wanted = {
            "fr": "cavepi::front_right_thruster_link_cavepi",
            "fl": "cavepi::front_left_thruster_link_cavepi",
            "rr": "cavepi::rear_right_thruster_link_cavepi",
            "rl": "cavepi::rear_left_thruster_link_cavepi",
            "base": "cavepi::base_link",
        }

        found = {}
        for k, n in wanted.items():
            try:
                found[k] = names.index(n)
            except ValueError:
                found[k] = None

        if any(found[k] is None for k in found):
            # Not ready yet; Gazebo might not have published all links
            return False

        self.idx.update(found)
        rospy.loginfo("Cached link indices: %s", str(self.idx))
        return True

    def _rot_from_pose(self, pose):
        """Return scipy Rotation from geometry_msgs/Pose orientation."""
        q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                     dtype=np.float64)
        # Normalize safely
        n = np.linalg.norm(q)
        if n < 1e-12 or not np.isfinite(n):
            # Identity if invalid
            return R.from_quat([0.0, 0.0, 0.0, 1.0])
        q /= n
        return R.from_quat(q)

    def _publish_zero_thrust(self):
        z = Wrench()
        self.front_right_thrust_pub.publish(z)
        self.front_left_thrust_pub.publish(z)
        self.rear_right_thrust_pub.publish(z)
        self.rear_left_thrust_pub.publish(z)

    # -------------------------
    # Callbacks
    # -------------------------
    def input_callback(self, msg):
        # Shape commands
        surge = self._shape_cmd(getattr(msg, "surge", None))
        heave = self._shape_cmd(getattr(msg, "heave", None))
        roll = self._shape_cmd(getattr(msg, "roll", None))
        yaw = self._shape_cmd(getattr(msg, "yaw", None))

        # Mark armed if we require first command
        if self.require_first_command:
            if (surge != 0.0) or (heave != 0.0) or (roll != 0.0) or (yaw != 0.0):
                self._have_cmd = True
        else:
            self._have_cmd = True

        # Compute force scale (symmetric)
        surge_force = self.max_forward_force if surge >= 0.0 else self.max_reverse_force
        heave_force = self.max_forward_force if heave <= 0.0 else self.max_reverse_force

        # Rear thrusters control surge+yaw
        self.rear_left_thrust_raw = surge_force * (surge + yaw)
        self.rear_right_thrust_raw = surge_force * (surge - yaw)

        # Front thrusters control heave+roll
        self.front_left_thrust_raw = heave_force * (heave + roll)
        self.front_right_thrust_raw = heave_force * (heave - roll)

    def states_callback(self, msg):
        # Ensure indices exist
        if not self._ensure_indices(msg.name):
            return

        base_i = self.idx["base"]
        fr_i = self.idx["fr"]
        fl_i = self.idx["fl"]
        rr_i = self.idx["rr"]
        rl_i = self.idx["rl"]

        # Base pose/orientation
        base_pose = msg.pose[base_i]
        pos = base_pose.position
        ori = base_pose.orientation

        # Broadcast odom->base_link
        self.tf_broadcaster.sendTransform(
            (pos.x, pos.y, pos.z),
            (ori.x, ori.y, ori.z, ori.w),
            rospy.Time.now(),
            "base_link",
            "odom"
        )

        # Deviation error
        dev = Float32()
        dev.data = float(pos.y)
        self.deviation_error_pub.publish(dev)

        # If not armed yet, keep thrust zero (prevents stale forces at startup)
        if self.require_first_command and not self._have_cmd:
            self._publish_zero_thrust()
        else:
            # Publish thrusts in WORLD frame using each thruster link orientation.
            # We define the desired thrust vector in the THRUSTER-LINK frame:
            #   - Front thrusters: +Z
            #   - Rear thrusters: +X
            fr_R = self._rot_from_pose(msg.pose[fr_i])
            fl_R = self._rot_from_pose(msg.pose[fl_i])
            rr_R = self._rot_from_pose(msg.pose[rr_i])
            rl_R = self._rot_from_pose(msg.pose[rl_i])

            # Desired in-link vectors
            fr_vec_link = np.array([0.0, 0.0, self.front_right_thrust_raw], dtype=np.float64)
            fl_vec_link = np.array([0.0, 0.0, self.front_left_thrust_raw], dtype=np.float64)
            rr_vec_link = np.array([self.rear_right_thrust_raw, 0.0, 0.0], dtype=np.float64)
            rl_vec_link = np.array([self.rear_left_thrust_raw, 0.0, 0.0], dtype=np.float64)

            # Rotate to world
            fr_vec_world = fr_R.apply(fr_vec_link)
            fl_vec_world = fl_R.apply(fl_vec_link)
            rr_vec_world = rr_R.apply(rr_vec_link)
            rl_vec_world = rl_R.apply(rl_vec_link)

            # Publish
            w = Wrench()
            w.force.x, w.force.y, w.force.z = fr_vec_world.tolist()
            self.front_right_thrust_pub.publish(w)

            w = Wrench()
            w.force.x, w.force.y, w.force.z = fl_vec_world.tolist()
            self.front_left_thrust_pub.publish(w)

            w = Wrench()
            w.force.x, w.force.y, w.force.z = rr_vec_world.tolist()
            self.rear_right_thrust_pub.publish(w)

            w = Wrench()
            w.force.x, w.force.y, w.force.z = rl_vec_world.tolist()
            self.rear_left_thrust_pub.publish(w)

        # Drag wrench
        if not self.enable_drag:
            return

        # Base rotation: body->world
        base_R = self._rot_from_pose(base_pose)
        R_bw = base_R.as_matrix()            # body -> world
        R_wb = R_bw.T                        # world -> body

        # Velocities in world (Gazebo gives twist in world frame for LinkStates)
        v_w = np.array([msg.twist[base_i].linear.x,
                        msg.twist[base_i].linear.y,
                        msg.twist[base_i].linear.z], dtype=np.float64)

        w_w = np.array([msg.twist[base_i].angular.x,
                        msg.twist[base_i].angular.y,
                        msg.twist[base_i].angular.z], dtype=np.float64)

        if (not np.all(np.isfinite(v_w))) or (not np.all(np.isfinite(w_w))):
            return

        # Convert to body frame
        v_b = R_wb @ v_w
        w_b = R_wb @ w_w

        # Quadratic drag force in body axes: F = -0.5*rho*Cd*A*v*|v|
        drag_force_b = -0.5 * self.rho * self.drag_coef_force * self.area * v_b * np.abs(v_b)

        # Rotational damping torque (stable): tau = -k * w
        if self.enable_rot_damping:
            drag_torque_b = -self.rot_damping * w_b
        else:
            drag_torque_b = np.zeros(3, dtype=np.float64)

        # Cap outputs to avoid destabilizing physics
        drag_force_b = np.clip(drag_force_b, -self.max_drag_force, self.max_drag_force)
        drag_torque_b = np.clip(drag_torque_b, -self.max_drag_torque, self.max_drag_torque)

        # Convert back to world
        drag_force_w = R_bw @ drag_force_b
        drag_torque_w = R_bw @ drag_torque_b

        drag = Wrench()
        drag.force.x, drag.force.y, drag.force.z = drag_force_w.tolist()
        drag.torque.x, drag.torque.y, drag.torque.z = drag_torque_w.tolist()
        self.drag_force_pub.publish(drag)


if __name__ == "__main__":
    try:
        node = cavepiControlNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
