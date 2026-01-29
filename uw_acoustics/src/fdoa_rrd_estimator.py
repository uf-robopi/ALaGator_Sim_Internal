#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================


import rospy
import numpy as np
from std_msgs.msg import Float32
from gazebo_msgs.msg import LinkStates, ModelStates
from tf.transformations import quaternion_matrix

def rotmat_from_quat_xyzw(q):
    # q = [x,y,z,w]
    return quaternion_matrix(q)[:3, :3]

class FDOAFromKinematics:
    def __init__(self):
        rospy.init_node("fdoa_from_kinematics")

        # --- Environment constants ---
        self.c  = float(rospy.get_param("~sound_speed", 1482.0))  # m/s
        self.f0 = float(rospy.get_param("~f0_hz", 150.0))         # Hz (0 -> don't publish delta-f)

        # --- Source (will be updated from /gazebo/model_states) ---
        # Initial fallback from params until model_states callback arrives
        self.r_src = np.array([
            float(rospy.get_param("~source_x", 0.6)),
            float(rospy.get_param("~source_y", -2.8)),
            float(rospy.get_param("~source_z", -59.1)),
        ], dtype=float)
        self.v_src = np.array([
            float(rospy.get_param("~source_vx", 0.0)),
            float(rospy.get_param("~source_vy", 0.0)),
            float(rospy.get_param("~source_vz", 0.0)),
        ], dtype=float)
        self.source_model_name = rospy.get_param("~source_model_name", "mobile_acoustic_source")
        self.have_src_state = False

        # --- Fixed hydrophone (world frame) ---
        self.r_fix = np.array([
            float(rospy.get_param("~static_x", 0.0)),
            float(rospy.get_param("~static_y", 0.0)),
            float(rospy.get_param("~static_z", -59.06)),
        ], dtype=float)
        self.v_fix = np.array([
            float(rospy.get_param("~static_vx", 0.0)),
            float(rospy.get_param("~static_vy", 0.0)),
            float(rospy.get_param("~static_vz", 0.0)),
        ], dtype=float)

        # --- Mobile hydrophone offset on the ROV (base_link frame) ---
        off = rospy.get_param("~rov_hydro_offset_xyz", [0., 0., 0.])
        self.offset_b = np.array(off, dtype=float)

        # --- Gazebo naming ---
        self.base_link_name = rospy.get_param("~base_link_name", "nemesys::base_link")

        # --- Publishers ---
        self.pub_fdoa_mps = rospy.Publisher("/hydrophones/fdoa_mps", Float32, queue_size=10)
        self.pub_df_hz    = rospy.Publisher("/hydrophones/delta_f_hz", Float32, queue_size=10)
        self.pub_rr_rov   = rospy.Publisher("/hydrophones/rov_range_rate_mps", Float32, queue_size=10)
        self.pub_rr_fix   = rospy.Publisher("/hydrophones/fix_range_rate_mps", Float32, queue_size=10)

        # --- Subscribers ---
        rospy.Subscriber("/gazebo/link_states",  LinkStates,  self.link_cb,   queue_size=10)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_cb,  queue_size=10)

        rospy.loginfo("[FDOA-Kin] Started. Using kinematic FDOA (range-rate difference, m/s). "
                      "Source from /gazebo/model_states model='%s'", self.source_model_name)

    # Update source position/velocity from model_states
    def model_cb(self, msg: ModelStates):
        try:
            j = msg.name.index(self.source_model_name)
        except ValueError:
            rospy.logwarn_throttle(5.0, "[FDOA-Kin] Source model '%s' not in /gazebo/model_states yet...",
                                   self.source_model_name)
            return

        # Pose (unused here)
        p = msg.pose[j].position
        # Twist (world frame)
        v = msg.twist[j].linear

        self.r_src = np.array([p.x, p.y, p.z], dtype=float)
        self.v_src = np.array([v.x, v.y, v.z], dtype=float)
        self.have_src_state = True

    # Main computation when we get ROV base_link state
    def link_cb(self, msg: LinkStates):
        # Need ROV base_link first
        try:
            i = msg.name.index(self.base_link_name)
        except ValueError:
            return

        # Base pose
        p = msg.pose[i].position
        q = msg.pose[i].orientation
        r_base = np.array([p.x, p.y, p.z], dtype=float)
        q_xyzw = np.array([q.x, q.y, q.z, q.w], dtype=float)
        R_wb   = rotmat_from_quat_xyzw(q_xyzw)  # world<-base

        # Base twist (world frame)
        v = msg.twist[i].linear
        w = msg.twist[i].angular
        v_base = np.array([v.x, v.y, v.z], dtype=float)
        w_base = np.array([w.x, w.y, w.z], dtype=float)

        # Mobile hydrophone position & velocity in world:
        o_w  = R_wb.dot(self.offset_b)                   # offset in world
        r_rov = r_base + o_w
        v_rov = v_base + np.cross(w_base, o_w)           # rigid-body velocity at offset point

        # --- Range-rate for each receiver (positive = increasing range), units: m/s ---
        rr_rov = self.range_rate(r_rx=r_rov, v_rx=v_rov, r_tx=self.r_src, v_tx=self.v_src)
        rr_fix = self.range_rate(r_rx=self.r_fix, v_rx=self.v_fix, r_tx=self.r_src, v_tx=self.v_src)

        # FDOA in m/s = (rov range-rate) - (fixed range-rate)
        fdoa_mps = rr_rov - rr_fix  # [m/s]

        # Publish (units: m/s)
        self.pub_rr_rov.publish(Float32(data=float(rr_rov)))
        self.pub_rr_fix.publish(Float32(data=float(rr_fix)))
        self.pub_fdoa_mps.publish(Float32(data=float(fdoa_mps)))

        # Optional Δf (Hz): Δf = -(f0/c) * Δ(range-rate)
        delta_f = 0.0
        if self.f0 > 0.0:
            delta_f = -(self.f0 / self.c) * fdoa_mps
            self.pub_df_hz.publish(Float32(data=float(delta_f)))

        rospy.loginfo_throttle(5.0,
            f"[FDOA-Kin] rr_fix={rr_fix:.3f} m/s, rr_rov={rr_rov:.3f} m/s, "
            f"FDOA={fdoa_mps:.3f} m/s, Δf={delta_f:.3f} Hz")

    @staticmethod
    def range_rate(r_rx, v_rx, r_tx, v_tx):
        """
        d/dt || r_rx - r_tx || = u · (v_rx - v_tx),
        where u = (r_rx - r_tx)/||r_rx - r_tx|| is the LOS unit vector from TX to RX.
        Returns m/s.
        """
        dr = r_rx - r_tx
        d  = np.linalg.norm(dr) + 1e-12  # avoid 0
        u  = dr / d
        return float(np.dot(u, (v_rx - v_tx)))

if __name__ == "__main__":
    try:
        FDOAFromKinematics()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
