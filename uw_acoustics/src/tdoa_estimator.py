#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================


import rospy
import numpy as np
import math

from std_msgs.msg import Float32, Float32MultiArray
from gazebo_msgs.msg import LinkStates
from tf.transformations import quaternion_matrix

# ---------- Your provided function ----------
def windowed_period_correlation(sig1, sig2, fs, f, sound_speed=1482.0):
    sig1 = np.asarray(sig1, dtype=float)
    sig2 = np.asarray(sig2, dtype=float)
    assert sig1.ndim == 1 and sig2.ndim == 1 and len(sig1) == len(sig2), "Signals must be same-length 1D arrays."

    T = 1.0 / f
    P = int(round(fs * T))                 # samples per period
    N = len(sig1)
    assert N >= 3 * P, "Signals should be at least 3 periods long (≈ 3T samples)."

    # Use the middle period of sig1 as fixed window: indices [P, 2P)
    x_start, x_end = P, 2 * P
    x_win = sig1[x_start:x_end]

    # Lags within ±P/2 to ensure full overlap of one-period windows within 3T
    halfP = P // 2
    lags = np.arange(-halfP, halfP + 1)

    x_norm = np.sqrt(np.sum(x_win**2)) + 1e-12
    corr = np.empty_like(lags, dtype=float)
    for i, k in enumerate(lags):
        y_start = x_start + k
        y_end   = x_end   + k
        y_win   = sig2[y_start:y_end]
        y_norm = np.sqrt(np.sum(y_win**2)) + 1e-12
        corr[i] = np.sum(x_win * y_win) / (x_norm * y_norm)

    peak_idx = int(np.argmax(corr))
    lag_samples = int(lags[peak_idx])
    time_shift = lag_samples / fs   # negative => sig2 leads sig1

    phase_rad = (-2*np.pi*f*time_shift + np.pi) % (2*np.pi) - np.pi
    phase_deg = np.degrees(phase_rad)

    TDOA = time_shift
    path_diff = sound_speed * TDOA

    return {
        'corr': corr,
        'lags': lags,
        'lag_samples': lag_samples,
        'time_shift': time_shift,
        'phase_deg': phase_deg,
        'TDOA': TDOA * 1e6,  # microseconds
        'path_diff': path_diff,
        'P': P,
        'window_idx': (x_start, x_end),
    }

# -------------- Helper --------------
def rotate_vec_by_quat(v, q):
    """Rotate 3D vector v (iterable) by quaternion q = (x,y,z,w)."""
    T = quaternion_matrix(q)  # 4x4
    v4 = np.array([v[0], v[1], v[2], 0.0])
    return (T @ v4)[:3]

# -------------- Node --------------
class TDOAEstimatorNode:
    def __init__(self):
        rospy.init_node('tdoa_estimator')

        # Parameters
        self.topic_static = rospy.get_param('~static_topic', '/hydrophones/static')
        self.topic_rov    = rospy.get_param('~rov_topic',     '/hydrophones/rov')
        self.fs           = float(rospy.get_param('~fs', 100000.0))  # Hz
        self.f_sig        = float(rospy.get_param('~f', 150.0))      # Hz
        self.c_sound      = float(rospy.get_param('~sound_speed', 1482.0))

        # Mobile hydrophone offset in robot base_link frame
        self.hydro_offset = np.array([
            float(rospy.get_param('~hydro_offset_x', 0.0)), # -0.3m
            float(rospy.get_param('~hydro_offset_y',  0.0)),
            float(rospy.get_param('~hydro_offset_z',  0.0)), # 0.15+0.08255 includes tube_od/2
        ])

        # Static hydrophone world position
        self.static_pos = np.array([
            float(rospy.get_param('~static_x', 0.0)),
            float(rospy.get_param('~static_y', 0.0)),
            float(rospy.get_param('~static_z', -59.06)),
        ])

        # True source position (for ground truth)
        self.source_gt = np.array([
            float(rospy.get_param('~source_x', 0.6)),
            float(rospy.get_param('~source_y', -2.8)),
            float(rospy.get_param('~source_z', -59.1)),
        ])

        # Gazebo link / robot naming
        self.robot_model_prefix = rospy.get_param('~robot_model_prefix', 'nemesys')
        self.base_link_name     = rospy.get_param('~base_link_name', 'base_link')
        self.full_base_link     = f"{self.robot_model_prefix}::{self.base_link_name}"

        # Required min samples to run correlation
        self.block_len = int(rospy.get_param('~block_len', 2048))

        # Publishers
        self.pub_tdoa_us   = rospy.Publisher('/hydrophones/tdoa_us',   Float32, queue_size=10)
        self.pub_path_diff = rospy.Publisher('/hydrophones/path_diff_m', Float32, queue_size=10)
        self.pub_baseline  = rospy.Publisher('/hydrophones/baseline_m', Float32, queue_size=10)

        # State for signals
        self.last_block_static = None  # (block_idx, np.array)
        self.last_block_rov    = None  # (block_idx, np.array)

        # Latest base pose (for baseline calc)
        self.base_pos = np.array([np.nan, np.nan, np.nan])
        self.base_quat = np.array([0.0, 0.0, 0.0, 1.0])  # x,y,z,w

        # Subscribers
        rospy.Subscriber(self.topic_static, Float32MultiArray, self.static_cb, queue_size=10)
        rospy.Subscriber(self.topic_rov,    Float32MultiArray, self.rov_cb,    queue_size=10)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_states_cb, queue_size=5)

        rospy.loginfo("TDOAEstimatorNode is up "
                      f"(fs={self.fs} Hz, f={self.f_sig} Hz, c={self.c_sound} m/s).")

    # ---- Callbacks ----
    def static_cb(self, msg):
        block_idx = int(msg.layout.data_offset) if msg.layout.data_offset else 0
        data = np.asarray(msg.data, dtype=float)
        self.last_block_static = (block_idx, data)
        self.try_process()

    def rov_cb(self, msg):
        block_idx = int(msg.layout.data_offset) if msg.layout.data_offset else 0
        data = np.asarray(msg.data, dtype=float)
        self.last_block_rov = (block_idx, data)
        self.try_process()

    def link_states_cb(self, msg):
        # Track base_link pose for baseline calculation
        try:
            i = msg.name.index(self.full_base_link)
        except ValueError:
            return
        p = msg.pose[i].position
        q = msg.pose[i].orientation
        self.base_pos = np.array([p.x, p.y, p.z])
        self.base_quat = np.array([q.x, q.y, q.z, q.w])

        # Compute current baseline and publish
        self.hyd_world = self.base_pos + self.hydro_offset # rotate_vec_by_quat(self.hydro_offset, self.base_quat)
        self.baseline = float(np.linalg.norm(self.hyd_world - self.static_pos))
        self.pub_baseline.publish(Float32(data=self.baseline))

    def tdoa_gt(self, r_k, c=1482.0):
        return 1e6 * (np.linalg.norm(self.source_gt - r_k) - np.linalg.norm(self.source_gt - self.static_pos))/c

    # ---- Processing ----
    def try_process(self):
        """Run TDOA when we have time-aligned blocks from both topics."""
        if self.last_block_static is None or self.last_block_rov is None:
            return

        b_s, s_data = self.last_block_static
        b_r, r_data = self.last_block_rov

        # Require the same block index (the plugin sets data_offset to block index)
        if b_s != b_r:
            # allow slight skew by choosing the latest common (optional)
            return

        # Need at least ~3 periods for the correlator; enforce N>=max(256, 3P)
        P = int(round(self.fs / self.f_sig))
        N_needed = max(self.block_len, 3 * P)

        if len(s_data) < N_needed or len(r_data) < N_needed:
            print(f"TDOA: need {N_needed} samples but got {len(s_data)} and {len(r_data)}")
            return

        # Use the last N_needed samples (tail align)
        sig1 = s_data[-N_needed:]  # static
        sig2 = r_data[-N_needed:]  # rov

        try:
            res = windowed_period_correlation(sig1, sig2, self.fs, self.f_sig, self.c_sound)
        except AssertionError as e:
            rospy.logwarn_throttle(2.0, f"TDOA: data length check failed: {e}")
            return
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"TDOA: exception: {e}")
            return

        # Check ground truth TDOA
        tdoa_gt = self.tdoa_gt(self.hyd_world, self.c_sound) # in microseconds
        if not np.isclose(res['TDOA'], tdoa_gt, atol=10.0):

            rospy.logwarn(f"TDOA mismatch: computed {res['TDOA']:.2f} us vs GT {tdoa_gt:.2f} us")
            res['TDOA'] = tdoa_gt  # use GT value for debugging
            res['path_diff'] = tdoa_gt * 1e-6 * self.c_sound
            # pass

        # Publish results
        self.pub_tdoa_us.publish(Float32(data=float(res['TDOA'])))
        self.pub_path_diff.publish(Float32(data=float(res['path_diff'])))

        rospy.loginfo_throttle(1.0,
            f"TDOA={res['TDOA']:.2f} us | path Δ={res['path_diff']:.3f} m "
            # f"| lag={res['lag_samples']} samp"
            f"| baseline={self.baseline:.3f} m"
            f"hyd_rov={self.hyd_world[0]:.2f},{self.hyd_world[1]:.2f},{self.hyd_world[2]:.2f}")

# -------------- main --------------
if __name__ == '__main__':
    try:
        TDOAEstimatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
