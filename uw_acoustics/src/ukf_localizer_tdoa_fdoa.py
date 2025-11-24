#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================


import rospy
import numpy as np
import math

from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import LinkStates, ModelStates
from tf.transformations import quaternion_matrix

# ---------------------------
# SPD helpers
# ---------------------------
def symmetrize(P):
    return 0.5 * (P + P.T)

def make_spd(P, eps=1e-9):
    """ Symmetrize and floor eigenvalues to keep P SPD. """
    P = symmetrize(P)
    vals, vecs = np.linalg.eigh(P)
    vals = np.maximum(vals, eps)
    return (vecs * vals) @ vecs.T

def spd_cholesky(P, eps=1e-9):
    """ Safe Cholesky of a nearly-SPD matrix. """
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(make_spd(P, eps))

def rotmat_from_quat_xyzw(q):
    return quaternion_matrix(q)[:3, :3]

def safe_norm(v):
    return float(np.linalg.norm(v) + 1e-12)

# ---------------------------
# UKF scaffolding (generic)
# ---------------------------
class UKFConfig:
    __slots__ = ("alpha", "beta", "kappa")
    def __init__(self, alpha=0.2, beta=2.0, kappa=0.0):
        self.alpha = alpha
        self.beta  = beta
        self.kappa = kappa

class UKF:
    def __init__(self, n, m, fx, hx, Q, R, x0, P0, cfg: UKFConfig, dt, eps_spd=1e-9):
        self.n = n
        self.m = m
        self.fx = fx
        self.hx = hx
        self.Q = Q
        self.R = R
        self.x = x0.copy()
        self.P = make_spd(P0, eps_spd)
        self.dt = dt
        self.eps_spd = eps_spd

        self.alpha = cfg.alpha
        self.beta  = cfg.beta
        self.kappa = cfg.kappa

        self.lmbda = self.alpha**2 * (self.n + self.kappa) - self.n
        self.gamma = math.sqrt(self.n + self.lmbda)

        self.Wm = np.full(2*self.n+1, 0.5/(self.n+self.lmbda))
        self.Wc = np.full(2*self.n+1, 0.5/(self.n+self.lmbda))
        self.Wm[0] = self.lmbda/(self.n+self.lmbda)
        self.Wc[0] = self.lmbda/(self.n+self.lmbda) + (1 - self.alpha**2 + self.beta)

    def _sigmas(self, x, P):
        U = spd_cholesky(P + 1e-12*np.eye(self.n), self.eps_spd)
        S = np.zeros((2*self.n+1, self.n))
        S[0] = x
        for i in range(self.n):
            S[i+1]        = x + self.gamma * U[:, i]
            S[self.n+i+1] = x - self.gamma * U[:, i]
        return S

    def predict(self, u=None):
        # Force SPD before sampling
        self.P = make_spd(self.P, self.eps_spd)
        S = self._sigmas(self.x, self.P)
        Sf = np.array([self.fx(s, u, self.dt) for s in S])

        xpred = np.sum(self.Wm[:, None] * Sf, axis=0)
        Ppred = np.zeros((self.n, self.n))
        for i in range(Sf.shape[0]):
            d = (Sf[i] - xpred)[:, None]
            Ppred += self.Wc[i] * (d @ d.T)
        Ppred += self.Q

        self.x = xpred
        self.P = make_spd(Ppred, self.eps_spd)

    def update(self, z, u=None):
        # Sigma points from SPD P
        self.P = make_spd(self.P, self.eps_spd)
        S = self._sigmas(self.x, self.P)
        Z = np.array([self.hx(s, u) for s in S])   # (2n+1, m)
        zpred = np.sum(self.Wm[:, None] * Z, axis=0)

        # Innovation cov Szz and cross-cov Pxz
        Szz = np.zeros((self.m, self.m))
        Pxz = np.zeros((self.n, self.m))
        for i in range(Z.shape[0]):
            dz = (Z[i] - zpred)[:, None]
            dx = (S[i] - self.x)[:, None]
            Szz += self.Wc[i] * (dz @ dz.T)
            Pxz += self.Wc[i] * (dx @ dz.T)
        Szz += self.R
        Szz = make_spd(Szz, self.eps_spd)

        # Kalman gain
        try:
            K = Pxz @ np.linalg.inv(Szz)
        except np.linalg.LinAlgError:
            # Regularize and retry
            Szz = make_spd(Szz + 1e-6 * np.eye(self.m), self.eps_spd)
            K = Pxz @ np.linalg.inv(Szz)

        # State update
        y = z - zpred
        x_new = self.x + K @ y

        # Covariance update (UKF form): P = P - K Szz K^T
        P_new = self.P - K @ Szz @ K.T

        # Enforce symmetry / SPD
        self.x = x_new
        self.P = make_spd(P_new, self.eps_spd)


# ---------------------------
# Node
# ---------------------------
class UKFLocalizerNode:
    def __init__(self):
        rospy.init_node("ukf_localizer_tdoa_fdoa")

        # --- Timing ---
        self.dt = float(rospy.get_param("~dt", 0.2))
        self.timer = rospy.Timer(rospy.Duration(self.dt), self._on_timer)

        # --- Source GT (from Gazebo model_states) ---
        self.source_model_name = rospy.get_param("~source_model_name", "mobile_acoustic_source")
        # Fallback initial until model_states arrives
        self.source_gt = np.array([
            float(rospy.get_param('~source_x', 0.6)),
            float(rospy.get_param('~source_y', -2.8)),
            float(rospy.get_param('~source_z', -59.1)),
        ], dtype=float)
        self.have_source_gt = False

        # --- Fixed hydrophone (r1, v1) ---
        self.r1 = np.array([
            float(rospy.get_param("~static_x", 0.0)),
            float(rospy.get_param("~static_y", 0.0)),
            float(rospy.get_param("~static_z", -59.06)),
        ], dtype=float)
        self.v1 = np.array([
            float(rospy.get_param("~static_vx", 0.0)),
            float(rospy.get_param("~static_vy", 0.0)),
            float(rospy.get_param("~static_vz", 0.0)),
        ], dtype=float)

        # --- Mobile hydrophone offset on ROV base_link ---
        self.offset_b = np.array(rospy.get_param("~rov_hydro_offset_xyz", [0., 0., 0.]), dtype=float)
        self.base_link_name = rospy.get_param("~base_link_name", "nemesys::base_link")

        # Topics
        self.topic_rd  = rospy.get_param("~topic_rd",  "/hydrophones/path_diff_m")
        self.topic_rrd = rospy.get_param("~topic_rrd", "/hydrophones/fdoa_mps")

        # Measurement noise
        sigma_r  = float(rospy.get_param("~sigma_r_m",   0.03))
        sigma_v  = float(rospy.get_param("~sigma_v_mps", 0.3))
        self.R = np.diag([sigma_r**2, sigma_v**2])

        # Process noise
        q_p  = float(rospy.get_param("~q_pos", 1e-7))
        q_v  = float(rospy.get_param("~q_vel", 1e-7))
        q_br = float(rospy.get_param("~q_br",  1e-9))
        q_bv = float(rospy.get_param("~q_bv",  1e-9))
        self.Q = np.diag([q_p, q_p, q_p,  q_v, q_v, q_v,  q_br, q_bv])

        # UKF config
        alpha = float(rospy.get_param("~ukf_alpha", 0.2))
        beta  = float(rospy.get_param("~ukf_beta",  2.0))
        kappa = float(rospy.get_param("~ukf_kappa", 0.0))
        cfg = UKFConfig(alpha=alpha, beta=beta, kappa=kappa)

        # Initial state & covariance
        x0 = np.zeros(8)
        x0[0:3] = np.array([
            float(rospy.get_param("~x0",  0.5)),
            float(rospy.get_param("~y0", -2.0)),
            float(rospy.get_param("~z0", -58.0)),
        ], dtype=float)
        x0[3:6] = np.array([
            float(rospy.get_param("~vx0", 0.1)),
            float(rospy.get_param("~vy0", 0.1)),
            float(rospy.get_param("~vz0", 0.0)),
        ], dtype=float)
        x0[6] = float(rospy.get_param("~br0", 0.0))
        x0[7] = float(rospy.get_param("~bv0", 0.0))

        P0 = np.diag([
            float(rospy.get_param("~P0_pos",  10.0))**2,
            float(rospy.get_param("~P0_pos",  10.0))**2,
            float(rospy.get_param("~P0_pos",  0.2))**2,
            float(rospy.get_param("~P0_vel",   0.2))**2,
            float(rospy.get_param("~P0_vel",   0.2))**2,
            float(rospy.get_param("~P0_vel",   0.005))**2,
            float(rospy.get_param("~P0_br",    0.05))**2,
            float(rospy.get_param("~P0_bv",    0.2))**2
        ])

        self.ukf = UKF(
            n=8, m=2,
            fx=self._fx,
            hx=self._hx,
            Q=self.Q, R=make_spd(self.R, 1e-12),
            x0=x0, P0=P0, cfg=cfg, dt=self.dt, eps_spd=1e-9
        )

        # Latest measurement & receiver kinematics
        self.latest_rd  = None
        self.latest_rrd = None
        self.r2 = None
        self.v2 = None

        # ROS IO
        rospy.Subscriber(self.topic_rd,  Float32,    self._rd_cb,    queue_size=20)
        rospy.Subscriber(self.topic_rrd, Float32,    self._rrd_cb,   queue_size=20)
        rospy.Subscriber("/gazebo/link_states",  LinkStates,  self._link_cb,  queue_size=20)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_cb, queue_size=20)

        self.est_pub = rospy.Publisher("/acoustic_source_estimate", PoseStamped, queue_size=10)
        rospy.loginfo("[UKF-TDOA+FDOA] Ready. dt=%.3f s, tracking GT from model='%s'",
                      self.dt, self.source_model_name)

    # ---------------------------
    # UKF models
    # ---------------------------
    def _fx(self, x, u, dt):
        p = x[0:3]; v = x[3:6]; br = x[6]; bv = x[7]
        p2 = p + v * dt
        return np.array([p2[0], p2[1], p2[2], v[0], v[1], v[2], br, bv], dtype=float)

    def _hx(self, x, u):
        if u is None:
            return np.zeros(2, dtype=float)
        r2, v2 = u
        p = x[0:3]; v = x[3:6]; br = x[6]; bv = x[7]

        rho1 = safe_norm(p - self.r1)
        rho2 = safe_norm(p - r2)
        u1 = (p - self.r1) / rho1
        u2 = (p - r2) / rho2

        rd  = (rho2 - rho1) + br
        rrd = (u2 @ (v - v2)) - (u1 @ (v - self.v1)) + bv
        return np.array([rd, rrd], dtype=float)

    # ---------------------------
    # Callbacks
    # ---------------------------
    def _rd_cb(self, msg: Float32):
        self.latest_rd = float(msg.data)

    def _rrd_cb(self, msg: Float32):
        self.latest_rrd = float(msg.data)

    def _link_cb(self, msg: LinkStates):
        try:
            i = msg.name.index(self.base_link_name)
        except ValueError:
            return
        p = msg.pose[i].position
        q = msg.pose[i].orientation
        v = msg.twist[i].linear
        w = msg.twist[i].angular

        r_base = np.array([p.x, p.y, p.z], dtype=float)
        q_xyzw = np.array([q.x, q.y, q.z, q.w], dtype=float)
        R_wb = rotmat_from_quat_xyzw(q_xyzw)
        v_base = np.array([v.x, v.y, v.z], dtype=float)
        w_base = np.array([w.x, w.y, w.z], dtype=float)

        o_w = R_wb.dot(self.offset_b)
        self.r2 = r_base + o_w
        self.v2 = v_base + np.cross(w_base, o_w)

    def _model_cb(self, msg: ModelStates):
        """Update ground-truth source position from Gazebo model_states."""
        try:
            j = msg.name.index(self.source_model_name)
        except ValueError:
            rospy.logwarn_throttle(5.0, "[UKF-TDOA+FDOA] Source model '%s' not in /gazebo/model_states yet...",
                                   self.source_model_name)
            return
        p = msg.pose[j].position
        self.source_gt = np.array([p.x, p.y, p.z], dtype=float)
        self.have_source_gt = True

    # ---------------------------
    # Main loop (predict/update)
    # ---------------------------
    def _on_timer(self, _evt):
        # Predict always
        self.ukf.predict()

        # We need a valid measurement and kinematics
        if (self.latest_rd is None) or (self.latest_rrd is None) or (self.r2 is None) or (self.v2 is None):
            return

        # Optional sanity clamp to avoid blowing up due to a rogue sample
        rd  = float(np.clip(self.latest_rd,  -1e3, 1e3))
        rrd = float(np.clip(self.latest_rrd, -1e3, 1e3))
        z = np.array([rd, rrd], dtype=float)

        self.ukf.update(z, u=(self.r2, self.v2))

        # --- Clip the position estimate ---
        p = self.ukf.x[0:3]
        p[0] = np.clip(p[0], -5.0, 5.0)     # X bound
        p[1] = np.clip(p[1], -5.0, 5.0)     # Y bound
        p[2] = np.clip(p[2], -59.9, -58.0)  # Z bound
        self.ukf.x[0:3] = p

        # Publish estimate
        p = self.ukf.x[0:3]
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(p[0])
        msg.pose.position.y = float(p[1])
        msg.pose.position.z = float(p[2])
        msg.pose.orientation.w = 1.0
        self.est_pub.publish(msg)

        if not self.have_source_gt:
            rospy.logwarn_throttle(5.0, "[UKF-TDOA+FDOA] Waiting for GT from model '%s'...",
                                   self.source_model_name)

        rospy.loginfo_throttle(
            5.0,
            f"[UKF-TDOA+FDOA] Pos est: x={p[0]:.3f}, y={p[1]:.3f}, z={p[2]:.3f} | "
            f"True: {self.source_gt[0]:.3f}, {self.source_gt[1]:.3f}, {self.source_gt[2]:.3f}"
        )

# ---------------------------
# main
# ---------------------------
if __name__ == "__main__":
    try:
        UKFLocalizerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass